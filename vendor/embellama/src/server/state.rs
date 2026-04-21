// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Application state for the HTTP server
//!
//! This module defines the shared state that is passed to HTTP handlers.
//! The state must be Send + Sync + Clone for use with Axum.

use crate::server::dispatcher::Dispatcher;
use crate::{EmbeddingEngine, EngineConfig, ModelConfig};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Number of worker threads
    pub worker_count: usize,
    /// Maximum pending requests per worker
    pub queue_size: usize,
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Request timeout duration
    pub request_timeout: std::time::Duration,
    /// Engine configuration (includes model configuration with `n_seq_max`, pooling, etc.)
    pub engine_config: EngineConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Create a minimal default engine config
        // Note: This will fail validation as model_path is required
        // Users must provide a proper EngineConfig via builder
        let engine_config = EngineConfig::default();

        Self {
            worker_count: num_cpus::get(),
            queue_size: 100,
            host: "127.0.0.1".to_string(),
            port: 8080,
            request_timeout: std::time::Duration::from_secs(60),
            engine_config,
        }
    }
}

impl ServerConfig {
    /// Create a new builder for server configuration
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::default()
    }
}

/// Builder for `ServerConfig`
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    worker_count: Option<usize>,
    queue_size: Option<usize>,
    host: Option<String>,
    port: Option<u16>,
    request_timeout: Option<std::time::Duration>,
    engine_config: Option<EngineConfig>,
}

impl ServerConfigBuilder {
    /// Set the engine configuration
    #[must_use]
    pub fn engine_config(mut self, config: EngineConfig) -> Self {
        self.engine_config = Some(config);
        self
    }

    /// Convenience method: Set model path (creates `ModelConfig` → `EngineConfig`)
    ///
    /// This is a convenience method for simple cases. For more control,
    /// build an `EngineConfig` separately and use `.engine_config()`.
    #[must_use]
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        let path_str = path.into();

        // Extract model name from path (filename without extension)
        let model_name = Path::new(&path_str)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string();

        // Build a basic ModelConfig → EngineConfig
        if let Ok(model_config) = ModelConfig::builder()
            .with_model_path(path_str)
            .with_model_name(&model_name)
            .build()
            && let Ok(engine_config) = EngineConfig::builder()
                .with_model_config(model_config)
                .build()
        {
            self.engine_config = Some(engine_config);
        }

        self
    }

    /// Convenience method: Set model name (updates existing `EngineConfig`'s model name)
    #[must_use]
    pub fn model_name(self, _name: impl Into<String>) -> Self {
        // Deprecated pattern - users should build EngineConfig separately
        // For backward compatibility, this is a no-op
        self
    }

    /// Set the number of worker threads
    #[must_use]
    pub fn worker_count(mut self, count: usize) -> Self {
        self.worker_count = Some(count);
        self
    }

    /// Set the queue size per worker
    #[must_use]
    pub fn queue_size(mut self, size: usize) -> Self {
        self.queue_size = Some(size);
        self
    }

    /// Set the server host address
    #[must_use]
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set the server port
    #[must_use]
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set the request timeout duration
    #[must_use]
    pub fn request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Build the `ServerConfig`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The `engine_config` is not provided
    /// - Worker count is 0 or greater than 128
    /// - Queue size is 0 or greater than 10000
    pub fn build(self) -> crate::Result<ServerConfig> {
        let default = ServerConfig::default();

        let engine_config = self
            .engine_config
            .ok_or_else(|| crate::Error::ConfigurationError {
                message: "Engine configuration is required. Use .engine_config() or .model_path()"
                    .to_string(),
            })?;

        // Validate the engine config
        engine_config.validate()?;

        let worker_count = self.worker_count.unwrap_or(default.worker_count);
        let queue_size = self.queue_size.unwrap_or(default.queue_size);

        // Validate worker count
        if worker_count == 0 {
            return Err(crate::Error::ConfigurationError {
                message: "Worker count must be at least 1".to_string(),
            });
        }
        if worker_count > 128 {
            return Err(crate::Error::ConfigurationError {
                message: "Worker count cannot exceed 128".to_string(),
            });
        }

        // Validate queue size
        if queue_size == 0 {
            return Err(crate::Error::ConfigurationError {
                message: "Queue size must be at least 1".to_string(),
            });
        }
        if queue_size > 10000 {
            return Err(crate::Error::ConfigurationError {
                message: "Queue size cannot exceed 10000".to_string(),
            });
        }

        Ok(ServerConfig {
            worker_count,
            queue_size,
            host: self.host.unwrap_or(default.host),
            port: self.port.unwrap_or(default.port),
            request_timeout: self.request_timeout.unwrap_or(default.request_timeout),
            engine_config,
        })
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Request dispatcher
    pub dispatcher: Arc<Dispatcher>,
    /// Server configuration
    pub config: Arc<ServerConfig>,
    /// Embedding engine instance
    pub engine: Arc<Mutex<EmbeddingEngine>>,
}

impl AppState {
    /// Create a new application state
    ///
    /// # Arguments
    /// * `config` - Server configuration
    ///
    /// # Returns
    /// A new `AppState` instance or an error
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Engine configuration validation fails
    /// - Engine initialization fails
    pub fn new(config: ServerConfig) -> crate::Result<Self> {
        // Use the engine config directly from ServerConfig
        let engine = EmbeddingEngine::get_or_init(config.engine_config.clone())?;

        // Extract n_seq_max from the model configuration
        let n_seq_max = config.engine_config.model_config.n_seq_max.unwrap_or(8);

        info!(
            "Creating AppState with model '{}' (n_seq_max: {})",
            config.engine_config.model_config.model_name, n_seq_max
        );

        // Create the dispatcher with inference worker
        // Use n_seq_max from model config as the batch size to match model's parallel processing capacity
        let dispatcher = Dispatcher::new(n_seq_max as usize, config.queue_size);

        Ok(Self {
            dispatcher: Arc::new(dispatcher),
            config: Arc::new(config),
            engine,
        })
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        &self.config.engine_config.model_config.model_name
    }

    /// Check if the server is ready
    pub fn is_ready(&self) -> bool {
        self.dispatcher.is_ready()
    }
}
