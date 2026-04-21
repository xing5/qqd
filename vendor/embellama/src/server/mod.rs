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

//! Server module for Embellama HTTP API
//!
//! This module provides an OpenAI-compatible REST API for the embedding engine.
//! It uses a worker pool architecture to handle the `!Send` constraint of LlamaContext.
//!
//! # Library Usage
//!
//! The server can be embedded into other applications as a library feature:
//!
//! ```no_run
//! use embellama::server::{
//!     AppState, ServerConfig, create_router, run_server,
//!     EngineConfig, ModelConfig
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Build engine configuration first
//!     let engine_config = EngineConfig::builder()
//!         .with_model_path("/path/to/model.gguf")
//!         .with_model_name("my-model")
//!         .build()?;
//!
//!     // Option 1: Use the convenient run_server function
//!     let config = ServerConfig::builder()
//!         .engine_config(engine_config.clone())
//!         .port(8080)
//!         .build()?;
//!
//!     run_server(config.clone()).await?;
//!
//!     // Option 2: Create router for custom integration
//!     let state = AppState::new(config)?;
//!     let router = create_router(state);
//!     // Add your own routes or middleware...
//!
//!     Ok(())
//! }
//! ```

pub mod api_types;
pub mod cache_handlers;
pub mod channel;
pub mod dispatcher;
pub mod handlers;
pub mod inference_worker;
pub mod middleware;
pub mod state;
pub mod worker;

// Phase 4 modules
#[cfg(feature = "server")]
pub mod backpressure;
#[cfg(feature = "server")]
pub mod metrics;
#[cfg(feature = "server")]
pub mod rate_limiter;

// Re-exports for convenience
pub use middleware::{
    API_KEY_HEADER, ApiKeyConfig, MAX_REQUEST_SIZE, REQUEST_ID_HEADER, authenticate_api_key,
    extract_request_id, inject_request_id, limit_request_size,
};
pub use state::{AppState, ServerConfig, ServerConfigBuilder};

// Re-export configuration types needed for building ServerConfig
pub use crate::{EngineConfig, ModelConfig, NormalizationMode, PoolingStrategy};

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
};
use serde_json::json;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use uuid::Uuid;

/// Model provider trait for custom model loading strategies
///
/// This trait allows users to implement custom model loading logic,
/// such as downloading models from cloud storage or managing multiple models.
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    /// Get the path to a model file by name
    ///
    /// # Arguments
    /// * `model_name` - Name of the model to load
    ///
    /// # Returns
    /// Path to the model file or an error
    async fn get_model_path(&self, model_name: &str) -> crate::Result<PathBuf>;

    /// List available models
    ///
    /// # Returns
    /// List of available model information with metadata extracted from GGUF files,
    /// including actual embedding dimensions, max tokens, and file size
    async fn list_models(&self) -> crate::Result<Vec<crate::ModelInfo>>;
}

/// Default file-based model provider
///
/// This provider serves a single GGUF model file and extracts metadata
/// such as embedding dimensions and context length directly from the file.
pub struct FileModelProvider {
    model_path: PathBuf,
    model_name: String,
}

impl FileModelProvider {
    /// Create a new file-based model provider
    pub fn new(model_path: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_name: model_name.into(),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for FileModelProvider {
    async fn get_model_path(&self, model_name: &str) -> crate::Result<PathBuf> {
        if model_name == self.model_name {
            Ok(self.model_path.clone())
        } else {
            Err(crate::Error::ModelNotFound {
                name: model_name.to_string(),
            })
        }
    }

    async fn list_models(&self) -> crate::Result<Vec<crate::ModelInfo>> {
        // Get file size
        let model_size = match std::fs::metadata(&self.model_path) {
            Ok(metadata) => Some(metadata.len()),
            Err(e) => {
                warn!(
                    "Failed to get file size for {}: {}",
                    self.model_path.display(),
                    e
                );
                None
            }
        };

        // Extract model metadata from GGUF file using shared function
        let (dimensions, max_tokens) = match crate::extract_gguf_metadata(&self.model_path) {
            Ok(metadata) => {
                info!(
                    "Successfully extracted metadata: dimensions={}, max_tokens={}",
                    metadata.embedding_dimensions, metadata.context_size
                );
                (metadata.embedding_dimensions, metadata.context_size)
            }
            Err(e) => {
                warn!(
                    "Failed to extract GGUF metadata from {}: {}",
                    self.model_path.display(),
                    e
                );
                // Return default values if metadata extraction fails
                (0, 512)
            }
        };

        Ok(vec![crate::ModelInfo {
            name: self.model_name.clone(),
            dimensions,
            max_tokens,
            model_size: model_size.and_then(|s| s.try_into().ok()),
        }])
    }
}

/// Create a router with the default configuration
///
/// This function creates an Axum router with all the standard routes and middleware
/// configured. Users can add additional routes or middleware before starting the server.
///
/// # Arguments
/// * `state` - Application state containing the embedding engine and configuration
///
/// # Returns
/// Configured Axum router ready to be served
///
/// # Example
/// ```no_run
/// # use embellama::server::{AppState, ServerConfig, create_router};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ServerConfig::builder()
///     .model_path("/path/to/model.gguf")
///     .build()?;
///
/// let state = AppState::new(config)?;
/// let router = create_router(state);
///
/// // Add custom routes
/// let router = router.route("/custom", axum::routing::get(|| async { "Custom route" }));
///
/// // Start server...
/// # Ok(())
/// # }
/// ```
pub fn create_router(state: AppState) -> Router<()> {
    // TODO: Consider adding commonly-needed middleware by default:
    // - inject_request_id: Adds request ID to all requests for tracing
    // - limit_request_size: Prevents excessive payload sizes
    // Users can still add these manually if needed
    // Health check handler
    async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
        if state.is_ready() {
            (
                StatusCode::OK,
                Json(json!({
                    "status": "healthy",
                    "model": state.model_name(),
                    "version": env!("CARGO_PKG_VERSION"),
                })),
            )
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "unhealthy",
                    "error": "Service not ready",
                })),
            )
        }
    }

    // Build router with routes and middleware
    Router::new()
        .route("/health", get(health_handler))
        // OpenAI-compatible API routes
        .route(
            "/v1/embeddings",
            axum::routing::post(handlers::embeddings_handler),
        )
        .route("/v1/models", get(handlers::list_models_handler))
        .route("/v1/rerank", axum::routing::post(handlers::rerank_handler))
        // Cache management endpoints
        .route("/cache/stats", get(cache_handlers::cache_stats_handler))
        .route(
            "/cache/clear",
            axum::routing::post(cache_handlers::cache_clear_handler),
        )
        .route(
            "/cache/warm",
            axum::routing::post(cache_handlers::cache_warm_handler),
        )
        // Prefix cache endpoints
        .route(
            "/v1/embeddings/prefix",
            axum::routing::post(cache_handlers::prefix_register_handler),
        )
        .route(
            "/v1/embeddings/prefix",
            get(cache_handlers::prefix_list_handler),
        )
        .route(
            "/v1/embeddings/prefix",
            axum::routing::delete(cache_handlers::prefix_clear_handler),
        )
        .route(
            "/v1/embeddings/prefix/stats",
            get(cache_handlers::prefix_stats_handler),
        )
        .layer(
            tower::ServiceBuilder::new()
                // Add tracing/logging middleware
                .layer(TraceLayer::new_for_http().make_span_with(
                    |request: &axum::http::Request<_>| {
                        let request_id = Uuid::new_v4();
                        tracing::info_span!(
                            "http_request",
                            request_id = %request_id,
                            method = %request.method(),
                            uri = %request.uri(),
                        )
                    },
                ))
                // Add CORS middleware
                .layer(CorsLayer::permissive()),
        )
        .with_state(state)
}

/// Run the server with the provided configuration
///
/// This is a convenience function that creates the application state,
/// builds the router, and starts the server with graceful shutdown handling.
///
/// # Arguments
/// * `config` - Server configuration
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
///
/// Returns an error if:
/// - Application state creation fails
/// - Server binding fails
/// - Server startup fails
///
/// # Example
/// ```no_run
/// # use embellama::server::{ServerConfig, run_server};
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ServerConfig::builder()
///     .model_path("/path/to/model.gguf")
///     .model_name("my-model")
///     .host("0.0.0.0")
///     .port(8080)
///     .build()?;
///
/// run_server(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_server(config: ServerConfig) -> crate::Result<()> {
    let model_path = &config.engine_config.model_config.model_path;
    let model_name = &config.engine_config.model_config.model_name;

    info!("Starting Embellama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Model: {} ({})", model_path.display(), model_name);
    info!(
        "Workers: {}, Queue size: {}",
        config.worker_count, config.queue_size
    );

    // Create application state
    let state = AppState::new(config.clone())?;

    // Build the router
    let app = create_router(state);

    // Create socket address
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Invalid address: {e}")))?;

    info!("Server listening on http://{}", addr);

    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Failed to bind to address: {e}")))?;

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("Server error: {e}")))?;

    info!("Server shutdown complete");
    Ok(())
}

/// Signal handler for graceful shutdown
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, shutting down");
        }
        () = terminate => {
            info!("Received terminate signal, shutting down");
        }
    }
}
