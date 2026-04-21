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

//! Embedding engine module for the embellama library.
//!
//! This module provides the main `EmbeddingEngine` struct which serves as
//! the primary interface for the library, managing model lifecycle and
//! providing high-level embedding generation APIs.

use crate::batch::BatchProcessorBuilder;
use crate::cache::embedding_cache::EmbeddingCache;
use crate::cache::prefix_cache::PrefixCache;
use crate::cache::token_cache::TokenCache;
use crate::cache::{CacheStats, CacheStore};
use crate::config::{EngineConfig, NormalizationMode, TruncateTokens};
use crate::error::{Error, Result};
use crate::model::EmbeddingModel;
use llama_cpp_2::llama_backend::LlamaBackend;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info, instrument};

// Global singleton instance of the engine
static INSTANCE: RwLock<Option<Arc<Mutex<EmbeddingEngine>>>> = RwLock::new(None);

// Lock to protect singleton initialization
static INIT_LOCK: Mutex<()> = Mutex::new(());

// Global singleton backend instance
static BACKEND: OnceLock<Arc<Mutex<LlamaBackend>>> = OnceLock::new();

// Thread-local storage for models due to !Send constraint of LlamaContext
thread_local! {
    static THREAD_MODELS: RefCell<HashMap<String, EmbeddingModel>> = RefCell::new(HashMap::new());
}

// Thread-local reference to the token cache for fast access
thread_local! {
    static THREAD_TOKEN_CACHE: RefCell<Option<Arc<TokenCache>>> = const { RefCell::new(None) };
}

/// The main entry point for the embellama library.
///
/// `EmbeddingEngine` manages the lifecycle of embedding models and provides
/// a high-level API for generating embeddings. It supports loading multiple
/// models and switching between them.
///
/// # Important
///
/// Due to the `!Send` constraint of `LlamaContext`, each thread maintains
/// its own copy of loaded models. This means:
/// - Models are loaded per-thread when first accessed
/// - Memory usage scales with number of threads × number of models
/// - Model loading may happen multiple times across threads
///
/// # Example
///
/// ```ignore
/// use embellama::{EmbeddingEngine, EngineConfig};
///
/// let config = EngineConfig::builder()
///     .with_model_path("path/to/model.gguf")
///     .with_model_name("my-model")
///     .build()?;
///
/// let engine = EmbeddingEngine::new(config)?;
/// let embedding = engine.embed("my-model", "Hello, world!")?;
/// ```
pub struct EmbeddingEngine {
    /// Shared reference to the llama backend instance
    backend: Arc<Mutex<LlamaBackend>>,
    /// Registry of model configurations
    model_configs: Arc<RwLock<HashMap<String, EngineConfig>>>,
    /// Default model name if none specified
    default_model: Option<String>,
    /// Embedding cache for performance optimization
    embedding_cache: Option<Arc<EmbeddingCache>>,
    /// Token cache for caching tokenization results
    token_cache: Option<Arc<TokenCache>>,
    /// Prefix cache for KV cache optimization
    prefix_cache: Option<Arc<PrefixCache>>,
}

impl EmbeddingEngine {
    /// Gets or creates the singleton `LlamaBackend` instance.
    ///
    /// This ensures only one backend is created per process, avoiding the
    /// `BackendAlreadyInitialized` error when creating multiple engines.
    fn get_or_create_backend() -> Result<Arc<Mutex<LlamaBackend>>> {
        if let Some(backend) = BACKEND.get() {
            return Ok(Arc::clone(backend));
        }

        // Initialize the backend for the first time
        let mut backend = LlamaBackend::init().map_err(|e| {
            let error_str = format!("{e}");
            if error_str.contains("BackendAlreadyInitialized") {
                Error::ConfigurationError {
                    message: "LlamaBackend already initialized. This is an internal error."
                        .to_string(),
                }
            } else {
                Error::ModelInitError {
                    message: "Failed to initialize llama backend".to_string(),
                    source: Some(anyhow::anyhow!("{e}")),
                }
            }
        })?;
        backend.void_logs();

        let backend_arc = Arc::new(Mutex::new(backend));
        // Try to set it, but if another thread beat us to it, use theirs
        match BACKEND.set(Arc::clone(&backend_arc)) {
            Ok(()) => Ok(backend_arc),
            Err(_) => Ok(Arc::clone(BACKEND.get().unwrap())),
        }
    }

    /// Gets or initializes the singleton embedding engine with the given configuration.
    ///
    /// If the engine is already initialized, returns the existing instance.
    /// The configuration is only used for the first initialization.
    ///
    /// # Arguments
    ///
    /// * `config` - The engine configuration (used only on first call)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the engine instance or an error.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The configuration is invalid (on first initialization)
    /// - Model loading fails (on first initialization)
    /// - The initialization lock is poisoned
    #[instrument(skip(config), fields(model_name = %config.model_config.model_name))]
    pub fn get_or_init(config: EngineConfig) -> Result<Arc<Mutex<Self>>> {
        // Fast path: check if already initialized
        {
            let instance_guard = INSTANCE.read();
            if let Some(ref instance) = *instance_guard {
                debug!("Returning existing engine instance");
                return Ok(Arc::clone(instance));
            }
        }

        // Slow path: initialize the singleton
        let _lock = INIT_LOCK.lock().map_err(|_| Error::LockPoisoned)?;

        // Double-check after acquiring lock
        {
            let instance_guard = INSTANCE.read();
            if let Some(ref instance) = *instance_guard {
                debug!("Returning existing engine instance (after lock)");
                return Ok(Arc::clone(instance));
            }
        }

        // Create new instance
        info!("Initializing singleton embedding engine");
        let engine = Self::new_internal(config)?;
        let arc_engine = Arc::new(Mutex::new(engine));

        // Store the instance
        {
            let mut instance_guard = INSTANCE.write();
            *instance_guard = Some(Arc::clone(&arc_engine));
        }

        Ok(arc_engine)
    }

    /// Gets the existing engine instance if it has been initialized.
    ///
    /// # Returns
    ///
    /// Returns `Some(engine)` if initialized, `None` otherwise.
    pub fn instance() -> Option<Arc<Mutex<Self>>> {
        let instance_guard = INSTANCE.read();
        instance_guard.as_ref().map(Arc::clone)
    }

    /// Resets the singleton instance (test-only).
    ///
    /// This method is only available in test builds and should be called
    /// at the start of tests that need a fresh engine state.
    ///
    /// # Safety
    ///
    /// This method should only be called when no other code is using the engine.
    /// Tests using this must be marked with `#[serial]` to prevent parallel execution.
    ///
    /// # Panics
    ///
    /// Panics if the mutex lock cannot be acquired
    #[cfg(test)]
    pub fn reset() {
        let _lock = INIT_LOCK.lock().unwrap();

        // Clear thread-local models first
        THREAD_MODELS.with(|models| {
            models.borrow_mut().clear();
        });

        // Take and drop the instance to ensure backend is dropped
        let mut instance_guard = INSTANCE.write();
        if let Some(instance) = instance_guard.take() {
            // Check if we're the only reference
            if Arc::strong_count(&instance) > 1 {
                // Other references exist - this is likely a test error
                // Put it back and panic
                *instance_guard = Some(instance);
                panic!(
                    "Cannot reset engine: other references exist. Ensure tests are marked with #[serial]"
                );
            }
            // Explicitly drop the instance (and its backend)
            drop(instance);
            debug!("Dropped engine instance and backend");
        }

        // instance_guard is now None
        info!("Engine singleton reset - backend dropped");
    }

    /// Convenience method for tests to get a fresh instance.
    ///
    /// Resets the singleton and initializes with the given config.
    ///
    /// # Errors
    ///
    /// Returns an error if engine creation fails
    #[cfg(test)]
    pub fn fresh_instance(config: EngineConfig) -> Result<Arc<Mutex<Self>>> {
        Self::reset();
        Self::get_or_init(config)
    }

    /// Internal method to create a new engine instance.
    ///
    /// This is the actual implementation, separated from the singleton logic.
    fn new_internal(config: EngineConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        let model_name = config.model_config.model_name.clone();
        info!("Initializing embedding engine with model: {}", model_name);

        // Get or create the shared backend
        let backend = Self::get_or_create_backend()?;
        info!("Llama backend ready");

        // Initialize caches if enabled
        let (embedding_cache, token_cache, prefix_cache) = if let Some(cache_config) = &config.cache
        {
            if cache_config.enabled {
                info!(
                    "Initializing embedding cache with {} max entries",
                    cache_config.embedding_cache_size
                );
                let embedding_cache = Some(Arc::new(EmbeddingCache::new(
                    cache_config.embedding_cache_size as u64,
                    cache_config.ttl_seconds,
                )));

                info!(
                    "Initializing token cache with {} max entries",
                    cache_config.token_cache_size
                );
                let token_cache = Some(Arc::new(TokenCache::with_ttl(
                    cache_config.token_cache_size,
                    Some(cache_config.ttl_seconds),
                )));

                // Initialize prefix cache if enabled
                let prefix_cache = if cache_config.prefix_cache_enabled {
                    info!(
                        "Initializing prefix cache with {} max sessions",
                        cache_config.prefix_cache_size
                    );
                    Some(Arc::new(
                        PrefixCache::new(
                            cache_config.prefix_cache_size,
                            cache_config.ttl_seconds,
                            5,    // Frequency threshold for automatic caching
                            None, // No persistent storage for now
                        )
                        .map_err(|e| Error::ConfigurationError {
                            message: format!("Failed to create prefix cache: {e}"),
                        })?,
                    ))
                } else {
                    None
                };

                (embedding_cache, token_cache, prefix_cache)
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        // Create the engine with the initial model config
        let mut model_configs = HashMap::new();
        model_configs.insert(model_name.clone(), config);

        let engine = Self {
            backend,
            model_configs: Arc::new(RwLock::new(model_configs)),
            default_model: Some(model_name.clone()),
            embedding_cache,
            token_cache: token_cache.clone(),
            prefix_cache,
        };

        // Store token cache reference in thread-local storage
        if let Some(ref cache) = token_cache {
            THREAD_TOKEN_CACHE.with(|tc| {
                *tc.borrow_mut() = Some(Arc::clone(cache));
            });
        }

        // Load the model in the current thread
        engine.ensure_model_loaded(&model_name)?;

        info!("Embedding engine initialized successfully");
        Ok(engine)
    }

    /// Creates a new embedding engine with the given configuration.
    ///
    /// **Note**: This now uses the singleton pattern internally. Use `get_or_init()`
    /// for explicit singleton access.
    ///
    /// # Arguments
    ///
    /// * `config` - The engine configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the engine or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Use the internal method directly for backward compatibility
        // This allows tests to create instances without singleton
        Self::new_internal(config)
    }

    /// Loads a model with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The model configuration
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - A model with the same name is already loaded
    /// - Model loading fails
    #[instrument(skip(self, config), fields(model_name = %config.model_config.model_name))]
    pub fn load_model(&mut self, config: EngineConfig) -> Result<()> {
        // Validate configuration
        config.validate()?;

        let model_name = config.model_config.model_name.clone();

        // Check if model already exists
        {
            let configs = self.model_configs.read();
            if configs.contains_key(&model_name) {
                return Err(Error::ConfigurationError {
                    message: format!("Model '{model_name}' is already loaded"),
                });
            }
        }

        // Add configuration to registry
        {
            let mut configs = self.model_configs.write();
            configs.insert(model_name.clone(), config);
        }

        // Set as default if it's the first model
        if self.default_model.is_none() {
            self.default_model = Some(model_name.clone());
        }

        info!("Model '{}' configuration added to registry", model_name);
        Ok(())
    }

    /// Unregisters a model from the registry, preventing future loads.
    ///
    /// This removes the model configuration from the registry but does not
    /// affect already-loaded model instances in threads.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to unregister
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found in the registry.
    #[instrument(skip(self))]
    pub fn unregister_model(&mut self, model_name: &str) -> Result<()> {
        // Remove from config registry
        {
            let mut configs = self.model_configs.write();
            if !configs.contains_key(model_name) {
                return Err(Error::ModelNotFound {
                    name: model_name.to_string(),
                });
            }
            configs.remove(model_name);
        }

        // Update default model if needed
        if self.default_model.as_ref() == Some(&model_name.to_string()) {
            let configs = self.model_configs.read();
            self.default_model = configs.keys().next().cloned();
        }

        info!("Model '{}' unregistered from config registry", model_name);
        Ok(())
    }

    /// Drops a model from the current thread's cache.
    ///
    /// This removes the model instance from the current thread but keeps
    /// its configuration in the registry, allowing it to be reloaded later.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to drop from thread
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not registered.
    #[instrument(skip(self))]
    pub fn drop_model_from_thread(&self, model_name: &str) -> Result<()> {
        // First check if model is registered
        {
            let configs = self.model_configs.read();
            if !configs.contains_key(model_name) {
                return Err(Error::ModelNotFound {
                    name: model_name.to_string(),
                });
            }
        }

        // Remove from thread-local storage
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            if models.remove(model_name).is_some() {
                info!("Model '{}' dropped from current thread", model_name);
            } else {
                debug!("Model '{}' was not loaded in current thread", model_name);
            }
        });

        Ok(())
    }

    /// Unloads a model completely (unregisters and drops from thread).
    ///
    /// This is a convenience method that combines `unregister_model` and
    /// `drop_model_from_thread`. It maintains backward compatibility with
    /// the original `unload_model` behavior.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to unload
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    #[instrument(skip(self))]
    pub fn unload_model(&mut self, model_name: &str) -> Result<()> {
        // Drop from current thread first (while config still exists)
        self.drop_model_from_thread(model_name)?;

        // Then unregister from config
        self.unregister_model(model_name)?;

        info!("Model '{}' fully unloaded", model_name);
        Ok(())
    }

    /// Ensures a model is loaded in the current thread.
    ///
    /// This is an internal method that handles thread-local model loading.
    fn ensure_model_loaded(&self, model_name: &str) -> Result<()> {
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();

            // Check if model is already loaded in this thread
            if models.contains_key(model_name) {
                debug!("Model '{}' already loaded in current thread", model_name);
                return Ok(());
            }

            // Get configuration from registry
            let config = {
                let configs = self.model_configs.read();
                configs
                    .get(model_name)
                    .ok_or_else(|| Error::ModelNotFound {
                        name: model_name.to_string(),
                    })?
                    .clone()
            };

            info!("Loading model '{}' in current thread", model_name);

            // Use the model configuration from EngineConfig
            let backend_guard = self.backend.lock().map_err(|_| Error::LockPoisoned)?;
            let model = EmbeddingModel::new(&backend_guard, &config.model_config)?;
            drop(backend_guard); // Release lock as soon as we're done

            // Update the stored engine config with resolved pooling/normalization values
            // so that cache keys and other downstream reads reflect the actual model semantics.
            {
                let resolved = model.config();
                let mut configs = self.model_configs.write();
                if let Some(stored) = configs.get_mut(model_name) {
                    stored.model_config.pooling_strategy = resolved.pooling_strategy;
                    stored.model_config.normalization_mode = resolved.normalization_mode;
                }
            }

            // Store in thread-local map
            models.insert(model_name.to_string(), model);

            info!(
                "Model '{}' loaded successfully in current thread",
                model_name
            );
            Ok(())
        })
    }

    /// Generates an embedding for a single text using the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `text` - The text to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model is not found
    /// - Embedding generation fails
    ///
    /// # Panics
    ///
    /// This function may panic if:
    /// - The model configuration is not found after validation (internal inconsistency)
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed(&self, model_name: Option<&str>, text: &str) -> Result<Vec<f32>> {
        // Determine which model to use
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        // Get model config (needed for both caching and truncation)
        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        // Get truncation setting from config
        let truncate = config
            .embedding
            .as_ref()
            .map_or(TruncateTokens::No, |e| e.truncate_tokens);

        // Check cache first if enabled
        if let Some(cache) = &self.embedding_cache {
            // Compute cache key
            let key = EmbeddingCache::compute_key(
                text,
                &model_name,
                config.model_config.pooling_strategy.unwrap_or_default(),
                config.model_config.normalization_mode.unwrap_or_default(),
            );

            // Check cache
            if let Some(embedding) = cache.get(&key) {
                debug!("Cache hit for text of length {}", text.len());
                return Ok(embedding);
            }
            debug!("Cache miss for text of length {}", text.len());
        }

        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;

        // Generate embedding using thread-local model with token cache
        let embedding = THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
                })?;

            // Check prefix cache if enabled
            if let Some(ref prefix_cache) = self.prefix_cache {
                // First tokenize to check for prefix matches
                let tokens = model.tokenize(text)?;
                let token_ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();

                // Try to find a matching prefix
                if let Some((_prefix_len, _session_data)) =
                    prefix_cache.find_prefix_session(text, &token_ids)
                {
                    debug!("Prefix cache hit for text of length {}", text.len());
                    // Use the prefix-aware embedding generation
                    // Pass the whole prefix_cache - the method will find the session internally
                    return THREAD_TOKEN_CACHE.with(|tc| {
                        let cache_ref = tc.borrow();
                        model.generate_embedding_with_prefix(
                            text,
                            Some(prefix_cache.as_ref()),
                            cache_ref.as_deref(),
                            truncate,
                        )
                    });
                }

                // Analyze for future caching opportunities
                // > TODO: Implement automatic prefix detection and registration
                // This would require tracking patterns over time
            }

            // Get thread-local token cache reference
            THREAD_TOKEN_CACHE.with(|tc| {
                let cache_ref = tc.borrow();
                if let Some(ref cache) = *cache_ref {
                    model.generate_embedding_cached(text, Some(cache.as_ref()), truncate)
                } else {
                    model.generate_embedding(text)
                }
            })
        })?;

        // Update cache with result if enabled
        if let Some(cache) = &self.embedding_cache {
            // Get model config again for cache key (already validated above)
            let config = self.model_configs.read();
            let config = config.get(&model_name).unwrap();

            let key = EmbeddingCache::compute_key(
                text,
                &model_name,
                config.model_config.pooling_strategy.unwrap_or_default(),
                config.model_config.normalization_mode.unwrap_or_default(),
            );

            cache.insert(key, embedding.clone());
            debug!("Cached embedding for text of length {}", text.len());
        }

        Ok(embedding)
    }

    /// Generates per-token (multi-vector) embeddings for a single text.
    ///
    /// Returns one embedding vector per token, suitable for ColBERT-style late
    /// interaction reranking. Each vector is individually normalized.
    ///
    /// This method does not use the embedding cache (since multi-vector results
    /// have a different shape than cached single-vector embeddings).
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `text` - The text to generate per-token embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, one per token.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found or embedding generation fails.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn embed_multi(&self, model_name: Option<&str>, text: &str) -> Result<Vec<Vec<f32>>> {
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        let truncate = config
            .embedding
            .as_ref()
            .map_or(TruncateTokens::No, |e| e.truncate_tokens);

        self.ensure_model_loaded(&model_name)?;

        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
                })?;

            THREAD_TOKEN_CACHE.with(|tc| {
                let cache_ref = tc.borrow();
                model.generate_multi_embedding(text, cache_ref.as_deref(), truncate)
            })
        })
    }

    /// Generates per-token (multi-vector) embeddings for a batch of texts.
    ///
    /// Returns one `Vec<Vec<f32>>` per input text — each containing one embedding
    /// vector per token. Suitable for ColBERT-style late interaction reranking.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `texts` - Texts to generate per-token embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of multi-vector embeddings, one per input text.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found or embedding generation fails.
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn embed_batch_multi(
        &self,
        model_name: Option<&str>,
        texts: &[&str],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        let truncate = config
            .embedding
            .as_ref()
            .map_or(TruncateTokens::No, |e| e.truncate_tokens);

        self.ensure_model_loaded(&model_name)?;

        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
                })?;

            // Tokenize all texts
            let token_sequences: Vec<Vec<_>> = texts
                .iter()
                .map(|text| model.tokenize(text))
                .collect::<Result<Vec<_>>>()?;

            model.process_batch_tokens_multi(&token_sequences, truncate)
        })
    }

    /// Generates embeddings for a batch of texts using the specified model.
    ///
    /// This method processes multiple texts efficiently using parallel processing
    /// for tokenization and post-processing while respecting the single-threaded
    /// constraint of model inference.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to use (or None for default)
    /// * `texts` - A vector of texts to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model is not found
    /// - Any embedding generation fails
    ///
    /// # Panics
    ///
    /// This function may panic if:
    /// - A cached result is unexpectedly None after successful cache population
    #[instrument(skip(self, texts), fields(batch_size = texts.len()))]
    pub fn embed_batch(&self, model_name: Option<&str>, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Determine which model to use
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        // Get model configuration
        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        // Get truncation setting from config
        let truncate = config
            .embedding
            .as_ref()
            .map_or(TruncateTokens::No, |e| e.truncate_tokens);

        // If cache is enabled, check for cached embeddings
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        if let Some(cache) = &self.embedding_cache {
            for (i, text) in texts.iter().enumerate() {
                let key = EmbeddingCache::compute_key(
                    text,
                    &model_name,
                    config.model_config.pooling_strategy.unwrap_or_default(),
                    config.model_config.normalization_mode.unwrap_or_default(),
                );

                if let Some(embedding) = cache.get(&key) {
                    debug!("Batch cache hit for text {} of length {}", i, text.len());
                    results.push(Some(embedding));
                } else {
                    debug!("Batch cache miss for text {} of length {}", i, text.len());
                    results.push(None);
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }

            // If all are cached, return early
            if uncached_texts.is_empty() {
                debug!("All {} texts found in cache", texts.len());
                return Ok(results.into_iter().map(|r| r.unwrap()).collect());
            }

            debug!(
                "Processing {} uncached texts out of {}",
                uncached_texts.len(),
                texts.len()
            );
        } else {
            // No cache, process all texts
            uncached_texts = texts.to_vec();
        }

        // Ensure model is loaded in current thread
        self.ensure_model_loaded(&model_name)?;

        // Create batch processor with model configuration
        let batch_processor = BatchProcessorBuilder::default()
            .with_max_batch_size(64) // Default batch size
            .with_normalization(
                config.model_config.normalization_mode.unwrap_or_default()
                    != NormalizationMode::None,
            )
            .with_pooling_strategy(config.model_config.pooling_strategy.unwrap_or_default())
            .build();

        // Process uncached texts using the BatchProcessor
        let new_embeddings = THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
                })?;

            batch_processor.process_batch(model, &uncached_texts, truncate)
        })?;

        // Update cache and results
        if let Some(cache) = &self.embedding_cache {
            for (idx, embedding) in new_embeddings.into_iter().enumerate() {
                let text = uncached_texts[idx];
                let key = EmbeddingCache::compute_key(
                    text,
                    &model_name,
                    config.model_config.pooling_strategy.unwrap_or_default(),
                    config.model_config.normalization_mode.unwrap_or_default(),
                );

                cache.insert(key, embedding.clone());

                // Update results at the correct position
                let original_idx = uncached_indices[idx];
                results[original_idx] = Some(embedding);
            }

            // Convert results to final output
            Ok(results.into_iter().map(|r| r.unwrap()).collect())
        } else {
            // No cache, return new embeddings directly
            Ok(new_embeddings)
        }
    }

    /// Reranks documents against a query using a cross-encoder reranking model.
    ///
    /// The model must be configured with `PoolingStrategy::Rank`. Each document
    /// is scored against the query, and results are returned sorted by relevance
    /// (descending).
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the reranking model (or None for default)
    /// * `query` - The query text
    /// * `documents` - Documents to rerank
    /// * `top_n` - Optional limit on number of results returned
    /// * `normalize` - Whether to apply sigmoid normalization to \[0, 1\]
    ///
    /// # Returns
    ///
    /// Returns a vector of `RerankResult` sorted by relevance score (descending).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found, not configured for reranking,
    /// or inference fails.
    #[instrument(skip(self, query, documents), fields(query_len = query.len(), n_docs = documents.len()))]
    pub fn rerank(
        &self,
        model_name: Option<&str>,
        query: &str,
        documents: &[&str],
        top_n: Option<usize>,
        normalize: bool,
    ) -> Result<Vec<crate::config::RerankResult>> {
        let model_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        let config = self
            .model_configs
            .read()
            .get(&model_name)
            .ok_or_else(|| Error::ModelNotFound {
                name: model_name.clone(),
            })?
            .clone();

        let truncate = config
            .embedding
            .as_ref()
            .map_or(TruncateTokens::No, |e| e.truncate_tokens);

        self.ensure_model_loaded(&model_name)?;

        let raw_scores = THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();
            let model = models
                .get_mut(&model_name)
                .ok_or_else(|| Error::ModelNotFound {
                    name: model_name.clone(),
                })?;

            model.generate_rerank_scores_batch(query, documents, truncate)
        })?;

        let mut results: Vec<crate::config::RerankResult> = raw_scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| {
                let relevance_score = if normalize {
                    // Sigmoid normalization: 1 / (1 + e^(-x))
                    1.0 / (1.0 + (-score).exp())
                } else {
                    score
                };
                crate::config::RerankResult {
                    index,
                    relevance_score,
                }
            })
            .collect();

        // Reject NaN scores that would corrupt sort order
        for r in &results {
            if r.relevance_score.is_nan() {
                return Err(Error::EmbeddingGenerationError {
                    message: "Model produced NaN relevance score".to_string(),
                    source: None,
                });
            }
        }

        // Sort by relevance score descending (total_cmp provides well-defined ordering)
        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));

        // Apply top_n filtering
        if let Some(n) = top_n {
            results.truncate(n);
        }

        Ok(results)
    }

    /// Lists all currently loaded models.
    ///
    /// Note: This returns models registered in the engine, not necessarily
    /// loaded in the current thread.
    ///
    /// # Returns
    ///
    /// Returns a vector of model names.
    pub fn list_models(&self) -> Vec<String> {
        let configs = self.model_configs.read();
        configs.keys().cloned().collect()
    }

    /// Get model configurations with their metadata.
    ///
    /// Returns a vector of tuples containing (`model_name`, `context_size`).
    pub fn get_model_details(&self) -> Vec<(String, Option<u32>)> {
        let configs = self.model_configs.read();
        configs
            .iter()
            .map(|(name, config)| {
                // Get context_size from the model configuration
                let context_size = config
                    .model_config
                    .context_size
                    .or(config.model_config.n_ctx);
                (name.clone(), context_size)
            })
            .collect()
    }

    /// Gets cache statistics if caching is enabled.
    ///
    /// # Returns
    ///
    /// Returns cache statistics if caching is enabled, None otherwise.
    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.embedding_cache.as_ref().map(|cache| cache.stats())
    }

    /// Clears the embedding cache if enabled.
    ///
    /// This removes all cached embeddings and resets statistics.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.embedding_cache {
            cache.clear();
            info!("Embedding cache cleared");
        }
        if let Some(cache) = &self.token_cache {
            cache.clear();
            info!("Token cache cleared");
        }
        if let Some(cache) = &self.prefix_cache {
            cache.clear();
            info!("Prefix cache cleared");
        }
    }

    /// Warms up the cache by pre-computing embeddings for the given texts.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (None uses default)
    /// * `texts` - The texts to pre-compute embeddings for
    ///
    /// # Returns
    ///
    /// Returns Ok(()) if successful, or an error if pre-computation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails for any text.
    pub fn warm_cache(&self, model_name: Option<&str>, texts: &[&str]) -> Result<()> {
        if self.embedding_cache.is_none() {
            return Ok(()); // No-op if cache is disabled
        }

        info!("Warming cache with {} texts", texts.len());
        for text in texts {
            // This will compute and cache the embedding
            self.embed(model_name, text)?;
        }
        info!("Cache warmed successfully");
        Ok(())
    }

    /// Checks if caching is enabled.
    ///
    /// # Returns
    ///
    /// Returns true if caching is enabled, false otherwise.
    pub fn is_cache_enabled(&self) -> bool {
        self.embedding_cache.is_some()
    }

    /// Checks if a model is registered in the config registry.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is registered, false otherwise.
    pub fn is_model_registered(&self, model_name: &str) -> bool {
        let configs = self.model_configs.read();
        configs.contains_key(model_name)
    }

    /// Checks if a model is loaded in the current thread.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to check
    ///
    /// # Returns
    ///
    /// Returns true if the model is loaded in the current thread, false otherwise.
    pub fn is_model_loaded_in_thread(&self, model_name: &str) -> bool {
        THREAD_MODELS.with(|models| {
            let models = models.borrow();
            models.contains_key(model_name)
        })
    }

    /// Gets the default model name.
    ///
    /// # Returns
    ///
    /// Returns the default model name if set.
    pub fn default_model(&self) -> Option<String> {
        self.default_model.clone()
    }

    /// Sets the default model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to set as default
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not loaded.
    pub fn set_default_model(&mut self, model_name: &str) -> Result<()> {
        if !self.is_model_registered(model_name) {
            return Err(Error::ModelNotFound {
                name: model_name.to_string(),
            });
        }
        self.default_model = Some(model_name.to_string());
        Ok(())
    }

    /// Gets information about a loaded model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model
    ///
    /// # Returns
    ///
    /// Returns model information if the model is loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found
    pub fn model_info(&self, model_name: &str) -> Result<ModelInfo> {
        // Ensure model is loaded in current thread
        self.ensure_model_loaded(model_name)?;

        THREAD_MODELS.with(|models| {
            let models = models.borrow();
            let model = models.get(model_name).ok_or_else(|| Error::ModelNotFound {
                name: model_name.to_string(),
            })?;

            Ok(ModelInfo {
                name: model_name.to_string(),
                dimensions: model.embedding_dimensions(),
                max_tokens: model.max_sequence_length(),
                model_size: model.model_size(),
            })
        })
    }

    /// Warms up a model by generating a test embedding.
    ///
    /// This can be useful to ensure the model is fully loaded and ready
    /// before processing actual requests.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to warm up (or None for default)
    ///
    /// # Errors
    ///
    /// Returns an error if warmup fails.
    pub fn warmup_model(&self, model_name: Option<&str>) -> Result<()> {
        let resolved_name = model_name
            .map(std::string::ToString::to_string)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| Error::ConfigurationError {
                message: "No model specified and no default model set".to_string(),
            })?;

        // Ensure the model is loaded so the resolved config is available
        self.ensure_model_loaded(&resolved_name)?;

        // Check the resolved pooling strategy to pick the right warmup path
        let is_reranker = {
            let configs = self.model_configs.read();
            configs
                .get(&resolved_name)
                .and_then(|c| c.model_config.pooling_strategy)
                == Some(crate::config::PoolingStrategy::Rank)
        };

        if is_reranker {
            let _ = self.rerank(
                Some(&resolved_name),
                "warmup query",
                &["warmup document"],
                None,
                false,
            )?;
        } else {
            let _ = self.embed(
                Some(&resolved_name),
                "This is a warmup text for model initialization.",
            )?;
        }

        debug!("Model warmed up successfully");
        Ok(())
    }

    /// Performs explicit cleanup of all models in the current thread.
    ///
    /// With global tracing subscriber, this method is now optional but can
    /// still be useful for explicit resource management in tests.
    pub fn cleanup_thread_models(&self) {
        THREAD_MODELS.with(|models| {
            let mut models = models.borrow_mut();

            // Clear all models from the thread
            let count = models.len();
            models.clear();

            if count > 0 {
                info!("Cleared {} thread-local models", count);
            }
        });
    }

    // Prefix cache management methods

    /// Registers a text prefix for KV cache optimization.
    ///
    /// This method allows manual registration of common text prefixes for caching.
    /// The KV cache state will be saved and reused for texts that share this prefix.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The model to use (None uses default)
    /// * `prefix` - The prefix text to cache
    ///
    /// # Returns
    ///
    /// Returns Ok(()) if successful, or an error if registration fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No model is specified and no default model is set
    /// - The model is not found
    /// - Embedding generation for the prefix fails
    ///
    /// # Performance Notes
    ///
    /// - Only beneficial for prefixes >100 tokens due to session loading overhead
    /// - Best for code embeddings with common imports, template documents
    /// - Memory usage: ~100MB per cached prefix (typical models)
    pub fn register_prefix(&self, model_name: Option<&str>, prefix: &str) -> Result<()> {
        if let Some(cache) = &self.prefix_cache {
            // Determine which model to use
            let model_name = model_name
                .map(std::string::ToString::to_string)
                .or_else(|| self.default_model.clone())
                .ok_or_else(|| Error::ConfigurationError {
                    message: "No model specified and no default model set".to_string(),
                })?;

            // Ensure model is loaded in current thread
            self.ensure_model_loaded(&model_name)?;

            // Generate session state for the prefix
            let (tokens, session_data) = THREAD_MODELS.with(|models| {
                let mut models = models.borrow_mut();
                let model = models
                    .get_mut(&model_name)
                    .ok_or_else(|| Error::ModelNotFound {
                        name: model_name.clone(),
                    })?;

                // Tokenize the prefix
                let tokens = model.tokenize(prefix)?;

                // Generate embedding to populate KV cache
                model.generate_embedding(prefix)?;

                // Save the session state
                let session_data = model.save_session_state()?;

                Ok::<_, Error>((tokens, session_data))
            })?;

            // Register with prefix cache
            // Convert tokens to u32 for the cache API
            let token_ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();
            cache.register_prefix(prefix, &token_ids, session_data)?;

            info!(
                "Registered prefix of {} tokens for caching",
                token_ids.len()
            );
            Ok(())
        } else {
            Err(Error::ConfigurationError {
                message: "Prefix cache is not enabled".to_string(),
            })
        }
    }

    /// Gets prefix cache statistics.
    ///
    /// # Returns
    ///
    /// Returns statistics about the prefix cache if enabled, None otherwise.
    pub fn get_prefix_cache_stats(&self) -> Option<crate::cache::prefix_cache::PrefixCacheStats> {
        self.prefix_cache.as_ref().map(|cache| cache.stats())
    }

    /// Clears the prefix cache.
    ///
    /// This removes all cached prefix sessions and resets statistics.
    pub fn clear_prefix_cache(&self) {
        if let Some(cache) = &self.prefix_cache {
            cache.clear();
            info!("Prefix cache cleared");
        }
    }

    /// Lists all cached prefixes.
    ///
    /// # Returns
    ///
    /// Returns a vector of prefix information if cache is enabled, empty vector otherwise.
    pub fn list_cached_prefixes(&self) -> Vec<String> {
        if let Some(_cache) = &self.prefix_cache {
            // > TODO: Implement a method in PrefixCache to list cached prefixes
            // For now, return empty vector
            vec![]
        } else {
            vec![]
        }
    }

    /// Checks if prefix caching is enabled.
    ///
    /// # Returns
    ///
    /// Returns true if prefix caching is enabled, false otherwise.
    pub fn is_prefix_cache_enabled(&self) -> bool {
        self.prefix_cache.is_some()
    }
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum token count
    pub max_tokens: usize,
    /// Approximate model size in bytes (None if unable to calculate)
    pub model_size: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_config() -> EngineConfig {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model.gguf");
        fs::write(&model_path, b"dummy model file").unwrap();

        EngineConfig::builder()
            .with_model_path(model_path)
            .with_model_name("test-model")
            .build()
            .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        // This test would require a real GGUF model file
        // For now, we just test that the structure compiles correctly
    }

    #[test]
    fn test_model_listing() {
        // Test would require real model files
    }

    #[test]
    #[ignore = "Requires actual GGUF model file"]
    fn test_embedding_generation() {
        let config = create_test_config();
        let engine = EmbeddingEngine::new(config).unwrap();

        let text = "Hello, world!";
        let embedding = engine.embed(None, text).unwrap();

        assert!(!embedding.is_empty());
    }
}
