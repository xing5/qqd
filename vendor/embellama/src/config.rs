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

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

/// Configuration for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,

    /// Name identifier for the model
    pub model_name: String,

    /// Context size (number of tokens)
    pub n_ctx: Option<u32>,

    /// Batch size for prompt processing (max usable context per sequence)
    /// If not set, defaults to `context_size`
    /// This is the effective maximum context size for single sequences
    pub n_batch: Option<u32>,

    /// Micro-batch size for prompt processing (physical batch size)
    /// If not set, defaults to `n_batch`
    /// Must be <= `n_batch`
    pub n_ubatch: Option<u32>,

    /// Number of threads for CPU inference
    pub n_threads: Option<usize>,

    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: Option<u32>,

    /// Use memory mapping for model loading
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mmap: bool,

    /// Use memory locking to prevent swapping
    /// NOTE: This setting is not yet supported by llama-cpp-2 API
    pub use_mlock: bool,

    /// Normalization mode for embeddings
    pub normalization_mode: Option<NormalizationMode>,

    /// Pooling strategy for embeddings
    pub pooling_strategy: Option<PoolingStrategy>,

    /// Maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    pub n_seq_max: Option<u32>,

    /// Context size override (defaults to `n_ctx` if not specified)
    /// This controls the KV cache/attention cache size for better performance
    pub context_size: Option<u32>,

    /// Enable KV cache optimization for batch processing
    /// This includes batch reordering and similar-length grouping
    pub enable_kv_optimization: bool,
}

impl ModelConfig {
    /// Create a new configuration builder
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }

    /// Create configuration with backend auto-detection
    pub fn with_backend_detection() -> ModelConfigBuilder {
        let backend = crate::backend::detect_best_backend();
        let mut builder = ModelConfigBuilder::new();

        // Set GPU layers based on backend
        if let Some(gpu_layers) = backend.recommended_gpu_layers() {
            builder = builder.with_n_gpu_layers(gpu_layers);
        }

        builder
    }

    /// Validate the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model path is empty
    /// - The model file does not exist
    /// - The model file extension is not `.gguf`
    /// - Invalid thread count (0)
    /// - Invalid batch size (0)
    /// - Invalid `n_seq_max` (0 or > 64)
    pub fn validate(&self) -> Result<()> {
        if self.model_path.as_os_str().is_empty() {
            return Err(Error::config("Model path cannot be empty"));
        }

        if !self.model_path.exists() {
            return Err(Error::config(format!(
                "Model file does not exist: {}",
                self.model_path.display()
            )));
        }

        // Canonicalize the path to resolve symlinks and detect path traversal
        let canonical = self.model_path.canonicalize().map_err(|e| {
            Error::config(format!(
                "Cannot resolve model path '{}': {e}",
                self.model_path.display()
            ))
        })?;

        // Ensure the canonical path still ends with .gguf (catches symlink tricks)
        if canonical.extension().and_then(|e| e.to_str()) != Some("gguf") {
            return Err(Error::config(format!(
                "Model path resolves to non-GGUF file: {}",
                canonical.display()
            )));
        }

        if self.model_name.trim().is_empty() {
            return Err(Error::config("Model name cannot be empty"));
        }

        if self.n_ctx == Some(0) {
            return Err(Error::config("Context size must be greater than 0"));
        }

        if self.context_size == Some(0) {
            return Err(Error::config("Context size must be greater than 0"));
        }

        if self.n_batch == Some(0) {
            return Err(Error::config("Batch size must be greater than 0"));
        }

        if self.n_ubatch == Some(0) {
            return Err(Error::config("Micro-batch size must be greater than 0"));
        }

        // Validate n_batch <= context_size if both are set
        if let (Some(context_size), Some(n_batch)) = (self.context_size, self.n_batch) {
            if n_batch > context_size {
                return Err(Error::config(
                    "Batch size (n_batch) cannot exceed context size",
                ));
            }
        }

        // Validate n_ubatch <= n_batch if both are set
        if let (Some(n_batch), Some(n_ubatch)) = (self.n_batch, self.n_ubatch) {
            if n_ubatch > n_batch {
                return Err(Error::config(
                    "Micro-batch size (n_ubatch) cannot exceed batch size (n_batch)",
                ));
            }
        }

        if self.n_threads == Some(0) {
            return Err(Error::config("Number of threads must be greater than 0"));
        }

        if let Some(n_seq) = self.n_seq_max {
            if n_seq == 0 {
                return Err(Error::config("n_seq_max must be greater than 0"));
            }
            if n_seq > 64 {
                return Err(Error::config(
                    "n_seq_max cannot exceed 64 (llama.cpp limit)",
                ));
            }
        }

        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_name: String::new(),
            n_ctx: None,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_gpu_layers: None,
            use_mmap: true,
            use_mlock: false,
            normalization_mode: None,
            pooling_strategy: None,
            n_seq_max: None,
            context_size: None,
            enable_kv_optimization: false,
        }
    }
}

/// Builder for creating `ModelConfig` instances
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    /// Create a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }

    /// Set the model path
    #[must_use]
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name
    #[must_use]
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the context size
    #[must_use]
    pub fn with_n_ctx(mut self, ctx: u32) -> Self {
        self.config.n_ctx = Some(ctx);
        self
    }

    /// Set the batch size for prompt processing
    #[must_use]
    pub fn with_n_batch(mut self, batch: u32) -> Self {
        self.config.n_batch = Some(batch);
        self
    }

    /// Set the micro-batch size for prompt processing
    #[must_use]
    pub fn with_n_ubatch(mut self, ubatch: u32) -> Self {
        self.config.n_ubatch = Some(ubatch);
        self
    }

    /// Set the number of threads
    #[must_use]
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.n_threads = Some(threads);
        self
    }

    /// Set the number of GPU layers
    #[must_use]
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = Some(layers);
        self
    }

    /// Set whether to use memory mapping
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    /// Set the normalization mode for embeddings
    #[must_use]
    pub fn with_normalization_mode(mut self, mode: NormalizationMode) -> Self {
        self.config.normalization_mode = Some(mode);
        self
    }

    /// Set the pooling strategy
    #[must_use]
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling_strategy = Some(strategy);
        self
    }

    /// Set the maximum number of sequences for batch processing
    /// Default: 1, max: 64 (llama.cpp limit)
    #[must_use]
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.n_seq_max = Some(n_seq_max);
        self
    }

    /// Set the context size (KV cache size)
    #[must_use]
    pub fn with_context_size(mut self, context_size: u32) -> Self {
        self.config.context_size = Some(context_size);
        self
    }

    /// Enable or disable KV cache optimization
    #[must_use]
    pub fn with_kv_optimization(mut self, enable: bool) -> Self {
        self.config.enable_kv_optimization = enable;
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
    pub fn build(self) -> Result<ModelConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the embedding engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    /// Model configuration (contains path, name, and all model-specific settings)
    pub model_config: ModelConfig,

    /// Whether to use GPU acceleration if available
    pub use_gpu: bool,

    /// Batch size for processing
    pub batch_size: Option<usize>,

    /// Maximum number of tokens per input
    pub max_tokens: Option<usize>,

    /// Memory limit in MB (None for unlimited)
    pub memory_limit_mb: Option<usize>,

    /// Enable verbose logging
    pub verbose: bool,

    /// Seed for reproducibility (None for random)
    pub seed: Option<u32>,

    /// Temperature for sampling (not typically used for embeddings)
    pub temperature: Option<f32>,

    /// Cache configuration
    pub cache: Option<CacheConfig>,

    /// Embedding-specific configuration (truncation, etc.)
    pub embedding: Option<EmbeddingConfig>,
}

/// Pooling strategy for combining token embeddings
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pooling across all tokens
    #[default]
    Mean,
    /// Use only the \[CLS\] token embedding
    Cls,
    /// Max pooling across all tokens
    Max,
    /// Mean pooling with sqrt(length) normalization
    MeanSqrt,
    /// Use only the last token embedding (EOS token) - required for decoder models like Qwen
    Last,
    /// No pooling — return per-token embeddings for late interaction / ColBERT-style reranking.
    /// When this strategy is selected, use `embed_multi` / `embed_batch_multi` on the engine
    /// (or `generate_multi_embedding` on the model) to get `Vec<Vec<f32>>` output.
    /// The standard `embed` / `generate_embedding` methods will return an error.
    None,
    /// Rank pooling for cross-encoder reranking models.
    /// Returns a single scalar relevance score per query-document pair.
    /// Use `rerank` on the engine (or `generate_rerank_score` / `generate_rerank_scores_batch`
    /// on the model). The standard `embed` / `generate_embedding` methods will return an error.
    Rank,
}

/// A single reranking result for a query-document pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RerankResult {
    /// Original index of the document in the input list
    pub index: usize,
    /// Relevance score (higher = more relevant)
    pub relevance_score: f32,
}

/// Normalization mode for embedding vectors
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormalizationMode {
    /// No normalization (-1 in llama-server)
    None,
    /// Max absolute normalization scaled to int16 range (0 in llama-server)
    MaxAbs,
    /// L2 (Euclidean) normalization - default (2 in llama-server)
    #[default]
    L2,
    /// P-norm with custom exponent (N in llama-server)
    PNorm(i32),
}

/// Token truncation strategy for embedding inputs
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TruncateTokens {
    /// Don't truncate - let the model handle the tokens (may error if exceeds max)
    #[default]
    No,
    /// Truncate to model's `effective_max_tokens` (automatically adapts to model capacity)
    Yes,
    /// Truncate to specific limit (must be > 0 and <= `effective_max_tokens` at runtime)
    Limit(u32),
}

/// Configuration for caching system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Maximum number of entries in token cache
    pub token_cache_size: usize,
    /// Maximum number of entries in embedding cache
    pub embedding_cache_size: usize,
    /// Maximum memory usage in megabytes
    pub max_memory_mb: usize,
    /// Time-to-live for cache entries in seconds
    pub ttl_seconds: u64,
    /// Whether to enable metrics collection
    pub enable_metrics: bool,
    /// Whether prefix caching is enabled
    pub prefix_cache_enabled: bool,
    /// Maximum number of cached prefix sessions
    pub prefix_cache_size: usize,
    /// Minimum prefix length in tokens to consider for caching
    pub min_prefix_length: usize,
    /// Frequency threshold for automatic prefix caching
    pub prefix_frequency_threshold: usize,
    /// TTL for prefix cache sessions in seconds
    pub prefix_ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            token_cache_size: 10_000,
            embedding_cache_size: 10_000,
            max_memory_mb: 1024,
            ttl_seconds: 3600,
            enable_metrics: true,
            prefix_cache_enabled: false, // Disabled by default for gradual rollout
            prefix_cache_size: 100,
            min_prefix_length: 100,
            prefix_frequency_threshold: 5,
            prefix_ttl_seconds: 7200, // 2 hours
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration builder
    pub fn builder() -> CacheConfigBuilder {
        CacheConfigBuilder::new()
    }

    /// Validate the cache configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid:
    /// - Token cache size is 0
    /// - Embedding cache size is 0
    /// - TTL is 0
    pub fn validate(&self) -> Result<()> {
        if self.token_cache_size == 0 {
            return Err(Error::config("Token cache size must be greater than 0"));
        }
        if self.embedding_cache_size == 0 {
            return Err(Error::config("Embedding cache size must be greater than 0"));
        }
        if self.max_memory_mb == 0 {
            return Err(Error::config("Max memory must be greater than 0"));
        }
        if self.ttl_seconds == 0 {
            return Err(Error::config("TTL must be greater than 0"));
        }
        if self.prefix_cache_enabled {
            if self.prefix_cache_size == 0 {
                return Err(Error::config("Prefix cache size must be greater than 0"));
            }
            if self.min_prefix_length < 50 {
                return Err(Error::config(
                    "Minimum prefix length must be at least 50 tokens",
                ));
            }
            if self.prefix_frequency_threshold == 0 {
                return Err(Error::config(
                    "Prefix frequency threshold must be greater than 0",
                ));
            }
            if self.prefix_ttl_seconds == 0 {
                return Err(Error::config("Prefix TTL must be greater than 0"));
            }
        }
        Ok(())
    }
}

/// Builder for `CacheConfig`
pub struct CacheConfigBuilder {
    config: CacheConfig,
}

impl Default for CacheConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    /// Set whether caching is enabled
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set token cache size
    #[must_use]
    pub fn with_token_cache_size(mut self, size: usize) -> Self {
        self.config.token_cache_size = size;
        self
    }

    /// Set embedding cache size
    #[must_use]
    pub fn with_embedding_cache_size(mut self, size: usize) -> Self {
        self.config.embedding_cache_size = size;
        self
    }

    /// Set maximum memory usage in MB
    #[must_use]
    pub fn with_max_memory_mb(mut self, mb: usize) -> Self {
        self.config.max_memory_mb = mb;
        self
    }

    /// Set TTL in seconds
    #[must_use]
    pub fn with_ttl_seconds(mut self, seconds: u64) -> Self {
        self.config.ttl_seconds = seconds;
        self
    }

    /// Set metrics collection enabled
    #[must_use]
    pub fn with_enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Enable prefix caching
    #[must_use]
    pub fn with_prefix_cache_enabled(mut self, enabled: bool) -> Self {
        self.config.prefix_cache_enabled = enabled;
        self
    }

    /// Set prefix cache size
    #[must_use]
    pub fn with_prefix_cache_size(mut self, size: usize) -> Self {
        self.config.prefix_cache_size = size;
        self
    }

    /// Set minimum prefix length
    #[must_use]
    pub fn with_min_prefix_length(mut self, length: usize) -> Self {
        self.config.min_prefix_length = length;
        self
    }

    /// Set prefix frequency threshold
    #[must_use]
    pub fn with_prefix_frequency_threshold(mut self, threshold: usize) -> Self {
        self.config.prefix_frequency_threshold = threshold;
        self
    }

    /// Set prefix TTL in seconds
    #[must_use]
    pub fn with_prefix_ttl_seconds(mut self, seconds: u64) -> Self {
        self.config.prefix_ttl_seconds = seconds;
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
    pub fn build(self) -> Result<CacheConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration for embedding-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Token truncation strategy
    pub truncate_tokens: TruncateTokens,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            truncate_tokens: TruncateTokens::No,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new embedding configuration builder
    pub fn builder() -> EmbeddingConfigBuilder {
        EmbeddingConfigBuilder::new()
    }

    /// Validate the embedding configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid:
    /// - Limit is 0
    pub fn validate(&self) -> Result<()> {
        if let TruncateTokens::Limit(n) = self.truncate_tokens {
            if n == 0 {
                return Err(Error::config("Truncation limit must be greater than 0"));
            }
        }
        Ok(())
    }
}

/// Builder for `EmbeddingConfig`
pub struct EmbeddingConfigBuilder {
    config: EmbeddingConfig,
}

impl Default for EmbeddingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig::default(),
        }
    }

    /// Set the truncation strategy
    #[must_use]
    pub fn with_truncate_tokens(mut self, truncate: TruncateTokens) -> Self {
        self.config.truncate_tokens = truncate;
        self
    }

    /// Set truncation to a specific limit (convenience method)
    #[must_use]
    pub fn with_truncate_limit(mut self, limit: u32) -> Self {
        self.config.truncate_tokens = TruncateTokens::Limit(limit);
        self
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
    pub fn build(self) -> Result<EmbeddingConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl EngineConfig {
    /// Create a new configuration builder
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::new()
    }

    /// Create configuration with backend auto-detection
    pub fn with_backend_detection() -> EngineConfigBuilder {
        let backend = crate::backend::detect_best_backend();
        let mut builder = EngineConfigBuilder::new();

        // Set GPU configuration based on backend
        if backend.is_gpu_accelerated() {
            builder = builder.with_use_gpu(true);
            if let Some(gpu_layers) = backend.recommended_gpu_layers() {
                builder = builder.with_n_gpu_layers(gpu_layers);
            }
        }

        builder
    }

    /// Validate the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model configuration is invalid
    /// - Invalid batch size (0)
    /// - Invalid max tokens (0)
    /// - Cache configuration is invalid
    pub fn validate(&self) -> Result<()> {
        // Validate the model configuration
        self.model_config.validate()?;

        // Validate engine-specific fields
        if self.batch_size == Some(0) {
            return Err(Error::config("Batch size must be greater than 0"));
        }

        if self.max_tokens == Some(0) {
            return Err(Error::config("Max tokens must be greater than 0"));
        }

        // Validate cache configuration if present
        if let Some(ref cache) = self.cache {
            cache.validate()?;
        }

        // Validate embedding configuration if present
        if let Some(ref embedding) = self.embedding {
            embedding.validate()?;
        }

        Ok(())
    }

    /// Load configuration from environment variables
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
    pub fn from_env() -> Result<Self> {
        let mut builder = EngineConfigBuilder::new();

        if let Ok(path) = env::var("EMBELLAMA_MODEL_PATH") {
            builder = builder.with_model_path(path);
        }

        if let Ok(name) = env::var("EMBELLAMA_MODEL_NAME") {
            builder = builder.with_model_name(name);
        }

        if let Ok(size) = env::var("EMBELLAMA_CONTEXT_SIZE") {
            let size = size
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_CONTEXT_SIZE value"))?;
            builder = builder.with_context_size(size);
        }

        if let Ok(threads) = env::var("EMBELLAMA_N_THREADS") {
            let threads = threads
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_N_THREADS value"))?;
            builder = builder.with_n_threads(threads);
        }

        if let Ok(use_gpu) = env::var("EMBELLAMA_USE_GPU") {
            let use_gpu = use_gpu
                .parse()
                .map_err(|_| Error::config("Invalid EMBELLAMA_USE_GPU value"))?;
            builder = builder.with_use_gpu(use_gpu);
        }

        builder.build()
    }
}

/// Builder for creating `EngineConfig` instances
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Create a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
        }
    }

    /// Set the entire model configuration
    #[must_use]
    pub fn with_model_config(mut self, model_config: ModelConfig) -> Self {
        self.config.model_config = model_config;
        self
    }

    // Convenience methods that modify the nested model_config for backward compatibility

    /// Set the model path (convenience method)
    #[must_use]
    pub fn with_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.model_config.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the model name (convenience method)
    #[must_use]
    pub fn with_model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.config.model_config.model_name = name.into();
        self
    }

    /// Set the context size (convenience method)
    #[must_use]
    pub fn with_context_size(mut self, size: usize) -> Self {
        self.config.model_config.context_size = u32::try_from(size).ok();
        self
    }

    /// Set the batch size for prompt processing (convenience method)
    #[must_use]
    pub fn with_n_batch(mut self, batch: u32) -> Self {
        self.config.model_config.n_batch = Some(batch);
        self
    }

    /// Set the micro-batch size for prompt processing (convenience method)
    #[must_use]
    pub fn with_n_ubatch(mut self, ubatch: u32) -> Self {
        self.config.model_config.n_ubatch = Some(ubatch);
        self
    }

    /// Set the number of threads (convenience method)
    #[must_use]
    pub fn with_n_threads(mut self, threads: usize) -> Self {
        self.config.model_config.n_threads = Some(threads);
        self
    }

    /// Set the number of GPU layers (convenience method)
    #[must_use]
    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.model_config.n_gpu_layers = Some(layers);
        self
    }

    /// Set the normalization mode for embeddings (convenience method)
    #[must_use]
    pub fn with_normalization_mode(mut self, mode: NormalizationMode) -> Self {
        self.config.model_config.normalization_mode = Some(mode);
        self
    }

    /// Set the pooling strategy (convenience method)
    #[must_use]
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.config.model_config.pooling_strategy = Some(strategy);
        self
    }

    /// Set whether to use memory mapping (convenience method)
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.config.model_config.use_mmap = use_mmap;
        self
    }

    /// Set whether to use memory locking (convenience method)
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.model_config.use_mlock = use_mlock;
        self
    }

    /// Set the maximum number of sequences for batch processing (convenience method)
    /// Default: 1, max: 64 (llama.cpp limit)
    #[must_use]
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.config.model_config.n_seq_max = Some(n_seq_max);
        self
    }

    // Engine-specific methods

    /// Set whether to use GPU
    #[must_use]
    pub fn with_use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.use_gpu = use_gpu;
        self
    }

    /// Set the batch size
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = Some(size);
        self
    }

    /// Set the maximum tokens
    #[must_use]
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set the memory limit in MB
    #[must_use]
    pub fn with_memory_limit_mb(mut self, limit_mb: usize) -> Self {
        self.config.memory_limit_mb = Some(limit_mb);
        self
    }

    /// Set verbose logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the seed for reproducibility
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set cache configuration
    #[must_use]
    pub fn with_cache_config(mut self, cache: CacheConfig) -> Self {
        self.config.cache = Some(cache);
        self
    }

    /// Enable caching with default configuration
    #[must_use]
    pub fn with_cache_enabled(mut self) -> Self {
        self.config.cache = Some(CacheConfig::default());
        self
    }

    /// Disable caching
    #[must_use]
    pub fn with_cache_disabled(mut self) -> Self {
        self.config.cache = None;
        self
    }

    /// Set embedding configuration
    #[must_use]
    pub fn with_embedding_config(mut self, embedding: EmbeddingConfig) -> Self {
        self.config.embedding = Some(embedding);
        self
    }

    /// Set truncation strategy (convenience method)
    #[must_use]
    pub fn with_truncate_tokens(mut self, truncate: TruncateTokens) -> Self {
        let embedding = self
            .config
            .embedding
            .get_or_insert_with(EmbeddingConfig::default);
        embedding.truncate_tokens = truncate;
        self
    }

    /// Set truncation to a specific limit (convenience method)
    #[must_use]
    pub fn with_truncate_limit(self, limit: u32) -> Self {
        self.with_truncate_tokens(TruncateTokens::Limit(limit))
    }

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails
    pub fn build(self) -> Result<EngineConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for EngineConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_model_config_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_n_ctx(512)
            .with_n_threads(4)
            .with_n_gpu_layers(0)
            .build()
            .unwrap();

        assert_eq!(config.model_path, model_path);
        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.n_ctx, Some(512));
        assert_eq!(config.n_threads, Some(4));
        assert_eq!(config.n_gpu_layers, Some(0));
    }

    #[test]
    fn test_model_config_validation() {
        let result = ModelConfig::builder().with_model_name("test").build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::ConfigurationError { .. }
        ));

        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_ctx(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_with_model_config() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Build a model config
        let model_config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_context_size(1024)
            .with_n_threads(8)
            .build()
            .unwrap();

        // Build engine config using the model config
        let engine_config = EngineConfig::builder()
            .with_model_config(model_config.clone())
            .build()
            .unwrap();

        // Verify the model config is properly stored
        assert_eq!(engine_config.model_config.model_path, model_path);
        assert_eq!(engine_config.model_config.model_name, "test-model");
        assert_eq!(engine_config.model_config.context_size, Some(1024));
        assert_eq!(engine_config.model_config.n_threads, Some(8));
    }

    #[test]
    fn test_config_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .with_context_size(512)
            .with_n_threads(4)
            .with_use_gpu(true)
            .build()
            .unwrap();

        assert_eq!(config.model_config.model_path, model_path);
        assert_eq!(config.model_config.model_name, "test-model");
        assert_eq!(config.model_config.context_size, Some(512));
        assert_eq!(config.model_config.n_threads, Some(4));
        assert!(config.use_gpu);
    }

    #[test]
    fn test_config_validation_empty_path() {
        let result = EngineConfig::builder().with_model_name("test").build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::ConfigurationError { .. }
        ));
    }

    #[test]
    fn test_config_validation_nonexistent_file() {
        let result = EngineConfig::builder()
            .with_model_path("/nonexistent/path/model.gguf")
            .with_model_name("test")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_empty_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = EngineConfig::builder().with_model_path(model_path).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_invalid_values() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(0)
            .build();
        assert!(result.is_err());

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_threads(0)
            .build();
        assert!(result.is_err());

        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_batch_size(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pooling_strategy_default() {
        assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_engine_config_full_builder() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("full-test")
            .with_context_size(2048)
            .with_n_threads(16)
            .with_use_gpu(true)
            .with_n_gpu_layers(32)
            .with_normalization_mode(NormalizationMode::L2)
            .with_pooling_strategy(PoolingStrategy::Cls)
            .with_batch_size(128)
            .build()
            .unwrap();

        assert_eq!(config.model_config.context_size, Some(2048));
        assert_eq!(config.model_config.n_threads, Some(16));
        assert!(config.use_gpu);
        assert_eq!(config.model_config.n_gpu_layers, Some(32));
        assert_eq!(
            config.model_config.normalization_mode,
            Some(NormalizationMode::L2)
        );
        assert_eq!(
            config.model_config.pooling_strategy,
            Some(PoolingStrategy::Cls)
        );
        assert_eq!(config.batch_size, Some(128));
    }

    #[test]
    fn test_model_config_defaults() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build()
            .unwrap();

        // Check that defaults are None
        assert!(config.n_ctx.is_none());
        assert!(config.n_threads.is_none());
        assert!(config.n_gpu_layers.is_none());
    }

    #[test]
    fn test_engine_config_defaults() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build()
            .unwrap();

        // Check defaults
        assert!(config.model_config.context_size.is_none());
        assert!(config.model_config.n_threads.is_none());
        assert!(!config.use_gpu);
        assert!(config.model_config.n_gpu_layers.is_none());
        assert_eq!(config.model_config.normalization_mode, None);
        assert_eq!(config.model_config.pooling_strategy, None);
        assert!(config.batch_size.is_none());
    }

    #[test]
    fn test_all_pooling_strategies() {
        let strategies = vec![
            PoolingStrategy::Mean,
            PoolingStrategy::Cls,
            PoolingStrategy::Max,
            PoolingStrategy::MeanSqrt,
            PoolingStrategy::Last,
            PoolingStrategy::None,
            PoolingStrategy::Rank,
        ];

        for strategy in strategies {
            let dir = tempdir().unwrap();
            let model_path = dir.path().join("model.gguf");
            fs::write(&model_path, b"dummy").unwrap();

            let config = EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name(format!("test-{strategy:?}"))
                .with_pooling_strategy(strategy)
                .build()
                .unwrap();

            assert_eq!(config.model_config.pooling_strategy, Some(strategy));
        }
    }

    #[test]
    fn test_model_config_path_types() {
        let dir = tempdir().unwrap();

        // Test with PathBuf
        let model_path_buf = dir.path().join("model1.gguf");
        fs::write(&model_path_buf, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path_buf)
            .with_model_name("test1")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path_buf);

        // Test with &Path
        let model_path = dir.path().join("model2.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(model_path.as_path())
            .with_model_name("test2")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path);

        // Test with String
        let model_path_str = dir.path().join("model3.gguf");
        fs::write(&model_path_str, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(model_path_str.to_str().unwrap())
            .with_model_name("test3")
            .build()
            .unwrap();
        assert_eq!(config.model_path, model_path_str);
    }

    #[test]
    fn test_config_validation_large_values() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Test with very large context size
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(1_000_000)
            .build()
            .unwrap();
        assert_eq!(config.model_config.context_size, Some(1_000_000));

        // Test with large thread count
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_n_threads(256)
            .build()
            .unwrap();
        assert_eq!(config.model_config.n_threads, Some(256));

        // Test with large batch size
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_batch_size(10000)
            .build()
            .unwrap();
        assert_eq!(config.batch_size, Some(10000));
    }

    #[test]
    fn test_config_clone() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let original = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_context_size(512)
            .build()
            .unwrap();

        let cloned = original.clone();
        assert_eq!(
            cloned.model_config.model_path,
            original.model_config.model_path
        );
        assert_eq!(
            cloned.model_config.model_name,
            original.model_config.model_name
        );
        assert_eq!(
            cloned.model_config.context_size,
            original.model_config.context_size
        );
    }

    #[test]
    fn test_model_config_debug_format() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("debug-test")
            .build()
            .unwrap();

        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ModelConfig"));
        assert!(debug_str.contains("debug-test"));
    }

    #[test]
    fn test_special_characters_in_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Test with special characters in name
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model_v2.0")
            .build()
            .unwrap();
        assert_eq!(config.model_config.model_name, "test-model_v2.0");

        // Test with Unicode characters
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("模型-测试")
            .build()
            .unwrap();
        assert_eq!(config.model_config.model_name, "模型-测试");
    }

    #[test]
    fn test_whitespace_in_model_name() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Empty name should fail
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("")
            .build();
        assert!(result.is_err());

        // Whitespace-only name should fail
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("   ")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_config_consistency() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // GPU layers without GPU flag should still be valid
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_use_gpu(false)
            .with_n_gpu_layers(10)
            .build()
            .unwrap();

        assert!(!config.use_gpu);
        assert_eq!(config.model_config.n_gpu_layers, Some(10));
    }

    // ============================================================================
    // Truncation Tests
    // ============================================================================

    #[test]
    fn test_truncate_tokens_default() {
        assert_eq!(TruncateTokens::default(), TruncateTokens::No);
    }

    #[test]
    fn test_truncate_tokens_variants() {
        // Test equality
        assert_eq!(TruncateTokens::No, TruncateTokens::No);
        assert_eq!(TruncateTokens::Yes, TruncateTokens::Yes);
        assert_eq!(TruncateTokens::Limit(50), TruncateTokens::Limit(50));
        assert_ne!(TruncateTokens::No, TruncateTokens::Yes);
        assert_ne!(TruncateTokens::Limit(50), TruncateTokens::Limit(100));
    }

    #[test]
    fn test_embedding_config_validation_limit_zero() {
        let config = EmbeddingConfig {
            truncate_tokens: TruncateTokens::Limit(0),
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must be greater than 0")
        );
    }

    #[test]
    fn test_embedding_config_validation_limit_positive() {
        let config = EmbeddingConfig {
            truncate_tokens: TruncateTokens::Limit(100),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_embedding_config_validation_yes_and_no() {
        let config_no = EmbeddingConfig {
            truncate_tokens: TruncateTokens::No,
        };
        assert!(config_no.validate().is_ok());

        let config_yes = EmbeddingConfig {
            truncate_tokens: TruncateTokens::Yes,
        };
        assert!(config_yes.validate().is_ok());
    }

    #[test]
    fn test_embedding_config_builder() {
        let config = EmbeddingConfig::builder()
            .with_truncate_tokens(TruncateTokens::Yes)
            .build()
            .unwrap();
        assert_eq!(config.truncate_tokens, TruncateTokens::Yes);
    }

    #[test]
    fn test_embedding_config_builder_limit() {
        let config = EmbeddingConfig::builder()
            .with_truncate_limit(500)
            .build()
            .unwrap();
        assert_eq!(config.truncate_tokens, TruncateTokens::Limit(500));
    }

    #[test]
    fn test_embedding_config_builder_validation() {
        // Valid config should build successfully
        let result = EmbeddingConfig::builder().with_truncate_limit(100).build();
        assert!(result.is_ok());

        // Invalid config (Limit(0)) should fail
        let result = EmbeddingConfig::builder().with_truncate_limit(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_config_with_truncate_tokens() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_truncate_tokens(TruncateTokens::Yes)
            .build()
            .unwrap();

        assert!(config.embedding.is_some());
        assert_eq!(
            config.embedding.unwrap().truncate_tokens,
            TruncateTokens::Yes
        );
    }

    #[test]
    fn test_engine_config_with_truncate_limit() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_truncate_limit(256)
            .build()
            .unwrap();

        assert!(config.embedding.is_some());
        assert_eq!(
            config.embedding.unwrap().truncate_tokens,
            TruncateTokens::Limit(256)
        );
    }

    #[test]
    fn test_engine_config_with_embedding_config() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let embedding_config = EmbeddingConfig {
            truncate_tokens: TruncateTokens::Limit(512),
        };

        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_embedding_config(embedding_config.clone())
            .build()
            .unwrap();

        assert!(config.embedding.is_some());
        assert_eq!(
            config.embedding.unwrap().truncate_tokens,
            TruncateTokens::Limit(512)
        );
    }

    #[test]
    fn test_engine_config_validation_with_truncation() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        // Valid truncation should pass
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_truncate_limit(100)
            .build();
        assert!(result.is_ok());

        // Invalid truncation (Limit(0)) should fail
        let result = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .with_truncate_limit(0)
            .build();
        assert!(result.is_err());
    }

    // ============================================================================
    // Path Canonicalization Tests (validates symlink resolution in validate())
    // ============================================================================

    #[cfg(unix)]
    #[test]
    fn test_symlink_to_non_gguf_file_rejected() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();

        // Create a real non-GGUF file
        let real_file = dir.path().join("not_a_model.txt");
        fs::write(&real_file, b"not a model").unwrap();

        // Create a symlink with .gguf extension pointing to the non-GGUF file
        let fake_gguf = dir.path().join("sneaky.gguf");
        symlink(&real_file, &fake_gguf).unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&fake_gguf)
            .with_model_name("test")
            .build();

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-GGUF"),
            "Expected 'non-GGUF' in error, got: {err_msg}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_to_gguf_file_accepted() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();

        // Create a real GGUF file
        let real_gguf = dir.path().join("real_model.gguf");
        fs::write(&real_gguf, b"dummy gguf").unwrap();

        // Create a symlink pointing to the real GGUF file
        let link_gguf = dir.path().join("link_model.gguf");
        symlink(&real_gguf, &link_gguf).unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&link_gguf)
            .with_model_name("test")
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_normal_gguf_path_still_validates() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        fs::write(&model_path, b"dummy").unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_non_gguf_extension_rejected() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.bin");
        fs::write(&model_path, b"dummy").unwrap();

        let result = ModelConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test")
            .build();

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-GGUF"),
            "Expected 'non-GGUF' in error, got: {err_msg}"
        );
    }
}
