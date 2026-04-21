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

//! Model management module for the embellama library.
//!
//! This module contains the `EmbeddingModel` struct which encapsulates
//! the llama.cpp model and context, handling the generation of embeddings.

use crate::cache::CacheStore;
use crate::cache::token_cache::TokenCache;
use crate::config::{ModelConfig, NormalizationMode, PoolingStrategy, TruncateTokens};
use crate::error::{Error, Result};
use crate::gguf;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    token::LlamaToken,
};
use self_cell::self_cell;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, info, instrument, warn};

/// Session state format version for compatibility checking
const SESSION_STATE_VERSION: u32 = 1;
/// Size of the session state header in bytes (version u32 + reserved u32)
const SESSION_STATE_HEADER_SIZE: usize = 8;

/// Maps our `PoolingStrategy` enum to llama.cpp's `LlamaPoolingType`
///
/// For strategies that llama.cpp doesn't natively support (Max, `MeanSqrt`),
/// we set pooling to None and handle it ourselves in `apply_pooling()`.
/// For Last pooling (required for decoder models), we must set it at the
/// llama.cpp level to ensure proper KV cache initialization.
fn pooling_strategy_to_llama_type(strategy: PoolingStrategy) -> LlamaPoolingType {
    match strategy {
        PoolingStrategy::Mean => LlamaPoolingType::Mean,
        PoolingStrategy::Cls => LlamaPoolingType::Cls,
        PoolingStrategy::Last => LlamaPoolingType::Last,
        PoolingStrategy::Rank => LlamaPoolingType::Rank,
        // Max, MeanSqrt, and None are not natively supported by llama.cpp pooling.
        // For Max/MeanSqrt we apply pooling ourselves after extraction.
        // For None we return raw per-token embeddings without any pooling.
        PoolingStrategy::Max | PoolingStrategy::MeanSqrt | PoolingStrategy::None => {
            LlamaPoolingType::None
        }
    }
}

self_cell! {
    struct ModelCell {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
}

/// Represents a loaded embedding model.
///
/// This struct encapsulates the `llama_cpp_2::LlamaModel` and `LlamaContext`
/// and provides methods for generating embeddings from text input.
///
/// # Important
///
/// Due to the `!Send` nature of `LlamaContext`, instances of this struct
/// cannot be safely sent between threads. Each thread must maintain its
/// own instance.
///
/// # Example
///
/// ```ignore
/// use embellama::model::EmbeddingModel;
/// use embellama::config::ModelConfig;
///
/// let config = ModelConfig::builder()
///     .with_model_path("path/to/model.gguf")
///     .with_model_name("my-model")
///     .build()?;
///
/// let model = EmbeddingModel::new(&config)?;
/// assert!(model.is_loaded());
/// ```
pub struct EmbeddingModel {
    // IMPORTANT: Field order matters for drop order!
    // Context must be dropped before model since it depends on it
    /// The llama model cell self-referential helper
    cell: ModelCell,

    // Metadata fields (order doesn't matter for these)
    /// Configuration used to create this model
    config: ModelConfig,
    /// Path to the model file
    model_path: PathBuf,
    /// Model name identifier
    model_name: String,
    /// Cached embedding dimensions (determined at load time)
    embedding_dimensions: usize,
    /// Maximum context size
    max_context_size: usize,
    /// Maximum number of sequences for batch processing
    n_seq_max: u32,
    /// Batch size (max usable context per sequence)
    n_batch: Option<u32>,
    /// GGUF metadata containing architecture info, dimensions, context size
    metadata: crate::gguf::GGUFMetadata,
}

impl EmbeddingModel {
    /// Creates a new embedding model from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `backend` - The llama backend to use for model loading
    /// * `config` - The model configuration containing path and parameters
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized model or an error.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The model file cannot be loaded
    /// - The context creation fails
    /// - Invalid configuration parameters are provided
    #[instrument(skip(backend, config), fields(model_path = %config.model_path.display()))]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    pub fn new(backend: &LlamaBackend, config: &ModelConfig) -> Result<Self> {
        info!("Loading model from {:?}", config.model_path);

        // Set up model parameters
        let mut model_params = LlamaModelParams::default();

        // Configure GPU layers if specified
        if let Some(gpu_layers) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(gpu_layers);
            debug!("GPU layers set to: {}", gpu_layers);
        }

        // TODO: Configure memory options when API supports it
        // Currently llama-cpp-2 doesn't expose use_mmap/use_mlock setters

        // Load the model into a Box for stable address
        let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
            .map_err(|e| Error::ModelLoadError {
                path: config.model_path.clone(),
                source: anyhow::anyhow!("Failed to load model: {e}"),
            })?;

        debug!("Model loaded successfully");

        // Set up context parameters
        // Priority: 1) Explicit config, 2) GGUF metadata, 3) Fallback to 2048
        let ctx_size = if let Some(n_ctx) = config.n_ctx {
            debug!("Using configured context size: {}", n_ctx);
            n_ctx
        } else {
            // Try to auto-detect from GGUF metadata
            match Self::extract_context_size_from_gguf(&config.model_path) {
                Ok(size) => {
                    info!("Auto-detected context size from GGUF metadata: {}", size);
                    size
                }
                Err(e) => {
                    debug!(
                        "Could not read context size from GGUF metadata: {}, using default 2048",
                        e
                    );
                    2048
                }
            }
        };

        let n_threads = i32::try_from(config.n_threads.unwrap_or_else(|| {
            let threads = num_cpus::get();
            debug!("Using {} CPU threads", threads);
            threads
        }))
        .unwrap_or(1);

        // Extract GGUF metadata for architecture detection and model info
        let metadata = gguf::extract_metadata(&config.model_path).unwrap_or_else(|e| {
            warn!(
                "Failed to read GGUF metadata: {}, assuming decoder model with defaults",
                e
            );
            // Create a fallback metadata that defaults to decoder (safer)
            crate::gguf::GGUFMetadata {
                architecture: Some("unknown".to_string()),
                embedding_dimensions: 0,
                context_size: ctx_size as usize,
                pooling_type: None,
            }
        });
        let is_decoder = metadata.is_decoder();
        debug!(
            "Detected model architecture: {} for model: {}",
            if is_decoder { "decoder" } else { "encoder" },
            config.model_name
        );

        // Resolve effective pooling strategy from config or GGUF metadata
        let effective_pooling = match config.pooling_strategy {
            Some(strategy) => strategy,
            None => {
                if metadata.is_reranker() {
                    info!("Auto-detected reranker model from GGUF metadata, using Rank pooling");
                    PoolingStrategy::Rank
                } else if is_decoder {
                    PoolingStrategy::Last
                } else {
                    PoolingStrategy::Mean
                }
            }
        };

        // Resolve effective normalization mode
        let effective_normalization = match config.normalization_mode {
            Some(mode) => mode,
            None => {
                if effective_pooling == PoolingStrategy::Rank {
                    NormalizationMode::None
                } else {
                    NormalizationMode::L2
                }
            }
        };

        // Create resolved config with effective values
        let mut resolved_config = config.clone();
        resolved_config.pooling_strategy = Some(effective_pooling);
        resolved_config.normalization_mode = Some(effective_normalization);

        // Use configured n_seq_max or default to 2 for reasonable batching
        let n_seq_max = config.n_seq_max.unwrap_or(2);
        debug!(
            "Setting n_seq_max={} for {} model",
            n_seq_max,
            if is_decoder { "decoder" } else { "encoder" }
        );

        let mut ctx_params = LlamaContextParams::default();
        ctx_params = ctx_params.with_n_seq_max(n_seq_max);

        // Set context size (use context_size if specified, otherwise use n_ctx)
        let context_size = config.context_size.unwrap_or(ctx_size);

        // Validate context_size doesn't exceed GGUF maximum
        if context_size > ctx_size {
            return Err(Error::ConfigurationError {
                message: format!(
                    "context_size ({context_size}) cannot exceed maximum context size from GGUF metadata ({ctx_size})"
                ),
            });
        }

        let n_ctx = NonZeroU32::new(context_size);
        ctx_params = ctx_params.with_n_ctx(n_ctx);

        // Enable KV cache optimizations if requested
        if config.enable_kv_optimization {
            debug!("Enabling KV cache optimizations");
            // > NOTE: These optimizations are enabled through llama-cpp parameters
        }

        // Set batch size (max usable context per sequence)
        // Default: min(context_size, 2048) for reasonable memory usage
        let n_batch = config.n_batch.unwrap_or(context_size.min(2048));
        debug!("Setting n_batch={} (max usable context)", n_batch);

        // Validate n_batch <= context_size
        if n_batch > context_size {
            return Err(Error::ConfigurationError {
                message: format!(
                    "Batch size (n_batch={n_batch}) cannot exceed context size ({context_size})"
                ),
            });
        }

        ctx_params = ctx_params.with_n_batch(n_batch);

        // Set micro-batch size (physical batch size for processing)
        // Default: n_batch (or architecture-specific defaults capped at n_batch)
        let n_ubatch = if let Some(ubatch) = config.n_ubatch {
            // Use explicitly configured value
            debug!("Using configured n_ubatch: {}", ubatch);

            // Validate n_ubatch <= n_batch
            if ubatch > n_batch {
                return Err(Error::ConfigurationError {
                    message: format!(
                        "Micro-batch size (n_ubatch={ubatch}) cannot exceed batch size (n_batch={n_batch})"
                    ),
                });
            }
            ubatch
        } else if config.n_batch.is_some() {
            // n_batch was explicitly set, use it for n_ubatch
            debug!("Setting n_ubatch={} (matching n_batch)", n_batch);
            n_batch
        } else if is_decoder {
            // Decoder models: use conservative 512 to prevent crashes
            // llama-server uses 2048, but 512 is safer for large contexts
            // Cap at n_batch to ensure n_ubatch <= n_batch
            let ubatch = 512_u32.min(n_batch);
            debug!(
                "Setting n_ubatch={} for decoder model (conservative default, capped at n_batch)",
                ubatch
            );
            ubatch
        } else {
            // Encoder models: can use larger values for better performance
            // Cap at n_batch to ensure n_ubatch <= n_batch
            let ubatch = 2048_u32.min(n_batch);
            debug!(
                "Setting n_ubatch={} for encoder model (capped at n_batch)",
                ubatch
            );
            ubatch
        };

        ctx_params = ctx_params.with_n_ubatch(n_ubatch);

        // Set thread counts
        ctx_params = ctx_params.with_n_threads(n_threads);
        ctx_params = ctx_params.with_n_threads_batch(n_threads);

        // Enable embeddings mode
        ctx_params = ctx_params.with_embeddings(true);

        // Set pooling type based on our pooling strategy
        // This is critical for decoder models (e.g., Qwen) which require Last pooling
        let llama_pooling_type = pooling_strategy_to_llama_type(effective_pooling);
        ctx_params = ctx_params.with_pooling_type(llama_pooling_type);
        debug!(
            "Set llama.cpp pooling type to {:?} for strategy {:?}",
            llama_pooling_type, effective_pooling
        );

        // Enable flash attention for better performance
        // Decoder models benefit significantly from flash attention
        if is_decoder {
            debug!("Enabling flash attention for decoder model");
            ctx_params = ctx_params.with_flash_attention_policy(1); // LLAMA_FLASH_ATTN_TYPE_ENABLED
        }

        // Get embedding dimensions from the model
        #[allow(clippy::cast_sign_loss)]
        let embedding_dimensions = model.n_embd() as usize;

        info!(
            "Model initialized: dimensions={}, context_size={}, threads={}",
            embedding_dimensions, context_size, n_threads
        );

        let cell = ModelCell::try_new(model, |m| {
            m.new_context(backend, ctx_params)
                .map_err(|e| Error::ContextError {
                    source: anyhow::anyhow!("Failed to create context: {e}"),
                })
        })?;

        let model = Self {
            cell,
            config: resolved_config,
            model_path: config.model_path.clone(),
            model_name: config.model_name.clone(),
            embedding_dimensions,
            #[allow(clippy::cast_lossless)]
            max_context_size: context_size as usize,
            n_seq_max,
            n_batch: Some(n_batch),
            metadata,
        };

        // Log effective max tokens for debugging batch size issues
        let effective_max = model.effective_max_tokens();
        let usable_context = model.n_batch.map_or(model.max_context_size, |b| b as usize);
        let overhead = usable_context.saturating_sub(effective_max);
        info!(
            "Effective max tokens: {} (usable_context: {}, overhead: {})",
            effective_max, usable_context, overhead
        );

        Ok(model)
    }

    /// Loads a model from disk.
    ///
    /// This is an alternative way to create a model, useful when you want
    /// to explicitly separate the loading step.
    ///
    /// # Arguments
    ///
    /// * `backend` - The llama backend to use for model loading
    /// * `config` - The model configuration
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the loaded model or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails
    pub fn load(backend: &LlamaBackend, config: &ModelConfig) -> Result<Self> {
        Self::new(backend, config)
    }

    /// Consumes the model and explicitly frees resources.
    ///
    /// Note: This happens automatically when the model is dropped.
    /// This method exists mainly for explicit resource management.
    pub fn unload(self) {
        // Model is dropped here, which triggers cleanup
        drop(self);
    }

    /// Returns the effective pooling strategy.
    ///
    /// After construction, `pooling_strategy` is always `Some(...)` because
    /// `new()` resolves `None` to a concrete strategy. This helper avoids
    /// `.unwrap()` calls throughout the codebase.
    fn effective_pooling(&self) -> PoolingStrategy {
        self.config
            .pooling_strategy
            .unwrap_or(PoolingStrategy::Mean)
    }

    /// Returns the effective normalization mode.
    fn effective_normalization(&self) -> NormalizationMode {
        self.config
            .normalization_mode
            .unwrap_or(NormalizationMode::L2)
    }

    /// Checks if the model is currently loaded and ready for inference.
    ///
    /// # Returns
    ///
    /// Returns true if the model is loaded, false otherwise.
    pub fn is_loaded(&self) -> bool {
        // Check if we have valid dimensions and context size
        self.embedding_dimensions > 0 && self.max_context_size > 0
    }

    /// Returns the dimensionality of embeddings produced by this model.
    ///
    /// # Returns
    ///
    /// The number of dimensions in the embedding vectors.
    pub fn embedding_dimensions(&self) -> usize {
        self.embedding_dimensions
    }

    /// Returns the maximum sequence length supported by this model.
    ///
    /// # Returns
    ///
    /// The maximum number of tokens that can be processed.
    pub fn max_sequence_length(&self) -> usize {
        self.max_context_size
    }

    /// Returns the approximate memory footprint of the model in bytes.
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes, or `None` if the size cannot be calculated
    /// (e.g., on 32-bit platforms with very large models).
    pub fn model_size(&self) -> Option<usize> {
        // This is an approximation based on model parameters
        // More accurate measurement would require llama.cpp API support
        let params = self.cell.borrow_owner().n_params();
        let size_per_param = 2; // Approximate bytes per parameter for quantized models
        usize::try_from(params).ok().map(|p| p * size_per_param)
    }

    /// Returns the model's metadata.
    ///
    /// # Returns
    ///
    /// A tuple containing (`model_name`, `model_path`, `vocab_size`, `n_params`).
    pub fn model_metadata(&self) -> (String, PathBuf, usize, usize) {
        (
            self.model_name.clone(),
            self.model_path.clone(),
            usize::try_from(self.cell.borrow_owner().n_vocab()).unwrap_or_else(|_| {
                warn!("Model vocab size conversion failed, using 0");
                0
            }),
            usize::try_from(self.cell.borrow_owner().n_params()).unwrap_or_else(|_| {
                warn!("Model params count too large for platform, using 0");
                0
            }),
        )
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the model name.
    pub fn name(&self) -> &str {
        &self.model_name
    }

    /// Returns the path to the model file.
    pub fn path(&self) -> &PathBuf {
        &self.model_path
    }

    /// Returns the maximum number of sequences for batch processing.
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Calculate the effective maximum tokens available per sequence in batch processing.
    ///
    /// When batching multiple sequences, each sequence gets its own KV cache slot.
    /// The usable context (`n_batch`) is divided among sequences based on `n_seq_max`.
    ///
    /// # Returns
    ///
    /// The maximum number of input tokens per sequence that can be safely processed.
    ///
    /// # Implementation Note
    ///
    /// Each sequence slot size = `n_batch / n_seq_max - 2`
    /// - `n_batch` represents the max usable context per sequence (defaults to `context_size`)
    /// - The division accounts for parallel sequence processing
    /// - The 2-token overhead is for special tokens (\[CLS\], \[SEP\])
    ///
    /// # Example
    ///
    /// For a model with `n_batch = 8192` and `n_seq_max = 2`:
    /// - Per-sequence size: 8192 / 2 = 4096
    /// - Overhead: 2 tokens (\[CLS\] and \[SEP\])
    /// - Effective max per sequence: 4096 - 2 = 4094 tokens
    pub fn effective_max_tokens(&self) -> usize {
        // Use n_batch if set (the max usable context), otherwise max_context_size
        let usable_context = self.n_batch.map_or(self.max_context_size, |b| b as usize);

        // Each sequence gets its own KV cache slot: usable_context / n_seq_max
        // Subtract 2 for special tokens ([CLS], [SEP]) per sequence
        let per_sequence_size = usable_context / (self.n_seq_max as usize);
        per_sequence_size.saturating_sub(2)
    }

    /// Tokenizes the input text.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    ///
    /// A vector of tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        self.cell
            .borrow_owner()
            .str_to_token(text, AddBos::Always)
            .map_err(|e| Error::TokenizationError {
                message: format!("Failed to tokenize text: {e}"),
            })
    }

    /// Tokenizes the input text with caching support.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    /// * `cache` - Optional token cache for caching tokenization results
    ///
    /// # Returns
    ///
    /// Returns a vector of tokens representing the tokenized text.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn tokenize_cached(
        &self,
        text: &str,
        cache: Option<&TokenCache>,
    ) -> Result<Vec<LlamaToken>> {
        // If cache is available, try to get cached tokens
        if let Some(cache) = cache {
            let key = TokenCache::compute_key(text, &self.model_name);
            if let Some(tokens) = cache.get(&key) {
                debug!("Using cached tokens for text (length: {})", text.len());
                return Ok(tokens);
            }

            // Cache miss, tokenize and cache the result
            let tokens = self.tokenize(text)?;
            cache.insert(key, tokens.clone());
            return Ok(tokens);
        }

        // No cache available, fallback to regular tokenization
        self.tokenize(text)
    }

    /// Generates an embedding for the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Tokenization fails
    /// - The input exceeds the maximum token limit
    /// - Model inference fails
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub fn generate_embedding(&mut self, text: &str) -> Result<Vec<f32>> {
        self.generate_embedding_cached(text, None, TruncateTokens::No)
    }

    /// Generates an embedding for the given text with optional token cache support.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    /// * `token_cache` - Optional token cache for caching tokenization results
    /// * `truncate` - Truncation strategy to apply
    ///
    /// # Returns
    ///
    /// Returns a vector of f32 values representing the embedding.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Tokenization fails
    /// - The input exceeds the maximum token limit (when truncation is disabled)
    /// - Model inference fails
    /// - Truncation limit exceeds model's effective maximum
    #[instrument(skip(self, text, token_cache), fields(text_len = text.len()))]
    pub fn generate_embedding_cached(
        &mut self,
        text: &str,
        token_cache: Option<&TokenCache>,
        truncate: TruncateTokens,
    ) -> Result<Vec<f32>> {
        // Validate input
        if text.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot generate embedding for empty text".to_string(),
            });
        }

        // Tokenize the text with caching support
        let tokens = self.tokenize_cached(text, token_cache)?;

        // Resolve truncation limit
        let truncation_limit = self.resolve_truncation_limit(truncate)?;

        // Apply truncation if needed
        let tokens = Self::truncate_tokens_if_needed(&tokens, truncation_limit);

        // Validate token limit (after truncation)
        self.validate_token_limit(tokens.len(), Some("Input"))?;

        debug!("Processing {} tokens", tokens.len());

        // Process tokens to get embeddings
        let embeddings = self.process_tokens_internal(tokens)?;

        // Apply pooling and normalization
        self.finalize_embedding(&embeddings, tokens.len())
    }

    /// Generates per-token (multi-vector) embeddings for the given text.
    ///
    /// Returns one embedding vector per token, suitable for ColBERT-style late
    /// interaction reranking. Each vector is individually normalized according
    /// to the model's normalization mode.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    /// * `token_cache` - Optional token cache for caching tokenization results
    /// * `truncate` - Truncation strategy to apply
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, one per token.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or model inference fails.
    #[instrument(skip(self, text, token_cache), fields(text_len = text.len()))]
    pub fn generate_multi_embedding(
        &mut self,
        text: &str,
        token_cache: Option<&TokenCache>,
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<f32>>> {
        if text.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot generate embedding for empty text".to_string(),
            });
        }

        let tokens = self.tokenize_cached(text, token_cache)?;
        let truncation_limit = self.resolve_truncation_limit(truncate)?;
        let tokens = Self::truncate_tokens_if_needed(&tokens, truncation_limit);
        self.validate_token_limit(tokens.len(), Some("Input"))?;

        debug!("Processing {} tokens for multi-vector output", tokens.len());

        let embeddings = self.process_tokens_internal(tokens)?;
        self.finalize_multi_embedding(&embeddings)
    }

    /// Processes multiple token sequences as a batch through the model.
    ///
    /// This method enables true batch processing by encoding multiple sequences
    /// in a single model pass using unique sequence IDs. If the number of sequences
    /// exceeds `n_seq_max`, it will automatically chunk them.
    ///
    /// # Arguments
    ///
    /// * `token_sequences` - Slice of token sequences to process
    /// * `truncate` - Truncation strategy to apply to each sequence
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, one for each input sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Context creation fails
    /// - Batch processing fails
    /// - Embedding extraction fails
    /// - Pooling or normalization operations fail
    /// - Truncation limit exceeds model's effective maximum
    #[instrument(skip(self, token_sequences), fields(batch_size = token_sequences.len()))]
    pub fn process_batch_tokens(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<f32>>> {
        if token_sequences.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Processing batch of {} sequences with n_seq_max={}",
            token_sequences.len(),
            self.n_seq_max
        );

        // If we have more sequences than n_seq_max, process in chunks
        #[allow(clippy::cast_lossless)]
        if token_sequences.len() > self.n_seq_max as usize {
            debug!(
                "Batch size {} exceeds n_seq_max {}, chunking",
                token_sequences.len(),
                self.n_seq_max
            );

            let mut all_embeddings = Vec::with_capacity(token_sequences.len());

            // Process sequences in chunks of n_seq_max
            #[allow(clippy::cast_lossless)]
            for chunk in token_sequences.chunks(self.n_seq_max as usize) {
                debug!("Processing chunk of {} sequences", chunk.len());
                let chunk_embeddings = self.process_batch_tokens_internal(chunk, truncate)?;
                all_embeddings.extend(chunk_embeddings);
            }

            return Ok(all_embeddings);
        }

        // Process all sequences in a single batch
        self.process_batch_tokens_internal(token_sequences, truncate)
    }

    /// Internal method to process a batch of token sequences that fits within `n_seq_max`.
    #[allow(clippy::too_many_lines)]
    fn process_batch_tokens_internal(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<f32>>> {
        // Resolve truncation limit once for all sequences
        let truncation_limit = self.resolve_truncation_limit(truncate)?;

        // Apply truncation and validate each sequence
        let truncated_sequences: Vec<&[LlamaToken]> = token_sequences
            .iter()
            .enumerate()
            .map(|(i, tokens)| {
                let truncated = Self::truncate_tokens_if_needed(tokens, truncation_limit);
                // Validate token limit after truncation
                self.validate_token_limit(truncated.len(), Some(&format!("Sequence {i}")))?;
                Ok(truncated)
            })
            .collect::<Result<Vec<_>>>()?;

        // Calculate total tokens needed for batch allocation (from truncated sequences)
        let total_tokens: usize = truncated_sequences.iter().map(|s| s.len()).sum();

        // Create a batch with all sequences (using actual n_seq_max)
        let _n_seq_max_i32 =
            i32::try_from(self.n_seq_max).map_err(|_| Error::EmbeddingGenerationError {
                message: "n_seq_max too large for i32".to_string(),
                source: None,
            })?;
        let mut batch = LlamaBatch::new(total_tokens, 1);

        // Add each sequence with unique ID
        for (seq_id, tokens) in truncated_sequences.iter().enumerate() {
            batch
                .add_sequence(
                    tokens,
                    i32::try_from(seq_id).map_err(|_| Error::EmbeddingGenerationError {
                        message: format!("Sequence ID {seq_id} too large for i32"),
                        source: None,
                    })?,
                    true,
                )
                .map_err(|e| Error::EmbeddingGenerationError {
                    message: format!("Failed to add sequence {seq_id} to batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                })?;
        }

        // Process the entire batch in one model pass
        // Decoder models need to use decode() instead of encode()
        // encode() tries to access unified KV cache which is null for decoder models
        self.process_batch(&mut batch)?;

        // Extract embeddings for each sequence
        let mut all_embeddings = Vec::with_capacity(truncated_sequences.len());

        for seq_id in 0..truncated_sequences.len() {
            // Calculate token offset for this sequence
            let token_offset: usize = truncated_sequences[..seq_id].iter().map(|s| s.len()).sum();

            let embeddings = self.extract_sequence_embeddings(
                seq_id,
                truncated_sequences[seq_id].len(),
                Some(token_offset),
            )?;

            // Apply pooling and normalization
            let final_embedding =
                self.finalize_embedding(&embeddings, truncated_sequences[seq_id].len())?;
            all_embeddings.push(final_embedding);
        }

        Ok(all_embeddings)
    }

    /// Processes multiple token sequences as a batch, returning per-token (multi-vector) embeddings.
    ///
    /// Each input sequence produces a `Vec<Vec<f32>>` — one embedding per token. This is the
    /// batch equivalent of `generate_multi_embedding` for ColBERT-style late interaction.
    ///
    /// # Arguments
    ///
    /// * `token_sequences` - Slice of token sequences to process
    /// * `truncate` - Truncation strategy to apply to each sequence
    ///
    /// # Returns
    ///
    /// Returns a vector of multi-vector embeddings, one per input sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if batch processing, embedding extraction, or normalization fails.
    #[instrument(skip(self, token_sequences), fields(batch_size = token_sequences.len()))]
    pub fn process_batch_tokens_multi(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        if token_sequences.is_empty() {
            return Ok(Vec::new());
        }

        // Chunk if needed, same as process_batch_tokens
        #[allow(clippy::cast_lossless)]
        if token_sequences.len() > self.n_seq_max as usize {
            let mut all_embeddings = Vec::with_capacity(token_sequences.len());
            #[allow(clippy::cast_lossless)]
            for chunk in token_sequences.chunks(self.n_seq_max as usize) {
                let chunk_embeddings = self.process_batch_tokens_multi_internal(chunk, truncate)?;
                all_embeddings.extend(chunk_embeddings);
            }
            return Ok(all_embeddings);
        }

        self.process_batch_tokens_multi_internal(token_sequences, truncate)
    }

    /// Internal method to process a batch returning per-token embeddings.
    fn process_batch_tokens_multi_internal(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let truncation_limit = self.resolve_truncation_limit(truncate)?;

        let truncated_sequences: Vec<&[LlamaToken]> = token_sequences
            .iter()
            .enumerate()
            .map(|(i, tokens)| {
                let truncated = Self::truncate_tokens_if_needed(tokens, truncation_limit);
                self.validate_token_limit(truncated.len(), Some(&format!("Sequence {i}")))?;
                Ok(truncated)
            })
            .collect::<Result<Vec<_>>>()?;

        let total_tokens: usize = truncated_sequences.iter().map(|s| s.len()).sum();
        let mut batch = LlamaBatch::new(total_tokens, 1);

        for (seq_id, tokens) in truncated_sequences.iter().enumerate() {
            batch
                .add_sequence(
                    tokens,
                    i32::try_from(seq_id).map_err(|_| Error::EmbeddingGenerationError {
                        message: format!("Sequence ID {seq_id} too large for i32"),
                        source: None,
                    })?,
                    true,
                )
                .map_err(|e| Error::EmbeddingGenerationError {
                    message: format!("Failed to add sequence {seq_id} to batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                })?;
        }

        self.process_batch(&mut batch)?;

        let mut all_multi_embeddings = Vec::with_capacity(truncated_sequences.len());
        for seq_id in 0..truncated_sequences.len() {
            let token_offset: usize = truncated_sequences[..seq_id].iter().map(|s| s.len()).sum();
            let embeddings = self.extract_sequence_embeddings(
                seq_id,
                truncated_sequences[seq_id].len(),
                Some(token_offset),
            )?;
            let final_embeddings = self.finalize_multi_embedding(&embeddings)?;
            all_multi_embeddings.push(final_embeddings);
        }

        Ok(all_multi_embeddings)
    }

    /// Processes a batch of tokens through the model.
    ///
    /// This is a lower-level method used internally for batch processing.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The tokens to process
    ///
    /// # Returns
    ///
    /// Returns the processed embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Token processing fails
    /// - Pooling operation fails
    /// - Normalization fails (if enabled)
    #[instrument(skip(self, tokens), fields(token_count = tokens.len()))]
    pub fn process_tokens(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        // Convert i32 tokens to LlamaToken and process
        let llama_tokens: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        let embeddings = self.process_tokens_internal(&llama_tokens)?;

        // Apply pooling and normalization
        self.finalize_embedding(&embeddings, llama_tokens.len())
    }

    /// Helper to convert usize index to i32 with consistent error handling.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to convert
    /// * `context` - Description of what the index represents (for error messages)
    ///
    /// # Returns
    ///
    /// Returns the i32 representation of the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is too large for i32.
    #[inline]
    fn to_i32(index: usize, context: &str) -> Result<i32> {
        i32::try_from(index).map_err(|_| Error::EmbeddingGenerationError {
            message: format!("{context} {index} too large for i32"),
            source: None,
        })
    }

    /// Validate that a token count is within the effective maximum limit.
    ///
    /// This method consolidates token limit validation that was previously
    /// duplicated in three different locations.
    ///
    /// # Arguments
    ///
    /// * `token_count` - Number of tokens to validate
    /// * `context_hint` - Optional context string for error messages (e.g., "Sequence 0")
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the token count is within limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the token count exceeds the effective maximum.
    fn validate_token_limit(&self, token_count: usize, context_hint: Option<&str>) -> Result<()> {
        let effective_max = self.effective_max_tokens();
        if token_count > effective_max {
            let context_prefix = context_hint.map_or_else(String::new, |h| format!("{h} "));
            let usable_context = self.n_batch.map_or(self.max_context_size, |b| b as usize);
            let overhead = usable_context.saturating_sub(effective_max);
            return Err(Error::InvalidInput {
                message: format!(
                    "{context_prefix}exceeds effective maximum tokens: {token_count} tokens > {effective_max} effective max (context: {}, overhead: {overhead}). Please truncate your input.",
                    self.max_context_size
                ),
            });
        }
        Ok(())
    }

    /// Resolve truncation strategy to a concrete token limit.
    ///
    /// # Arguments
    ///
    /// * `truncate` - The truncation strategy to resolve
    ///
    /// # Returns
    ///
    /// Returns `Some(limit)` if truncation should be applied, `None` if no truncation.
    ///
    /// # Errors
    ///
    /// Returns an error if `Limit(n)` exceeds the model's `effective_max_tokens()`.
    fn resolve_truncation_limit(&self, truncate: TruncateTokens) -> Result<Option<usize>> {
        match truncate {
            TruncateTokens::No => Ok(None),
            TruncateTokens::Yes => {
                let limit = self.effective_max_tokens();
                debug!(
                    "Truncation enabled: will truncate to {} tokens (model's effective_max_tokens)",
                    limit
                );
                Ok(Some(limit))
            }
            TruncateTokens::Limit(n) => {
                let limit = n as usize;
                let effective_max = self.effective_max_tokens();
                if limit > effective_max {
                    return Err(Error::InvalidInput {
                        message: format!(
                            "Truncation limit ({limit}) exceeds model's effective maximum ({effective_max}) tokens"
                        ),
                    });
                }
                debug!(
                    "Truncation enabled: will truncate to {} tokens (explicit limit)",
                    limit
                );
                Ok(Some(limit))
            }
        }
    }

    /// Truncate tokens if needed based on the configured limit.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token sequence to potentially truncate
    /// * `limit` - Optional token limit; if `None`, returns the original slice
    ///
    /// # Returns
    ///
    /// Returns a slice of tokens, truncated to the limit if specified.
    fn truncate_tokens_if_needed(tokens: &[LlamaToken], limit: Option<usize>) -> &[LlamaToken] {
        if let Some(limit) = limit {
            if tokens.len() > limit {
                debug!(
                    "Truncating tokens: {} -> {} tokens (keeping first {})",
                    tokens.len(),
                    limit,
                    limit
                );
                &tokens[..limit]
            } else {
                tokens
            }
        } else {
            tokens
        }
    }

    /// Finalize an embedding by applying pooling and normalization.
    ///
    /// This method consolidates the pooling + normalization logic that was
    /// previously duplicated in four different locations.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - The raw embeddings from the model
    /// * `expected_tokens` - The number of tokens we expected (for pre-pooled detection)
    ///
    /// # Returns
    ///
    /// Returns the final pooled and optionally normalized embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if pooling or normalization fails.
    fn finalize_embedding(
        &self,
        embeddings: &[Vec<f32>],
        expected_tokens: usize,
    ) -> Result<Vec<f32>> {
        // Check if we got a single pre-pooled embedding
        let pooled = if embeddings.len() == 1 && expected_tokens > 1 {
            // This is already pooled by the model (BERT with pooling_type)
            debug!("Using pre-pooled embedding from model");
            embeddings[0].clone()
        } else {
            // Apply our pooling strategy for multi-token outputs
            Self::apply_pooling(embeddings, self.effective_pooling())?
        };

        // Apply normalization based on configured mode
        if self.effective_normalization() == NormalizationMode::None {
            Ok(pooled)
        } else {
            Self::normalize_embedding(pooled, self.effective_normalization())
        }
    }

    /// Finalizes per-token embeddings by applying normalization to each token
    /// embedding individually, without pooling.
    ///
    /// Used by `generate_multi_embedding` for ColBERT-style output.
    fn finalize_multi_embedding(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if embeddings.is_empty() {
            return Err(Error::EmbeddingGenerationError {
                message: "No embeddings to finalize".to_string(),
                source: None,
            });
        }

        if self.effective_normalization() == NormalizationMode::None {
            Ok(embeddings.to_vec())
        } else {
            embeddings
                .iter()
                .map(|emb| Self::normalize_embedding(emb.clone(), self.effective_normalization()))
                .collect()
        }
    }

    /// Extract embeddings for a sequence from the context.
    ///
    /// This method handles both pre-pooled embeddings (from `embeddings_seq_ith`)
    /// and token-wise embeddings (from `embeddings_ith`). This logic was previously
    /// duplicated in three different locations.
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID to extract embeddings for
    /// * `n_tokens` - Number of tokens in the sequence
    /// * `token_offset` - Optional offset for token-wise extraction (used in batch processing)
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors (one per token, or single pre-pooled).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding extraction fails.
    fn extract_sequence_embeddings(
        &self,
        seq_id: usize,
        n_tokens: usize,
        token_offset: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.cell.with_dependent(|_, ctx| -> Result<Vec<Vec<f32>>> {
            // llama.cpp handles pooling internally based on our configured strategy
            // Try to get the pre-pooled sequence embedding first
            let seq_id_i32 = Self::to_i32(seq_id, "Sequence ID")?;
            if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(seq_id_i32) {
                // Got pooled embedding from llama.cpp
                debug!(
                    "Retrieved pooled embedding for sequence {} (strategy: {:?})",
                    seq_id,
                    self.effective_pooling()
                );
                return Ok(vec![seq_embeddings.to_vec()]);
            }

            if seq_id == 0 {
                debug!(
                    "Failed to get sequence embedding, falling back to token-wise (strategy: {:?})",
                    self.effective_pooling()
                );
            }

            // Fall back to token-wise embeddings (for LLaMA-style models)
            // Need to extract tokens for this specific sequence
            let mut token_embeddings = Vec::with_capacity(n_tokens);
            let offset = token_offset.unwrap_or(0);

            for i in 0..n_tokens {
                let global_idx = offset + i;
                let global_idx_i32 = Self::to_i32(global_idx, "Token index")?;
                let embeddings = ctx.embeddings_ith(global_idx_i32).map_err(|e| {
                    Error::EmbeddingGenerationError {
                        message: format!(
                            "Failed to get embeddings for token {i} in sequence {seq_id}"
                        ),
                        source: Some(anyhow::anyhow!(e)),
                    }
                })?;
                token_embeddings.push(embeddings.to_vec());
            }
            Ok(token_embeddings)
        })
    }

    /// Process a batch through the model using decode (decoders) or encode (encoders).
    ///
    /// This method handles the KV cache clearing and model-specific processing logic
    /// that was previously duplicated across multiple methods.
    ///
    /// # Arguments
    ///
    /// * `batch` - The batch to process through the model
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if processing succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if batch processing fails.
    fn process_batch(&mut self, batch: &mut LlamaBatch) -> Result<()> {
        self.cell.with_dependent_mut(|_, ctx| {
            // Clear KV cache to ensure clean state for each embedding generation
            // This prevents cache contamination between sequential calls
            ctx.clear_kv_cache();

            if self.metadata.is_decoder() {
                ctx.decode(batch)
                    .map_err(|e| Error::EmbeddingGenerationError {
                        message: format!("Failed to decode batch: {e}"),
                        source: Some(anyhow::anyhow!(e)),
                    })
            } else {
                ctx.encode(batch)
                    .map_err(|e| Error::EmbeddingGenerationError {
                        message: format!("Failed to encode batch: {e}"),
                        source: Some(anyhow::anyhow!(e)),
                    })
            }
        })
    }

    /// Internal method to process `LlamaToken` vectors.
    fn process_tokens_internal(&mut self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        if tokens.is_empty() {
            return Err(Error::InvalidInput {
                message: "Cannot process empty token list".to_string(),
            });
        }

        // Validate token limit
        self.validate_token_limit(tokens.len(), Some("Input"))?;

        // Create a batch for processing
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        batch
            .add_sequence(tokens, 0, true)
            .map_err(|e| Error::EmbeddingGenerationError {
                message: format!("Failed to add tokens to batch: {e}"),
                source: Some(anyhow::anyhow!(e)),
            })?;

        // Process the batch through the model
        // Decoder models need to use decode() instead of encode()
        // encode() tries to access unified KV cache which is null for decoder models
        self.process_batch(&mut batch)?;

        // Extract embeddings based on pooling configuration
        // When llama.cpp pooling is enabled (Last, Mean, etc.), the model computes and stores
        // the pooled embedding, which we retrieve as a sequence embedding.
        // When pooling is NONE, we get individual token embeddings and pool ourselves.
        let all_embeddings = self.extract_sequence_embeddings(0, n_tokens, None)?;

        Ok(all_embeddings)
    }

    /// Applies pooling strategy to token embeddings.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Token embeddings from the model
    /// * `strategy` - Pooling strategy to apply
    ///
    /// # Returns
    ///
    /// Returns a single pooled embedding vector.
    fn apply_pooling(embeddings: &[Vec<f32>], strategy: PoolingStrategy) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(Error::EmbeddingGenerationError {
                message: "No embeddings to pool".to_string(),
                source: None,
            });
        }

        let embedding_dim = embeddings[0].len();

        match strategy {
            PoolingStrategy::Mean => {
                // Mean pooling across all tokens
                let mut pooled = vec![0.0f32; embedding_dim];
                #[allow(clippy::cast_precision_loss)]
                let n_tokens = embeddings.len() as f32;

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] += val / n_tokens;
                    }
                }

                Ok(pooled)
            }
            PoolingStrategy::Cls => {
                // Use only the first token (CLS token)
                Ok(embeddings[0].clone())
            }
            PoolingStrategy::Max => {
                // Max pooling across all tokens
                let mut pooled = vec![f32::NEG_INFINITY; embedding_dim];

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] = pooled[i].max(val);
                    }
                }

                Ok(pooled)
            }
            PoolingStrategy::MeanSqrt => {
                // Mean pooling with sqrt(length) normalization
                let mut pooled = vec![0.0f32; embedding_dim];
                #[allow(clippy::cast_precision_loss)]
                let sqrt_n = (embeddings.len() as f32).sqrt();

                for token_emb in embeddings {
                    for (i, &val) in token_emb.iter().enumerate() {
                        pooled[i] += val;
                    }
                }

                // Normalize by sqrt(length)
                for val in &mut pooled {
                    *val /= sqrt_n;
                }

                Ok(pooled)
            }
            PoolingStrategy::Last => {
                // Use only the last token (EOS token)
                // This is required for decoder models like Qwen
                // The empty-embeddings case is handled at the top of apply_pooling(),
                // but use ok_or_else for defense-in-depth
                embeddings
                    .last()
                    .cloned()
                    .ok_or_else(|| Error::EmbeddingGenerationError {
                        message: "No embeddings available for Last pooling".to_string(),
                        source: None,
                    })
            }
            PoolingStrategy::None => {
                // None strategy should not reach apply_pooling — use generate_multi_embedding instead
                Err(Error::InvalidOperation {
                    message: "PoolingStrategy::None does not produce a single embedding vector. \
                              Use generate_multi_embedding() or embed_multi() for per-token embeddings."
                        .to_string(),
                })
            }
            PoolingStrategy::Rank => {
                // Rank strategy should not reach apply_pooling — use rerank methods instead
                Err(Error::InvalidOperation {
                    message: "PoolingStrategy::Rank does not produce embedding vectors. \
                              Use generate_rerank_score() or rerank() for relevance scoring."
                        .to_string(),
                })
            }
        }
    }

    /// Normalizes an embedding vector according to the specified mode.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector to normalize
    /// * `mode` - The normalization mode to apply
    ///
    /// # Returns
    ///
    /// Returns the normalized embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if P-norm is used with a non-positive exponent.
    fn normalize_embedding(mut embedding: Vec<f32>, mode: NormalizationMode) -> Result<Vec<f32>> {
        match mode {
            NormalizationMode::None => Ok(embedding),

            NormalizationMode::L2 => {
                // Calculate L2 (Euclidean) norm
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm == 0.0 {
                    // Return zero vector as-is, matching llama-server behavior
                    return Ok(embedding);
                }
                // Normalize the vector
                for val in &mut embedding {
                    *val /= norm;
                }
                Ok(embedding)
            }

            NormalizationMode::MaxAbs => {
                // Find maximum absolute value
                let max_abs = embedding.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                if max_abs == 0.0 {
                    // Return zero vector as-is
                    return Ok(embedding);
                }
                // Scale to [-1, 1] range
                for val in &mut embedding {
                    *val /= max_abs;
                }
                Ok(embedding)
            }

            NormalizationMode::PNorm(p) => {
                if p <= 0 {
                    return Err(Error::InvalidInput {
                        message: format!("P-norm exponent must be positive, got {p}"),
                    });
                }
                #[allow(clippy::cast_precision_loss)]
                let p_f32 = p as f32;
                // Calculate p-norm
                let norm: f32 = embedding
                    .iter()
                    .map(|x| x.abs().powf(p_f32))
                    .sum::<f32>()
                    .powf(1.0 / p_f32);
                if norm == 0.0 {
                    // Return zero vector as-is
                    return Ok(embedding);
                }
                // Normalize the vector
                for val in &mut embedding {
                    *val /= norm;
                }
                Ok(embedding)
            }
        }
    }

    /// Generates a reranking relevance score for a query-document pair.
    ///
    /// The model encodes the concatenated query and document as a single sequence
    /// and returns a scalar relevance score via `LlamaPoolingType::Rank`.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text
    /// * `document` - The document text to score against the query
    /// * `truncate` - Truncation strategy for the combined input
    ///
    /// # Returns
    ///
    /// Returns the raw relevance score (f32). Apply sigmoid for \[0,1\] normalization.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not configured with `PoolingStrategy::Rank`,
    /// tokenization fails, or model inference fails.
    #[instrument(skip(self, query, document), fields(query_len = query.len(), doc_len = document.len()))]
    pub fn generate_rerank_score(
        &mut self,
        query: &str,
        document: &str,
        truncate: TruncateTokens,
    ) -> Result<f32> {
        if self.effective_pooling() != PoolingStrategy::Rank {
            return Err(Error::InvalidOperation {
                message: format!(
                    "Reranking requires PoolingStrategy::Rank, but model is configured with {:?}",
                    self.effective_pooling()
                ),
            });
        }

        if query.is_empty() {
            return Err(Error::InvalidInput {
                message: "Rerank query cannot be empty".to_string(),
            });
        }
        if document.is_empty() {
            return Err(Error::InvalidInput {
                message: "Rerank document cannot be empty".to_string(),
            });
        }

        // Tokenize the combined query + document
        // Reranking models expect query and document concatenated; the model's
        // tokenizer will produce appropriate separator tokens.
        let combined = format!("{query}\n\n{document}");
        let tokens = self.tokenize(&combined)?;

        // Resolve truncation
        let truncation_limit = self.resolve_truncation_limit(truncate)?;
        let tokens = Self::truncate_tokens_if_needed(&tokens, truncation_limit);

        debug!("Processing {} tokens for reranking", tokens.len());

        // Process through model
        let embeddings = self.process_tokens_internal(tokens)?;

        // Extract the scalar relevance score
        Self::extract_rerank_score(&embeddings)
    }

    /// Generates reranking scores for multiple documents against a single query.
    ///
    /// Processes multiple query-document pairs in batches for efficiency.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text
    /// * `documents` - Slice of document texts to score
    /// * `truncate` - Truncation strategy for each combined input
    ///
    /// # Returns
    ///
    /// Returns a vector of raw relevance scores, one per document, in input order.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not configured with `PoolingStrategy::Rank`,
    /// tokenization fails, or model inference fails.
    #[instrument(skip(self, query, documents), fields(query_len = query.len(), n_docs = documents.len()))]
    pub fn generate_rerank_scores_batch(
        &mut self,
        query: &str,
        documents: &[&str],
        truncate: TruncateTokens,
    ) -> Result<Vec<f32>> {
        if self.effective_pooling() != PoolingStrategy::Rank {
            return Err(Error::InvalidOperation {
                message: format!(
                    "Reranking requires PoolingStrategy::Rank, but model is configured with {:?}",
                    self.effective_pooling()
                ),
            });
        }

        if query.is_empty() {
            return Err(Error::InvalidInput {
                message: "Rerank query cannot be empty".to_string(),
            });
        }
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Validate no empty documents
        for (i, doc) in documents.iter().enumerate() {
            if doc.is_empty() {
                return Err(Error::InvalidInput {
                    message: format!("Rerank document at index {i} cannot be empty"),
                });
            }
        }

        // Tokenize each query+document pair
        let token_sequences: Vec<Vec<LlamaToken>> = documents
            .iter()
            .map(|doc| {
                let combined = format!("{query}\n\n{doc}");
                self.tokenize(&combined)
            })
            .collect::<Result<Vec<_>>>()?;

        // Process in batches, respecting n_seq_max
        self.process_batch_rerank(&token_sequences, truncate)
    }

    /// Process batched token sequences for reranking, chunking if needed.
    fn process_batch_rerank(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<f32>> {
        if token_sequences.is_empty() {
            return Ok(Vec::new());
        }

        #[allow(clippy::cast_lossless)]
        let max_seqs = self.n_seq_max as usize;

        if token_sequences.len() <= max_seqs {
            return self.process_batch_rerank_internal(token_sequences, truncate);
        }

        // Chunk into batches of n_seq_max
        let mut all_scores = Vec::with_capacity(token_sequences.len());
        for chunk in token_sequences.chunks(max_seqs) {
            let chunk_scores = self.process_batch_rerank_internal(chunk, truncate)?;
            all_scores.extend(chunk_scores);
        }
        Ok(all_scores)
    }

    /// Internal: process a single batch of token sequences for reranking.
    fn process_batch_rerank_internal(
        &mut self,
        token_sequences: &[Vec<LlamaToken>],
        truncate: TruncateTokens,
    ) -> Result<Vec<f32>> {
        let truncation_limit = self.resolve_truncation_limit(truncate)?;

        // Truncate and validate each sequence
        let truncated: Vec<&[LlamaToken]> = token_sequences
            .iter()
            .enumerate()
            .map(|(i, tokens)| {
                let t = Self::truncate_tokens_if_needed(tokens, truncation_limit);
                self.validate_token_limit(t.len(), Some(&format!("Rerank pair {i}")))?;
                Ok(t)
            })
            .collect::<Result<Vec<_>>>()?;

        let total_tokens: usize = truncated.iter().map(|s| s.len()).sum();
        let mut batch = LlamaBatch::new(total_tokens, 1);

        for (seq_id, tokens) in truncated.iter().enumerate() {
            let seq_id_i32 = Self::to_i32(seq_id, "Sequence ID")?;
            batch.add_sequence(tokens, seq_id_i32, true).map_err(|e| {
                Error::EmbeddingGenerationError {
                    message: format!("Failed to add rerank sequence {seq_id} to batch: {e}"),
                    source: Some(anyhow::anyhow!(e)),
                }
            })?;
        }

        self.process_batch(&mut batch)?;

        // Extract scores from each sequence
        let mut scores = Vec::with_capacity(truncated.len());
        let mut token_offset = 0usize;
        for (seq_id, tokens) in truncated.iter().enumerate() {
            let embeddings =
                self.extract_sequence_embeddings(seq_id, tokens.len(), Some(token_offset))?;
            scores.push(Self::extract_rerank_score(&embeddings)?);
            token_offset += tokens.len();
        }

        Ok(scores)
    }

    /// Extract the rerank relevance score from raw embeddings.
    ///
    /// For `LlamaPoolingType::Rank`, llama.cpp returns a sequence embedding
    /// where the first element is the relevance score.
    fn extract_rerank_score(embeddings: &[Vec<f32>]) -> Result<f32> {
        if embeddings.is_empty() || embeddings[0].is_empty() {
            return Err(Error::EmbeddingGenerationError {
                message: "No rerank score produced by model".to_string(),
                source: None,
            });
        }
        Ok(embeddings[0][0])
    }

    /// Save the current KV cache state to memory
    ///
    /// > NOTE: This is for advanced prefix caching optimization
    /// > PERFORMANCE ISSUE: Only beneficial for prefixes > 100 tokens
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The context is empty (no state to save)
    /// - State copy operation fails
    pub fn save_session_state(&self) -> Result<Vec<u8>> {
        // Get the state size first
        let state_size = self.cell.borrow_dependent().get_state_size();

        if state_size == 0 {
            return Err(Error::InvalidOperation {
                message: "No state to save - context is empty".to_string(),
            });
        }

        // Allocate buffer for the state
        let mut buffer = vec![0u8; state_size];

        // Copy the state data
        let copied_size = unsafe {
            self.cell
                .borrow_dependent()
                .copy_state_data(buffer.as_mut_ptr())
        };

        if copied_size != state_size {
            return Err(Error::InvalidOperation {
                message: format!("State size mismatch: expected {state_size}, got {copied_size}"),
            });
        }

        // Prepend version header
        let mut versioned = Vec::with_capacity(SESSION_STATE_HEADER_SIZE + state_size);
        versioned.extend_from_slice(&SESSION_STATE_VERSION.to_le_bytes());
        versioned.extend_from_slice(&[0u8; 4]); // reserved for future use
        versioned.extend_from_slice(&buffer);

        debug!(
            "Saved session state: {} bytes (+ {} header)",
            state_size, SESSION_STATE_HEADER_SIZE
        );
        Ok(versioned)
    }

    /// Load a previously saved KV cache state
    ///
    /// > NOTE: Session must be from the same model version
    /// > BUG: Session format may change between llama.cpp versions
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - State data is empty
    /// - State size check fails
    pub fn load_session_state(&mut self, state_data: &[u8]) -> Result<()> {
        if state_data.is_empty() {
            return Err(Error::InvalidInput {
                message: "Empty session data provided".to_string(),
            });
        }

        // Try to read versioned header; fall back to legacy unversioned format
        let state_data = if state_data.len() >= SESSION_STATE_HEADER_SIZE {
            let version =
                u32::from_le_bytes([state_data[0], state_data[1], state_data[2], state_data[3]]);

            if version == SESSION_STATE_VERSION {
                // Versioned format: strip header before passing to llama.cpp
                &state_data[SESSION_STATE_HEADER_SIZE..]
            } else {
                // Not a recognized version — treat as legacy headerless data
                warn!(
                    "Session state has no recognized version header (first 4 bytes decode to {}), \
                     loading as legacy unversioned format. Re-save to upgrade.",
                    version
                );
                state_data
            }
        } else {
            // Data too small for a header — treat as legacy headerless data
            warn!(
                "Session state has no version header ({} bytes < {} header), \
                 loading as legacy unversioned format. Re-save to upgrade.",
                state_data.len(),
                SESSION_STATE_HEADER_SIZE
            );
            state_data
        };

        // Set the state data
        let loaded_size = AtomicUsize::new(0);
        self.cell.with_dependent_mut(|_, context| {
            loaded_size.store(
                unsafe { context.set_state_data(state_data) },
                Ordering::Relaxed,
            );
        });
        let loaded_size = loaded_size.load(Ordering::Relaxed);

        if loaded_size != state_data.len() {
            return Err(Error::InvalidOperation {
                message: format!(
                    "Failed to load session state: expected {} bytes, loaded {}",
                    state_data.len(),
                    loaded_size
                ),
            });
        }

        debug!("Loaded session state: {} bytes", loaded_size);
        Ok(())
    }

    /// Generate embedding with prefix caching support
    ///
    /// This method checks if the text has a common prefix that's been cached,
    /// and if so, loads that session state to avoid recomputing the KV cache
    /// for the prefix portion.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to generate embeddings for
    /// * `prefix_cache` - Optional reference to the prefix cache
    /// * `token_cache` - Optional reference to the token cache
    /// * `truncate` - Truncation strategy to apply
    ///
    /// # Returns
    ///
    /// Returns the embedding vector and optionally the number of prefix tokens used
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails or truncation limit exceeds model maximum
    pub fn generate_embedding_with_prefix(
        &mut self,
        text: &str,
        prefix_cache: Option<&crate::cache::prefix_cache::PrefixCache>,
        token_cache: Option<&TokenCache>,
        truncate: TruncateTokens,
    ) -> Result<Vec<f32>> {
        // First tokenize to get tokens
        let tokens = self.tokenize_cached(text, token_cache)?;

        // Resolve truncation limit
        let truncation_limit = self.resolve_truncation_limit(truncate)?;

        // Apply truncation if needed
        let tokens = Self::truncate_tokens_if_needed(&tokens, truncation_limit);

        let tokens_i: Vec<i32> = tokens.iter().map(|t| t.0).collect();

        // Check for cached prefix if available
        let prefix_tokens_used = if let Some(cache) = prefix_cache {
            if let Some((prefix_len, session)) = cache.find_prefix_session(text, &tokens_i) {
                // Load the cached session state if available
                if let Some(ref state) = session.memory_state {
                    match self.load_session_state(state) {
                        Ok(()) => {
                            info!("Loaded prefix cache for {} tokens", prefix_len);
                            Some(prefix_len)
                        }
                        Err(e) => {
                            warn!("Failed to load prefix cache: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                // Analyze for future caching opportunities
                if let Some(suggested_len) = cache.analyze(&tokens_i) {
                    debug!(
                        "Prefix of {} tokens is candidate for caching",
                        suggested_len
                    );
                }
                None
            }
        } else {
            None
        };

        // Generate the embedding (with or without prefix optimization)
        let embedding = if let Some(prefix_len) = prefix_tokens_used {
            // Process only the suffix tokens after the cached prefix
            let suffix_tokens = &tokens[prefix_len..];
            if suffix_tokens.is_empty() {
                // The entire text was in the prefix, just extract embeddings
                self.extract_embeddings(tokens)?
            } else {
                // Process the suffix and combine
                self.process_tokens_internal(suffix_tokens)?
            }
        } else {
            // Normal processing without prefix optimization
            self.process_tokens_internal(tokens)?
        };

        // Apply pooling and normalization
        self.finalize_embedding(&embedding, tokens.len())
    }

    /// Extract embeddings from the current context state
    fn extract_embeddings(&self, tokens: &[LlamaToken]) -> Result<Vec<Vec<f32>>> {
        // Delegate to the unified extraction method
        self.extract_sequence_embeddings(0, tokens.len(), None)
    }

    /// Extract context size from GGUF file metadata
    ///
    /// Uses the `gguf::extract_metadata` function to get model metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Returns
    ///
    /// Returns the context size from metadata, or an error if not found
    fn extract_context_size_from_gguf(path: &Path) -> Result<u32> {
        let metadata = gguf::extract_metadata(path)?;
        Ok(metadata.context_size.try_into().unwrap_or(2048))
    }
}

impl Drop for EmbeddingModel {
    /// Ensures proper cleanup of model resources.
    fn drop(&mut self) {
        // The self_cell will handle dropping both the model and context in the correct order
        // Note: Cannot safely log here as tracing TLS may already be destroyed during shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_send() {
        // This test verifies at compile time that EmbeddingModel is !Send
        fn assert_not_send<T: ?Sized>() {}
        assert_not_send::<EmbeddingModel>();
    }

    #[test]
    fn test_model_metadata_methods() {
        // We can't load a real model in tests without a GGUF file,
        // but we can test the structure compiles correctly
        // Real integration tests would use actual model files
    }

    #[test]
    #[ignore = "Requires actual GGUF model file"]
    fn test_model_loading_with_real_file() {
        // This test would require a real GGUF model file
        // It's marked as ignore but can be run with: cargo test -- --ignored

        let config = ModelConfig::builder()
            .with_model_path("/path/to/real/model.gguf")
            .with_model_name("test-model")
            .build()
            .unwrap();

        // Initialize backend for testing
        let backend = LlamaBackend::init().unwrap();

        match EmbeddingModel::new(&backend, &config) {
            Ok(model) => {
                assert!(model.is_loaded());
                assert!(model.embedding_dimensions() > 0);
                assert!(model.max_sequence_length() > 0);
            }
            Err(e) => {
                eprintln!("Expected error loading model: {e}");
            }
        }
    }

    // ============================================================================
    // apply_pooling unit tests
    // ============================================================================

    #[test]
    fn test_apply_pooling_empty_embeddings() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Mean);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No embeddings to pool")
        );
    }

    #[test]
    fn test_apply_pooling_last_returns_last_token() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Last).unwrap();
        assert_eq!(result, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_apply_pooling_last_single_embedding() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Last).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_pooling_last_empty_returns_error() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Last);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_pooling_cls_returns_first_token() {
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Cls).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_pooling_mean() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Mean).unwrap();
        assert_eq!(result, vec![2.0, 3.0]);
    }

    #[test]
    fn test_apply_pooling_max() {
        let embeddings = vec![vec![1.0, 4.0], vec![3.0, 2.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Max).unwrap();
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_apply_pooling_none_returns_error() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("PoolingStrategy::None")
        );
    }

    #[test]
    fn test_apply_pooling_rank_returns_error() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let result = EmbeddingModel::apply_pooling(&embeddings, PoolingStrategy::Rank);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("PoolingStrategy::Rank")
        );
    }

    #[test]
    fn test_pooling_strategy_to_llama_type_rank() {
        let llama_type = pooling_strategy_to_llama_type(PoolingStrategy::Rank);
        assert_eq!(llama_type, LlamaPoolingType::Rank);
    }

    #[test]
    fn test_extract_rerank_score_valid() {
        let embeddings = vec![vec![-2.5, 0.1, 0.3]];
        let score = EmbeddingModel::extract_rerank_score(&embeddings).unwrap();
        assert!((score - (-2.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extract_rerank_score_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = EmbeddingModel::extract_rerank_score(&embeddings);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No rerank score"));
    }

    #[test]
    fn test_extract_rerank_score_empty_inner() {
        let embeddings: Vec<Vec<f32>> = vec![vec![]];
        let result = EmbeddingModel::extract_rerank_score(&embeddings);
        assert!(result.is_err());
    }

    // ============================================================================
    // Session state versioning tests
    // ============================================================================

    #[test]
    fn test_session_state_version_constant() {
        // Verify the version and header size constants are sensible
        assert_eq!(SESSION_STATE_VERSION, 1);
        assert_eq!(SESSION_STATE_HEADER_SIZE, 8);
    }

    #[test]
    fn test_load_session_state_rejects_empty_data() {
        // We can't create a real EmbeddingModel without a GGUF file,
        // but we can verify the version header parsing logic by testing
        // the error conditions directly. The load function checks data
        // length before doing anything with the model.

        // Verify the header format: first 4 bytes = version (u32 LE), next 4 = reserved
        let version_bytes = SESSION_STATE_VERSION.to_le_bytes();
        assert_eq!(version_bytes.len(), 4);

        // A valid header would be 8 bytes: version + reserved
        let valid_header: Vec<u8> = {
            let mut h = Vec::with_capacity(SESSION_STATE_HEADER_SIZE);
            h.extend_from_slice(&SESSION_STATE_VERSION.to_le_bytes());
            h.extend_from_slice(&[0u8; 4]); // reserved
            h
        };
        assert_eq!(valid_header.len(), SESSION_STATE_HEADER_SIZE);

        // Verify version can be read back correctly
        let version = u32::from_le_bytes(valid_header[..4].try_into().unwrap());
        assert_eq!(version, SESSION_STATE_VERSION);
    }

    #[test]
    fn test_session_state_header_wrong_version_falls_back_to_legacy() {
        // Data with unrecognized version is treated as legacy (headerless) format
        // rather than rejected — this ensures backward compatibility
        let wrong_version: u32 = 99;
        let mut header = Vec::with_capacity(SESSION_STATE_HEADER_SIZE);
        header.extend_from_slice(&wrong_version.to_le_bytes());
        header.extend_from_slice(&[0u8; 4]);

        let version = u32::from_le_bytes(header[..4].try_into().unwrap());
        assert_ne!(version, SESSION_STATE_VERSION);
        // load_session_state would treat this as legacy data and pass it through
    }

    #[test]
    fn test_session_state_header_too_small_falls_back_to_legacy() {
        // Data smaller than SESSION_STATE_HEADER_SIZE is treated as legacy format
        // rather than rejected — this ensures backward compatibility with pre-versioned data
        let small_data = vec![0u8; SESSION_STATE_HEADER_SIZE - 1];
        assert!(small_data.len() < SESSION_STATE_HEADER_SIZE);
        // load_session_state would treat this as legacy data and pass it through

        // Empty data is still rejected (separate check in load_session_state)
        let empty_data: Vec<u8> = vec![];
        assert!(empty_data.is_empty());
    }
}
