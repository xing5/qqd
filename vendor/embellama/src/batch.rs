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

//! Batch processing module for the embellama library.
//!
//! This module provides efficient batch processing capabilities for generating
//! embeddings for multiple texts simultaneously. It leverages parallel processing
//! for pre/post-processing while respecting the single-threaded constraint of
//! model inference.

use crate::config::{PoolingStrategy, TruncateTokens};
use crate::error::{Error, Result};
use crate::model::EmbeddingModel;
use llama_cpp_2::token::LlamaToken;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{debug, instrument};

/// Represents a batch of texts to be processed.
pub struct BatchProcessor {
    /// Maximum number of texts to process in a single batch
    #[allow(dead_code)]
    max_batch_size: usize,
    /// Optional progress callback for tracking batch processing
    progress_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    /// Whether to normalize embeddings
    #[allow(dead_code)]
    normalize: bool,
    /// Pooling strategy to apply
    #[allow(dead_code)]
    pooling_strategy: PoolingStrategy,
}

impl BatchProcessor {
    /// Creates a new batch processor with the specified maximum batch size.
    ///
    /// # Arguments
    ///
    /// * `max_batch_size` - Maximum number of texts to process in a single batch
    ///
    /// # Returns
    ///
    /// Returns a new `BatchProcessor` instance.
    pub fn new(max_batch_size: usize) -> Self {
        BatchProcessor {
            max_batch_size,
            progress_callback: None,
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }

    /// Creates a batch processor with custom configuration.
    pub fn builder() -> BatchProcessorBuilder {
        BatchProcessorBuilder::default()
    }

    /// Processes a batch of texts to generate embeddings.
    ///
    /// This function implements the following pipeline:
    /// 1. Parallel tokenization of all texts (using rayon)
    /// 2. True batch model inference (multiple sequences in single pass)
    /// 3. Parallel post-processing and normalization (handled by model)
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model to use
    /// * `texts` - Vector of texts to process
    /// * `truncate` - Truncation strategy to apply to all texts
    ///
    /// # Returns
    ///
    /// Returns a vector of embedding vectors, maintaining the input order.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Any text fails tokenization
    /// - Model inference fails
    /// - Memory allocation fails
    /// - Truncation limit exceeds model's effective maximum
    #[instrument(skip(self, model, texts), fields(batch_size = texts.len()))]
    pub fn process_batch(
        &self,
        model: &mut EmbeddingModel,
        texts: &[&str],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Processing batch of {} texts", texts.len());

        // Progress tracking
        let progress_counter = Arc::new(AtomicUsize::new(0));
        let total = texts.len();

        // Step 1: Parallel validation
        Self::parallel_validate(texts)?;

        // Step 2: Parallel tokenization using the real tokenizer
        let token_sequences = Self::parallel_tokenize_real(model, texts, truncate)?;

        // Step 3: Check if we need to chunk the batch based on n_seq_max
        // Each sequence is validated individually against effective_max_tokens in process_batch_tokens_internal
        let n_seq_max = model.n_seq_max() as usize;

        debug!(
            "Batch validation: {} sequences, n_seq_max: {}",
            token_sequences.len(),
            n_seq_max
        );

        let embeddings = if token_sequences.len() <= n_seq_max {
            // Process all sequences in a single batch
            debug!(
                "Processing {} sequences in single batch",
                token_sequences.len()
            );

            let batch_embeddings = model.process_batch_tokens(&token_sequences, truncate)?;

            // Update progress
            let current = progress_counter.fetch_add(texts.len(), Ordering::Relaxed);
            if let Some(ref callback) = self.progress_callback {
                callback(current + texts.len(), total);
            }

            batch_embeddings
        } else {
            // Need to chunk into smaller batches based on n_seq_max
            debug!(
                "Chunking batch: {} sequences (n_seq_max {})",
                token_sequences.len(),
                n_seq_max
            );

            let mut all_embeddings = Vec::with_capacity(texts.len());
            let mut current_batch = Vec::new();

            for seq in token_sequences {
                // Check if adding this sequence would exceed n_seq_max
                if !current_batch.is_empty() && current_batch.len() >= n_seq_max {
                    // Process current batch
                    let batch_embeddings = model.process_batch_tokens(&current_batch, truncate)?;
                    let batch_len = batch_embeddings.len();
                    all_embeddings.extend(batch_embeddings);

                    // Update progress
                    let current = progress_counter.fetch_add(batch_len, Ordering::Relaxed);
                    if let Some(ref callback) = self.progress_callback {
                        callback(current + batch_len, total);
                    }

                    // Start new batch
                    current_batch = vec![seq];
                } else {
                    // Add to current batch
                    current_batch.push(seq);
                }
            }

            // Process remaining batch
            if !current_batch.is_empty() {
                let batch_embeddings = model.process_batch_tokens(&current_batch, truncate)?;
                let batch_len = batch_embeddings.len();

                // Update progress
                let current = progress_counter.fetch_add(batch_len, Ordering::Relaxed);
                if let Some(ref callback) = self.progress_callback {
                    callback(current + batch_len, total);
                }

                all_embeddings.extend(batch_embeddings);
            }

            all_embeddings
        };

        debug!("Completed batch processing of {} texts", texts.len());
        Ok(embeddings)
    }

    /// Sets a progress callback for batch processing.
    ///
    /// The callback will be called with (`current_index`, `total_count`) during processing.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that receives progress updates
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
    }

    /// Validates texts in parallel.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of texts to validate
    ///
    /// # Returns
    ///
    /// Returns Ok if all texts are valid, Err otherwise.
    #[instrument(skip(texts), fields(count = texts.len()))]
    fn parallel_validate(texts: &[&str]) -> Result<()> {
        debug!("Validating {} texts in parallel", texts.len());

        // Parallel validation with error handling
        texts.par_iter().try_for_each(|text| {
            if text.is_empty() {
                return Err(Error::InvalidInput {
                    message: "Cannot process empty text".to_string(),
                });
            }
            // Could add more validation here (e.g., max length check)
            Ok(())
        })?;

        Ok(())
    }

    /// Tokenizes texts using the real model tokenizer.
    ///
    /// Note: Due to !Send constraint on model, tokenization happens sequentially
    /// but validation can still happen in parallel.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for tokenization
    /// * `texts` - Vector of texts to tokenize
    ///
    /// # Returns
    ///
    /// Returns a vector of tokenized sequences.
    #[instrument(skip(model, texts), fields(count = texts.len()))]
    fn parallel_tokenize_real(
        model: &EmbeddingModel,
        texts: &[&str],
        truncate: TruncateTokens,
    ) -> Result<Vec<Vec<LlamaToken>>> {
        debug!("Starting tokenization of {} texts", texts.len());

        let max_seq_len = model.max_sequence_length();

        // First, validate all texts in parallel
        let validation_results: Result<Vec<_>> = texts
            .par_iter()
            .map(|text| {
                if text.is_empty() {
                    Err(Error::InvalidInput {
                        message: "Cannot tokenize empty text".to_string(),
                    })
                } else {
                    Ok(())
                }
            })
            .collect();
        validation_results?;

        // Then tokenize sequentially (due to !Send constraint on model)
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            // Use real tokenizer from model
            let mut tokens = model.tokenize(text)?;

            // Apply truncation or check token limit
            match truncate {
                TruncateTokens::Yes => {
                    if tokens.len() > max_seq_len {
                        tokens.truncate(max_seq_len);
                    }
                }
                TruncateTokens::Limit(n) => {
                    let limit = n as usize;
                    if tokens.len() > limit {
                        tokens.truncate(limit);
                    }
                }
                TruncateTokens::No => {
                    if tokens.len() > max_seq_len {
                        return Err(Error::InvalidInput {
                            message: format!(
                                "Text exceeds maximum token limit: {} > {}",
                                tokens.len(),
                                max_seq_len
                            ),
                        });
                    }
                }
            }

            results.push(tokens);
        }

        debug!("Completed tokenization");
        Ok(results)
    }
}

/// Builder for creating configured `BatchProcessor` instances.
#[derive(Default)]
pub struct BatchProcessorBuilder {
    max_batch_size: Option<usize>,
    normalize: bool,
    pooling_strategy: PoolingStrategy,
    progress_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

impl BatchProcessorBuilder {
    /// Sets the maximum batch size.
    #[must_use]
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Sets whether to normalize embeddings.
    #[must_use]
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets the pooling strategy.
    #[must_use]
    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = strategy;
        self
    }

    /// Sets the progress callback.
    #[must_use]
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Builds the `BatchProcessor`.
    pub fn build(self) -> BatchProcessor {
        BatchProcessor {
            max_batch_size: self.max_batch_size.unwrap_or(32),
            progress_callback: self.progress_callback,
            normalize: self.normalize,
            pooling_strategy: self.pooling_strategy,
        }
    }
}

/// Utilities for batch processing optimization.
pub mod utils {
    // TODO: Phase 4 - Uncomment when implementing
    // use super::*;

    /// Calculates optimal batch size based on available memory and model size.
    ///
    /// # Arguments
    ///
    /// * `model_size_mb` - Size of the model in megabytes
    /// * `embedding_dim` - Dimension of embeddings
    /// * `available_memory_mb` - Available memory in megabytes
    ///
    /// # Returns
    ///
    /// Returns the recommended batch size.
    #[allow(dead_code)]
    pub fn calculate_optimal_batch_size(
        _model_size_mb: usize,
        _embedding_dim: usize,
        _available_memory_mb: usize,
    ) -> usize {
        // TODO: Phase 4 - Implement batch size calculation
        32 // Default placeholder
    }

    /// Chunks a large vector of texts into smaller batches.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of texts to chunk
    /// * `chunk_size` - Size of each chunk
    ///
    /// # Returns
    ///
    /// Returns an iterator over chunks.
    #[allow(dead_code)]
    pub fn chunk_texts<'a>(
        texts: &'a [&'a str],
        chunk_size: usize,
    ) -> impl Iterator<Item = &'a [&'a str]> {
        texts.chunks(chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::utils;

    #[test]
    fn test_chunk_texts() {
        let texts = vec!["a", "b", "c", "d", "e"];
        let chunks: Vec<_> = utils::chunk_texts(&texts, 2).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], &["a", "b"]);
        assert_eq!(chunks[1], &["c", "d"]);
        assert_eq!(chunks[2], &["e"]);
    }

    #[test]
    #[ignore = "Will be enabled in Phase 4"]
    fn test_batch_processing() {
        // TODO: Phase 4 - Add actual batch processing tests
    }
}
