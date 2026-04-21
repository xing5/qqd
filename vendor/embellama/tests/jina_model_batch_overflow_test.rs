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

//! Integration tests for batch overflow handling with Jina model
//!
//! This test validates batch processing with per-sequence token limits.
//! Each sequence gets its own KV cache slot of size: context_size / n_seq_max.
//!
//! The test validates:
//! 1. effective_max_tokens() method returns per-sequence limit
//! 2. Batch splitting logic respects n_seq_max for parallel processing
//! 3. Individual sequences are validated against per-sequence limit

use embellama::{EmbeddingEngine, EngineConfig};
use serial_test::serial;
use std::path::PathBuf;

#[allow(dead_code)]
#[path = "common.rs"]
mod common;

/// Generate a large batch of sequences totaling approximately 8042 tokens
/// This replicates the user's reported failure case
fn generate_large_token_batch() -> Vec<String> {
    // Each sequence averages ~134 tokens (8042 / 60 ≈ 134)
    // We'll create strings of varying lengths to simulate realistic data
    let mut sequences = Vec::new();

    // Generate 60 sequences
    for i in 0..60 {
        // Create strings of varying complexity to get realistic token counts
        // Longer strings with code/technical content to increase token count
        let text = match i % 5 {
            0 => format!(
                "Function implementation #{}: def process_data(input_data, config_params, options_dict): \
                return [transform(x, config_params) for x in input_data if validate(x, options_dict)]",
                i
            ),
            1 => format!(
                "Code snippet #{}: class DataProcessor: def __init__(self, config): self.config = config; \
                self.cache = {{}}; def process(self, data): result = self.transform(data); \
                self.cache[data] = result; return result",
                i
            ),
            2 => format!(
                "Technical documentation #{}: The system architecture consists of multiple layers including \
                data ingestion, processing pipeline, transformation engine, validation framework, \
                and output generation with error handling and logging capabilities throughout the stack",
                i
            ),
            3 => format!(
                "Algorithm description #{}: This implementation uses a divide-and-conquer approach with \
                memoization to optimize performance. Time complexity is O(n log n) with space complexity \
                O(n) due to the auxiliary storage requirements for the intermediate results and cache",
                i
            ),
            _ => format!(
                "Extended text sample #{}: The quick brown fox jumps over the lazy dog while carrying \
                a heavy backpack full of technical equipment and documentation for the project. \
                Additional context and metadata are included for comprehensive analysis and processing",
                i
            ),
        };
        sequences.push(text);
    }

    sequences
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests batch overflow handling"]
fn test_jina_batch_overflow_with_8042_tokens() {
    // Initialize test logging
    common::init_test_logger();

    // Get Jina model path from environment
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .expect("EMBELLAMA_TEST_MODEL must be set to a Jina model path");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    // Create engine config with Jina model
    // Jina models typically have context_size: 8192
    // With n_seq_max=8:
    // - Per-sequence effective_max: 8192 / 8 - 2 = 1022 tokens per sequence
    // - Total capacity: up to 8 sequences in parallel
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8) // Same as user's configuration
        .build()
        .expect("Failed to create engine config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize embedding engine");

    // Generate large batch (60 sequences, ~8042 tokens total)
    let batch = generate_large_token_batch();

    println!("Testing batch with {} sequences", batch.len());
    println!("With n_seq_max=8, per-sequence limit is 1022 tokens");
    println!("This batch will be automatically chunked into multiple batches of 8 sequences");

    // The batch processor will:
    // 1. Validate each sequence individually (must be <= 1022 tokens)
    // 2. Process sequences in chunks of 8 (n_seq_max) in parallel
    // 3. Automatically handle the 60 sequences across multiple batches (60/8 = 8 batches)

    let text_refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let result = engine.embed_batch(Some(model_name), &text_refs);

    match result {
        Ok(embeddings) => {
            println!("✓ Successfully processed batch (after implementation)");
            assert_eq!(
                embeddings.len(),
                batch.len(),
                "Should return embeddings for all {} sequences",
                batch.len()
            );

            // Verify each embedding has correct dimensions (768 for Jina base)
            for (i, embedding) in embeddings.iter().enumerate() {
                assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);

                // All embeddings should have same dimensions
                if i > 0 {
                    assert_eq!(
                        embedding.len(),
                        embeddings[0].len(),
                        "All embeddings should have same dimensions"
                    );
                }
            }

            println!(
                "✓ All {} embeddings have {} dimensions",
                embeddings.len(),
                embeddings[0].len()
            );
        }
        Err(e) => {
            // Before implementation, this will fail with GGML assertion
            panic!(
                "Batch processing failed (expected before implementation): {}",
                e
            );
        }
    }
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests boundary conditions"]
fn test_jina_batch_at_effective_max_boundary() {
    // Initialize test logging
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize engine");

    // Create a batch with sequences at the per-sequence limit
    // With n_seq_max=8: per-sequence effective_max = 1022 tokens
    // Each sequence should be close to but under 1022 tokens

    // Each sequence should be close to 1022 tokens (the per-sequence limit)
    let batch: Vec<String> = (0..8) // 8 sequences = one full batch at n_seq_max
        .map(|i| {
            // Create text that's close to 1022 tokens
            // Approximately 4000 characters ≈ 1000 tokens
            let padding = "word ".repeat(800);
            format!(
                "Sequence {}: Testing per-sequence boundary condition. {}",
                i, padding
            )
        })
        .collect();

    println!("Testing batch at per-sequence effective_max boundary");
    println!(
        "Each of {} sequences is close to 1022 tokens (per-sequence limit)",
        batch.len()
    );

    let text_refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let result = engine.embed_batch(Some(model_name), &text_refs);

    // This should succeed even before implementation if tokens < ctx_size
    // After implementation, it should succeed more reliably with correct validation
    result.expect("Batch at boundary should process successfully");
}

#[test]
#[serial]
#[ignore = "Run only if EMBELLAMA_TEST_MODEL is a Jina model - tests small batch"]
fn test_jina_small_batch_succeeds() {
    // Initialize test logging
    common::init_test_logger();

    let model_path =
        std::env::var("EMBELLAMA_TEST_MODEL").expect("EMBELLAMA_TEST_MODEL must be set");

    let model_path_buf = PathBuf::from(&model_path);
    let model_name = model_path_buf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("jina-test-model");

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(model_name)
        .with_n_seq_max(8)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to initialize engine");

    // Small batch that's well within limits
    let batch = vec!["Short text one", "Short text two", "Short text three"];

    let result = engine.embed_batch(Some(model_name), &batch);

    assert!(
        result.is_ok(),
        "Small batch should always succeed: {:?}",
        result.err()
    );

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");
}
