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

//! Integration tests for token truncation functionality
//!
//! These tests verify that the truncation feature works correctly end-to-end,
//! including configuration, model-level truncation, batch processing, and caching.

mod common;

use embellama::{EmbeddingEngine, EngineConfig, TruncateTokens};
use serial_test::serial;
use std::path::PathBuf;

/// Get model path or skip test
fn get_model_path() -> Option<PathBuf> {
    std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

// ============================================================================
// Basic Truncation Tests
// ============================================================================

#[test]
#[serial]
fn test_truncation_no() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-no")
        .with_truncate_tokens(TruncateTokens::No)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Short text should work fine
    let short_text = "This is a short text.";
    let embedding = engine.embed(None, short_text).unwrap();
    assert!(!embedding.is_empty());

    // Very long text should fail with No truncation
    let long_text = common::generate_text_with_approx_tokens(10000);
    let result = engine.embed(None, &long_text);
    assert!(
        result.is_err(),
        "Expected error for overly long text with TruncateTokens::No"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("exceeds effective maximum"),
        "Error should mention exceeding maximum: {err}"
    );
}

#[test]
#[serial]
fn test_truncation_yes() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-yes")
        .with_truncate_tokens(TruncateTokens::Yes)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Very long text should succeed with Yes truncation
    let long_text = common::generate_text_with_approx_tokens(10000);
    let embedding = engine.embed(None, &long_text).unwrap();
    assert!(!embedding.is_empty());

    // Verify embedding is normalized (assuming L2 normalization)
    common::assert_normalized_strict(&embedding);
}

#[test]
#[serial]
fn test_truncation_limit_valid() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    // Use a small limit that should be within the model's capacity
    let truncate_limit = 100;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-limit")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Text that would exceed the limit
    let long_text = common::generate_text_with_approx_tokens(500);
    let embedding = engine.embed(None, &long_text).unwrap();
    assert!(!embedding.is_empty());

    // Verify embedding is valid
    common::assert_normalized_strict(&embedding);
}

#[test]
#[serial]
fn test_truncation_limit_exceeds_max() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    // Use a very large limit that exceeds the model's effective_max_tokens
    let truncate_limit = 100_000;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-limit-exceeds")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Any text should fail because the limit exceeds effective_max_tokens
    let text = "This is a test.";
    let result = engine.embed(None, text);
    assert!(
        result.is_err(),
        "Expected error when truncation limit exceeds effective_max_tokens"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Truncation limit") && err.to_string().contains("exceeds"),
        "Error should mention truncation limit exceeding maximum: {err}"
    );
}

// ============================================================================
// Truncation Behavior Tests
// ============================================================================

#[test]
#[serial]
fn test_truncation_preserves_prefix() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let truncate_limit = 50;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-prefix")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Create two texts that share the same prefix
    let prefix = "The quick brown fox jumps over the lazy dog. This is the common prefix. ";
    let text1 = format!(
        "{prefix}This is unique suffix 1 with additional content to ensure truncation happens."
    );
    let text2 =
        format!("{prefix}This is unique suffix 2 with different additional content for testing.");

    let embedding1 = engine.embed(None, &text1).unwrap();
    let embedding2 = engine.embed(None, &text2).unwrap();

    // Both embeddings should be valid
    assert!(!embedding1.is_empty());
    assert!(!embedding2.is_empty());
    assert_eq!(embedding1.len(), embedding2.len());

    // The embeddings should be similar (high cosine similarity) since they share
    // the truncated prefix
    let similarity = common::cosine_similarity(&embedding1, &embedding2);
    assert!(
        similarity > 0.9,
        "Embeddings with same truncated prefix should be very similar. Got similarity: {similarity}"
    );
}

#[test]
#[serial]
fn test_truncation_exact_boundary() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let truncate_limit = 100;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-boundary")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Generate text that's approximately at the truncation boundary
    let text = common::generate_text_with_approx_tokens(100);
    let embedding = engine.embed(None, &text).unwrap();

    assert!(!embedding.is_empty());
    common::assert_normalized_strict(&embedding);
}

#[test]
#[serial]
fn test_truncation_below_limit() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let truncate_limit = 500;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-below")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Short text well below the limit
    let short_text = "This is a very short text.";
    let embedding1 = engine.embed(None, short_text).unwrap();

    // Same short text should produce identical embedding (no truncation applied)
    let embedding2 = engine.embed(None, short_text).unwrap();

    common::assert_embeddings_identical(&embedding1, &embedding2);
}

#[test]
#[serial]
fn test_truncation_embeddings_valid() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-valid")
        .with_truncate_tokens(TruncateTokens::Yes)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Generate embeddings for various lengths
    let long_text1 = common::generate_text_with_approx_tokens(500);
    let long_text2 = common::generate_text_with_approx_tokens(2000);
    let texts = vec![
        "Short.",
        "Medium length text for testing embeddings.",
        &long_text1,
        &long_text2,
    ];

    for (i, text) in texts.iter().enumerate() {
        let embedding = engine
            .embed(None, text)
            .unwrap_or_else(|e| panic!("Failed to generate embedding for text {i}: {e}"));

        // Verify embedding is valid
        assert!(!embedding.is_empty(), "Embedding {i} is empty");
        assert!(embedding.len() > 0, "Embedding {i} has zero dimensions");

        // Verify embedding is normalized
        common::assert_normalized_strict(&embedding);

        // Verify no NaN or Inf values
        for (j, &val) in embedding.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Embedding {i} has non-finite value at index {j}: {val}"
            );
        }
    }
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

#[test]
#[serial]
fn test_batch_truncation_per_sequence() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-batch-truncation")
        .with_truncate_tokens(TruncateTokens::Yes)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Create batch with varying lengths, all exceeding normal limits
    let texts = vec![
        common::generate_text_with_approx_tokens(5000),
        common::generate_text_with_approx_tokens(3000),
        common::generate_text_with_approx_tokens(8000),
    ];
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let embeddings = engine.embed_batch(None, &text_refs).unwrap();

    assert_eq!(embeddings.len(), texts.len());

    // Each embedding should be valid
    for (i, embedding) in embeddings.iter().enumerate() {
        assert!(!embedding.is_empty(), "Batch embedding {i} is empty");
        common::assert_normalized_strict(embedding);
    }
}

#[test]
#[serial]
fn test_batch_truncation_mixed_lengths() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let truncate_limit = 200;

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-batch-mixed")
        .with_truncate_limit(truncate_limit)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // Mix of short and long texts
    let long1 = common::generate_text_with_approx_tokens(50);
    let long2 = common::generate_text_with_approx_tokens(500);
    let long3 = common::generate_text_with_approx_tokens(1000);
    let texts = vec!["Short text.", &long1, &long2, "Another short one.", &long3];

    let embeddings = engine.embed_batch(None, &texts).unwrap();

    assert_eq!(embeddings.len(), texts.len());

    // All embeddings should have same dimensionality
    let dim = embeddings[0].len();
    for embedding in &embeddings {
        assert_eq!(embedding.len(), dim);
        common::assert_normalized_strict(embedding);
    }
}

// ============================================================================
// Cache Integration Tests
// ============================================================================

#[test]
#[serial]
fn test_truncation_with_token_cache() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    use embellama::CacheConfig;

    let cache_config = CacheConfig::builder()
        .with_enabled(true)
        .with_token_cache_size(1000)
        .build()
        .unwrap();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-token-cache")
        .with_truncate_tokens(TruncateTokens::Yes)
        .with_cache_config(cache_config)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let long_text = common::generate_text_with_approx_tokens(5000);

    // First call - should populate cache
    let embedding1 = engine.embed(None, &long_text).unwrap();

    // Second call - should use cached tokens
    let embedding2 = engine.embed(None, &long_text).unwrap();

    // Should produce identical embeddings
    common::assert_embeddings_identical(&embedding1, &embedding2);
}

#[test]
#[serial]
fn test_truncation_with_embedding_cache() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    use embellama::CacheConfig;

    let cache_config = CacheConfig::builder()
        .with_enabled(true)
        .with_embedding_cache_size(1000)
        .build()
        .unwrap();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-embedding-cache")
        .with_truncate_tokens(TruncateTokens::Yes)
        .with_cache_config(cache_config)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let long_text = common::generate_text_with_approx_tokens(5000);

    // First call - should populate cache
    let embedding1 = engine.embed(None, &long_text).unwrap();

    // Second call - should use cached embedding
    let embedding2 = engine.embed(None, &long_text).unwrap();

    // Should produce identical embeddings
    common::assert_embeddings_identical(&embedding1, &embedding2);
}

#[test]
#[serial]
fn test_truncation_deterministic() {
    common::init_test_logger();

    let Some(model_path) = get_model_path() else {
        eprintln!("⚠️  Skipping test: Set EMBELLAMA_TEST_MODEL to run this test");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-truncation-deterministic")
        .with_truncate_limit(150)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let text = common::generate_text_with_approx_tokens(1000);

    // Generate embedding multiple times
    let embedding1 = engine.embed(None, &text).unwrap();
    let embedding2 = engine.embed(None, &text).unwrap();
    let embedding3 = engine.embed(None, &text).unwrap();

    // All should be identical (deterministic)
    common::assert_embeddings_identical(&embedding1, &embedding2);
    common::assert_embeddings_identical(&embedding2, &embedding3);
}
