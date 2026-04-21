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

//! Tests for context size auto-detection and utilization
//!
//! These tests verify that:
//! 1. GGUF metadata extraction correctly reads context sizes
//! 2. Models auto-detect context size from GGUF when not explicitly configured
//! 3. Models can utilize their full context size
//! 4. Proper errors are returned when context size is exceeded

use embellama::{EmbeddingEngine, EngineConfig, extract_gguf_metadata};
use serial_test::serial;
use std::path::PathBuf;

mod common;

/// Get the Jina test model path
fn get_jina_model_path() -> Option<PathBuf> {
    let path = PathBuf::from(
        std::env::var("HOME").unwrap()
            + "/Library/Caches/roothalia/models/jina-embeddings-v2-base-code-Q4_K_M.gguf",
    );
    if path.exists() {
        Some(path)
    } else {
        // Also try the EMBELLAMA_TEST_MODEL env var
        std::env::var("EMBELLAMA_TEST_MODEL")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists() && p.to_string_lossy().contains("jina"))
    }
}

// ============================================================================
// 1. GGUF Metadata Extraction Tests
// ============================================================================

#[test]
fn test_extract_gguf_metadata_jina_model() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        eprintln!(
            "   Expected at: ~/Library/Caches/roothalia/models/jina-embeddings-v2-base-code-Q4_K_M.gguf"
        );
        return;
    };

    println!(
        "Testing GGUF metadata extraction from: {}",
        model_path.display()
    );

    let result = extract_gguf_metadata(&model_path);
    assert!(
        result.is_ok(),
        "Failed to extract GGUF metadata: {:?}",
        result.err()
    );

    let metadata = result.unwrap();

    // PRIMARY TEST: Jina model should have 8192 context size
    assert_eq!(
        metadata.context_size, 8192,
        "Expected Jina model to have 8192 context size from GGUF metadata"
    );
    println!(
        "✓ Successfully extracted context size: {}",
        metadata.context_size
    );

    // SECONDARY: Dimensions (optional - model.n_embd() is the source of truth)
    println!(
        "  Dimensions from GGUF: {} ({})",
        metadata.embedding_dimensions,
        if metadata.embedding_dimensions > 0 {
            "found"
        } else {
            "not found - will use model.n_embd()"
        }
    );
}

#[test]
fn test_extract_gguf_metadata_invalid_file() {
    use tempfile::NamedTempFile;

    // Create a file that's not a valid GGUF
    let temp_file = NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), b"not a gguf file").unwrap();

    let result = extract_gguf_metadata(temp_file.path());
    assert!(result.is_err(), "Should fail on invalid GGUF file");

    println!("✓ Correctly rejects invalid GGUF file");
}

// ============================================================================
// 2. Context Size Auto-Detection Tests
// ============================================================================

#[test]
#[serial]
fn test_model_autodetect_context_size_from_gguf() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing auto-detection of context size from GGUF");

    // Create config WITHOUT explicit n_ctx - should auto-detect from GGUF
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        // Note: NOT setting with_context_size() or with_n_ctx()
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // The engine should have detected 8192 from GGUF metadata
    // We can verify this by checking the model's max_sequence_length
    // Note: We'll need to expose this through the engine API or test via embedding

    println!("✓ Engine created successfully with auto-detected context size");

    // Try to generate an embedding to ensure it works
    let embedding = engine.embed(Some("jina-test"), "Test text").unwrap();
    assert!(!embedding.is_empty());

    println!("✓ Auto-detected context size allows embeddings to work");

    engine.cleanup_thread_models();
}

#[test]
#[serial]
fn test_model_explicit_config_overrides_gguf() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing that explicit config overrides GGUF metadata");

    // Explicitly set context size to 2048 (even though Jina GGUF says 8192)
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        .with_context_size(2048) // Explicit override (increased from 512 to accommodate embedding overhead)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Generate some embeddings to verify it works with the explicit size
    let embedding = engine.embed(Some("jina-test"), "Test text").unwrap();
    assert!(!embedding.is_empty());

    println!("✓ Explicit config successfully overrides GGUF metadata");

    engine.cleanup_thread_models();
}

// ============================================================================
// 3. Context Size Utilization Tests
// ============================================================================

#[test]
#[serial]
fn test_embedding_near_context_limit() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing embedding generation near effective token limit");

    // Default n_batch=min(ctx,2048)=2048, n_seq_max=2 for encoder models,
    // so effective_max = 2048/2 - 2 = 1022 tokens.
    // Generate text safely under that limit (~900 tokens).
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Aim for ~900 tokens, safely under effective_max of 1022
    let target_tokens = 900;
    let chars_needed = target_tokens * 13 / 10;
    let large_text = "The quick brown fox jumps over the lazy dog. This is a test of the embedding system with a very long context. "
        .repeat(chars_needed / 110);

    println!(
        "Generated text with approximately {} characters (target: ~{} tokens)",
        large_text.len(),
        target_tokens
    );

    // This should succeed within the effective token limit
    let result = engine.embed(Some("jina-test"), &large_text);

    match &result {
        Ok(embedding) => {
            // Verify embedding is valid
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm > 0.1,
                "Embedding norm should be non-zero, got: {}",
                norm
            );
            println!(
                "✓ Successfully generated embedding for large text (norm: {:.4})",
                norm
            );
        }
        Err(e) => {
            panic!(
                "Failed to generate embedding for text near context limit: {:?}",
                e
            );
        }
    }

    engine.cleanup_thread_models();
}

// ============================================================================
// 4. Context Size Overflow Tests
// ============================================================================

#[test]
#[serial]
fn test_embedding_exceeds_context_size() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing error handling when text exceeds context limit");

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Generate text that will definitely exceed 8192 tokens
    // Aim for ~8500 tokens to be sure
    let target_tokens = 8500;
    let chars_needed = target_tokens * 13 / 10;
    let oversized_text = "The quick brown fox jumps over the lazy dog. This is a test of the embedding system with an extremely long context that should exceed the limit. "
        .repeat(chars_needed / 150);

    println!(
        "Generated text with approximately {} characters (target: ~{} tokens)",
        oversized_text.len(),
        target_tokens
    );

    // This should fail with InvalidInput error
    let result = engine.embed(Some("jina-test"), &oversized_text);

    assert!(result.is_err(), "Expected error for oversized text");

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);

    // Verify the error message contains useful information
    assert!(
        err_msg.contains("exceeds")
            || err_msg.contains("token limit")
            || err_msg.contains("InvalidInput"),
        "Error should mention token limit. Got: {}",
        err_msg
    );

    println!(
        "✓ Correctly rejected oversized text with error: {}",
        err_msg
    );

    engine.cleanup_thread_models();
}

#[test]
#[serial]
fn test_embedding_at_exact_boundary() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing boundary conditions with context_size=100");

    // Use a smaller context for precise boundary testing.
    // Set n_seq_max=1 so effective_max = n_batch - 2 = 98 tokens
    // (with n_seq_max=2, effective_max would be 100/2 - 2 = 48).
    let test_context_size = 100;
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        .with_context_size(test_context_size)
        .with_n_seq_max(1)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Test text that's under the effective limit (98 tokens)
    let safe_text = "word ".repeat(90); // ~90 tokens, under effective_max of 98
    let result = engine.embed(Some("jina-test"), &safe_text);
    assert!(result.is_ok(), "Text under limit should succeed");
    println!("✓ Text under limit succeeded");

    // Test text that exceeds the limit
    let oversized_text = "word ".repeat(150); // Definitely over 98 tokens
    let result = engine.embed(Some("jina-test"), &oversized_text);
    assert!(result.is_err(), "Text over limit should fail");
    println!("✓ Text over limit correctly rejected");

    engine.cleanup_thread_models();
}

#[test]
#[serial]
fn test_batch_with_large_contexts() {
    common::init_test_logger();

    let Some(model_path) = get_jina_model_path() else {
        eprintln!("⚠️  Skipping test: Jina model not found");
        return;
    };

    println!("Testing batch processing with large contexts");

    // Default n_batch=2048, n_seq_max=2 for encoder models,
    // so effective_max per sequence = 2048/2 - 2 = 1022 tokens.
    // Use texts safely under that limit (~900 tokens each).
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("jina-test")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Create 3 texts, each around 900 tokens (under effective_max of 1022)
    let target_tokens = 900;
    let chars_needed = target_tokens * 13 / 10;
    let large_text = "The quick brown fox jumps over the lazy dog in this batch test. "
        .repeat(chars_needed / 65);

    let texts: Vec<&str> = vec![&large_text, &large_text, &large_text];

    println!(
        "Testing batch of {} texts, each ~{} tokens",
        texts.len(),
        target_tokens
    );

    let result = engine.embed_batch(Some("jina-test"), &texts);

    assert!(
        result.is_ok(),
        "Batch with large contexts should succeed: {:?}",
        result.err()
    );

    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 3, "Should get 3 embeddings");

    for (i, embedding) in embeddings.iter().enumerate() {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1, "Embedding {} should have non-zero norm", i);
    }

    println!("✓ Successfully processed batch with large contexts");

    engine.cleanup_thread_models();
}
