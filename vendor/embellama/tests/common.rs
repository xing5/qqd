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

//! Common test utilities and fixtures
//!
//! Each integration test file that includes `mod common;` compiles this module
//! independently. Functions not used by a particular test file would trigger
//! dead_code warnings, so we allow them at the module level.

#![allow(dead_code)]

use embellama::{EngineConfig, NormalizationMode, PoolingStrategy};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Creates a dummy model file for testing purposes
///
/// # Panics
///
/// Panics if temp directory creation or file writing fails
#[must_use]
pub fn create_dummy_model() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = dir.path().join("test_model.gguf");

    // Create a minimal GGUF file structure (simplified for testing)
    // > NOTE: This is not a valid GGUF file but sufficient for path validation tests
    fs::write(&model_path, b"GGUF\x00\x00\x00\x04dummy_model_content")
        .expect("Failed to write dummy model");

    (dir, model_path)
}

/// Creates a test configuration with sensible defaults
///
/// # Panics
///
/// Panics if configuration building fails
#[must_use]
pub fn create_test_config(model_path: PathBuf) -> EngineConfig {
    EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_n_threads(2)
        .with_normalization_mode(NormalizationMode::L2)
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .expect("Failed to create test config")
}

/// Gets the path to a real test model if available
/// Returns None if `EMBELLAMA_TEST_MODEL` environment variable is not set
pub fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("EMBELLAMA_TEST_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Checks if real model tests should be run
#[must_use]
pub fn should_run_model_tests() -> bool {
    get_test_model_path().is_some()
}

/// Initialize test logging
/// Uses global tracing subscriber to prevent TLS issues
///
/// # Panics
///
/// Panics if logger initialization fails
pub fn init_test_logger() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive("embellama=debug".parse().unwrap()),
            )
            .with_test_writer()
            .init();
    });
}

/// Generate sample texts for batch testing
#[must_use]
pub fn generate_sample_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Sample text number {i} for testing embeddings"))
        .collect()
}

/// Assert that two embedding vectors are approximately equal
///
/// # Panics
///
/// Panics if embeddings have different dimensions or values differ by more than tolerance
pub fn assert_embeddings_equal(emb1: &[f32], emb2: &[f32], tolerance: f32) {
    assert_eq!(
        emb1.len(),
        emb2.len(),
        "Embeddings have different dimensions"
    );

    for (i, (a, b)) in emb1.iter().zip(emb2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < tolerance,
            "Embedding values differ at index {i}: {a} vs {b} (diff: {diff})"
        );
    }
}

/// Calculate L2 norm of an embedding vector
#[must_use]
pub fn calculate_l2_norm(embedding: &[f32]) -> f32 {
    embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Calculate cosine similarity between two vectors
///
/// For normalized vectors, this is equivalent to the dot product.
///
/// # Panics
///
/// Panics if vectors have different dimensions
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate dot product between two vectors
///
/// For normalized vectors, this equals cosine similarity.
///
/// # Panics
///
/// Panics if vectors have different dimensions
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Assert embeddings are identical (bit-for-bit equality)
///
/// This is stricter than assert_embeddings_equal as it requires exact float equality.
/// Use this to test determinism and cache consistency.
///
/// # Panics
///
/// Panics if embeddings differ in any way
pub fn assert_embeddings_identical(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "Embeddings have different dimensions");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x, y, "Embeddings differ at index {}: {} != {}", i, x, y);
    }
}

/// Assert embedding is normalized with strict tolerance (1e-6)
///
/// Use this for validating that embeddings are properly normalized.
///
/// # Panics
///
/// Panics if L2 norm is not within 1e-6 of 1.0
pub fn assert_normalized_strict(embedding: &[f32]) {
    let norm = calculate_l2_norm(embedding);
    const STRICT_TOLERANCE: f32 = 1e-6;
    assert!(
        (norm - 1.0).abs() < STRICT_TOLERANCE,
        "Embedding not normalized: L2 norm = {} (expected 1.0 ± {})",
        norm,
        STRICT_TOLERANCE
    );
}

/// Generate text that tokenizes to approximately N tokens
///
/// Uses repetitive pattern to be predictable. This is useful for testing
/// context size limits.
///
/// # Arguments
///
/// * `target_tokens` - Approximate number of tokens desired
///
/// # Returns
///
/// A string that should tokenize to roughly `target_tokens` tokens
///
/// # Note
///
/// This uses a rough estimate of ~1.3 characters per token for English text.
/// Actual token count may vary depending on the tokenizer.
#[must_use]
pub fn generate_text_with_approx_tokens(target_tokens: usize) -> String {
    // Rough estimate: ~1.3 chars per token for English
    let chars_needed = target_tokens * 13 / 10;
    "The quick brown fox jumps over the lazy dog. This is a test sentence for context size testing. "
        .repeat(chars_needed / 95)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dummy_model() {
        let (_dir, model_path) = create_dummy_model();
        assert!(model_path.exists());
        assert!(
            model_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        );
    }

    #[test]
    fn test_generate_sample_texts() {
        let texts = generate_sample_texts(5);
        assert_eq!(texts.len(), 5);
        assert!(texts[0].contains("Sample text number 0"));
        assert!(texts[4].contains("Sample text number 4"));
    }

    #[test]
    fn test_l2_norm_calculation() {
        let embedding = vec![0.6, 0.8]; // 3-4-5 triangle
        let norm = calculate_l2_norm(&embedding);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_assert_embeddings_equal() {
        let emb1 = vec![0.1, 0.2, 0.3];
        let emb2 = vec![0.1001, 0.2001, 0.3001];
        assert_embeddings_equal(&emb1, &emb2, 0.001);
    }

    #[test]
    fn test_generate_text_with_approx_tokens() {
        let text = generate_text_with_approx_tokens(1000);
        // Should generate roughly 1000 tokens worth of text
        // At ~1.3 chars per token, that's ~1300 chars
        assert!(text.len() > 1000, "Text should be at least 1000 chars");
        assert!(text.len() < 2000, "Text should be less than 2000 chars");
    }

    #[test]
    fn test_cosine_similarity() {
        // Test with normalized vectors
        let a = vec![0.6, 0.8];
        let b = vec![0.8, 0.6];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.96).abs() < 0.01); // 0.6*0.8 + 0.8*0.6 = 0.96

        // Test identical vectors
        let c = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&c, &c) - 1.0).abs() < 0.001);

        // Test orthogonal vectors
        let d = vec![1.0, 0.0];
        let e = vec![0.0, 1.0];
        assert!(cosine_similarity(&d, &e).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 0.001); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_equals_cosine_for_normalized() {
        // For normalized vectors, dot product == cosine similarity
        let a = vec![0.6, 0.8]; // Already normalized
        let b = vec![0.8, 0.6]; // Already normalized

        let dot = dot_product(&a, &b);
        let cos = cosine_similarity(&a, &b);

        assert!((dot - cos).abs() < 0.001);
    }

    #[test]
    fn test_assert_embeddings_identical_passes() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];
        assert_embeddings_identical(&a, &b); // Should not panic
    }

    #[test]
    #[should_panic(expected = "Embeddings differ")]
    fn test_assert_embeddings_identical_fails() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.30001];
        assert_embeddings_identical(&a, &b); // Should panic
    }

    #[test]
    fn test_assert_normalized_strict_passes() {
        let embedding = vec![0.6, 0.8]; // sqrt(0.36 + 0.64) = 1.0
        assert_normalized_strict(&embedding); // Should not panic
    }

    #[test]
    #[should_panic(expected = "not normalized")]
    fn test_assert_normalized_strict_fails() {
        let embedding = vec![1.0, 1.0]; // sqrt(2) ≠ 1.0
        assert_normalized_strict(&embedding); // Should panic
    }
}
