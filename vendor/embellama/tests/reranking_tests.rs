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

//! Integration tests for the reranking feature.
//!
//! Tests that require a real reranking model are gated behind the
//! `EMBELLAMA_TEST_RERANK_MODEL` environment variable.
//! Use a cross-encoder GGUF model such as bge-reranker-v2-m3,
//! jina-reranker-v1-turbo-en, or jina-reranker-v2-base-multilingual.

mod common;

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode, PoolingStrategy, RerankResult};
use serial_test::serial;
use std::path::PathBuf;

/// Gets the path to a reranking test model if available
fn get_rerank_model_path() -> Option<PathBuf> {
    std::env::var("EMBELLAMA_TEST_RERANK_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Checks if reranking model tests should be run
fn should_run_rerank_tests() -> bool {
    get_rerank_model_path().is_some()
}

// ============================================================================
// Unit-style tests (no model required)
// ============================================================================

#[test]
fn test_rerank_result_struct() {
    let result = RerankResult {
        index: 2,
        relevance_score: 0.95,
    };
    assert_eq!(result.index, 2);
    assert!((result.relevance_score - 0.95).abs() < f32::EPSILON);
}

#[test]
fn test_rerank_result_serialization() {
    let result = RerankResult {
        index: 0,
        relevance_score: 0.75,
    };
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("\"index\":0"));
    assert!(json.contains("\"relevance_score\":0.75"));

    // Deserialize back
    let deserialized: RerankResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.index, result.index);
    assert!((deserialized.relevance_score - result.relevance_score).abs() < f32::EPSILON);
}

#[test]
fn test_rerank_result_equality() {
    let a = RerankResult {
        index: 1,
        relevance_score: 0.5,
    };
    let b = RerankResult {
        index: 1,
        relevance_score: 0.5,
    };
    assert_eq!(a, b);
}

#[test]
fn test_pooling_strategy_rank_is_not_default() {
    assert_ne!(PoolingStrategy::default(), PoolingStrategy::Rank);
    assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
}

#[test]
fn test_pooling_strategy_rank_serde() {
    let strategy = PoolingStrategy::Rank;
    let json = serde_json::to_string(&strategy).unwrap();
    let deserialized: PoolingStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, PoolingStrategy::Rank);
}

#[test]
fn test_sigmoid_normalization_properties() {
    // Sigmoid properties: f(0) = 0.5, f(large) -> 1, f(-large) -> 0
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!(sigmoid(10.0) > 0.999);
    assert!(sigmoid(-10.0) < 0.001);
    assert!(sigmoid(100.0) > 0.999_999);
    assert!(sigmoid(-100.0) < 0.000_001);

    // Monotonically increasing
    assert!(sigmoid(-1.0) < sigmoid(0.0));
    assert!(sigmoid(0.0) < sigmoid(1.0));
    assert!(sigmoid(1.0) < sigmoid(2.0));
}

#[test]
fn test_engine_config_with_rank_pooling() {
    let (_dir, model_path) = common::create_dummy_model();

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .build()
        .unwrap();

    assert_eq!(
        config.model_config.pooling_strategy,
        Some(PoolingStrategy::Rank)
    );
    assert_eq!(
        config.model_config.normalization_mode,
        Some(NormalizationMode::None)
    );
}

// ============================================================================
// Integration tests (require real reranking model)
// ============================================================================

#[test]
#[serial]
fn test_rerank_basic() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .with_n_seq_max(4)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let query = "What is the capital of France?";
    let documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The weather is nice today.",
    ];

    let results = engine
        .rerank(Some("reranker"), query, &documents, None, true)
        .unwrap();

    // Should return all 3 documents
    assert_eq!(results.len(), 3);

    // Results should be sorted by relevance (descending)
    for i in 1..results.len() {
        assert!(
            results[i - 1].relevance_score >= results[i].relevance_score,
            "Results should be sorted descending: {} >= {}",
            results[i - 1].relevance_score,
            results[i].relevance_score
        );
    }

    // The most relevant document should be about Paris (index 0)
    assert_eq!(results[0].index, 0, "Paris document should be ranked first");

    // All normalized scores should be in [0, 1]
    for r in &results {
        assert!(
            r.relevance_score >= 0.0 && r.relevance_score <= 1.0,
            "Normalized score should be in [0, 1]: {}",
            r.relevance_score
        );
    }
}

#[test]
#[serial]
fn test_rerank_top_n() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .with_n_seq_max(4)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let query = "machine learning";
    let documents = [
        "Deep learning is a subset of machine learning.",
        "The stock market rose today.",
        "Neural networks power modern AI.",
        "I went to the grocery store.",
    ];

    let results = engine
        .rerank(Some("reranker"), query, &documents, Some(2), true)
        .unwrap();

    assert_eq!(results.len(), 2, "Should return only top 2 results");

    // Top results should be sorted descending
    assert!(results[0].relevance_score >= results[1].relevance_score);
}

#[test]
#[serial]
fn test_rerank_without_normalization() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .with_n_seq_max(4)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let query = "What is the capital of France?";
    let documents = ["Paris is the capital of France.", "Berlin is in Germany."];

    let results = engine
        .rerank(Some("reranker"), query, &documents, None, false)
        .unwrap();

    // Raw scores can be any value (not necessarily in [0, 1])
    assert_eq!(results.len(), 2);

    // Still sorted descending
    assert!(results[0].relevance_score >= results[1].relevance_score);
}

#[test]
#[serial]
fn test_rerank_empty_documents_returns_empty() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let documents: &[&str] = &[];
    let results = engine
        .rerank(Some("reranker"), "query", documents, None, true)
        .unwrap();

    assert!(results.is_empty());
}

#[test]
#[serial]
fn test_rerank_single_document() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let results = engine
        .rerank(
            Some("reranker"),
            "test query",
            &["single document"],
            None,
            true,
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
    assert!(results[0].relevance_score >= 0.0 && results[0].relevance_score <= 1.0);
}

#[test]
#[serial]
fn test_rerank_batch_exceeds_n_seq_max() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    // Use n_seq_max=2 to force chunking with 5 documents
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .with_n_seq_max(2)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let query = "What is machine learning?";
    let documents = [
        "Machine learning is a branch of AI.",
        "The sun is a star.",
        "Deep learning uses neural networks.",
        "I like pizza.",
        "Supervised learning uses labeled data.",
    ];

    let results = engine
        .rerank(Some("reranker"), query, &documents, None, true)
        .unwrap();

    // All 5 documents should be scored despite n_seq_max=2
    assert_eq!(results.len(), 5);

    // Each original index should appear exactly once
    let mut seen_indices: Vec<usize> = results.iter().map(|r| r.index).collect();
    seen_indices.sort();
    assert_eq!(seen_indices, vec![0, 1, 2, 3, 4]);
}

#[test]
#[serial]
fn test_embed_on_rank_model_fails() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // embed() on a Rank model should fail
    let result = engine.embed(Some("reranker"), "test text");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("PoolingStrategy::Rank")
    );
}

#[test]
#[serial]
fn test_rerank_on_embedding_model_fails() {
    if !common::should_run_model_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = common::get_test_model_path().unwrap();
    // Load an embedding model with Mean pooling
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("embedder")
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    // rerank() on a non-Rank model should fail
    let result = engine.rerank(Some("embedder"), "query", &["document"], None, true);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("PoolingStrategy::Rank")
    );
}

// ============================================================================
// Auto-detection tests (require real reranking model with GGUF pooling_type=4)
// ============================================================================

#[test]
#[serial]
fn test_rerank_auto_detect_from_gguf() {
    if !should_run_rerank_tests() {
        eprintln!("Skipping: EMBELLAMA_TEST_RERANK_MODEL not set");
        return;
    }
    common::init_test_logger();

    let model_path = get_rerank_model_path().unwrap();

    // Do NOT set pooling_strategy or normalization_mode — let auto-detection work
    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker-auto")
        .with_n_seq_max(4)
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).unwrap();

    let query = "What is the capital of France?";
    let documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The weather is nice today.",
    ];

    // rerank() should work without explicit PoolingStrategy::Rank
    // because GGUF metadata pooling_type=4 is auto-detected
    let results = engine
        .rerank(Some("reranker-auto"), query, &documents, None, true)
        .unwrap();

    assert_eq!(results.len(), 3);

    // Results should be sorted by relevance (descending)
    for i in 1..results.len() {
        assert!(
            results[i - 1].relevance_score >= results[i].relevance_score,
            "Results should be sorted descending"
        );
    }

    // The most relevant document should be about Paris (index 0)
    assert_eq!(results[0].index, 0, "Paris document should be ranked first");
}
