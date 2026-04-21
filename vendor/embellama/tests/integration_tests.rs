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

//! Integration tests for the embellama library

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode, PoolingStrategy};
use serial_test::serial;
use std::fs;
use std::path::PathBuf;
use std::sync::Once;
use tempfile::tempdir;

/// Gets the path to the test model (only call when env var is known to be set)
fn get_test_model_path() -> PathBuf {
    PathBuf::from(std::env::var("EMBELLAMA_TEST_MODEL").unwrap())
}

// Initialize global tracing once for all tests
static INIT: Once = Once::new();

fn init_test_tracing() {
    INIT.call_once(|| {
        // Use global tracing subscriber for tests
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init();
    });
}

/// Creates a dummy model file for testing
fn create_test_model_file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("test_model.gguf");
    fs::write(&model_path, b"dummy model file").unwrap();
    (dir, model_path)
}

/// Creates a test configuration
fn create_test_config(model_path: PathBuf, name: &str) -> EngineConfig {
    EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name(name)
        .with_n_threads(1)
        .with_normalization_mode(NormalizationMode::L2)
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .unwrap()
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_engine_creation_and_embedding() {
    init_test_tracing();

    let model_path = get_test_model_path();

    let normalize = true; // Store this before moving config
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_normalization_mode(if normalize {
            NormalizationMode::L2
        } else {
            NormalizationMode::None
        })
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Test single embedding
    let text = "Hello, world!";
    let embedding = engine
        .embed(None, text)
        .expect("Failed to generate embedding");

    assert!(!embedding.is_empty());
    assert!(!embedding.is_empty());

    // Check that embeddings are not all zeros (norm should be non-zero)
    // Note: Some models like MiniLM don't normalize by default
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm > 0.1,
        "Embedding norm too low (likely zeros), got norm: {norm}"
    );

    // If normalization is enabled in config, check it's close to 1.0
    if normalize {
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be normalized to 1.0, got norm: {norm}"
        );
    }

    // Clean up thread-local models before test ends
    engine.cleanup_thread_models();
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_batch_embeddings() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let texts = vec![
        "First document about technology",
        "Second document about science",
        "Third document about mathematics",
    ];

    let embeddings = engine
        .embed_batch(None, &texts)
        .expect("Failed to generate batch embeddings");

    assert_eq!(embeddings.len(), texts.len());

    // Check that all embeddings have the same dimension
    let dim = embeddings[0].len();
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), dim, "Embedding {i} has different dimension");
    }

    // Clean up thread-local models before test ends
    engine.cleanup_thread_models();
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_multiple_models() {
    init_test_tracing();
    let model_path = get_test_model_path();

    // Create engine with first model
    let config1 = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("model1")
        .build()
        .unwrap();

    let mut engine = EmbeddingEngine::new(config1).expect("Failed to create engine");

    // Load second model with different config
    let config2 = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("model2")
        .with_normalization_mode(NormalizationMode::None) // Different config
        .build()
        .unwrap();

    engine
        .load_model(config2)
        .expect("Failed to load second model");

    // Test that both models are listed
    let models = engine.list_models();
    assert_eq!(models.len(), 2);
    assert!(models.contains(&"model1".to_string()));
    assert!(models.contains(&"model2".to_string()));

    // Generate embeddings with both models
    let text = "Test text";
    let emb1 = engine
        .embed(Some("model1"), text)
        .expect("Failed with model1");
    let emb2 = engine
        .embed(Some("model2"), text)
        .expect("Failed with model2");

    assert!(!emb1.is_empty());
    assert!(!emb2.is_empty());

    // Clean up thread-local models before test ends
    engine.cleanup_thread_models();
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_error_handling() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Test empty text
    let result = engine.embed(None, "");
    assert!(result.is_err());

    // Test non-existent model
    let result = engine.embed(Some("non-existent"), "test");
    assert!(result.is_err());

    // Clean up thread-local models before test ends
    engine.cleanup_thread_models();
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_granular_unload_operations() {
    init_test_tracing();
    let model_path = get_test_model_path();

    // Create engine with a model
    let config = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("test-model")
        .build()
        .unwrap();

    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Load a second model
    let config2 = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("test-model-2")
        .build()
        .unwrap();

    engine
        .load_model(config2)
        .expect("Failed to load second model");

    // Verify both models are listed
    assert_eq!(engine.list_models().len(), 2);

    // Test drop_model_from_thread - model should still be registered
    engine
        .drop_model_from_thread("test-model")
        .expect("Failed to drop from thread");
    assert_eq!(engine.list_models().len(), 2); // Still registered

    // Should be able to use the model again (it will reload)
    let emb = engine
        .embed(Some("test-model"), "test text")
        .expect("Failed to embed after drop");
    assert!(!emb.is_empty());

    // Test unregister_model - removes from registry but not necessarily from thread
    engine
        .unregister_model("test-model-2")
        .expect("Failed to unregister");
    assert_eq!(engine.list_models().len(), 1); // Only one model left
    assert!(!engine.list_models().contains(&"test-model-2".to_string()));

    // Should not be able to use unregistered model
    let result = engine.embed(Some("test-model-2"), "test text");
    assert!(result.is_err());

    // Test full unload_model (backward compatibility)
    let config3 = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model-3")
        .build()
        .unwrap();

    engine
        .load_model(config3)
        .expect("Failed to load third model");
    assert_eq!(engine.list_models().len(), 2);

    engine
        .unload_model("test-model-3")
        .expect("Failed to unload");
    assert_eq!(engine.list_models().len(), 1);
    assert!(!engine.list_models().contains(&"test-model-3".to_string()));

    // Should not be able to use unloaded model
    let result = engine.embed(Some("test-model-3"), "test text");
    assert!(result.is_err());
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_model_registration_checking() {
    init_test_tracing();
    let model_path = get_test_model_path();

    // Create engine with a model
    let config = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("test-model")
        .build()
        .unwrap();

    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Check model is registered
    assert!(engine.is_model_registered("test-model"));
    assert!(!engine.is_model_registered("non-existent"));

    // The first model (from EmbeddingEngine::new) is loaded immediately
    assert!(engine.is_model_loaded_in_thread("test-model"));

    // Generate embedding (model is already loaded)
    let _ = engine
        .embed(Some("test-model"), "test text")
        .expect("Failed to embed");

    // Model should still be loaded in thread
    assert!(engine.is_model_loaded_in_thread("test-model"));
    assert!(engine.is_model_registered("test-model"));

    // Drop from thread
    engine
        .drop_model_from_thread("test-model")
        .expect("Failed to drop from thread");
    assert!(!engine.is_model_loaded_in_thread("test-model"));
    assert!(engine.is_model_registered("test-model")); // Still registered

    // Unregister model
    engine
        .unregister_model("test-model")
        .expect("Failed to unregister");
    assert!(!engine.is_model_registered("test-model"));
    assert!(!engine.is_model_loaded_in_thread("test-model"));

    // Test with multiple models - additional models are lazy loaded
    let config1 = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("model1")
        .build()
        .unwrap();

    let config2 = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("model2")
        .build()
        .unwrap();

    engine.load_model(config1).expect("Failed to load model1");
    engine.load_model(config2).expect("Failed to load model2");

    assert!(engine.is_model_registered("model1"));
    assert!(engine.is_model_registered("model2"));
    // Additional models loaded via load_model() are lazy - not loaded in thread yet
    assert!(!engine.is_model_loaded_in_thread("model1"));
    assert!(!engine.is_model_loaded_in_thread("model2"));

    // Load model1 in thread
    let _ = engine
        .embed(Some("model1"), "test")
        .expect("Failed to embed");
    assert!(engine.is_model_loaded_in_thread("model1"));
    assert!(!engine.is_model_loaded_in_thread("model2"));

    // Load model2 in thread
    let _ = engine
        .embed(Some("model2"), "test")
        .expect("Failed to embed");
    assert!(engine.is_model_loaded_in_thread("model1"));
    assert!(engine.is_model_loaded_in_thread("model2"));
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_pooling_strategies() {
    init_test_tracing();
    let model_path = get_test_model_path();

    // Test different pooling strategies
    let strategies = vec![
        PoolingStrategy::Mean,
        PoolingStrategy::Cls,
        PoolingStrategy::Max,
        PoolingStrategy::MeanSqrt,
    ];

    // Create initial engine
    let config_initial = EngineConfig::builder()
        .with_model_path(model_path.clone())
        .with_model_name("model-Mean")
        .with_pooling_strategy(PoolingStrategy::Mean)
        .build()
        .unwrap();

    let mut engine = EmbeddingEngine::new(config_initial).expect("Failed to create engine");

    // Test first strategy with initial model
    let text = "Testing pooling strategy";
    let embedding = engine
        .embed(None, text)
        .expect("Failed to generate embedding");
    assert!(
        !embedding.is_empty(),
        "Embedding empty for strategy {:?}",
        PoolingStrategy::Mean
    );

    // Test remaining strategies by loading new models
    for strategy in strategies.into_iter().skip(1) {
        let config = EngineConfig::builder()
            .with_model_path(model_path.clone())
            .with_model_name(format!("model-{strategy:?}"))
            .with_pooling_strategy(strategy)
            .build()
            .unwrap();

        engine.load_model(config).expect("Failed to load model");

        let embedding = engine
            .embed(Some(&format!("model-{strategy:?}")), text)
            .expect("Failed to generate embedding");

        assert!(
            !embedding.is_empty(),
            "Embedding empty for strategy {strategy:?}"
        );
    }
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_model_warmup() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Warmup should not fail
    engine.warmup_model(None).expect("Warmup failed");

    // After warmup, embeddings should work
    let embedding = engine.embed(None, "test").expect("Failed after warmup");
    assert!(!embedding.is_empty());
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_model_info() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let info = engine
        .model_info("test-model")
        .expect("Failed to get model info");

    assert_eq!(info.name, "test-model");
    assert!(info.dimensions > 0);
    assert!(info.max_tokens > 0);
    assert!(info.model_size.unwrap_or(0) > 0);
}

#[test]
fn test_configuration_validation() {
    // Test invalid model path
    let result = EngineConfig::builder()
        .with_model_path("/non/existent/path.gguf")
        .with_model_name("test")
        .build();
    assert!(result.is_err());

    // Test empty model name
    let (_dir, model_path) = create_test_model_file();
    let result = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("")
        .build();
    assert!(result.is_err());
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = Arc::new(EmbeddingEngine::new(config).expect("Failed to create engine"));

    // Spawn multiple threads that use the same engine
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let engine = engine.clone();
            thread::spawn(move || {
                let text = format!("Thread {i} test text");
                let embedding = engine.embed(None, &text).expect("Failed in thread");
                assert!(!embedding.is_empty());
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Performance benchmark for single embedding
#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn bench_single_embedding() {
    let model_path = get_test_model_path();

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .build()
        .unwrap();

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Warmup
    engine.warmup_model(None).expect("Warmup failed");

    let text = "This is a sample text for benchmarking embedding generation performance.";

    let start = std::time::Instant::now();
    let iterations = 10;

    for _ in 0..iterations {
        let _ = engine
            .embed(None, text)
            .expect("Failed to generate embedding");
    }

    let duration = start.elapsed();
    let avg_time = duration / iterations;

    println!("Average time per embedding: {avg_time:?}");

    // Assert performance target (adjust based on hardware)
    assert!(
        avg_time.as_millis() < 1000,
        "Embedding generation too slow: {avg_time:?}"
    );
}

// ============================================================================
// Phase 4: Batch Processing Tests
// ============================================================================

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_batch_processing_basic() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_batch");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Test batch of texts
    let texts = vec![
        "First text for batch processing",
        "Second text with different content",
        "Third text to complete the batch",
    ];

    let embeddings = engine.embed_batch(Some("test_batch"), &texts).unwrap();

    // Verify batch results
    assert_eq!(embeddings.len(), texts.len());

    // Each embedding should have the same dimensions
    let dim = embeddings[0].len();
    assert!(dim > 0);
    for emb in &embeddings {
        assert_eq!(emb.len(), dim);
    }

    // Embeddings should be different for different texts
    assert_ne!(embeddings[0], embeddings[1]);
    assert_ne!(embeddings[1], embeddings[2]);
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_batch_processing_large() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_large_batch");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Create a large batch
    let mut texts = Vec::new();
    for i in 0..100 {
        texts.push(format!("Test document number {i}"));
    }
    let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

    let start = std::time::Instant::now();
    let embeddings = engine
        .embed_batch(Some("test_large_batch"), &text_refs)
        .unwrap();
    let duration = start.elapsed();

    println!("Processed {} texts in {:?}", texts.len(), duration);
    let text_count = u32::try_from(texts.len()).unwrap_or(u32::MAX);
    println!("Average time per text: {:?}", duration / text_count);

    assert_eq!(embeddings.len(), texts.len());
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_batch_processing_empty() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_empty_batch");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Test empty batch
    let texts: Vec<&str> = vec![];
    let embeddings = engine
        .embed_batch(Some("test_empty_batch"), &texts)
        .unwrap();

    assert!(embeddings.is_empty());
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
fn test_batch_processing_single_item() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_single_batch");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Test single item batch
    let texts = vec!["Single text item"];
    let batch_embeddings = engine
        .embed_batch(Some("test_single_batch"), &texts)
        .unwrap();

    // Compare with single embedding
    let single_embedding = engine.embed(Some("test_single_batch"), texts[0]).unwrap();

    assert_eq!(batch_embeddings.len(), 1);
    assert_eq!(batch_embeddings[0], single_embedding);
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
#[ignore = "Batch vs sequential processing produces slightly different results due to numerical precision differences in llama.cpp batching"]
fn test_batch_processing_order_preservation() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_order");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Test that order is preserved
    let texts = vec![
        "Alpha text",
        "Beta text",
        "Gamma text",
        "Delta text",
        "Epsilon text",
    ];

    let batch_embeddings = engine.embed_batch(Some("test_order"), &texts).unwrap();

    // Get individual embeddings
    let mut individual_embeddings = Vec::new();
    for text in &texts {
        individual_embeddings.push(engine.embed(Some("test_order"), text).unwrap());
    }

    // Verify order is preserved
    assert_eq!(batch_embeddings.len(), individual_embeddings.len());
    for (batch_emb, individual_emb) in batch_embeddings.iter().zip(individual_embeddings.iter()) {
        assert_eq!(batch_emb, individual_emb);
    }
}

#[serial]
#[test_with::env(EMBELLAMA_TEST_MODEL)]
#[ignore = "Batch vs sequential processing produces slightly different results due to numerical precision differences in llama.cpp batching"]
fn test_batch_vs_sequential_performance() {
    init_test_tracing();
    let model_path = get_test_model_path();

    let config = create_test_config(model_path, "test_perf");
    let engine = EmbeddingEngine::new(config).unwrap();

    // Create test texts
    let mut texts = Vec::new();
    for i in 0..20 {
        texts.push(format!("Performance test document {i}"));
    }
    let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

    // Measure sequential processing time
    let sequential_start = std::time::Instant::now();
    let mut sequential_embeddings = Vec::new();
    for text in &text_refs {
        sequential_embeddings.push(engine.embed(Some("test_perf"), text).unwrap());
    }
    let sequential_duration = sequential_start.elapsed();

    // Measure batch processing time
    let batch_start = std::time::Instant::now();
    let batch_embeddings = engine.embed_batch(Some("test_perf"), &text_refs).unwrap();
    let batch_duration = batch_start.elapsed();

    println!("Sequential processing: {sequential_duration:?}");
    println!("Batch processing: {batch_duration:?}");
    println!(
        "Speedup: {:.2}x",
        sequential_duration.as_secs_f64() / batch_duration.as_secs_f64()
    );

    // Verify results are the same
    assert_eq!(batch_embeddings.len(), sequential_embeddings.len());
    for (batch_emb, seq_emb) in batch_embeddings.iter().zip(sequential_embeddings.iter()) {
        assert_eq!(batch_emb, seq_emb);
    }

    // > NOTE: Batch processing should be faster for tokenization and post-processing
    // Model inference remains sequential due to !Send constraint
}
