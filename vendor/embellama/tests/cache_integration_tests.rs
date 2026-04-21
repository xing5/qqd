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

mod common;

use common::*;
use embellama::{CacheConfigBuilder, EmbeddingEngine, EngineConfigBuilder, NormalizationMode};
use std::time::Instant;

#[test]
#[ignore = "Requires model file"]
fn test_engine_with_cache_enabled() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    // Create engine with cache enabled
    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_config(
            CacheConfigBuilder::new()
                .with_enabled(true)
                .with_embedding_cache_size(100)
                .with_ttl_seconds(3600)
                .build()
                .expect("Failed to build cache config"),
        )
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Verify cache is enabled
    assert!(engine.is_cache_enabled());

    // Generate embedding twice for the same text
    let text = "This is a test for caching functionality";

    // First call - should miss cache
    let start = Instant::now();
    let embedding1 = engine
        .embed(None, text)
        .expect("Failed to generate embedding");
    let time_uncached = start.elapsed();

    // Second call - should hit cache
    let start = Instant::now();
    let embedding2 = engine
        .embed(None, text)
        .expect("Failed to generate embedding");
    let time_cached = start.elapsed();

    // Embeddings should be identical
    assert_eq!(embedding1, embedding2);

    // Cached call should be significantly faster
    // > NOTE: Cache lookup should be < 1ms, model inference is typically > 10ms
    assert!(
        time_cached < time_uncached / 5,
        "Cached call should be at least 5x faster. Uncached: {:?}, Cached: {:?}",
        time_uncached,
        time_cached
    );

    // Check cache statistics
    let stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entry_count, 1);
    assert!(stats.hit_rate > 0.0 && stats.hit_rate <= 1.0);
}

#[test]
#[ignore = "Requires model file"]
fn test_engine_with_cache_disabled() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    // Create engine with cache disabled
    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_disabled()
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Verify cache is disabled
    assert!(!engine.is_cache_enabled());

    // Generate embedding twice
    let text = "Test without caching";
    let embedding1 = engine
        .embed(None, text)
        .expect("Failed to generate embedding");
    let embedding2 = engine
        .embed(None, text)
        .expect("Failed to generate embedding");

    // Embeddings should still be identical (same input)
    assert_eq!(embedding1, embedding2);

    // No cache statistics should be available
    assert!(engine.get_cache_stats().is_none());
}

#[test]
#[ignore = "Requires model file"]
fn test_cache_batch_processing() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    // Create engine with cache
    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_enabled()
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Prepare batch with duplicates
    let texts = vec![
        "First unique text",
        "Second unique text",
        "First unique text", // Duplicate
        "Third unique text",
        "Second unique text", // Duplicate
    ];

    // First batch - all cache misses
    let embeddings1 = engine
        .embed_batch(None, &texts)
        .expect("Failed to generate batch embeddings");

    assert_eq!(embeddings1.len(), 5);

    // Check that duplicates have identical embeddings
    assert_eq!(embeddings1[0], embeddings1[2]); // First text duplicates
    assert_eq!(embeddings1[1], embeddings1[4]); // Second text duplicates

    // Second batch with same texts - should hit cache
    let start = Instant::now();
    let embeddings2 = engine
        .embed_batch(None, &texts)
        .expect("Failed to generate batch embeddings");
    let cached_time = start.elapsed();

    // All embeddings should be identical to first batch
    for (e1, e2) in embeddings1.iter().zip(embeddings2.iter()) {
        assert_eq!(e1, e2);
    }

    // Check cache statistics
    let stats = engine.get_cache_stats().expect("Cache should be enabled");
    // We have 3 unique texts, but called them multiple times
    assert_eq!(stats.entry_count, 3);
    // Second batch should have 5 hits (all texts were cached)
    assert!(stats.hits >= 5);

    println!("Batch cache lookup time: {:?}", cached_time);
}

#[test]
#[ignore = "Requires model file"]
fn test_cache_clearing() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_enabled()
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Generate some embeddings to populate cache
    let texts = vec!["text1", "text2", "text3"];
    for text in &texts {
        engine
            .embed(None, text)
            .expect("Failed to generate embedding");
    }

    // Verify cache has entries
    let stats_before = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats_before.entry_count, 3);

    // Clear cache
    engine.clear_cache();

    // Verify cache is empty
    let stats_after = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats_after.entry_count, 0);
    assert_eq!(stats_after.hits, 0);
    assert_eq!(stats_after.misses, 0);

    // Generate embedding again - should be a cache miss
    engine
        .embed(None, texts[0])
        .expect("Failed to generate embedding");
    let stats_final = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats_final.misses, 1);
    assert_eq!(stats_final.hits, 0);
}

#[test]
#[ignore = "Requires model file"]
fn test_cache_warm_up() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_enabled()
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Warm up cache with common texts
    let warm_up_texts = vec![
        "Frequently used text 1",
        "Frequently used text 2",
        "Frequently used text 3",
    ];

    engine
        .warm_cache(None, &warm_up_texts)
        .expect("Failed to warm cache");

    // Verify all texts are in cache
    let stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats.entry_count, 3);

    // Now using these texts should hit cache
    for text in &warm_up_texts {
        let start = Instant::now();
        engine
            .embed(None, text)
            .expect("Failed to generate embedding");
        let elapsed = start.elapsed();
        // Should be very fast (< 1ms typically)
        assert!(
            elapsed.as_millis() < 10,
            "Cache hit took too long: {:?}",
            elapsed
        );
    }

    // All should be hits
    let final_stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(final_stats.hits, 3);
}

#[test]
#[ignore = "Requires model file"]
fn test_cache_with_different_models() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    // Create engine with cache
    let config = EngineConfigBuilder::new()
        .with_model_path(model_path.clone())
        .with_model_name("model1")
        .with_cache_enabled()
        .build()
        .expect("Failed to build config");

    let mut engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Load a second model configuration (using same file for testing)
    let config2 = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("model2")
        .with_normalization_mode(NormalizationMode::None) // Different normalization
        .build()
        .expect("Failed to build config");

    engine
        .load_model(config2)
        .expect("Failed to load second model");

    let text = "Same text for different models";

    // Generate with first model
    let embedding1 = engine
        .embed(Some("model1"), text)
        .expect("Failed to generate embedding");

    // Generate with second model - should be cache miss (different model)
    let embedding2 = engine
        .embed(Some("model2"), text)
        .expect("Failed to generate embedding");

    // Different normalization (L2 vs None) should produce different embeddings
    assert_ne!(
        embedding1, embedding2,
        "Different normalization modes should produce different embeddings"
    );
    // But both should be in cache now
    let stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(stats.entry_count, 2); // Two different cache keys

    // Requesting again should hit cache for both
    let _ = engine
        .embed(Some("model1"), text)
        .expect("Failed to generate embedding");
    let _ = engine
        .embed(Some("model2"), text)
        .expect("Failed to generate embedding");

    let final_stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert_eq!(final_stats.hits, 2);
}

#[test]
fn test_cache_key_differences() {
    use embellama::PoolingStrategy;
    use embellama::cache::embedding_cache::EmbeddingCache;

    let text = "Test text";
    let model = "test-model";

    // Different configurations should produce different cache keys
    let key1 =
        EmbeddingCache::compute_key(text, model, PoolingStrategy::Mean, NormalizationMode::L2);
    let key2 =
        EmbeddingCache::compute_key(text, model, PoolingStrategy::Mean, NormalizationMode::None);
    let key3 =
        EmbeddingCache::compute_key(text, model, PoolingStrategy::Cls, NormalizationMode::L2);
    let key4 =
        EmbeddingCache::compute_key(text, model, PoolingStrategy::Max, NormalizationMode::L2);
    let key5 = EmbeddingCache::compute_key(
        text,
        "different-model",
        PoolingStrategy::Mean,
        NormalizationMode::L2,
    );

    // All keys should be different
    assert_ne!(key1, key2); // Different normalization
    assert_ne!(key1, key3); // Different pooling
    assert_ne!(key1, key4); // Different pooling
    assert_ne!(key1, key5); // Different model

    // Same configuration should produce same key
    let key1_duplicate =
        EmbeddingCache::compute_key(text, model, PoolingStrategy::Mean, NormalizationMode::L2);
    assert_eq!(key1, key1_duplicate);
}

#[test]
#[ignore = "Requires model file"]
fn test_mixed_batch_cache_hits() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping test: no test model configured");
        return;
    };

    let config = EngineConfigBuilder::new()
        .with_model_path(model_path)
        .with_model_name("test-model")
        .with_cache_enabled()
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Pre-populate cache with some texts
    let cached_texts = vec!["cached1", "cached2"];
    for text in &cached_texts {
        engine
            .embed(None, text)
            .expect("Failed to generate embedding");
    }

    // Create batch with mix of cached and new texts
    let mixed_batch = vec![
        "cached1", // Hit
        "new1",    // Miss
        "cached2", // Hit
        "new2",    // Miss
        "cached1", // Hit (duplicate)
    ];

    let embeddings = engine
        .embed_batch(None, &mixed_batch)
        .expect("Failed to generate batch");

    assert_eq!(embeddings.len(), 5);

    // Check cache statistics
    let stats = engine.get_cache_stats().expect("Cache should be enabled");

    // We should have 4 unique texts in cache now
    assert_eq!(stats.entry_count, 4);

    // The batch should have had 3 hits (cached1 twice, cached2 once)
    // and 2 misses (new1, new2)
    // Plus the original 2 misses from pre-population
    println!("Cache stats: {:?}", stats);
}
