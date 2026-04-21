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

use embellama::cache::CacheStore;
use embellama::cache::embedding_cache::EmbeddingCache;
use embellama::{NormalizationMode, PoolingStrategy};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_cache_basic_operations() {
    let cache = EmbeddingCache::new(100, 3600);
    let key = "test_key".to_string();
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    // Test insert and get
    cache.insert(key.clone(), embedding.clone());
    let retrieved = cache.get(&key);
    assert_eq!(retrieved, Some(embedding.clone()));

    // Test cache hit metrics
    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 0);

    // Test cache miss
    let missing_key = "missing_key".to_string();
    let result = cache.get(&missing_key);
    assert_eq!(result, None);

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_cache_key_computation() {
    // Test that cache keys are deterministic
    let text = "Hello, world!";
    let model = "test-model";
    let pooling = PoolingStrategy::Mean;
    let normalization = NormalizationMode::L2;

    let key1 = EmbeddingCache::compute_key(text, model, pooling, normalization);
    let key2 = EmbeddingCache::compute_key(text, model, pooling, normalization);
    assert_eq!(key1, key2);

    // Test that different inputs produce different keys
    let key3 = EmbeddingCache::compute_key("Different text", model, pooling, normalization);
    assert_ne!(key1, key3);

    let key4 = EmbeddingCache::compute_key(text, "different-model", pooling, normalization);
    assert_ne!(key1, key4);

    let key5 = EmbeddingCache::compute_key(text, model, PoolingStrategy::Cls, normalization);
    assert_ne!(key1, key5);

    let key6 = EmbeddingCache::compute_key(text, model, pooling, NormalizationMode::None);
    assert_ne!(key1, key6);
}

#[test_log::test]
fn test_cache_clear() {
    let cache = EmbeddingCache::new(100, 3600);

    // Insert multiple entries
    for i in 0..10 {
        let key = format!("key_{}", i);
        let embedding = vec![i as f32; 5];
        cache.insert(key, embedding);
    }

    let stats_before = cache.stats();
    assert!(stats_before.entry_count > 0);

    // Clear cache
    cache.clear();

    // Verify cache is empty
    let stats_after = cache.stats();
    assert_eq!(stats_after.entry_count, 0);

    // Verify entries are gone
    for i in 0..10 {
        let key = format!("key_{}", i);
        assert_eq!(cache.get(&key), None);
    }
}

#[test]
fn test_cache_capacity_limit() {
    // Create a small cache with capacity 5
    let cache = EmbeddingCache::new(5, 3600);

    // Insert 10 items (should trigger eviction)
    for i in 0..10 {
        let key = format!("key_{}", i);
        let embedding = vec![i as f32; 5];
        cache.insert(key, embedding);
    }

    // Cache should contain at most 5 items
    let stats = cache.stats();
    assert!(stats.entry_count <= 5);
}

#[test]
fn test_cache_warm_up() {
    let cache = EmbeddingCache::new(100, 3600);

    // Prepare entries for warming
    let entries: Vec<(String, Vec<f32>)> = (0..5)
        .map(|i| (format!("warm_key_{}", i), vec![i as f32; 5]))
        .collect();

    // Warm the cache
    cache.warm_cache(entries.clone());

    // Verify all entries are present
    for (key, expected_value) in entries {
        let retrieved = cache.get(&key);
        assert_eq!(retrieved, Some(expected_value));
    }
}

#[test]
fn test_cache_thread_safety() {
    let cache = Arc::new(EmbeddingCache::new(1000, 3600));
    let mut handles = vec![];

    // Spawn multiple threads to access cache concurrently
    for thread_id in 0..10 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let key = format!("thread_{}_key_{}", thread_id, i);
                let embedding = vec![thread_id as f32, i as f32];
                cache_clone.insert(key.clone(), embedding.clone());

                let retrieved = cache_clone.get(&key);
                assert_eq!(retrieved, Some(embedding));
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify cache statistics
    let stats = cache.stats();
    assert!(stats.entry_count > 0);
    assert!(stats.hits > 0);
}

#[test]
fn test_cache_memory_tracking() {
    let cache = EmbeddingCache::new(100, 3600);

    // Insert entries and verify memory tracking
    let embedding_small = vec![0.1; 10]; // 10 * 4 = 40 bytes
    let embedding_large = vec![0.2; 1000]; // 1000 * 4 = 4000 bytes

    cache.insert("small".to_string(), embedding_small);
    cache.insert("large".to_string(), embedding_large);

    let stats = cache.stats();
    // Memory should include data plus overhead
    assert!(stats.memory_bytes > 4040); // At least the raw data size
}

#[test]
fn test_cache_stats_accuracy() {
    let cache = EmbeddingCache::new(100, 3600);

    // Perform a series of operations
    cache.insert("key1".to_string(), vec![1.0]);
    cache.insert("key2".to_string(), vec![2.0]);
    cache.insert("key3".to_string(), vec![3.0]);

    // Hits
    cache.get(&"key1".to_string());
    cache.get(&"key2".to_string());

    // Misses
    cache.get(&"missing1".to_string());
    cache.get(&"missing2".to_string());
    cache.get(&"missing3".to_string());

    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 3);
    assert!(stats.entry_count > 0);

    // Calculate hit rate
    let expected_hit_rate = 2.0 / 5.0; // 2 hits out of 5 total requests
    assert!((stats.hit_rate - expected_hit_rate).abs() < 0.001);
}

#[test]
fn test_embedding_cache_with_real_embeddings() {
    let cache = EmbeddingCache::new(100, 3600);

    // Simulate real embedding dimensions (e.g., 384, 768, 1024)
    let embedding_384 = vec![0.1; 384];
    let embedding_768 = vec![0.2; 768];
    let embedding_1024 = vec![0.3; 1024];

    cache.insert("bert-base".to_string(), embedding_384.clone());
    cache.insert("bert-large".to_string(), embedding_768.clone());
    cache.insert("large-model".to_string(), embedding_1024.clone());

    // Verify retrieval
    assert_eq!(cache.get(&"bert-base".to_string()), Some(embedding_384));
    assert_eq!(cache.get(&"bert-large".to_string()), Some(embedding_768));
    assert_eq!(cache.get(&"large-model".to_string()), Some(embedding_1024));

    // Check memory usage is reasonable
    let stats = cache.stats();
    let expected_min_memory = (384 + 768 + 1024) * 4; // Just the embeddings
    assert!(stats.memory_bytes >= expected_min_memory as u64);
}

#[test]
#[ignore = "Takes ~2 seconds due to sleep for TTL expiration"]
fn test_cache_ttl_expiration() {
    // Create cache with 1 second TTL
    let cache = EmbeddingCache::new(100, 1);

    cache.insert("ephemeral".to_string(), vec![1.0, 2.0, 3.0]);

    // Should be present immediately
    assert!(cache.get(&"ephemeral".to_string()).is_some());

    // Wait for TTL to expire
    thread::sleep(Duration::from_secs(2));

    // Force moka to process pending expirations
    cache.inner_cache().run_pending_tasks();

    // Verify the shared (moka) cache has expired the entry.
    // We check the moka cache directly because the thread-local LRU cache
    // (which has no TTL) may still hold the value from the earlier get().
    assert!(
        cache.inner_cache().get(&"ephemeral".to_string()).is_none(),
        "Entry should have expired from shared cache after TTL"
    );
}
