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

use crate::cache::{CacheMetrics, CacheStats, CacheStore};
use dashmap::DashMap;
use llama_cpp_2::token::LlamaToken;
use lru::LruCache;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::trace;

// > NOTE: Thread-local cache for hot path optimization to minimize lock contention
// > PERFORMANCE ISSUE: Shared cache access should be minimized

thread_local! {
    // Thread-local LRU cache for frequently accessed tokens
    static LOCAL_TOKEN_CACHE: RefCell<LruCache<String, Vec<LlamaToken>>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(1000).unwrap()));
}

/// Two-tier token cache with thread-local and shared storage
pub struct TokenCache {
    /// Shared cache backing store
    shared: Arc<DashMap<String, Vec<LlamaToken>>>,
    /// Metrics for monitoring
    metrics: Arc<CacheMetrics>,
    /// Maximum size for shared cache
    max_size: usize,
    /// TTL for cache entries (optional)
    #[allow(dead_code)]
    ttl_seconds: Option<u64>,
}

impl TokenCache {
    /// Create a new token cache
    pub fn new(max_size: usize) -> Self {
        Self::with_ttl(max_size, None)
    }

    /// Create a new token cache with TTL
    pub fn with_ttl(max_size: usize, ttl_seconds: Option<u64>) -> Self {
        trace!(
            "Creating TokenCache with max_size: {}, ttl: {:?}",
            max_size, ttl_seconds
        );
        Self {
            shared: Arc::new(DashMap::with_capacity(max_size.min(100_000))),
            metrics: Arc::new(CacheMetrics::new()),
            max_size,
            ttl_seconds,
        }
    }

    /// Compute cache key from tokenization parameters
    pub fn compute_key(text: &str, model_name: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(model_name.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get metrics
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }
}

impl CacheStore<String, Vec<LlamaToken>> for TokenCache {
    fn get(&self, key: &String) -> Option<Vec<LlamaToken>> {
        // Check thread-local cache first (hot path)
        let result = LOCAL_TOKEN_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(tokens) = cache.get(key) {
                trace!("Token cache hit (thread-local) for key: {}", key);
                self.metrics.record_hit();
                return Some(tokens.clone());
            }
            None
        });

        if result.is_some() {
            return result;
        }

        // Check shared cache as fallback
        if let Some(entry) = self.shared.get(key) {
            let tokens = entry.clone();
            trace!("Token cache hit (shared) for key: {}", key);

            // Promote to thread-local cache
            LOCAL_TOKEN_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                cache.put(key.clone(), tokens.clone());
            });

            self.metrics.record_hit();
            Some(tokens)
        } else {
            trace!("Token cache miss for key: {}", key);
            self.metrics.record_miss();
            None
        }
    }

    fn insert(&self, key: String, value: Vec<LlamaToken>) {
        let token_count = value.len();
        trace!(
            "Inserting {} tokens into cache with key: {}",
            token_count, key
        );

        // Always insert into thread-local cache
        LOCAL_TOKEN_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.put(key.clone(), value.clone());
        });

        // Check if we should insert into shared cache
        if self.shared.len() < self.max_size {
            self.shared.insert(key, value);

            // Update memory estimate (approximate: 4 bytes per token + overhead)
            let memory_bytes = (token_count * 4 + 64) as u64;
            self.metrics
                .memory_bytes
                .fetch_add(memory_bytes, Ordering::Relaxed);
        } else {
            // > TODO: Implement eviction policy for shared cache
            trace!("Shared cache full, skipping insertion");
        }
    }

    fn clear(&self) {
        self.shared.clear();
        LOCAL_TOKEN_CACHE.with(|cache| cache.borrow_mut().clear());
        self.metrics.reset();
    }

    fn stats(&self) -> CacheStats {
        CacheStats::from_metrics(
            &self.metrics,
            self.shared.len().try_into().unwrap_or(u64::MAX),
        )
    }

    fn len(&self) -> usize {
        self.shared.len()
    }
}

impl TokenCache {
    /// Evict the oldest entries from the cache
    ///
    /// This method is used by the memory monitor to free memory under pressure.
    ///
    /// # Arguments
    /// * `count` - Number of entries to evict
    pub fn evict_oldest(&self, count: usize) {
        // Clear thread-local cache first
        LOCAL_TOKEN_CACHE.with(|cache| {
            cache.borrow_mut().clear();
        });

        if count == 0 {
            return;
        }

        let current_count = self.shared.len();
        if current_count == 0 {
            return;
        }

        // If we need to evict more than half the cache, just clear it
        if count >= current_count / 2 {
            self.clear();
        } else {
            // DashMap doesn't track insertion order, so we'll remove a portion
            // > NOTE: This is a simplified implementation. In production,
            // > you might want an LRU-based approach

            // Collect keys to remove (don't remove while iterating to avoid deadlock)
            let keys_to_remove: Vec<_> = self
                .shared
                .iter()
                .take(count)
                .map(|entry| entry.key().clone())
                .collect();

            // Now remove the collected keys
            let removed = keys_to_remove.len();
            for key in keys_to_remove {
                self.shared.remove(&key);
            }

            // Record evictions
            self.metrics.record_eviction(removed as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_cpp_2::token::LlamaToken;

    fn create_test_tokens() -> Vec<LlamaToken> {
        vec![LlamaToken(1), LlamaToken(2), LlamaToken(3)]
    }

    #[test]
    fn test_token_cache_creation() {
        let cache = TokenCache::new(1000);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_compute_key() {
        let key1 = TokenCache::compute_key("hello", "model1");
        let key2 = TokenCache::compute_key("hello", "model1");
        let key3 = TokenCache::compute_key("hello", "model2");
        let key4 = TokenCache::compute_key("world", "model1");

        // Same inputs should produce same key
        assert_eq!(key1, key2);

        // Different inputs should produce different keys
        assert_ne!(key1, key3); // different model
        assert_ne!(key1, key4); // different text
    }

    #[test]
    fn test_cache_miss() {
        let cache = TokenCache::new(100);
        let key = "test_key".to_string();

        let result = cache.get(&key);
        assert!(result.is_none());

        // Check miss was recorded
        assert_eq!(cache.metrics.misses.load(Ordering::Relaxed), 1);
        assert_eq!(cache.metrics.hits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_cache_hit() {
        let cache = TokenCache::new(100);
        let key = "test_key".to_string();
        let tokens = create_test_tokens();

        // Insert tokens
        cache.insert(key.clone(), tokens.clone());

        // Should get a hit
        let result = cache.get(&key);
        assert_eq!(result, Some(tokens));

        // Check hit was recorded
        assert_eq!(cache.metrics.hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_thread_local_caching() {
        let cache = TokenCache::new(100);
        let key = "test_key".to_string();
        let tokens = create_test_tokens();

        // Insert tokens (goes to thread-local and shared)
        cache.insert(key.clone(), tokens.clone());

        // First get should hit thread-local
        let result1 = cache.get(&key);
        assert_eq!(result1, Some(tokens.clone()));

        // Second get should also hit thread-local (fast path)
        let result2 = cache.get(&key);
        assert_eq!(result2, Some(tokens));

        // Both should be hits
        assert_eq!(cache.metrics.hits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_cache_clear() {
        let cache = TokenCache::new(100);
        let key = "test_key".to_string();
        let tokens = create_test_tokens();

        // Insert and verify
        cache.insert(key.clone(), tokens);
        assert_eq!(cache.len(), 1);

        // Clear cache
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Metrics should be reset after clear
        assert_eq!(cache.metrics.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cache.metrics.misses.load(Ordering::Relaxed), 0);

        // Should be a miss now
        let result = cache.get(&key);
        assert!(result.is_none());

        // After the get, we should have 1 miss
        assert_eq!(cache.metrics.misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_stats() {
        let cache = TokenCache::new(100);
        let key1 = "key1".to_string();
        let key2 = "key2".to_string();
        let tokens = create_test_tokens();

        // Insert some tokens
        cache.insert(key1.clone(), tokens.clone());
        cache.insert(key2, tokens);

        // Get one hit and one miss
        let _ = cache.get(&key1); // hit
        let _ = cache.get(&"missing".to_string()); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_ttl_cache_creation() {
        let cache = TokenCache::with_ttl(100, Some(3600));
        assert_eq!(cache.ttl_seconds, Some(3600));
        assert!(cache.is_empty());
    }
}
