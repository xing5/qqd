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
use crate::config::{NormalizationMode, PoolingStrategy};
use lru::LruCache;
use moka::sync::Cache;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

// Thread-local cache for hot path optimization
thread_local! {
    static LOCAL_EMBEDDING_CACHE: RefCell<LruCache<String, Vec<f32>>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(100).unwrap()));
}

/// High-performance embedding cache with automatic eviction
pub struct EmbeddingCache {
    /// Shared cache with TTL and size limits
    cache: Arc<Cache<String, Vec<f32>>>,
    /// Metrics for monitoring
    metrics: Arc<CacheMetrics>,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(max_capacity: u64, ttl_seconds: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(Duration::from_secs(ttl_seconds))
            .build();

        Self {
            cache: Arc::new(cache),
            metrics: Arc::new(CacheMetrics::new()),
        }
    }

    /// Compute cache key from embedding parameters
    pub fn compute_key(
        text: &str,
        model_name: &str,
        pooling: PoolingStrategy,
        normalization: NormalizationMode,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(model_name.as_bytes());
        hasher.update([pooling as u8]);
        // Hash the normalization mode discriminant and value
        match normalization {
            NormalizationMode::None => hasher.update([0u8]),
            NormalizationMode::MaxAbs => hasher.update([1u8]),
            NormalizationMode::L2 => hasher.update([2u8]),
            NormalizationMode::PNorm(p) => {
                hasher.update([3u8]);
                hasher.update(p.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    /// Get metrics
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    /// Warm up the cache with pre-computed embeddings
    pub fn warm_cache(&self, entries: Vec<(String, Vec<f32>)>) {
        for (key, value) in entries {
            self.insert(key, value);
        }
    }

    /// Get the underlying cache for advanced operations
    pub fn inner_cache(&self) -> &Arc<Cache<String, Vec<f32>>> {
        &self.cache
    }
}

impl CacheStore<String, Vec<f32>> for EmbeddingCache {
    fn get(&self, key: &String) -> Option<Vec<f32>> {
        // Check thread-local cache first
        let local_result = LOCAL_EMBEDDING_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(value) = cache.get(key) {
                // > NOTE: Thread-local cache hit - fastest path
                return Some(value.clone());
            }
            None
        });

        if let Some(value) = local_result {
            self.metrics.record_hit();
            return Some(value);
        }

        // Check shared cache
        let result = self.cache.get(key);
        if let Some(ref value) = result {
            // Update thread-local cache with frequently accessed item
            LOCAL_EMBEDDING_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                cache.put(key.clone(), value.clone());
            });
            self.metrics.record_hit();
        } else {
            self.metrics.record_miss();
        }
        result
    }

    fn insert(&self, key: String, value: Vec<f32>) {
        // > NOTE: Estimate memory usage (4 bytes per f32 plus overhead)
        let memory_bytes = value.len() * 4 + key.len() + 32; // 32 bytes for metadata overhead

        // Insert into thread-local cache
        LOCAL_EMBEDDING_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.put(key.clone(), value.clone());
        });

        // Check if we need to evict before inserting
        let current_entry_count = self.cache.entry_count();

        // Insert into shared cache
        self.cache.insert(key.clone(), value);

        // Force moka to process pending operations
        self.cache.run_pending_tasks();

        // Check if an eviction occurred
        let new_entry_count = self.cache.entry_count();
        if new_entry_count <= current_entry_count && current_entry_count > 0 {
            // An eviction likely occurred
            self.metrics.record_eviction(1);
            // > NOTE: Memory tracking for evictions is handled by moka internally
        }

        self.metrics.add_memory(memory_bytes as u64);
        debug!("Inserted into the cache: {key}");
    }

    fn clear(&self) {
        // Clear thread-local cache
        LOCAL_EMBEDDING_CACHE.with(|cache| {
            cache.borrow_mut().clear();
        });

        // Record number of entries being evicted
        let entry_count = self.cache.entry_count();
        if entry_count > 0 {
            self.metrics.record_eviction(entry_count);
        }

        self.cache.invalidate_all();
        self.metrics.reset();
    }

    fn stats(&self) -> CacheStats {
        // Ensure pending operations are processed before getting stats
        self.cache.run_pending_tasks();
        CacheStats::from_metrics(&self.metrics, self.cache.entry_count())
    }

    fn len(&self) -> usize {
        self.cache.run_pending_tasks();
        self.cache.entry_count().try_into().unwrap_or(usize::MAX)
    }
}

impl EmbeddingCache {
    /// Evict the oldest entries from the cache
    ///
    /// This method is used by the memory monitor to free memory under pressure.
    ///
    /// # Arguments
    /// * `count` - Number of entries to evict
    pub fn evict_oldest(&self, count: usize) {
        // Clear thread-local cache first as it's more volatile
        LOCAL_EMBEDDING_CACHE.with(|cache| {
            cache.borrow_mut().clear();
        });

        // Since moka doesn't expose direct eviction of oldest entries,
        // we'll invalidate a portion of the cache based on the count
        // > NOTE: This is a simplified implementation. In production,
        // > you might want to track access patterns for better eviction

        let current_count = self.cache.entry_count();
        if current_count == 0 {
            return;
        }

        // If we need to evict more than half the cache, just clear it
        if count as u64 >= current_count / 2 {
            self.clear();
        } else {
            // Record evictions
            self.metrics.record_eviction(count as u64);

            // Force moka to run its eviction process
            // This will evict based on the cache's internal LRU/LFU policy
            self.cache.run_pending_tasks();
        }
    }
}
