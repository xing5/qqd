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

/// Embedding cache module
pub mod embedding_cache;
pub mod memory_monitor;
/// Metrics module
pub mod metrics;
pub mod prefix_cache;
#[cfg(feature = "redis-cache")]
pub mod redis_backend;
/// Token cache module
pub mod token_cache;

use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use tracing::trace;

pub use metrics::CacheMetrics;

/// Statistics for cache performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Current memory usage in bytes
    pub memory_bytes: u64,
    /// Number of entries in the cache
    pub entry_count: u64,
    /// Hit rate (hits / (hits + misses))
    pub hit_rate: f64,
}

impl CacheStats {
    /// Create new cache stats from metrics
    pub fn from_metrics(metrics: &CacheMetrics, entry_count: u64) -> Self {
        trace!("Creating stats from metrics: {metrics:?}, entry_count: {entry_count}");
        let hits = metrics.hits.load(Ordering::Relaxed);
        let misses = metrics.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        #[allow(clippy::cast_precision_loss)]
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        Self {
            hits,
            misses,
            evictions: metrics.evictions.load(Ordering::Relaxed),
            memory_bytes: metrics.memory_bytes.load(Ordering::Relaxed),
            entry_count,
            hit_rate,
        }
    }
}

/// Trait for cache storage implementations
pub trait CacheStore<K, V> {
    /// Get a value from the cache
    fn get(&self, key: &K) -> Option<V>;

    /// Insert a value into the cache
    fn insert(&self, key: K, value: V);

    /// Clear all entries from the cache
    fn clear(&self);

    /// Get cache statistics
    fn stats(&self) -> CacheStats;

    /// Get the number of entries in the cache
    fn len(&self) -> usize;

    /// Check if the cache is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
