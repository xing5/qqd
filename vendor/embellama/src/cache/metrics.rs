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

#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic metrics for cache performance tracking
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Number of cache hits
    pub hits: AtomicU64,
    /// Number of cache misses
    pub misses: AtomicU64,
    /// Number of evictions
    pub evictions: AtomicU64,
    /// Current memory usage in bytes
    pub memory_bytes: AtomicU64,
}

impl CacheMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment hit count
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment miss count
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment eviction count
    pub fn record_eviction(&self, count: u64) {
        self.evictions.fetch_add(count, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn update_memory(&self, bytes: u64) {
        self.memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Add to memory usage
    pub fn add_memory(&self, bytes: u64) {
        self.memory_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Subtract from memory usage
    pub fn sub_memory(&self, bytes: u64) {
        self.memory_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.memory_bytes.store(0, Ordering::Relaxed);
    }

    /// Generate a metrics report
    pub fn report(&self) -> MetricsReport {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        MetricsReport {
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
            miss_rate: if total_requests > 0 {
                misses as f64 / total_requests as f64
            } else {
                0.0
            },
            total_requests,
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Report containing cache metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Miss rate (0.0 to 1.0)
    pub miss_rate: f64,
    /// Total number of cache requests
    pub total_requests: u64,
    /// Number of hits
    pub hits: u64,
    /// Number of misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Current memory usage in bytes
    pub memory_usage_bytes: u64,
}

impl MetricsReport {
    /// Get memory usage in megabytes
    pub fn memory_usage_mb(&self) -> f64 {
        self.memory_usage_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Check if the cache is effective (hit rate above threshold)
    pub fn is_effective(&self, threshold: f64) -> bool {
        self.hit_rate >= threshold
    }
}
