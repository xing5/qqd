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

//! Memory pressure monitoring and automatic cache eviction
//!
//! This module provides background monitoring of system memory and
//! triggers cache eviction when memory pressure exceeds configured thresholds.

use crate::cache::{CacheStore, embedding_cache::EmbeddingCache, token_cache::TokenCache};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(feature = "server")]
use std::time::Duration;
use sysinfo::System;
use tracing::info;
#[cfg(feature = "server")]
use tracing::{debug, warn};

/// Memory monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMonitorConfig {
    /// Enable memory monitoring
    pub enabled: bool,
    /// Memory threshold in MB - trigger eviction when available memory drops below this
    pub threshold_mb: usize,
    /// Percentage of cache entries to evict when threshold is exceeded (0.0 - 1.0)
    pub eviction_percentage: f32,
    /// Check interval in seconds
    pub check_interval_secs: u64,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_mb: 512,
            eviction_percentage: 0.2,
            check_interval_secs: 60,
        }
    }
}

/// Memory pressure monitor
pub struct MemoryMonitor {
    config: MemoryMonitorConfig,
    system: System,
    embedding_cache: Option<Arc<EmbeddingCache>>,
    token_cache: Option<Arc<TokenCache>>,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(config: MemoryMonitorConfig) -> Self {
        Self {
            config,
            system: System::new(),
            embedding_cache: None,
            token_cache: None,
        }
    }

    /// Set the embedding cache to monitor
    #[must_use]
    pub fn with_embedding_cache(mut self, cache: Arc<EmbeddingCache>) -> Self {
        self.embedding_cache = Some(cache);
        self
    }

    /// Set the token cache to monitor
    #[must_use]
    pub fn with_token_cache(mut self, cache: Arc<TokenCache>) -> Self {
        self.token_cache = Some(cache);
        self
    }

    /// Check current memory pressure
    pub fn check_pressure(&mut self) -> bool {
        self.system.refresh_memory();
        let available_mb = self.system.available_memory() / 1024 / 1024;
        available_mb < self.config.threshold_mb as u64
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&mut self) -> MemoryStats {
        self.system.refresh_memory();

        let total_bytes = self.system.total_memory();
        let available_bytes = self.system.available_memory();
        let used_bytes = total_bytes.saturating_sub(available_bytes);
        #[allow(clippy::cast_precision_loss)]
        let usage_percentage = if total_bytes > 0 {
            (used_bytes as f32 / total_bytes as f32) * 100.0
        } else {
            0.0
        };

        MemoryStats {
            total_bytes,
            available_bytes,
            used_bytes,
            usage_percentage,
        }
    }

    /// Evict cache entries to free memory
    pub fn evict_entries(&self) {
        let evict_count_embedding = if let Some(cache) = &self.embedding_cache {
            let stats = cache.stats();
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let to_evict = (stats.entry_count as f32 * self.config.eviction_percentage) as usize;

            if to_evict > 0 {
                info!(
                    "Evicting {} embedding cache entries due to memory pressure",
                    to_evict
                );
                cache.evict_oldest(to_evict);
            }

            to_evict
        } else {
            0
        };

        let evict_count_token = if let Some(cache) = &self.token_cache {
            let stats = cache.stats();
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let to_evict = (stats.entry_count as f32 * self.config.eviction_percentage) as usize;

            if to_evict > 0 {
                info!(
                    "Evicting {} token cache entries due to memory pressure",
                    to_evict
                );
                cache.evict_oldest(to_evict);
            }

            to_evict
        } else {
            0
        };

        if evict_count_embedding > 0 || evict_count_token > 0 {
            info!(
                "Memory pressure eviction complete: {} embedding entries, {} token entries",
                evict_count_embedding, evict_count_token
            );
        }
    }

    /// Start the monitoring loop (spawns a tokio task)
    #[cfg(feature = "server")]
    pub fn start_monitoring(mut self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if !self.config.enabled {
                debug!("Memory monitoring is disabled");
                return;
            }

            info!(
                "Starting memory monitor with threshold {}MB, check interval {}s",
                self.config.threshold_mb, self.config.check_interval_secs
            );

            let mut interval =
                tokio::time::interval(Duration::from_secs(self.config.check_interval_secs));

            loop {
                interval.tick().await;

                if self.check_pressure() {
                    let stats = self.get_memory_stats();
                    warn!(
                        "Memory pressure detected: {}MB available (threshold: {}MB), usage: {:.1}%",
                        stats.available_bytes / 1024 / 1024,
                        self.config.threshold_mb,
                        stats.usage_percentage
                    );

                    self.evict_entries();

                    // Check again after eviction
                    self.system.refresh_memory();
                    let new_available_mb = self.system.available_memory() / 1024 / 1024;
                    info!("Memory after eviction: {}MB available", new_available_mb);
                } else {
                    debug!(
                        "Memory check OK: {}MB available",
                        self.system.available_memory() / 1024 / 1024
                    );
                }
            }
        })
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    /// Total system memory in bytes
    pub total_bytes: u64,
    /// Available system memory in bytes
    pub available_bytes: u64,
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Memory usage percentage
    pub usage_percentage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let mut monitor = MemoryMonitor::new(MemoryMonitorConfig::default());
        let stats = monitor.get_memory_stats();

        assert!(stats.total_bytes > 0);
        assert!(stats.usage_percentage >= 0.0 && stats.usage_percentage <= 100.0);
        assert_eq!(stats.used_bytes, stats.total_bytes - stats.available_bytes);
    }

    #[test]
    fn test_config_defaults() {
        let config = MemoryMonitorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.threshold_mb, 512);
        assert_eq!(config.eviction_percentage, 0.2);
        assert_eq!(config.check_interval_secs, 60);
    }
}
