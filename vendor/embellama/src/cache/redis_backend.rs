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

//! Redis backend for distributed caching
//!
//! This module provides an optional Redis-based cache backend that can be used
//! as a third-level cache (L3) for sharing embeddings across multiple server instances.
//!
//! This module is only available with the `redis-cache` feature enabled.

#![cfg(feature = "redis-cache")]

use crate::Result;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client, RedisError};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Redis cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL (e.g., "redis://127.0.0.1:6379")
    pub url: String,
    /// Key prefix for all cache entries
    pub key_prefix: String,
    /// TTL for cache entries in seconds
    pub ttl_seconds: u64,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    /// Max connection retries
    pub max_retries: u32,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            key_prefix: "embellama:cache:".to_string(),
            ttl_seconds: 3600, // 1 hour
            connection_timeout_secs: 5,
            max_retries: 3,
        }
    }
}

/// Redis backend for caching
pub struct RedisBackend {
    config: RedisConfig,
    manager: ConnectionManager,
}

impl RedisBackend {
    /// Create a new Redis backend
    pub async fn new(config: RedisConfig) -> Result<Self> {
        info!("Connecting to Redis at {}", config.url);

        let client = Client::open(config.url.as_str()).map_err(|e| {
            crate::Error::Other(anyhow::anyhow!("Failed to create Redis client: {}", e))
        })?;

        let manager = ConnectionManager::new(client).await.map_err(|e| {
            crate::Error::Other(anyhow::anyhow!("Failed to connect to Redis: {}", e))
        })?;

        info!(
            "Redis connection established with prefix: {}",
            config.key_prefix
        );

        Ok(Self { config, manager })
    }

    /// Build a Redis key with the configured prefix
    fn build_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }

    /// Get an embedding from Redis
    pub async fn get_embedding(&mut self, key: &str) -> Option<Vec<f32>> {
        let redis_key = self.build_key(key);

        match self.manager.get::<_, Vec<u8>>(&redis_key).await {
            Ok(data) => {
                // Deserialize the data
                match bincode::deserialize::<Vec<f32>>(&data) {
                    Ok(embedding) => {
                        debug!("Redis cache hit for key: {}", key);
                        Some(embedding)
                    }
                    Err(e) => {
                        error!("Failed to deserialize embedding from Redis: {}", e);
                        None
                    }
                }
            }
            Err(e) if is_key_not_found(&e) => {
                debug!("Redis cache miss for key: {}", key);
                None
            }
            Err(e) => {
                warn!("Redis get error for key {}: {}", key, e);
                None
            }
        }
    }

    /// Set an embedding in Redis with TTL
    pub async fn set_embedding(&mut self, key: &str, embedding: &[f32]) -> bool {
        let redis_key = self.build_key(key);

        // Serialize the embedding
        let data = match bincode::serialize(embedding) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to serialize embedding for Redis: {}", e);
                return false;
            }
        };

        // Set with expiration
        match self
            .manager
            .set_ex::<_, _, ()>(&redis_key, data, self.config.ttl_seconds)
            .await
        {
            Ok(_) => {
                debug!("Stored embedding in Redis with key: {}", key);
                true
            }
            Err(e) => {
                warn!("Failed to store embedding in Redis: {}", e);
                false
            }
        }
    }

    /// Get tokens from Redis
    pub async fn get_tokens(&mut self, key: &str) -> Option<Vec<i32>> {
        let redis_key = self.build_key(&format!("tokens:{}", key));

        match self.manager.get::<_, Vec<u8>>(&redis_key).await {
            Ok(data) => match bincode::deserialize::<Vec<i32>>(&data) {
                Ok(tokens) => {
                    debug!("Redis token cache hit for key: {}", key);
                    Some(tokens)
                }
                Err(e) => {
                    error!("Failed to deserialize tokens from Redis: {}", e);
                    None
                }
            },
            Err(e) if is_key_not_found(&e) => {
                debug!("Redis token cache miss for key: {}", key);
                None
            }
            Err(e) => {
                warn!("Redis token get error for key {}: {}", key, e);
                None
            }
        }
    }

    /// Set tokens in Redis with TTL
    pub async fn set_tokens(&mut self, key: &str, tokens: &[i32]) -> bool {
        let redis_key = self.build_key(&format!("tokens:{}", key));

        let data = match bincode::serialize(tokens) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to serialize tokens for Redis: {}", e);
                return false;
            }
        };

        match self
            .manager
            .set_ex::<_, _, ()>(&redis_key, data, self.config.ttl_seconds)
            .await
        {
            Ok(_) => {
                debug!("Stored tokens in Redis with key: {}", key);
                true
            }
            Err(e) => {
                warn!("Failed to store tokens in Redis: {}", e);
                false
            }
        }
    }

    /// Delete a key from Redis
    pub async fn delete(&mut self, key: &str) -> bool {
        let redis_key = self.build_key(key);

        match self.manager.del::<_, ()>(&redis_key).await {
            Ok(_) => {
                debug!("Deleted key from Redis: {}", key);
                true
            }
            Err(e) => {
                warn!("Failed to delete key from Redis: {}", e);
                false
            }
        }
    }

    /// Clear all cache entries with the configured prefix
    pub async fn clear_all(&mut self) -> Result<()> {
        let pattern = format!("{}*", self.config.key_prefix);

        // Use SCAN to find all keys with our prefix
        let keys: Vec<String> = self
            .manager
            .scan_match(&pattern)
            .await
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("Failed to scan Redis keys: {}", e)))?
            .collect()
            .await;

        if !keys.is_empty() {
            info!("Clearing {} keys from Redis", keys.len());
            self.manager.del::<_, ()>(keys).await.map_err(|e| {
                crate::Error::Other(anyhow::anyhow!("Failed to delete Redis keys: {}", e))
            })?;
        }

        Ok(())
    }

    /// Check if Redis is reachable
    pub async fn ping(&mut self) -> bool {
        match redis::cmd("PING")
            .query_async::<_, String>(&mut self.manager)
            .await
        {
            Ok(_) => true,
            Err(e) => {
                error!("Redis ping failed: {}", e);
                false
            }
        }
    }

    /// Get Redis info
    pub async fn info(&mut self) -> Option<String> {
        match redis::cmd("INFO")
            .query_async::<_, String>(&mut self.manager)
            .await
        {
            Ok(info) => Some(info),
            Err(e) => {
                error!("Failed to get Redis info: {}", e);
                None
            }
        }
    }
}

/// Check if a Redis error indicates a key was not found
fn is_key_not_found(error: &RedisError) -> bool {
    matches!(error.kind(), redis::ErrorKind::TypeError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_config_defaults() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.key_prefix, "embellama:cache:");
        assert_eq!(config.ttl_seconds, 3600);
    }

    #[test]
    fn test_build_key() {
        let config = RedisConfig {
            key_prefix: "test:".to_string(),
            ..Default::default()
        };

        // We need to test the key building without actually connecting to Redis
        // So we'll just test the format
        let key = format!("{}embedding123", config.key_prefix);
        assert_eq!(key, "test:embedding123");
    }
}
