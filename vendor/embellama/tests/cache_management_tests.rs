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

//! Integration tests for cache management features

use embellama::cache::{CacheStore, embedding_cache::EmbeddingCache, token_cache::TokenCache};
use embellama::{CacheConfig, EmbeddingEngine, EngineConfig};
use serial_test::serial;
use std::sync::Arc;

/// Test cache eviction functionality
#[test]
#[serial]
fn test_cache_eviction() {
    // Create a small cache for testing
    let config = CacheConfig {
        enabled: true,
        embedding_cache_size: 10,
        token_cache_size: 10,
        ..Default::default()
    };

    let cache = Arc::new(EmbeddingCache::new(
        config.embedding_cache_size as u64,
        config.ttl_seconds,
    ));

    // Fill the cache
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = vec![i as f32; 100];
        cache.insert(key, value);
    }

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 10);

    // Evict half the entries
    cache.evict_oldest(5);

    let stats_after = cache.stats();
    // The exact count might vary due to internal cache behavior
    assert!(stats_after.entry_count <= 5);
}

/// Test token cache eviction
#[test]
#[serial]
fn test_token_cache_eviction() {
    use llama_cpp_2::token::LlamaToken;

    let cache = Arc::new(TokenCache::new(10));

    // Fill the cache
    for i in 0..10 {
        let key = format!("token_key_{}", i);
        // Create LlamaToken instances (they're just i32 wrappers)
        let tokens: Vec<LlamaToken> = (0..5).map(|j| LlamaToken::new(i + j)).collect();
        cache.insert(key, tokens);
    }

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 10);

    // Evict some entries
    cache.evict_oldest(3);

    let stats_after = cache.stats();
    assert!(stats_after.entry_count <= 7);
}

/// Test memory monitor functionality
#[test]
#[serial]
fn test_memory_monitor() {
    use embellama::cache::memory_monitor::{MemoryMonitor, MemoryMonitorConfig};

    let config = MemoryMonitorConfig {
        enabled: true,
        threshold_mb: 1, // Set very low threshold so it doesn't trigger
        eviction_percentage: 0.2,
        check_interval_secs: 1,
    };

    let mut monitor = MemoryMonitor::new(config);

    // Check that we can get memory stats
    let stats = monitor.get_memory_stats();
    assert!(stats.total_bytes > 0);
    assert!(stats.available_bytes > 0);
    assert!(stats.usage_percentage >= 0.0 && stats.usage_percentage <= 100.0);

    // Check pressure (should be false with high threshold)
    assert!(!monitor.check_pressure());
}

/// Test cache warm-up functionality
#[test]
#[serial]
fn test_cache_warmup() {
    // Skip this test if the model file doesn't exist
    let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
        .unwrap_or_else(|_| "models/test/all-minilm-l6-v2-q4_k_m.gguf".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping test_cache_warmup: model file not found");
        return;
    }

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("test-model")
        .with_cache_config(CacheConfig::default())
        .build()
        .expect("Failed to create engine config");

    let engine = EmbeddingEngine::get_or_init(config).expect("Failed to create engine");
    let engine = engine.lock().unwrap();

    // Warm the cache with some texts
    let texts = vec!["test text 1", "test text 2", "test text 3"];
    engine
        .warm_cache(None, &texts)
        .expect("Failed to warm cache");

    // Check that entries are in cache
    let stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert!(stats.entry_count >= 3);

    // Now these should be cache hits
    for text in texts {
        let _ = engine.embed(None, text).expect("Failed to get embedding");
    }

    let final_stats = engine.get_cache_stats().expect("Cache should be enabled");
    assert!(final_stats.hits >= 3);
}

/// Test cache statistics aggregation
#[test]
#[serial]
fn test_cache_stats_aggregation() {
    let config = CacheConfig {
        enabled: true,
        embedding_cache_size: 100,
        ..Default::default()
    };

    let cache = Arc::new(EmbeddingCache::new(
        config.embedding_cache_size as u64,
        config.ttl_seconds,
    ));

    // Insert some entries
    for i in 0..5 {
        cache.insert(format!("key_{}", i), vec![i as f32; 100]);
    }

    // Get some hits and misses
    for i in 0..10 {
        let key = format!("key_{}", i);
        let _ = cache.get(&key);
    }

    let stats = cache.stats();
    assert_eq!(stats.hits, 5);
    assert_eq!(stats.misses, 5);
    assert_eq!(stats.entry_count, 5);
    assert_eq!(stats.hit_rate, 0.5);
}

#[cfg(feature = "server")]
mod server_tests {
    use super::*;
    use axum::http::StatusCode;
    use axum_test::TestServer;
    use embellama::server::api_types::{CacheClearResponse, CacheStatsResponse, CacheWarmRequest};
    use embellama::server::{AppState, EngineConfig, ServerConfig, create_router};
    use serde_json::json;

    /// Create a test server with caching enabled
    async fn create_test_server() -> TestServer {
        let model_path = std::env::var("EMBELLAMA_TEST_MODEL")
            .unwrap_or_else(|_| "models/test/all-minilm-l6-v2-q4_k_m.gguf".to_string());

        if !std::path::Path::new(&model_path).exists() {
            panic!("Test model not found at: {}", model_path);
        }

        let engine_config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name("test-model")
            .build()
            .expect("Failed to create engine config");

        let config = ServerConfig::builder()
            .engine_config(engine_config)
            .worker_count(1)
            .queue_size(10)
            .build()
            .expect("Failed to create server config");

        let state = AppState::new(config).expect("Failed to create app state");
        let app = create_router(state);

        TestServer::new(app.into_make_service()).expect("Failed to create test server")
    }

    #[tokio::test]
    #[serial]
    async fn test_cache_stats_endpoint() {
        let server = create_test_server().await;

        let response = server.get("/cache/stats").await;

        assert_eq!(response.status_code(), StatusCode::OK);

        let stats: CacheStatsResponse = response.json();
        assert!(stats.enabled);
        assert!(stats.memory.total_bytes > 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_cache_clear_endpoint() {
        let server = create_test_server().await;

        let response = server.post("/cache/clear").await;

        assert_eq!(response.status_code(), StatusCode::OK);

        let clear_response: CacheClearResponse = response.json();
        assert_eq!(clear_response.status, "Cache cleared successfully");
    }

    #[tokio::test]
    #[serial]
    async fn test_cache_warm_endpoint() {
        let server = create_test_server().await;

        let request = CacheWarmRequest {
            texts: vec!["test1".to_string(), "test2".to_string()],
            model: None,
        };

        let response = server.post("/cache/warm").json(&request).await;

        assert_eq!(response.status_code(), StatusCode::OK);

        let warm_response: serde_json::Value = response.json();
        assert_eq!(warm_response["status"], "Cache warming completed");
        assert_eq!(warm_response["texts_processed"], 2);
    }

    #[tokio::test]
    #[serial]
    async fn test_cache_warm_validation() {
        let server = create_test_server().await;

        // Test with empty texts
        let request = CacheWarmRequest {
            texts: vec![],
            model: None,
        };

        let response = server.post("/cache/warm").json(&request).await;

        assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

        // Test with too many texts
        let request = CacheWarmRequest {
            texts: vec!["test".to_string(); 1001],
            model: None,
        };

        let response = server.post("/cache/warm").json(&request).await;

        assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);
    }
}

#[cfg(feature = "redis-cache")]
mod redis_tests {
    use super::*;
    use embellama::cache::redis_backend::{RedisBackend, RedisConfig};

    #[tokio::test]
    #[serial]
    async fn test_redis_config() {
        let config = RedisConfig::default();
        assert_eq!(config.key_prefix, "embellama:cache:");
        assert_eq!(config.ttl_seconds, 3600);

        // Note: Actual Redis connection tests would require a running Redis instance
        // and should be marked as ignored or conditional
    }

    #[tokio::test]
    #[serial]
    #[ignore = "Requires running Redis instance"]
    async fn test_redis_backend_connection() {
        let config = RedisConfig::default();
        let backend = RedisBackend::new(config).await;

        match backend {
            Ok(mut backend) => {
                assert!(backend.ping().await);
            }
            Err(_) => {
                eprintln!("Redis not available, skipping test");
            }
        }
    }
}
