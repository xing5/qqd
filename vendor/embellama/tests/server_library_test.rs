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

//! Tests for the server library API

#![cfg(feature = "server")]

mod server_test_helpers;

use embellama::server::{
    AppState, EngineConfig, FileModelProvider, ModelProvider, ServerConfig, create_router,
};
use server_test_helpers::*;
use std::path::PathBuf;

#[test]
fn test_server_config_builder() {
    let model_path = match get_test_model_path() {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping test_server_config_builder: test model not found");
            return;
        }
    };

    // Test building with explicit EngineConfig
    let engine_config = EngineConfig::builder()
        .with_model_path(model_path.to_string_lossy().to_string())
        .with_model_name("custom-model")
        .build()
        .expect("Should build engine config");

    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .host("0.0.0.0")
        .port(9000)
        .worker_count(4)
        .queue_size(200)
        .build()
        .expect("Should build with all fields");

    assert_eq!(config.engine_config.model_config.model_path, model_path);
    assert_eq!(config.engine_config.model_config.model_name, "custom-model");
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 9000);
    assert_eq!(config.worker_count, 4);
    assert_eq!(config.queue_size, 200);

    // Test that default values are applied
    let engine_config2 = EngineConfig::builder()
        .with_model_path(model_path.to_string_lossy().to_string())
        .with_model_name("test-model")
        .build()
        .expect("Should build engine config");

    let config2 = ServerConfig::builder()
        .engine_config(engine_config2)
        .build()
        .expect("Should build with defaults");

    assert_eq!(config2.host, "127.0.0.1");
    assert_eq!(config2.port, 8080);
}

#[test]
fn test_server_config_builder_requires_engine_config() {
    // Test that building without engine_config fails
    let result = ServerConfig::builder().build();

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Engine configuration is required"));
    }
}

#[tokio::test]
async fn test_file_model_provider() {
    let provider = FileModelProvider::new(PathBuf::from("/test/model.gguf"), "test-model");

    // Test get_model_path with matching name
    let path = provider.get_model_path("test-model").await.unwrap();
    assert_eq!(path, PathBuf::from("/test/model.gguf"));

    // Test get_model_path with non-matching name
    let result = provider.get_model_path("other-model").await;
    assert!(result.is_err());

    // Test list_models - even with non-existent file, it should return a result
    let models = provider.list_models().await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "test-model");
}

#[tokio::test]
async fn test_file_model_provider_with_real_gguf() {
    // Skip test if model file doesn't exist
    let model_path = if let Ok(path) = get_test_model_path() {
        path
    } else {
        eprintln!("Skipping test_file_model_provider_with_real_gguf: test model not found");
        return;
    };

    let provider = FileModelProvider::new(model_path.clone(), "minilm-test");

    // Test list_models with real GGUF file
    let models = provider.list_models().await.unwrap();
    assert_eq!(models.len(), 1);

    let model_info = &models[0];
    assert_eq!(model_info.name, "minilm-test");

    // MiniLM-L6-v2 should have 384 dimensions
    assert_eq!(
        model_info.dimensions, 384,
        "Expected MiniLM-L6-v2 to have 384 dimensions"
    );

    // Check that max_tokens is reasonable (should be > 0)
    assert!(
        model_info.max_tokens > 0,
        "Expected max_tokens to be greater than 0"
    );

    // Check that model_size is populated
    assert!(
        model_info.model_size.is_some(),
        "Expected model_size to be populated"
    );

    // The test model should be around 15MB
    if let Some(size) = model_info.model_size {
        assert!(size > 10_000_000, "Expected model size to be > 10MB");
        assert!(size < 50_000_000, "Expected model size to be < 50MB");
    }
}

#[tokio::test]
async fn test_create_router_with_custom_state() {
    let model_path = get_test_model_path().expect("Test model not found");

    let engine_config = EngineConfig::builder()
        .with_model_path(model_path.to_string_lossy().to_string())
        .with_model_name("library-test-model")
        .build()
        .expect("Should build engine config");

    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .worker_count(1)
        .build()
        .expect("Should build config");

    let state = AppState::new(config).expect("Should create app state");

    // Create router using library API
    let _router = create_router(state.clone());

    // The router should be created without panicking
    // In a real test, we'd start a test server and make requests
    assert_eq!(state.model_name(), "library-test-model");
}

#[tokio::test]
async fn test_nested_router_integration() {
    let model_path = get_test_model_path().expect("Test model not found");

    let engine_config = EngineConfig::builder()
        .with_model_path(model_path.to_string_lossy().to_string())
        .with_model_name("nested-test-model")
        .build()
        .expect("Should build engine config");

    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .worker_count(1)
        .build()
        .expect("Should build config");

    let state = AppState::new(config).expect("Should create app state");

    // Create base router
    let embedding_router = create_router(state);

    // Create main app router and nest the embedding router
    let app = axum::Router::new()
        .nest("/api/embeddings", embedding_router)
        .route("/", axum::routing::get(|| async { "Main app" }));

    // The nested router should work without conflicts
    // In a real test, we'd verify routes are accessible at /api/embeddings/v1/*

    // Just verify it compiles and doesn't panic
    let _ = app;
}
