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

//! Integration tests for backend detection and configuration

use embellama::{
    BackendInfo, BackendType, EmbeddingEngine, EngineConfig, detect_best_backend,
    get_compiled_backend,
};
use std::env;

/// Helper to get test model path
fn get_test_model_path() -> String {
    env::var("EMBELLAMA_TEST_MODEL").unwrap_or_else(|_| {
        panic!(
            "EMBELLAMA_TEST_MODEL environment variable not set.\n\
             Please run: just download-test-model"
        )
    })
}

#[test]
fn test_backend_detection() {
    let backend = detect_best_backend();
    let compiled = get_compiled_backend();

    println!("Detected backend: {:?}", backend);
    println!("Compiled backend: {:?}", compiled);

    // The detected backend should match the compiled backend
    assert_eq!(backend, compiled);

    // Verify we get a valid backend
    match backend {
        BackendType::Cpu
        | BackendType::OpenMP
        | BackendType::ROCm
        | BackendType::Cuda
        | BackendType::Metal
        | BackendType::Vulkan => {
            // All valid backends
        }
    }
}

#[test]
fn test_backend_info() {
    let info = BackendInfo::new();

    println!("Backend info:\n{}", info);

    // Verify platform is populated
    assert!(!info.platform.is_empty());

    // Verify backend matches compiled backend
    assert_eq!(info.backend, info.compiled_backend);

    // Check that we have at least the default feature
    #[cfg(feature = "openmp")]
    assert!(info.available_features.contains(&"openmp".to_string()));

    #[cfg(feature = "metal")]
    assert!(info.available_features.contains(&"metal".to_string()));

    #[cfg(feature = "cuda")]
    assert!(info.available_features.contains(&"cuda".to_string()));

    #[cfg(feature = "vulkan")]
    assert!(info.available_features.contains(&"vulkan".to_string()));
}

#[test]
fn test_backend_gpu_acceleration() {
    let backend = detect_best_backend();

    // Check GPU acceleration flag
    #[cfg(any(feature = "metal", feature = "cuda", feature = "vulkan"))]
    {
        if matches!(
            backend,
            BackendType::Metal | BackendType::Cuda | BackendType::Vulkan
        ) {
            assert!(backend.is_gpu_accelerated());
            assert_eq!(backend.recommended_gpu_layers(), Some(999));
        }
    }

    // CPU backends should not report GPU acceleration
    if matches!(backend, BackendType::Cpu | BackendType::OpenMP) {
        assert!(!backend.is_gpu_accelerated());
        assert_eq!(backend.recommended_gpu_layers(), None);
    }
}

#[test]
fn test_config_with_backend_detection() {
    use tempfile::NamedTempFile;

    // Create a temporary dummy GGUF file for testing
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let model_path = temp_file.path().with_extension("gguf");

    // Write minimal GGUF header to make it a valid file
    std::fs::write(&model_path, b"GGUF").expect("Failed to write temp file");

    // Test auto-detection configuration
    let config = EngineConfig::with_backend_detection()
        .with_model_path(model_path.to_str().unwrap())
        .with_model_name("test-backend")
        .build()
        .expect("Failed to build config");

    let backend = detect_best_backend();

    // If we have a GPU backend, GPU layers should be set
    if backend.is_gpu_accelerated() {
        assert!(config.use_gpu);
        assert_eq!(config.model_config.n_gpu_layers, Some(999));
    }

    // The config should be valid
    config.validate().expect("Config validation failed");
}

#[test]
#[cfg(feature = "metal")]
fn test_metal_backend_on_macos() {
    #[cfg(target_os = "macos")]
    {
        let backend = detect_best_backend();
        assert_eq!(backend, BackendType::Metal);
        assert!(backend.is_gpu_accelerated());
    }
}

#[test]
fn test_backend_display() {
    // Test Display implementation
    assert_eq!(BackendType::Cpu.to_string(), "CPU");
    assert_eq!(BackendType::OpenMP.to_string(), "OpenMP");
    assert_eq!(BackendType::ROCm.to_string(), "ROCm");
    assert_eq!(BackendType::Cuda.to_string(), "CUDA");
    assert_eq!(BackendType::Metal.to_string(), "Metal");
    assert_eq!(BackendType::Vulkan.to_string(), "Vulkan");
}

#[test]
#[ignore = "Requires model file"]
fn test_engine_with_backend_detection() {
    let model_path = get_test_model_path();

    // Create engine with backend auto-detection
    let config = EngineConfig::with_backend_detection()
        .with_model_path(&model_path)
        .with_model_name("test-backend")
        .with_context_size(2048) // Increased from 512 to accommodate embedding overhead
        .build()
        .expect("Failed to build config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Test that the engine works
    let text = "Test embedding with backend detection";
    let embedding = engine
        .embed(Some("test-backend"), text)
        .expect("Failed to generate embedding");

    assert!(!embedding.is_empty());
    println!("Generated embedding with {} dimensions", embedding.len());

    // Get backend info
    let info = BackendInfo::new();
    println!("Used backend: {}", info.backend);
}
