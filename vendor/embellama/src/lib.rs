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

#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate)]
#![doc = include_str!("../README.md")]

/// Backend detection and configuration
mod backend;

/// Error handling module
mod error;

/// Configuration module
mod config;

/// GGUF metadata extraction utilities
mod gguf;

/// Model management module
mod model;

/// Embedding engine module
mod engine;

/// Batch processing module
mod batch;

/// Cache module for token and embedding caching
pub mod cache;

/// Server module (feature-gated)
#[cfg(feature = "server")]
pub mod server;

// Re-export main types
pub use backend::{BackendInfo, BackendType, detect_best_backend, get_compiled_backend};
pub use batch::{BatchProcessor, BatchProcessorBuilder};
pub use config::{
    CacheConfig, CacheConfigBuilder, EmbeddingConfig, EmbeddingConfigBuilder, EngineConfig,
    EngineConfigBuilder, ModelConfig, ModelConfigBuilder, NormalizationMode, PoolingStrategy,
    RerankResult, TruncateTokens,
};
pub use engine::{EmbeddingEngine, ModelInfo};
pub use error::{Error, Result};
pub use gguf::{
    GGUFMetadata, clear_metadata_cache, extract_metadata as extract_gguf_metadata,
    metadata_cache_size,
};
pub use model::EmbeddingModel;

use llama_cpp_2::LogOptions;

use std::sync::Once;
use tracing::{debug, info};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Global initialization guard to ensure tracing is only initialized once
static INIT: Once = Once::new();

/// Initialize the library with default tracing subscriber
///
/// This function sets up a global tracing subscriber for the library.
/// It uses a global subscriber to ensure logging infrastructure outlives
/// all threads, preventing thread-local storage panics during cleanup.
///
/// Call this once at the start of your application.
///
/// # Example
///
/// ```
/// embellama::init();
/// ```
pub fn init() {
    init_with_env_filter("info");
}

/// Initialize the library with a custom environment filter
///
/// This function sets up a global tracing subscriber with a custom filter string.
/// The global subscriber ensures logging is available during all cleanup operations,
/// preventing thread-local storage issues.
///
/// # Arguments
///
/// * `filter` - Environment filter string (e.g., "info", "debug", "embellama=debug")
///
/// # Example
///
/// ```
/// embellama::init_with_env_filter("embellama=debug,info");
/// ```
pub fn init_with_env_filter(filter: &str) {
    use tracing_subscriber::{EnvFilter, fmt};

    INIT.call_once(|| {
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter));

        // Use init() to set a global default subscriber
        // This ensures the subscriber outlives all threads
        fmt().with_env_filter(env_filter).init();

        // Enable logging - with global tracing subscriber, this is safe
        llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

        info!("Embellama library initialized v{}", VERSION);
        debug!("Debug logging enabled");
    });
}

/// Get library version information
///
/// Returns a struct containing version and build information.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Library version
    pub version: &'static str,
    /// Rust compiler version used to build
    pub rustc_version: &'static str,
    /// Target architecture
    pub target_arch: &'static str,
    /// Target OS
    pub target_os: &'static str,
    /// Whether server feature is enabled
    pub server_enabled: bool,
}

impl VersionInfo {
    /// Create version info struct
    pub fn new() -> Self {
        Self {
            version: VERSION,
            rustc_version: "unknown",
            target_arch: std::env::consts::ARCH,
            target_os: std::env::consts::OS,
            server_enabled: cfg!(feature = "server"),
        }
    }
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Embellama v{}", self.version)?;
        writeln!(f, "  Target: {}-{}", self.target_arch, self.target_os)?;
        writeln!(
            f,
            "  Server: {}",
            if self.server_enabled {
                "enabled"
            } else {
                "disabled"
            }
        )?;
        Ok(())
    }
}

/// Get version information
pub fn version_info() -> VersionInfo {
    VersionInfo::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert_eq!(info.version, VERSION);
        assert!(!info.target_arch.is_empty());
        assert!(!info.target_os.is_empty());
    }

    #[test]
    fn test_version_display() {
        let info = version_info();
        let display = info.to_string();
        assert!(display.contains("Embellama"));
        assert!(display.contains(VERSION));
    }
}
