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

//! HTTP request handlers for cache management endpoints
//!
//! This module implements the handler functions for cache statistics,
//! clearing, and warming endpoints.

use crate::server::api_types::{
    CacheClearResponse, CacheStatsResponse, CacheWarmRequest, CacheWarmResponse, ErrorResponse,
    MemoryInfo, PrefixInfo, PrefixListResponse, PrefixRegisterRequest, PrefixRegisterResponse,
    PrefixStatsResponse,
};
use crate::server::state::AppState;
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info};

/// Returns a 500 error response for a poisoned engine lock.
fn lock_poisoned_response() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse::internal_error("Engine lock poisoned")),
    )
}

/// Handler for GET /cache/stats
///
/// Returns cache statistics and system memory information
pub async fn cache_stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Processing cache stats request");

    // Get cache stats from the engine
    let cache_stats = {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        engine.get_cache_stats()
    };

    // Get system memory info
    let memory = get_memory_info();

    let response = CacheStatsResponse {
        enabled: cache_stats.is_some(),
        stats: cache_stats,
        memory,
    };

    info!("Returning cache stats: enabled={}", response.enabled);
    (StatusCode::OK, Json(response)).into_response()
}

/// Handler for POST /cache/clear
///
/// Clears all caches and returns previous statistics
pub async fn cache_clear_handler(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Processing cache clear request");

    // Get current stats before clearing
    let previous_stats = {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        let cache_stats = engine.get_cache_stats();
        engine.clear_cache();
        cache_stats
    };

    let response = CacheClearResponse {
        status: "Cache cleared successfully".to_string(),
        previous_stats,
    };

    info!("Cache cleared successfully");
    (StatusCode::OK, Json(response)).into_response()
}

/// Handler for POST /cache/warm
///
/// Pre-computes embeddings for the provided texts to warm the cache
#[allow(clippy::too_many_lines)]
pub async fn cache_warm_handler(
    State(state): State<AppState>,
    Json(request): Json<CacheWarmRequest>,
) -> impl IntoResponse {
    debug!(
        "Processing cache warm request with {} texts",
        request.texts.len()
    );

    // Validate input
    if request.texts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request(
                "No texts provided for warming",
            )),
        )
            .into_response();
    }

    if request.texts.len() > 1000 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request(
                "Too many texts provided (max 1000)",
            )),
        )
            .into_response();
    }

    // Check cache status before warming
    let initial_stats = {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        engine.get_cache_stats()
    };

    // Warm the cache
    let result = {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        let texts: Vec<&str> = request
            .texts
            .iter()
            .map(std::string::String::as_str)
            .collect();
        engine.warm_cache(request.model.as_deref(), &texts)
    };

    match result {
        Ok(()) => {
            // Get stats after warming
            let final_stats = {
                let Ok(engine) = state.engine.lock() else {
                    return lock_poisoned_response().into_response();
                };
                engine.get_cache_stats()
            };

            // Calculate how many were already cached
            let already_cached = if let (Some(initial), Some(final_)) = (initial_stats, final_stats)
            {
                // If the cache entry count didn't increase by the full amount,
                // some were already cached
                let new_entries =
                    usize::try_from(final_.entry_count.saturating_sub(initial.entry_count))
                        .unwrap_or(0);
                request.texts.len().saturating_sub(new_entries)
            } else {
                0
            };

            let response = CacheWarmResponse {
                status: "Cache warming completed".to_string(),
                texts_processed: request.texts.len(),
                already_cached,
            };

            info!(
                "Cache warmed with {} texts ({} already cached)",
                request.texts.len(),
                already_cached
            );
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Failed to warm cache: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::internal_error(format!(
                    "Failed to warm cache: {e}"
                ))),
            )
                .into_response()
        }
    }
}

/// Get current system memory information using sysinfo
fn get_memory_info() -> MemoryInfo {
    use sysinfo::System;

    let mut system = System::new();
    system.refresh_memory();

    let total_bytes = system.total_memory();
    let available_bytes = system.available_memory();
    let used_bytes = total_bytes.saturating_sub(available_bytes);
    #[allow(clippy::cast_precision_loss)]
    let usage_percentage = if total_bytes > 0 {
        (used_bytes as f32 / total_bytes as f32) * 100.0
    } else {
        0.0
    };

    MemoryInfo {
        total_bytes,
        available_bytes,
        usage_percentage,
    }
}

// Prefix cache handlers

/// Handler for POST /v1/embeddings/prefix
///
/// Registers a new prefix for KV cache optimization
pub async fn prefix_register_handler(
    State(state): State<AppState>,
    Json(request): Json<PrefixRegisterRequest>,
) -> impl IntoResponse {
    debug!("Processing prefix register request");

    // Validate input
    if request.prefix.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request("Prefix cannot be empty")),
        )
            .into_response();
    }

    if request.prefix.len() > 10_000 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::invalid_request(
                "Prefix too long (max 10,000 characters)",
            )),
        )
            .into_response();
    }

    // Register the prefix
    let result = {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        engine.register_prefix(request.model.as_deref(), &request.prefix)
    };

    match result {
        Ok(()) => {
            // Estimate memory usage (rough approximation)
            let token_count = request.prefix.len() / 4; // Rough estimate
            let memory_usage = (token_count * 1024) as u64; // ~1KB per token estimate

            let response = PrefixRegisterResponse {
                status: "Prefix registered successfully".to_string(),
                token_count,
                memory_usage,
            };

            info!("Registered prefix of ~{} tokens", token_count);
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Failed to register prefix: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::internal_error(format!(
                    "Failed to register prefix: {e}"
                ))),
            )
                .into_response()
        }
    }
}

/// Handler for GET /v1/embeddings/prefix
///
/// Lists all cached prefixes
pub async fn prefix_list_handler(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Processing prefix list request");

    let Ok(engine) = state.engine.lock() else {
        return lock_poisoned_response().into_response();
    };
    let cached_prefixes = engine.list_cached_prefixes();

    // Convert to PrefixInfo format
    let prefixes: Vec<PrefixInfo> = cached_prefixes
        .iter()
        .map(|prefix| {
            let preview = if prefix.len() > 100 {
                format!("{}...", &prefix[..100])
            } else {
                prefix.clone()
            };

            PrefixInfo {
                key: format!("{:x}", Sha256::digest(prefix.as_bytes())),
                preview,
                token_count: prefix.len() / 4, // Rough estimate
                access_count: 0,               // > TODO: Get from actual cache
                age_seconds: 0,                // > TODO: Get from actual cache
            }
        })
        .collect();

    let response = PrefixListResponse {
        total_count: prefixes.len(),
        prefixes,
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// Handler for DELETE /v1/embeddings/prefix
///
/// Clears all cached prefixes
pub async fn prefix_clear_handler(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Processing prefix clear request");

    {
        let Ok(engine) = state.engine.lock() else {
            return lock_poisoned_response().into_response();
        };
        engine.clear_prefix_cache();
    }

    let response = CacheClearResponse {
        status: "Prefix cache cleared successfully".to_string(),
        previous_stats: None, // > TODO: Return actual stats
    };

    info!("Prefix cache cleared");
    (StatusCode::OK, Json(response)).into_response()
}

/// Handler for GET /v1/embeddings/prefix/stats
///
/// Returns prefix cache statistics
pub async fn prefix_stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    debug!("Processing prefix stats request");

    let Ok(engine) = state.engine.lock() else {
        return lock_poisoned_response().into_response();
    };
    let is_enabled = engine.is_prefix_cache_enabled();

    if let Some(stats) = engine.get_prefix_cache_stats() {
        #[allow(clippy::cast_precision_loss)]
        let hit_rate = if stats.total_hits + stats.total_misses > 0 {
            stats.total_hits as f64 / (stats.total_hits + stats.total_misses) as f64
        } else {
            0.0
        };

        let response = PrefixStatsResponse {
            enabled: true,
            session_count: stats.session_count,
            total_hits: stats.total_hits,
            total_misses: stats.total_misses,
            total_evictions: stats.total_evictions,
            memory_usage_bytes: stats.memory_usage_bytes,
            hit_rate,
        };

        (StatusCode::OK, Json(response)).into_response()
    } else {
        let response = PrefixStatsResponse {
            enabled: is_enabled,
            session_count: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
            memory_usage_bytes: 0,
            hit_rate: 0.0,
        };

        (StatusCode::OK, Json(response)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_info() {
        let info = get_memory_info();
        assert!(info.total_bytes > 0);
        assert!(info.usage_percentage >= 0.0 && info.usage_percentage <= 100.0);
    }
}
