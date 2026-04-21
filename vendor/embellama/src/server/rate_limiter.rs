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

//! Rate limiting for API endpoints
//!
//! This module provides token bucket rate limiting to prevent abuse
//! and ensure fair resource usage.

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use governor::{
    Quota, RateLimiter as GovernorRateLimiter,
    clock::{Clock, DefaultClock},
    state::{InMemoryState, NotKeyed},
};
use serde_json::json;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use crate::server::metrics;

/// Rate limiter using token bucket algorithm
pub type RateLimiter = Arc<GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Rate limiting configuration
#[derive(Clone, Debug)]
pub struct RateLimitConfig {
    /// Maximum requests per second
    pub requests_per_second: u32,
    /// Burst size (max tokens in bucket)
    pub burst_size: u32,
    /// Whether rate limiting is enabled
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 200,
            enabled: false,
        }
    }
}

impl RateLimitConfig {
    /// Create a new rate limiter from this config
    ///
    /// # Panics
    ///
    /// Panics if `requests_per_second` or `burst_size` is 0.
    pub fn build_limiter(&self) -> Option<RateLimiter> {
        if !self.enabled {
            return None;
        }

        let quota = Quota::per_second(
            NonZeroU32::new(self.requests_per_second).expect("Invalid requests_per_second"),
        )
        .allow_burst(NonZeroU32::new(self.burst_size).expect("Invalid burst_size"));

        Some(Arc::new(GovernorRateLimiter::new(
            quota,
            InMemoryState::default(),
            DefaultClock::default(),
        )))
    }
}

/// Rate limiting middleware
///
/// # Panics
///
/// Panics if the retry-after seconds value cannot be converted to a `HeaderValue`.
/// This should never happen in practice as the value is always a valid u64.
pub async fn rate_limit_middleware(
    State(limiter): State<Option<RateLimiter>>,
    request: Request,
    next: Next,
) -> Response {
    // If rate limiting is disabled, pass through
    let Some(limiter) = limiter else {
        return next.run(request).await;
    };

    // Try to acquire a token
    match limiter.check() {
        Ok(()) => {
            // Token acquired, proceed with request
            next.run(request).await
        }
        Err(not_until) => {
            // Rate limit exceeded
            let retry_after = not_until.wait_time_from(DefaultClock::default().now());
            let retry_after_secs = retry_after.as_secs();

            // Record metrics
            metrics::record_rate_limited("global", "token_exhausted");

            tracing::warn!(
                "Rate limit exceeded, retry after {} seconds",
                retry_after_secs
            );

            // Return 429 with Retry-After header
            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": {
                        "message": "Rate limit exceeded. Please retry later.",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit",
                        "retry_after_seconds": retry_after_secs
                    }
                })),
            )
                .into_response();

            response.headers_mut().insert(
                "Retry-After",
                HeaderValue::from_str(&retry_after_secs.to_string()).unwrap(),
            );

            response
        }
    }
}

/// Per-client rate limiter using IP addresses
pub struct ClientRateLimiter {
    /// Map of client IP to (rate limiter, last access time)
    limiters: dashmap::DashMap<String, (RateLimiter, std::time::Instant)>,
    /// Configuration for creating new limiters
    config: RateLimitConfig,
}

impl ClientRateLimiter {
    /// Create a new per-client rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            limiters: dashmap::DashMap::new(),
            config,
        }
    }

    /// Get or create a rate limiter for a client
    ///
    /// # Panics
    ///
    /// Panics if rate limiting is not enabled in the configuration.
    pub fn get_limiter(&self, client_id: &str) -> RateLimiter {
        let now = std::time::Instant::now();
        self.limiters
            .entry(client_id.to_string())
            .and_modify(|(_, last_access)| *last_access = now)
            .or_insert_with(|| {
                (
                    self.config
                        .build_limiter()
                        .expect("Rate limiting should be enabled"),
                    now,
                )
            })
            .0
            .clone()
    }

    /// Clean up old entries (call periodically)
    pub fn cleanup(&self, max_age: Duration) {
        let now = std::time::Instant::now();
        let before_size = self.limiters.len();

        // Remove entries older than max_age
        self.limiters
            .retain(|_key, (_limiter, last_access)| now.duration_since(*last_access) < max_age);

        let after_size = self.limiters.len();
        if before_size > after_size {
            tracing::debug!(
                "Cleaned {} expired entries from client rate limiter cache ({} remaining)",
                before_size - after_size,
                after_size
            );
        }

        // Warn if cache is still very large
        if after_size > 10000 {
            tracing::warn!(
                "Client rate limiter cache has {} entries after cleanup, consider decreasing max_age",
                after_size
            );
        }
    }
}

/// Extract client identifier from request (IP address)
pub fn extract_client_id(headers: &HeaderMap) -> String {
    // Try X-Forwarded-For first (for proxied requests)
    headers
        .get("X-Forwarded-For")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .or_else(|| {
            // Fall back to X-Real-IP
            headers
                .get("X-Real-IP")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string())
}
