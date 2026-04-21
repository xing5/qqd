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

//! Request preprocessing middleware for the server
//!
//! This module provides middleware for request size limiting, authentication,
//! and request ID injection.

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use subtle::ConstantTimeEq;
use uuid::Uuid;

/// Maximum request body size (default: 10MB)
pub const MAX_REQUEST_SIZE: usize = 10 * 1024 * 1024;

/// Request ID header name
pub const REQUEST_ID_HEADER: &str = "X-Request-Id";

/// API key header name
pub const API_KEY_HEADER: &str = "X-API-Key";

/// Middleware to inject request IDs
///
/// # Panics
///
/// Panics if the request ID string cannot be converted to a `HeaderValue`.
/// This should never happen in practice as UUID strings are always valid header values.
pub async fn inject_request_id(mut request: Request, next: Next) -> Response {
    // Check if request already has an ID
    let request_id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok())
        .unwrap_or_else(Uuid::new_v4);

    // Inject or update the request ID header
    request.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request_id.to_string()).unwrap(),
    );

    // Add request ID to span
    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %request.method(),
        uri = %request.uri(),
    );

    // Process request with span
    let _guard = span.enter();
    let mut response = next.run(request).await;

    // Add request ID to response headers
    response.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request_id.to_string()).unwrap(),
    );

    response
}

/// Middleware to enforce request size limits
///
/// # Errors
///
/// Returns `StatusCode::PAYLOAD_TOO_LARGE` if the request body size exceeds `MAX_REQUEST_SIZE`.
pub async fn limit_request_size(request: Request, next: Next) -> Result<Response, StatusCode> {
    // Check Content-Length header
    if let Some(content_length) = request.headers().get(header::CONTENT_LENGTH)
        && let Ok(length_str) = content_length.to_str()
        && let Ok(length) = length_str.parse::<usize>()
        && length > MAX_REQUEST_SIZE
    {
        tracing::warn!("Request size {} exceeds limit {}", length, MAX_REQUEST_SIZE);
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }

    Ok(next.run(request).await)
}

/// Configuration for API key authentication
#[derive(Clone)]
pub struct ApiKeyConfig {
    /// Required API key (None means no authentication required)
    pub api_key: Option<String>,
    /// Whether to allow requests without authentication
    pub allow_anonymous: bool,
}

impl Default for ApiKeyConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            allow_anonymous: true,
        }
    }
}

/// Middleware for API key authentication
pub async fn authenticate_api_key(
    State(config): State<ApiKeyConfig>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    // If no API key is configured, allow all requests
    let Some(required_key) = &config.api_key else {
        return next.run(request).await;
    };

    // Check for API key in headers
    let provided_key = headers.get(API_KEY_HEADER).and_then(|v| v.to_str().ok());

    match provided_key {
        Some(key) if key.as_bytes().ct_eq(required_key.as_bytes()).into() => {
            // Valid API key - using constant-time comparison to prevent timing attacks
            next.run(request).await
        }
        Some(_) => {
            // Invalid API key
            tracing::warn!("Invalid API key provided");
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_api_key",
                        "code": "unauthorized"
                    }
                })),
            )
                .into_response()
        }
        None if config.allow_anonymous => {
            // No key provided but anonymous allowed
            next.run(request).await
        }
        None => {
            // No key provided and anonymous not allowed
            tracing::warn!("Missing API key");
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "API key required",
                        "type": "missing_api_key",
                        "code": "unauthorized"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Extract request ID from headers
pub fn extract_request_id(headers: &HeaderMap) -> Uuid {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok())
        .unwrap_or_else(Uuid::new_v4)
}
