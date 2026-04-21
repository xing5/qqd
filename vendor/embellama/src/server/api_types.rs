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

//! OpenAI-compatible API types for the embeddings endpoint
//!
//! This module defines the request and response structures that match
//! the `OpenAI` API format for maximum compatibility.

use crate::cache::CacheStats;
use serde::{Deserialize, Serialize};

/// Input type for embeddings - can be a single string or array of strings
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputType {
    /// Single text input
    Single(String),
    /// Batch of text inputs
    Batch(Vec<String>),
}

impl InputType {
    /// Convert to the internal `TextInput` format
    pub fn into_text_input(self) -> crate::server::channel::TextInput {
        match self {
            Self::Single(text) => crate::server::channel::TextInput::Single(text),
            Self::Batch(texts) => crate::server::channel::TextInput::Batch(texts),
        }
    }
}

/// OpenAI-compatible embeddings request
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model identifier to use for embeddings
    pub model: String,
    /// Input text(s) to generate embeddings for
    pub input: InputType,
    /// Encoding format for the embeddings ("float" or "base64")
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    /// Optional dimensions for the embedding (for dimension reduction)
    pub dimensions: Option<usize>,
    /// Optional user identifier for tracking
    pub user: Option<String>,
    /// Optional truncation strategy (overrides model default)
    pub truncate: Option<crate::config::TruncateTokens>,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

/// OpenAI-compatible embeddings response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    /// Object type (always "list")
    pub object: String,
    /// Array of embedding data
    pub data: Vec<EmbeddingData>,
    /// Model used for generation
    pub model: String,
    /// Token usage statistics
    pub usage: Usage,
}

impl EmbeddingsResponse {
    /// Create a new embeddings response
    pub fn new(model: String, embeddings: Vec<Vec<f32>>, token_count: usize) -> Self {
        let data = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                index,
                object: "embedding".to_string(),
                embedding: EmbeddingValue::Float(embedding),
            })
            .collect();

        Self {
            object: "list".to_string(),
            data,
            model,
            usage: Usage {
                prompt_tokens: token_count,
                total_tokens: token_count,
            },
        }
    }

    /// Create a response with base64-encoded embeddings
    pub fn new_base64(model: String, embeddings: Vec<Vec<f32>>, token_count: usize) -> Self {
        let data = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| {
                let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                let base64 = STANDARD.encode(&bytes);

                EmbeddingData {
                    index,
                    object: "embedding".to_string(),
                    embedding: EmbeddingValue::Base64(base64),
                }
            })
            .collect();

        Self {
            object: "list".to_string(),
            data,
            model,
            usage: Usage {
                prompt_tokens: token_count,
                total_tokens: token_count,
            },
        }
    }
}

/// Individual embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Index of this embedding in the batch
    pub index: usize,
    /// Object type (always "embedding")
    pub object: String,
    /// The embedding vector
    pub embedding: EmbeddingValue,
}

/// Embedding value - either float array or base64 string
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    /// Float array representation
    Float(Vec<f32>),
    /// Base64-encoded representation
    Base64(String),
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Total tokens processed
    pub total_tokens: usize,
}

/// Error response format matching `OpenAI` API
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

/// Detailed error information
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    /// Error type (e.g., "`invalid_request_error`")
    #[serde(rename = "type")]
    pub error_type: String,
    /// Optional error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Create an invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "invalid_request_error".to_string(),
                code: None,
            },
        }
    }

    /// Create a model not found error
    pub fn model_not_found(model: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: format!("Model '{model}' not found"),
                error_type: "model_not_found_error".to_string(),
                code: Some("model_not_found".to_string()),
            },
        }
    }

    /// Create a rate limit error
    pub fn rate_limit() -> Self {
        Self {
            error: ErrorDetail {
                message: "Rate limit exceeded. Please try again later.".to_string(),
                error_type: "rate_limit_error".to_string(),
                code: Some("rate_limit_exceeded".to_string()),
            },
        }
    }

    /// Create an internal server error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: "internal_error".to_string(),
                code: Some("internal_server_error".to_string()),
            },
        }
    }
}

/// List models response
#[derive(Debug, Clone, Serialize)]
pub struct ListModelsResponse {
    /// Object type (always "list")
    pub object: String,
    /// Array of available models
    pub data: Vec<ModelData>,
}

/// Individual model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelData {
    /// Model identifier
    pub id: String,
    /// Object type (always "model")
    pub object: String,
    /// Unix timestamp of when the model was created
    pub created: i64,
    /// Owner of the model
    pub owned_by: String,
    /// Context size (max tokens) supported by the model
    pub context_size: Option<u32>,
}

impl ModelData {
    /// Create new model data
    pub fn new(id: String) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: 1_700_000_000, // Fixed timestamp for consistency
            owned_by: "embellama".to_string(),
            context_size: None,
        }
    }

    /// Create new model data with context size
    pub fn new_with_context(id: String, context_size: Option<u32>) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: 1_700_000_000, // Fixed timestamp for consistency
            owned_by: "embellama".to_string(),
            context_size,
        }
    }
}

/// Reranking request (inspired by Cohere Rerank API)
#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    /// Model identifier to use for reranking
    pub model: String,
    /// The query to rerank documents against
    pub query: String,
    /// List of documents to rerank
    pub documents: Vec<String>,
    /// Return only the top N results (None = return all)
    pub top_n: Option<usize>,
    /// Whether to apply sigmoid normalization to scores (default: true)
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}

fn default_normalize() -> bool {
    true
}

/// Reranking response
#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    /// Object type
    pub object: String,
    /// Reranking results sorted by relevance (descending)
    pub results: Vec<RerankResultData>,
    /// Model used
    pub model: String,
    /// Usage statistics
    pub usage: RerankUsage,
}

/// A single reranking result
#[derive(Debug, Clone, Serialize)]
pub struct RerankResultData {
    /// Original index of the document in the input list
    pub index: usize,
    /// Relevance score (higher = more relevant)
    pub relevance_score: f32,
}

/// Token usage for reranking
#[derive(Debug, Clone, Serialize)]
pub struct RerankUsage {
    /// Total documents processed
    pub total_documents: usize,
}

/// Add base64 encoding support
use base64::{Engine as _, engine::general_purpose::STANDARD};

/// Request to warm the cache with specific texts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarmRequest {
    /// Texts to pre-compute embeddings for
    pub texts: Vec<String>,
    /// Optional model name to use (defaults to engine's default model)
    pub model: Option<String>,
}

/// Response from cache warm operation
#[derive(Debug, Clone, Serialize)]
pub struct CacheWarmResponse {
    /// Status message
    pub status: String,
    /// Number of texts processed
    pub texts_processed: usize,
    /// Number of texts that were already cached
    pub already_cached: usize,
}

/// Cache statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsResponse {
    /// Cache enabled status
    pub enabled: bool,
    /// Cache statistics if available
    pub stats: Option<CacheStats>,
    /// System memory information
    pub memory: MemoryInfo,
}

/// System memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total system memory in bytes
    pub total_bytes: u64,
    /// Available system memory in bytes
    pub available_bytes: u64,
    /// Used memory percentage
    pub usage_percentage: f32,
}

/// Cache clear response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheClearResponse {
    /// Status message
    pub status: String,
    /// Previous cache statistics before clearing
    pub previous_stats: Option<CacheStats>,
}

// Prefix cache API types

/// Request to register a prefix for caching
#[derive(Debug, Clone, Deserialize)]
pub struct PrefixRegisterRequest {
    /// The prefix text to cache
    pub prefix: String,
    /// Optional model name (uses default if not specified)
    pub model: Option<String>,
}

/// Response for prefix registration
#[derive(Debug, Clone, Serialize)]
pub struct PrefixRegisterResponse {
    /// Status message
    pub status: String,
    /// Number of tokens in the prefix
    pub token_count: usize,
    /// Estimated memory usage in bytes
    pub memory_usage: u64,
}

/// Response listing cached prefixes
#[derive(Debug, Clone, Serialize)]
pub struct PrefixListResponse {
    /// List of cached prefix information
    pub prefixes: Vec<PrefixInfo>,
    /// Total number of cached prefixes
    pub total_count: usize,
}

/// Information about a cached prefix
#[derive(Debug, Clone, Serialize)]
pub struct PrefixInfo {
    /// Prefix key/identifier
    pub key: String,
    /// First 100 characters of the prefix text
    pub preview: String,
    /// Number of tokens in the prefix
    pub token_count: usize,
    /// Number of times accessed
    pub access_count: usize,
    /// Age in seconds
    pub age_seconds: u64,
}

/// Response with prefix cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct PrefixStatsResponse {
    /// Whether prefix cache is enabled
    pub enabled: bool,
    /// Number of cached sessions
    pub session_count: usize,
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Total evictions
    pub total_evictions: u64,
    /// Estimated memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Cache hit rate
    pub hit_rate: f64,
}
