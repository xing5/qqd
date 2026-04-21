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

//! Channel message types for worker communication
//!
//! This module defines the message types used for communication between
//! the HTTP handlers and worker threads via channels.

use tokio::sync::oneshot;
use uuid::Uuid;

/// Request sent to worker threads
#[derive(Debug)]
pub struct WorkerRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Model name to use
    pub model: String,
    /// Text input to process
    pub input: TextInput,
    /// Channel to send response back
    pub response_tx: oneshot::Sender<WorkerResponse>,
}

/// Response from worker threads
#[derive(Debug)]
pub struct WorkerResponse {
    /// Generated embeddings
    pub embeddings: Vec<Vec<f32>>,
    /// Total token count processed
    pub token_count: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Input text format
#[derive(Debug, Clone)]
pub enum TextInput {
    /// Single text to process
    Single(String),
    /// Batch of texts to process
    Batch(Vec<String>),
}

impl TextInput {
    /// Get the number of texts in the input
    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Batch(texts) => texts.len(),
        }
    }

    /// Check if the input is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(text) => text.is_empty(),
            Self::Batch(texts) => texts.is_empty(),
        }
    }
}
