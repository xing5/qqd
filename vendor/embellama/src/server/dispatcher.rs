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

//! Request dispatcher for routing to inference worker
//!
//! This module handles the routing of embedding requests to the dedicated
//! inference worker thread, which batches and processes them efficiently.

use crate::EmbeddingEngine;
use crate::server::channel::WorkerRequest;
use crate::server::inference_worker::{InferenceConfig, InferenceWorker};
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Dispatcher for routing requests to inference worker
pub struct Dispatcher {
    /// Channel to send requests to the inference worker
    inference_tx: mpsc::Sender<WorkerRequest>,
    /// Inference worker thread handle (not cloneable, so wrapped in Option)
    #[allow(dead_code)]
    handle: Option<thread::JoinHandle<()>>,
}

impl Clone for Dispatcher {
    fn clone(&self) -> Self {
        Self {
            inference_tx: self.inference_tx.clone(),
            handle: None, // Don't clone thread handle
        }
    }
}

impl Dispatcher {
    /// Create a new dispatcher with a dedicated inference worker
    ///
    /// # Arguments
    /// * `n_seq_max` - Maximum batch size (should match model's `n_seq_max`)
    /// * `queue_size` - Maximum pending requests in queue
    ///
    /// # Returns
    /// A new `Dispatcher` instance
    ///
    /// # Panics
    ///
    /// Panics if the `EmbeddingEngine` has not been initialized before creating the dispatcher
    pub fn new(n_seq_max: usize, queue_size: usize) -> Self {
        info!(
            "Creating dispatcher with inference worker (n_seq_max={})",
            n_seq_max
        );

        // Get the engine instance (should already be initialized)
        let engine = EmbeddingEngine::instance()
            .expect("EmbeddingEngine should be initialized before creating Dispatcher");

        // Create channel for inference worker
        let (tx, rx) = mpsc::channel::<WorkerRequest>(queue_size);

        // Configure inference worker
        let config = InferenceConfig {
            max_batch_size: n_seq_max,
            batch_timeout_ms: 10, // 10ms timeout for batching
        };

        // Spawn the single inference worker thread
        let handle = InferenceWorker::spawn(Arc::clone(&engine), rx, config);

        info!("Dispatcher created with dedicated inference worker");

        Self {
            inference_tx: tx,
            handle: Some(handle),
        }
    }

    /// Send a request to the inference worker
    ///
    /// # Arguments
    /// * `request` - The worker request to process
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    ///
    /// Returns an error if the request cannot be sent to the inference worker
    pub async fn send(&self, request: WorkerRequest) -> Result<(), String> {
        debug!("Routing request {:?} to inference worker", request.id);

        // Send to inference worker
        self.inference_tx.send(request).await.map_err(|e| {
            warn!("Inference worker channel full or closed: {}", e);
            format!("Failed to send request to inference worker: {e}")
        })
    }

    /// Check if the dispatcher is ready to accept requests
    pub fn is_ready(&self) -> bool {
        // Check if inference worker channel is open
        !self.inference_tx.is_closed()
    }

    /// Get the number of active workers (always 1 for inference worker)
    pub fn worker_count(&self) -> usize {
        1
    }

    /// Shutdown the inference worker gracefully
    ///
    /// This drops the sender channel, causing the inference worker to exit its loop
    pub fn shutdown(self) {
        info!("Shutting down dispatcher and inference worker");

        // Dropping self will drop inference_tx, signaling worker to stop
        drop(self);

        info!("Inference worker signaled to shutdown");
    }
}
