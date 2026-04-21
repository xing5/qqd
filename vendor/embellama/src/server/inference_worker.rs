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

//! Inference worker thread implementation
//!
//! This module implements a dedicated inference thread that owns the model
//! and batches requests for efficient processing. This ensures only one
//! model copy is loaded in memory while still supporting concurrent requests.

use crate::EmbeddingEngine;
use crate::server::channel::{TextInput, WorkerRequest, WorkerResponse};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Configuration for the inference worker
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum batch size (should match model's `n_seq_max`)
    pub max_batch_size: usize,
    /// Maximum time to wait for batch to fill (milliseconds)
    pub batch_timeout_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout_ms: 10,
        }
    }
}

/// Inference worker that processes embedding requests in batches
pub struct InferenceWorker {
    /// Shared embedding engine instance
    engine: Arc<Mutex<EmbeddingEngine>>,
    /// Request receiver channel
    receiver: mpsc::Receiver<WorkerRequest>,
    /// Configuration
    config: InferenceConfig,
}

impl InferenceWorker {
    /// Create a new inference worker
    ///
    /// # Arguments
    /// * `engine` - Shared embedding engine instance
    /// * `receiver` - Channel to receive requests
    /// * `config` - Inference configuration
    pub fn new(
        engine: Arc<Mutex<EmbeddingEngine>>,
        receiver: mpsc::Receiver<WorkerRequest>,
        config: InferenceConfig,
    ) -> Self {
        Self {
            engine,
            receiver,
            config,
        }
    }

    /// Run the inference worker main loop
    ///
    /// This method batches incoming requests and processes them efficiently
    /// using the model's batch processing capabilities.
    pub fn run(mut self) {
        info!("Inference worker starting");

        // Process requests in batches
        loop {
            // Collect a batch of requests
            let batch = self.collect_batch();

            if batch.is_empty() {
                // Channel closed and no more requests
                break;
            }

            debug!("Processing batch of {} requests", batch.len());
            let start = Instant::now();

            // Process the batch
            self.process_batch(batch);

            let elapsed = start.elapsed();
            debug!("Batch processed in {:?}", elapsed);
        }

        info!("Inference worker shutting down");
    }

    /// Collect a batch of requests up to `max_batch_size` with timeout
    fn collect_batch(&mut self) -> Vec<WorkerRequest> {
        let mut batch = Vec::new();
        let batch_start = Instant::now();
        let timeout = Duration::from_millis(self.config.batch_timeout_ms);

        // Get the first request (blocking)
        match self.receiver.blocking_recv() {
            Some(request) => batch.push(request),
            None => return batch, // Channel closed
        }

        // Try to collect more requests up to max_batch_size or timeout
        while batch.len() < self.config.max_batch_size {
            let elapsed = batch_start.elapsed();
            if elapsed >= timeout {
                break;
            }

            // Try non-blocking receive since we already have one request
            if let Ok(request) = self.receiver.try_recv() {
                batch.push(request);
            } else {
                // No more immediately available requests
                // Sleep briefly and check time
                std::thread::sleep(Duration::from_micros(100));
                if batch_start.elapsed() >= timeout {
                    break;
                }
            }
        }

        batch
    }

    /// Process a batch of requests
    fn process_batch(&self, batch: Vec<WorkerRequest>) {
        // Group requests by model (in case we support multiple models in future)
        let model_name = &batch[0].model;

        // Flatten all inputs into a single batch
        let mut all_texts = Vec::new();
        let mut request_metadata = Vec::new();

        for request in &batch {
            let start_idx = all_texts.len();

            match &request.input {
                TextInput::Single(text) => {
                    all_texts.push(text.as_str());
                    request_metadata.push((start_idx, 1));
                }
                TextInput::Batch(texts) => {
                    let count = texts.len();
                    for text in texts {
                        all_texts.push(text.as_str());
                    }
                    request_metadata.push((start_idx, count));
                }
            }
        }

        debug!(
            "Processing {} texts across {} requests",
            all_texts.len(),
            batch.len()
        );

        // Process all texts in a single batch
        let result = {
            let Ok(engine) = self.engine.lock() else {
                error!("Engine lock poisoned during batch processing");
                // Drop all response channels without sending — receivers get RecvError,
                // which is handled as an error by the callers
                for request in batch {
                    drop(request.response_tx);
                }
                return;
            };
            engine.embed_batch(Some(model_name), &all_texts)
        };

        // Send responses back to each requester
        match result {
            Ok(all_embeddings) => {
                for (request, (start_idx, count)) in batch.into_iter().zip(request_metadata.iter())
                {
                    // Extract embeddings for this request
                    let request_embeddings =
                        all_embeddings[*start_idx..*start_idx + *count].to_vec();

                    let response = WorkerResponse {
                        embeddings: request_embeddings.clone(),
                        token_count: request_embeddings.iter().map(Vec::len).sum::<usize>() / 10, // Rough estimate
                        processing_time_ms: 0, // Will be set by caller based on end-to-end time
                    };

                    if request.response_tx.send(response).is_err() {
                        warn!(
                            "Failed to send response for request {} (client may have timed out)",
                            request.id
                        );
                    }
                }
            }
            Err(e) => {
                error!("Batch processing failed: {}", e);

                // Send error responses to all requesters
                for request in batch {
                    let response = WorkerResponse {
                        embeddings: vec![],
                        token_count: 0,
                        processing_time_ms: 0,
                    };

                    let _ = request.response_tx.send(response);
                }
            }
        }
    }

    /// Spawn an inference worker in a new thread
    ///
    /// # Arguments
    /// * `engine` - Shared embedding engine instance
    /// * `receiver` - Channel to receive requests
    /// * `config` - Inference configuration
    ///
    /// # Returns
    /// Handle to the spawned thread
    pub fn spawn(
        engine: Arc<Mutex<EmbeddingEngine>>,
        receiver: mpsc::Receiver<WorkerRequest>,
        config: InferenceConfig,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let worker = InferenceWorker::new(engine, receiver, config);
            worker.run();
        })
    }
}
