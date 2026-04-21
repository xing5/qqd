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

//! Worker thread implementation for model inference
//!
//! This module implements worker threads that use the shared `EmbeddingEngine`.
//! Each thread will have its own thread-local model instance managed by the engine.

use crate::EmbeddingEngine;
use crate::server::channel::{TextInput, WorkerRequest, WorkerResponse};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Worker thread that processes embedding requests
pub struct Worker {
    /// Worker identifier
    id: usize,
    /// Shared embedding engine instance
    engine: Arc<Mutex<EmbeddingEngine>>,
    /// Request receiver channel
    receiver: mpsc::Receiver<WorkerRequest>,
}

impl Worker {
    /// Create a new worker
    ///
    /// # Arguments
    /// * `id` - Worker identifier
    /// * `engine` - Shared embedding engine instance
    /// * `receiver` - Channel to receive requests
    ///
    /// # Returns
    /// A new `Worker` instance
    pub fn new(
        id: usize,
        engine: Arc<Mutex<EmbeddingEngine>>,
        receiver: mpsc::Receiver<WorkerRequest>,
    ) -> Self {
        Self {
            id,
            engine,
            receiver,
        }
    }

    /// Run the worker main loop
    ///
    /// This method runs in a dedicated thread and processes requests
    /// until the channel is closed. The first request will trigger
    /// model loading in this thread via the engine's thread-local storage.
    pub fn run(mut self) {
        info!("Worker {} starting", self.id);

        // Use blocking recv since we're in a dedicated thread
        while let Some(request) = self.receiver.blocking_recv() {
            debug!("Worker {} processing request {:?}", self.id, request.id);
            let start = Instant::now();

            // Process the request using the shared engine
            let result = {
                let Ok(engine) = self.engine.lock() else {
                    error!("Worker {} engine lock poisoned", self.id);
                    // Drop response_tx without sending — receiver gets RecvError,
                    // which is handled as an error by the caller
                    drop(request.response_tx);
                    continue;
                };
                match &request.input {
                    TextInput::Single(text) => {
                        // Generate single embedding
                        engine
                            .embed(Some(&request.model), text)
                            .map(|embedding| vec![embedding])
                    }
                    TextInput::Batch(texts) => {
                        // Generate batch embeddings
                        let text_refs: Vec<&str> =
                            texts.iter().map(std::string::String::as_str).collect();
                        engine.embed_batch(Some(&request.model), &text_refs)
                    }
                }
            };

            // Create response based on result
            let response = match result {
                Ok(embeddings) => {
                    let token_count = embeddings.iter().map(std::vec::Vec::len).sum::<usize>() / 10; // Rough estimate

                    WorkerResponse {
                        embeddings,
                        token_count,
                        processing_time_ms: u64::try_from(start.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                    }
                }
                Err(e) => {
                    error!("Worker {} failed to generate embeddings: {}", self.id, e);
                    // Send empty response on error
                    // > TODO: Add error field to WorkerResponse for better error handling
                    WorkerResponse {
                        embeddings: vec![],
                        token_count: 0,
                        processing_time_ms: u64::try_from(start.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                    }
                }
            };

            // Send response back
            if request.response_tx.send(response).is_err() {
                warn!(
                    "Worker {} failed to send response for request {:?} (client may have timed out)",
                    self.id, request.id
                );
            }
        }

        info!("Worker {} shutting down", self.id);
    }

    /// Spawn a worker in a new thread
    ///
    /// # Arguments
    /// * `id` - Worker identifier
    /// * `engine` - Shared embedding engine instance
    /// * `receiver` - Channel to receive requests
    ///
    /// # Returns
    /// Handle to the spawned thread
    pub fn spawn(
        id: usize,
        engine: Arc<Mutex<EmbeddingEngine>>,
        receiver: mpsc::Receiver<WorkerRequest>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let worker = Worker::new(id, engine, receiver);
            worker.run();
        })
    }
}
