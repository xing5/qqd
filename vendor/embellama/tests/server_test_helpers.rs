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

//! Test helpers and utilities for server integration tests

#![cfg(feature = "server")]

use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Test server handle that ensures cleanup
pub struct TestServer {
    pub base_url: String,
    #[allow(dead_code)]
    pub port: u16,
    process: Option<Child>,
    _temp_dir: Option<TempDir>,
}

impl TestServer {
    /// Spawn a test server with the given configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No available port can be found
    /// - Server binary cannot be built
    /// - Server process cannot be started
    /// - Server fails to become ready within timeout
    ///
    /// # Panics
    ///
    /// Panics if the model path cannot be converted to a string
    pub async fn spawn(model_path: PathBuf, workers: usize) -> Result<Self, String> {
        Self::spawn_with_config(model_path, workers, Some(1)).await
    }

    /// Spawn a test server with the given configuration and optional n_seq_max
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No available port can be found
    /// - Server binary cannot be built
    /// - Server process cannot be started
    /// - Server fails to become ready within timeout
    ///
    /// # Panics
    ///
    /// Panics if the model path cannot be converted to a string
    pub async fn spawn_with_config(
        model_path: PathBuf,
        workers: usize,
        n_seq_max: Option<u32>,
    ) -> Result<Self, String> {
        // Find an available port
        let port = find_available_port()?;
        let base_url = format!("http://127.0.0.1:{port}");

        // Build the server binary
        let output = Command::new("cargo")
            .args(["build", "--features", "server", "--bin", "embellama-server"])
            .output()
            .map_err(|e| format!("Failed to build server: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "Failed to build server: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Build args - create owned strings first to avoid lifetime issues
        let port_str = port.to_string();
        let workers_str = workers.to_string();
        let n_seq_max_str = n_seq_max.map(|v| v.to_string());

        let mut args = vec![
            "run",
            "--features",
            "server",
            "--bin",
            "embellama-server",
            "--",
            "--model-path",
            model_path.to_str().unwrap(),
            "--model-name",
            "test-model",
            "--host",
            "127.0.0.1",
            "--port",
            port_str.as_str(),
            "--workers",
            workers_str.as_str(),
            "--log-level",
            "info",
        ];

        // Add n_seq_max if specified
        if let Some(ref n_seq_max_value) = n_seq_max_str {
            args.push("--n-seq-max");
            args.push(n_seq_max_value.as_str());
        }

        // Start the server process
        let child = Command::new("cargo")
            .args(&args)
            .spawn()
            .map_err(|e| format!("Failed to spawn server: {e}"))?;

        // Wait for server to be ready
        wait_for_server(&base_url, Duration::from_secs(30)).await?;

        Ok(TestServer {
            base_url,
            port,
            process: Some(child),
            _temp_dir: None,
        })
    }

    /// Get the base URL for the test server
    #[allow(dead_code)]
    #[must_use]
    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        // Kill the server process on drop
        if let Some(mut process) = self.process.take() {
            // Spawn a thread to avoid blocking in async context
            std::thread::spawn(move || {
                let _ = process.kill();
                let _ = process.wait();
            });
        }
    }
}

/// Find an available port
fn find_available_port() -> Result<u16, String> {
    TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to bind to port: {e}"))?
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|e| format!("Failed to get local address: {e}"))
}

/// Wait for the server to become ready
async fn wait_for_server(base_url: &str, timeout: Duration) -> Result<(), String> {
    let health_url = format!("{base_url}/health");
    let start = Instant::now();
    let client = reqwest::Client::new();

    loop {
        if start.elapsed() > timeout {
            return Err(format!("Server failed to start within {timeout:?}"));
        }

        // Use async client
        if let Ok(resp) = client.get(&health_url).send().await
            && resp.status().is_success()
        {
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Helper client for making API requests
pub struct TestClient {
    pub client: Client,
}

impl Default for TestClient {
    fn default() -> Self {
        Self::new()
    }
}

impl TestClient {
    /// Creates a new test client
    ///
    /// # Panics
    ///
    /// Panics if the HTTP client cannot be built
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    /// Make an embedding request
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails
    pub async fn embedding_request(
        &self,
        base_url: &str,
        model: &str,
        input: EmbeddingInput,
        encoding_format: Option<&str>,
    ) -> Result<Response, reqwest::Error> {
        let mut body = json!({
            "model": model,
            "input": input,
        });

        if let Some(format) = encoding_format {
            body["encoding_format"] = json!(format);
        }

        self.client
            .post(format!("{base_url}/v1/embeddings"))
            .json(&body)
            .send()
            .await
    }

    /// Get list of models
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails
    #[allow(dead_code)]
    pub async fn list_models(&self, base_url: &str) -> Result<Response, reqwest::Error> {
        self.client
            .get(format!("{base_url}/v1/models"))
            .send()
            .await
    }

    /// Check health endpoint
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails
    #[allow(dead_code)]
    pub async fn health_check(&self, base_url: &str) -> Result<Response, reqwest::Error> {
        self.client.get(format!("{base_url}/health")).send().await
    }
}

/// Input type for embedding requests
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

/// OpenAI-compatible embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub index: usize,
    pub object: String,
    pub embedding: EmbeddingValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Error response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// Generate test texts of various lengths
#[allow(dead_code)]
#[must_use]
pub fn generate_test_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            match i % 4 {
                0 => format!("Short text {i}"),
                1 => format!("This is a medium length text for testing embeddings. Number: {i}"),
                2 => format!("This is a much longer text that contains multiple sentences. It's designed to test how the embedding system handles larger inputs. We want to make sure that the model can process various text lengths efficiently. Test number: {i}"),
                _ => format!("Text with special chars: café, naïve, 日本語, emoji 🚀 #{i}"),
            }
        })
        .collect()
}

/// Validate embedding response structure
///
/// # Panics
///
/// Panics if the response structure is invalid, including:
/// - Wrong object type
/// - Incorrect number of embeddings
/// - Invalid embedding data
/// - Non-finite values in embeddings
/// - Invalid base64 encoding
/// - Invalid usage metrics
#[allow(dead_code)]
pub fn validate_embedding_response(response: &EmbeddingResponse, expected_count: usize) {
    use base64::Engine;

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), expected_count);

    for (i, data) in response.data.iter().enumerate() {
        assert_eq!(data.index, i);
        assert_eq!(data.object, "embedding");

        match &data.embedding {
            EmbeddingValue::Float(vec) => {
                assert!(!vec.is_empty(), "Embedding vector should not be empty");
                // Check all values are finite
                for val in vec {
                    assert!(val.is_finite(), "Embedding contains non-finite value");
                }
            }
            EmbeddingValue::Base64(s) => {
                assert!(!s.is_empty(), "Base64 embedding should not be empty");
                // Verify it's valid base64
                base64::engine::general_purpose::STANDARD
                    .decode(s)
                    .expect("Invalid base64 encoding");
            }
        }
    }

    // Validate usage metrics
    assert!(response.usage.prompt_tokens > 0);
    assert!(response.usage.total_tokens > 0);
}

/// Assert that embeddings are normalized (L2 norm ≈ 1.0)
///
/// # Panics
///
/// Panics if any embedding's L2 norm deviates from 1.0 by more than the specified tolerance
#[allow(dead_code)]
pub fn assert_embeddings_normalized(embeddings: &[Vec<f32>], tolerance: f32) {
    for embedding in embeddings {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < tolerance,
            "Embedding is not normalized. L2 norm: {norm} (expected: ~1.0)"
        );
    }
}

/// Helper to make concurrent requests
///
/// # Panics
///
/// Panics if any spawned task panics
#[allow(dead_code)]
pub async fn make_concurrent_requests(
    client: &TestClient,
    base_url: &str,
    count: usize,
) -> Vec<Result<Response, reqwest::Error>> {
    let mut handles = Vec::new();

    for i in 0..count {
        let client = client.client.clone();
        let url = format!("{base_url}/v1/embeddings");
        let text = format!("Concurrent request {i}");

        handles.push(tokio::spawn(async move {
            client
                .post(&url)
                .json(&json!({
                    "model": "test-model",
                    "input": text
                }))
                .send()
                .await
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    results
}

/// Measure request latency percentiles
///
/// # Panics
///
/// Panics if no latency measurements are collected
#[allow(dead_code)]
pub async fn measure_latencies(
    client: &TestClient,
    base_url: &str,
    requests: usize,
) -> LatencyStats {
    let mut latencies = Vec::new();

    for i in 0..requests {
        let start = Instant::now();
        let _ = client
            .embedding_request(
                base_url,
                "test-model",
                EmbeddingInput::Single(format!("Latency test {i}")),
                None,
            )
            .await;
        #[allow(clippy::cast_possible_truncation)]
        latencies.push(start.elapsed().as_millis() as u64);
    }

    latencies.sort_unstable();

    LatencyStats {
        p50: latencies[latencies.len() / 2],
        p95: latencies[latencies.len() * 95 / 100],
        p99: latencies[latencies.len() * 99 / 100],
        min: *latencies.first().unwrap(),
        max: *latencies.last().unwrap(),
        avg: latencies.iter().sum::<u64>() / u64::try_from(latencies.len()).unwrap_or(1),
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct LatencyStats {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub min: u64,
    pub max: u64,
    pub avg: u64,
}

/// Get path to test model
///
/// # Errors
///
/// Returns an error if the test model cannot be found
pub fn get_test_model_path() -> Result<PathBuf, String> {
    // First check environment variable
    if let Ok(path) = std::env::var("EMBELLAMA_TEST_MODEL") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }

    // Then check default test model location from justfile
    let default_path = PathBuf::from("models/test/all-minilm-l6-v2-q4_k_m.gguf");
    if default_path.exists() {
        return Ok(default_path);
    }

    Err("Test model not found. Run 'just download-test-model' first".to_string())
}
