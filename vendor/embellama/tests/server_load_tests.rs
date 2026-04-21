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

//! Load and performance tests for the Embellama server

#![cfg(feature = "server")]

mod server_test_helpers;

use reqwest::StatusCode;
use serial_test::serial;
use server_test_helpers::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

#[tokio::test]
#[serial]
async fn test_concurrent_requests_10() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    let mut tasks = JoinSet::new();
    let start = Instant::now();

    // Launch 10 concurrent requests
    for i in 0..10 {
        let client = client.clone();
        let base_url = server.base_url.clone();

        tasks.spawn(async move {
            let text = format!("Concurrent request {i}");
            let result = client
                .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                .await;
            (i, result)
        });
    }

    // Collect all results
    let mut success_count = 0;
    let mut failure_count = 0;

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok((_, Ok(response))) if response.status() == StatusCode::OK => {
                success_count += 1;
            }
            _ => {
                failure_count += 1;
            }
        }
    }

    let duration = start.elapsed();

    println!("10 concurrent requests completed in {duration:?}");
    println!("Success: {success_count}, Failures: {failure_count}");

    assert_eq!(success_count, 10, "All requests should succeed");
    assert_eq!(failure_count, 0, "No requests should fail");

    // Should complete reasonably quickly (adjust based on hardware)
    assert!(
        duration < Duration::from_secs(10),
        "Should complete within 10 seconds"
    );
}

#[tokio::test]
#[serial]
async fn test_concurrent_requests_50() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    let mut tasks = JoinSet::new();
    let start = Instant::now();

    // Launch 50 concurrent requests
    for i in 0..50 {
        let client = client.clone();
        let base_url = server.base_url.clone();

        tasks.spawn(async move {
            let text = format!("Stress test request {i}");
            let result = client
                .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                .await;
            (i, result)
        });
    }

    // Collect all results
    let mut success_count = 0;
    let mut failure_count = 0;

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok((_, Ok(response))) if response.status() == StatusCode::OK => {
                success_count += 1;
            }
            _ => {
                failure_count += 1;
            }
        }
    }

    let duration = start.elapsed();

    println!("50 concurrent requests completed in {duration:?}");
    println!("Success: {success_count}, Failures: {failure_count}");

    // With 4 workers, all 50 should eventually succeed
    assert!(success_count >= 45, "At least 45 requests should succeed");
    assert!(failure_count <= 5, "At most 5 requests may fail under load");
}

#[tokio::test]
#[serial]
#[ignore = "This test is resource-intensive"]
async fn test_concurrent_requests_100() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 8)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    let mut tasks = JoinSet::new();
    let start = Instant::now();

    // Launch 100 concurrent requests
    for i in 0..100 {
        let client = client.clone();
        let base_url = server.base_url.clone();

        tasks.spawn(async move {
            let text = format!("Heavy load test request {i}");
            let start = Instant::now();
            let result = client
                .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                .await;
            let latency = start.elapsed();
            (i, result, latency)
        });
    }

    // Collect all results
    let mut success_count = 0;
    let mut failure_count = 0;
    let mut latencies = Vec::new();

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok((_, Ok(response), latency)) if response.status() == StatusCode::OK => {
                success_count += 1;
                #[allow(clippy::cast_possible_truncation)]
                latencies.push(latency.as_millis() as u64);
            }
            _ => {
                failure_count += 1;
            }
        }
    }

    let duration = start.elapsed();

    // Calculate statistics
    latencies.sort_unstable();
    let p50 = latencies.get(latencies.len() / 2).copied().unwrap_or(0);
    let p95 = latencies
        .get(latencies.len() * 95 / 100)
        .copied()
        .unwrap_or(0);
    let p99 = latencies
        .get(latencies.len() * 99 / 100)
        .copied()
        .unwrap_or(0);

    println!("100 concurrent requests completed in {duration:?}");
    println!("Success: {success_count}, Failures: {failure_count}");
    println!("Latencies - P50: {p50}ms, P95: {p95}ms, P99: {p99}ms");

    // With proper queue management, most should succeed
    assert!(success_count >= 80, "At least 80 requests should succeed");
}

#[tokio::test]
#[serial]
async fn test_throughput_single_requests() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let num_requests = 20;
    let start = Instant::now();

    for i in 0..num_requests {
        let text = format!("Throughput test {i}");
        let response = client
            .embedding_request(
                &server.base_url,
                "test-model",
                EmbeddingInput::Single(text),
                None,
            )
            .await
            .expect("Request failed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    let duration = start.elapsed();
    let requests_per_second = f64::from(num_requests) / duration.as_secs_f64();

    println!("Sequential throughput: {requests_per_second:.2} requests/second");

    // Should handle at least 5 requests per second sequentially
    assert!(requests_per_second > 5.0, "Throughput too low");
}

#[tokio::test]
#[serial]
async fn test_throughput_batch_requests() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let num_batches = 10;
    let batch_size = 10;
    let start = Instant::now();
    let mut total_embeddings = 0;

    for i in 0..num_batches {
        let texts: Vec<String> = (0..batch_size)
            .map(|j| format!("Batch {i} text {j}"))
            .collect();

        let response = client
            .embedding_request(
                &server.base_url,
                "test-model",
                EmbeddingInput::Batch(texts),
                None,
            )
            .await
            .expect("Request failed");

        assert_eq!(response.status(), StatusCode::OK);
        total_embeddings += batch_size;
    }

    let duration = start.elapsed();
    let embeddings_per_second = f64::from(total_embeddings) / duration.as_secs_f64();

    println!("Batch throughput: {embeddings_per_second:.2} embeddings/second");
    println!("({total_embeddings} embeddings in {duration:?})");

    // Should process many embeddings per second in batches
    assert!(embeddings_per_second > 20.0, "Batch throughput too low");
}

#[tokio::test]
#[serial]
async fn test_latency_percentiles() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let stats = measure_latencies(&client, &server.base_url, 50).await;

    println!("\nLatency Statistics (50 requests):");
    println!("  P50: {}ms", stats.p50);
    println!("  P95: {}ms", stats.p95);
    println!("  P99: {}ms", stats.p99);
    println!("  Min: {}ms", stats.min);
    println!("  Max: {}ms", stats.max);
    println!("  Avg: {}ms", stats.avg);

    // Reasonable latency expectations (adjust based on hardware)
    assert!(stats.p50 < 500, "P50 latency should be under 500ms");
    assert!(stats.p95 < 1000, "P95 latency should be under 1s");
    assert!(stats.p99 < 2000, "P99 latency should be under 2s");
}

#[tokio::test]
#[serial]
#[ignore = "This test runs for a longer duration"]
async fn test_sustained_load() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    let duration = Duration::from_secs(30); // Run for 30 seconds
    let start = Instant::now();
    let mut tasks = JoinSet::new();
    let mut request_count = 0;

    println!("Running sustained load test for {duration:?}...");

    while start.elapsed() < duration {
        let client = client.clone();
        let base_url = server.base_url.clone();
        let req_id = request_count;

        tasks.spawn(async move {
            let text = format!("Sustained load request {req_id}");
            client
                .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                .await
        });

        request_count += 1;

        // Pace requests to maintain steady load
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Don't let too many accumulate
        while tasks.len() > 20 {
            let _ = tasks.join_next().await;
        }
    }

    // Collect remaining results
    let mut success_count = 0;
    let mut failure_count = 0;

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(response)) if response.status() == StatusCode::OK => {
                success_count += 1;
            }
            _ => {
                failure_count += 1;
            }
        }
    }

    let total_duration = start.elapsed();
    let success_rate = f64::from(success_count) / f64::from(request_count) * 100.0;

    println!("\nSustained load test results:");
    println!("  Duration: {total_duration:?}");
    println!("  Total requests: {request_count}");
    println!("  Successful: {success_count}");
    println!("  Failed: {failure_count}");
    println!("  Success rate: {success_rate:.2}%");

    // Should maintain high success rate under sustained load
    assert!(success_rate > 95.0, "Success rate should be above 95%");
}

#[tokio::test]
#[serial]
async fn test_mixed_workload() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    let mut tasks = JoinSet::new();
    let start = Instant::now();

    // Mix of single and batch requests
    for i in 0..30 {
        let client = client.clone();
        let base_url = server.base_url.clone();

        tasks.spawn(async move {
            if i % 3 == 0 {
                // Batch request
                let texts: Vec<String> = (0..5)
                    .map(|j| format!("Mixed batch {i} item {j}"))
                    .collect();
                client
                    .embedding_request(&base_url, "test-model", EmbeddingInput::Batch(texts), None)
                    .await
            } else {
                // Single request
                let text = format!("Mixed single request {i}");
                client
                    .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                    .await
            }
        });
    }

    // Collect results
    let mut success_count = 0;
    let mut failure_count = 0;

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(response)) if response.status() == StatusCode::OK => {
                success_count += 1;
            }
            _ => {
                failure_count += 1;
            }
        }
    }

    let duration = start.elapsed();

    println!("Mixed workload (30 requests) completed in {duration:?}");
    println!("Success: {success_count}, Failures: {failure_count}");

    assert_eq!(success_count, 30, "All mixed requests should succeed");
    assert_eq!(failure_count, 0, "No mixed requests should fail");
}

#[tokio::test]
#[serial]
async fn test_queue_saturation() {
    let model_path = get_test_model_path().expect("Test model not found");
    // Start with fewer workers to make queue saturation more likely
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = Arc::new(TestClient::new());

    // Send many requests quickly to saturate queues
    let mut tasks = JoinSet::new();

    for i in 0..200 {
        let client = client.clone();
        let base_url = server.base_url.clone();

        tasks.spawn(async move {
            let text = format!("Queue saturation test {i}");
            let start = Instant::now();
            let result = client
                .embedding_request(&base_url, "test-model", EmbeddingInput::Single(text), None)
                .await;
            let latency = start.elapsed();
            (result, latency)
        });

        // Don't wait between requests to create pressure
    }

    // Collect results
    let mut success_count = 0;
    let mut timeout_count = 0;
    let mut service_unavailable_count = 0;
    let mut max_latency = Duration::ZERO;

    while let Some(result) = tasks.join_next().await {
        if let Ok((Ok(response), latency)) = result {
            match response.status() {
                StatusCode::OK => {
                    success_count += 1;
                    if latency > max_latency {
                        max_latency = latency;
                    }
                }
                StatusCode::REQUEST_TIMEOUT => timeout_count += 1,
                StatusCode::SERVICE_UNAVAILABLE => service_unavailable_count += 1,
                _ => {}
            }
        }
    }

    println!("\nQueue saturation test results:");
    println!("  Successful: {success_count}");
    println!("  Timeouts: {timeout_count}");
    println!("  Service unavailable: {service_unavailable_count}");
    println!("  Max latency: {max_latency:?}");

    // Should handle overload gracefully
    assert!(success_count > 0, "Some requests should succeed");
    assert!(
        success_count + timeout_count + service_unavailable_count >= 180,
        "Most requests should be handled (success or proper error)"
    );
}

#[tokio::test]
#[serial]
async fn test_memory_stability() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Warm up
    for _ in 0..10 {
        let _ = client
            .embedding_request(
                &server.base_url,
                "test-model",
                EmbeddingInput::Single("Warmup".to_string()),
                None,
            )
            .await;
    }

    // Run many requests and check for consistent performance
    let mut batch_times = Vec::new();

    for batch in 0..5 {
        let batch_start = Instant::now();

        for i in 0..100 {
            let text = format!("Memory test batch {batch} request {i}");
            let response = client
                .embedding_request(
                    &server.base_url,
                    "test-model",
                    EmbeddingInput::Single(text),
                    None,
                )
                .await
                .expect("Request failed");

            assert_eq!(response.status(), StatusCode::OK);
        }

        let batch_time = batch_start.elapsed();
        batch_times.push(batch_time);

        println!("Batch {batch} completed in {batch_time:?}");
    }

    // Check that performance doesn't degrade significantly
    #[allow(clippy::cast_precision_loss)]
    let first_batch = batch_times[0].as_millis() as f64;
    #[allow(clippy::cast_precision_loss)]
    let last_batch = batch_times[4].as_millis() as f64;
    let degradation = (last_batch - first_batch) / first_batch * 100.0;

    println!("\nPerformance degradation: {degradation:.2}%");

    // Performance shouldn't degrade by more than 35%
    assert!(
        degradation < 35.0,
        "Performance degradation too high: {degradation:.2}%"
    );
}
