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

//! Prometheus metrics collection for the server
//!
//! This module provides observability through Prometheus metrics,
//! tracking request counts, latencies, queue depths, and worker utilization.

use prometheus::{
    CounterVec, Encoder, GaugeVec, HistogramVec, TextEncoder, register_counter_vec,
    register_gauge_vec, register_histogram_vec,
};
use std::sync::LazyLock;
use std::time::Duration;

// Request metrics
/// Counter for total number of requests by status and model
pub static REQUEST_COUNTER: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "embellama_requests_total",
        "Total number of requests by status",
        &["status", "model"]
    )
    .expect("Failed to register request counter")
});

/// Histogram for request processing duration in seconds
pub static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "embellama_request_duration_seconds",
        "Request processing time in seconds",
        &["model", "status"],
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]
    )
    .expect("Failed to register request duration histogram")
});

// Model inference metrics
/// Histogram for model inference duration in seconds
pub static INFERENCE_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "embellama_inference_duration_seconds",
        "Model inference time in seconds",
        &["model", "batch_size"],
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0
        ]
    )
    .expect("Failed to register inference duration histogram")
});

// Queue and worker metrics
/// Gauge for current depth of request queues per worker
pub static QUEUE_DEPTH: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "embellama_queue_depth",
        "Current depth of request queues",
        &["worker_id"]
    )
    .expect("Failed to register queue depth gauge")
});

/// Gauge for currently active requests per model
pub static ACTIVE_REQUESTS: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "embellama_active_requests",
        "Number of currently active requests",
        &["model"]
    )
    .expect("Failed to register active requests gauge")
});

/// Gauge for worker thread utilization (0.0 to 1.0)
pub static WORKER_UTILIZATION: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "embellama_worker_utilization",
        "Worker thread utilization (0-1)",
        &["worker_id"]
    )
    .expect("Failed to register worker utilization gauge")
});

// Rate limiting metrics
/// Counter for rate-limited requests by client and reason
pub static RATE_LIMITED_REQUESTS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "embellama_rate_limited_requests_total",
        "Total number of rate-limited requests",
        &["client", "reason"]
    )
    .expect("Failed to register rate limited requests counter")
});

// Error metrics
/// Counter for errors by type and model
pub static ERROR_COUNTER: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "embellama_errors_total",
        "Total number of errors by type",
        &["error_type", "model"]
    )
    .expect("Failed to register error counter")
});

/// Record a successful request
pub fn record_request_success(model: &str, duration: Duration) {
    REQUEST_COUNTER.with_label_values(&["success", model]).inc();
    REQUEST_DURATION
        .with_label_values(&[model, "success"])
        .observe(duration.as_secs_f64());
}

/// Record a failed request
pub fn record_request_failure(model: &str, duration: Duration, error_type: &str) {
    REQUEST_COUNTER.with_label_values(&["failure", model]).inc();
    REQUEST_DURATION
        .with_label_values(&[model, "failure"])
        .observe(duration.as_secs_f64());
    ERROR_COUNTER.with_label_values(&[error_type, model]).inc();
}

/// Record inference time
pub fn record_inference_time(model: &str, batch_size: usize, duration: Duration) {
    INFERENCE_DURATION
        .with_label_values(&[model, &batch_size.to_string()])
        .observe(duration.as_secs_f64());
}

/// Update queue depth for a worker
pub fn update_queue_depth(worker_id: usize, depth: usize) {
    #[allow(clippy::cast_precision_loss)]
    let depth_f64 = depth as f64;
    QUEUE_DEPTH
        .with_label_values(&[&worker_id.to_string()])
        .set(depth_f64);
}

/// Increment active requests
pub fn increment_active_requests(model: &str) {
    ACTIVE_REQUESTS.with_label_values(&[model]).inc();
}

/// Decrement active requests
pub fn decrement_active_requests(model: &str) {
    ACTIVE_REQUESTS.with_label_values(&[model]).dec();
}

/// Update worker utilization
pub fn update_worker_utilization(worker_id: usize, utilization: f64) {
    WORKER_UTILIZATION
        .with_label_values(&[&worker_id.to_string()])
        .set(utilization.clamp(0.0, 1.0));
}

/// Record a rate-limited request
pub fn record_rate_limited(client: &str, reason: &str) {
    RATE_LIMITED_REQUESTS
        .with_label_values(&[client, reason])
        .inc();
}

/// Export all metrics in Prometheus format
///
/// # Panics
///
/// Panics if encoding the metrics or converting to UTF-8 fails.
/// This should never happen in practice as the Prometheus encoder produces valid UTF-8.
pub fn export_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Initialize all metrics (called at server startup)
pub fn init_metrics() {
    // Force lazy initialization
    LazyLock::force(&REQUEST_COUNTER);
    LazyLock::force(&REQUEST_DURATION);
    LazyLock::force(&INFERENCE_DURATION);
    LazyLock::force(&QUEUE_DEPTH);
    LazyLock::force(&ACTIVE_REQUESTS);
    LazyLock::force(&WORKER_UTILIZATION);
    LazyLock::force(&RATE_LIMITED_REQUESTS);
    LazyLock::force(&ERROR_COUNTER);

    tracing::info!("Prometheus metrics initialized");
}
