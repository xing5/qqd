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

//! Backpressure handling and circuit breaker implementation
//!
//! This module provides mechanisms to handle system overload gracefully,
//! including circuit breakers and adaptive load shedding.

use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Internal state for circuit breaker (protected by single mutex)
struct CircuitBreakerInner {
    /// Current state of the circuit
    state: CircuitState,
    /// Number of consecutive failures
    failure_count: usize,
    /// Number of consecutive successes (in half-open state)
    success_count: usize,
    /// Timestamp when circuit was opened
    opened_at: Option<Instant>,
}

/// Circuit breaker for protecting downstream services
pub struct CircuitBreaker {
    /// All mutable state protected by single mutex
    inner: parking_lot::Mutex<CircuitBreakerInner>,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Total requests processed
    total_requests: AtomicU64,
    /// Total failures
    total_failures: AtomicU64,
}

/// Circuit breaker configuration
#[derive(Clone, Debug)]
pub struct CircuitBreakerConfig {
    /// Number of failures to open circuit
    pub failure_threshold: usize,
    /// Number of successes in half-open to close circuit
    pub success_threshold: usize,
    /// Duration to wait before transitioning from open to half-open
    pub timeout: Duration,
    /// Failure rate threshold (0.0 to 1.0)
    pub failure_rate_threshold: f64,
    /// Minimum number of requests before evaluating failure rate
    pub min_requests: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            failure_rate_threshold: 0.5,
            min_requests: 10,
        }
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: parking_lot::Mutex::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                opened_at: None,
            }),
            config,
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
        }
    }

    /// Check if request should be allowed
    pub fn should_allow(&self) -> bool {
        let mut inner = self.inner.lock();
        match inner.state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout has expired
                if let Some(opened_at) = inner.opened_at {
                    if opened_at.elapsed() >= self.config.timeout {
                        // Transition to half-open within the lock
                        inner.state = CircuitState::HalfOpen;
                        inner.success_count = 0;
                        tracing::info!("Circuit breaker half-open");
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut inner = self.inner.lock();
        match inner.state {
            CircuitState::Closed => {
                // Reset failure count on success
                inner.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                inner.success_count += 1;
                if inner.success_count >= self.config.success_threshold {
                    // Transition to closed
                    inner.state = CircuitState::Closed;
                    inner.opened_at = None;
                    inner.failure_count = 0;
                    inner.success_count = 0;
                    tracing::info!("Circuit breaker closed");
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle gracefully
            }
        }
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);

        let mut inner = self.inner.lock();
        match inner.state {
            CircuitState::Closed => {
                inner.failure_count += 1;

                // Check failure threshold
                if inner.failure_count >= self.config.failure_threshold {
                    // Transition to open
                    inner.state = CircuitState::Open;
                    inner.opened_at = Some(Instant::now());
                    inner.failure_count = 0;
                    inner.success_count = 0;
                    tracing::warn!("Circuit breaker opened");
                    return;
                }

                // Check failure rate
                let total_requests = self.total_requests.load(Ordering::Relaxed);
                if total_requests >= self.config.min_requests as u64 {
                    let total_failures = self.total_failures.load(Ordering::Relaxed);
                    #[allow(clippy::cast_precision_loss)]
                    let failure_rate = total_failures as f64 / total_requests as f64;

                    if failure_rate >= self.config.failure_rate_threshold {
                        // Transition to open
                        inner.state = CircuitState::Open;
                        inner.opened_at = Some(Instant::now());
                        inner.failure_count = 0;
                        inner.success_count = 0;
                        tracing::warn!("Circuit breaker opened due to high failure rate");
                    }
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state reopens circuit
                inner.state = CircuitState::Open;
                inner.opened_at = Some(Instant::now());
                inner.failure_count = 0;
                inner.success_count = 0;
                tracing::warn!("Circuit breaker reopened from half-open");
            }
            CircuitState::Open => {
                // Already open
            }
        }
    }

    // Note: transition methods are no longer needed as state changes happen atomically within the lock

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.inner.lock().state
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            failure_rate: {
                let total = self.total_requests.load(Ordering::Relaxed);
                if total > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        self.total_failures.load(Ordering::Relaxed) as f64 / total as f64
                    }
                } else {
                    0.0
                }
            },
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current state of the circuit breaker
    pub state: CircuitState,
    /// Total number of requests processed
    pub total_requests: u64,
    /// Total number of failed requests
    pub total_failures: u64,
    /// Current failure rate (0.0 to 1.0)
    pub failure_rate: f64,
}

/// Load shedder for adaptive backpressure
pub struct LoadShedder {
    /// Current load level (0.0 to 1.0)
    load_level: RwLock<f64>,
    /// Rejection probability curve
    rejection_curve: Box<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl Default for LoadShedder {
    fn default() -> Self {
        Self::with_curve(Box::new(|load| {
            // Linear curve: 0% rejection at 0.8 load, 100% at 1.0 load
            if load < 0.8 { 0.0 } else { (load - 0.8) * 5.0 }
        }))
    }
}

impl LoadShedder {
    /// Create a new load shedder with linear rejection curve
    pub fn new() -> Self {
        Self::default()
    }

    /// Create load shedder with custom rejection curve
    pub fn with_curve(curve: Box<dyn Fn(f64) -> f64 + Send + Sync>) -> Self {
        Self {
            load_level: RwLock::new(0.0),
            rejection_curve: curve,
        }
    }

    /// Update current load level
    pub fn update_load(&self, load: f64) {
        let clamped = load.clamp(0.0, 1.0);
        *self.load_level.write() = clamped;
    }

    /// Check if request should be shed
    pub fn should_shed(&self) -> bool {
        use rand::Rng;

        let load = *self.load_level.read();
        let rejection_prob = (self.rejection_curve)(load);

        // Random decision based on rejection probability
        let mut rng = rand::thread_rng();
        rng.gen_range(0.0..1.0) < rejection_prob
    }

    /// Get current load level
    pub fn load(&self) -> f64 {
        *self.load_level.read()
    }
}

/// System health indicator combining multiple signals
pub struct SystemHealth {
    /// Circuit breaker
    pub circuit_breaker: Arc<CircuitBreaker>,
    /// Load shedder
    pub load_shedder: Arc<LoadShedder>,
    /// Queue depth threshold
    pub queue_threshold: usize,
    /// Current queue depth
    queue_depth: AtomicUsize,
}

impl SystemHealth {
    /// Create a new system health monitor
    pub fn new(
        circuit_breaker: Arc<CircuitBreaker>,
        load_shedder: Arc<LoadShedder>,
        queue_threshold: usize,
    ) -> Self {
        Self {
            circuit_breaker,
            load_shedder,
            queue_threshold,
            queue_depth: AtomicUsize::new(0),
        }
    }

    /// Update queue depth
    pub fn update_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);

        // Update load shedder based on queue depth
        #[allow(clippy::cast_precision_loss)]
        let load = depth as f64 / self.queue_threshold as f64;
        self.load_shedder.update_load(load);
    }

    /// Check if system is healthy enough to accept request
    pub fn is_healthy(&self) -> bool {
        // Check circuit breaker
        if !self.circuit_breaker.should_allow() {
            return false;
        }

        // Check load shedding
        if self.load_shedder.should_shed() {
            return false;
        }

        // Check queue depth
        let depth = self.queue_depth.load(Ordering::Relaxed);
        depth < self.queue_threshold
    }

    /// Get health status
    pub fn status(&self) -> HealthStatus {
        HealthStatus {
            circuit_state: self.circuit_breaker.state(),
            load_level: self.load_shedder.load(),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            queue_threshold: self.queue_threshold,
            is_healthy: self.is_healthy(),
        }
    }
}

/// Health status information
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Current state of the circuit breaker
    pub circuit_state: CircuitState,
    /// Current system load level (0.0 to 1.0)
    pub load_level: f64,
    /// Current depth of the request queue
    pub queue_depth: usize,
    /// Maximum allowed queue depth threshold
    pub queue_threshold: usize,
    /// Overall health status of the system
    pub is_healthy: bool,
}
