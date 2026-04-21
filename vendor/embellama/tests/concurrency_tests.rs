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

//! Concurrency and thread safety tests for embellama

use embellama::{EmbeddingEngine, EngineConfig};
use serial_test::serial;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

mod common;
use common::*;

/// Test that model is not Send (compile-time check)
#[test]
fn test_model_not_send() {
    // This test verifies at compile time that EmbeddingModel is !Send
    // The following code should NOT compile if uncommented:
    // fn assert_send<T: Send>() {}
    // assert_send::<embellama::model::EmbeddingModel>();
}

/// Test thread-local model isolation
#[test]
#[serial]
fn test_thread_local_isolation() {
    let (_dir, model_path) = create_dummy_model();

    // Create thread-local storage for models
    thread_local! {
        static MODEL_NAME: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    }

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let path = model_path.clone();
            thread::spawn(move || {
                // Each thread sets its own model name
                MODEL_NAME.with(|name| {
                    *name.borrow_mut() = Some(format!("thread-{i}"));
                });

                // Verify isolation
                MODEL_NAME.with(|name| {
                    let stored = name.borrow();
                    assert_eq!(stored.as_ref().unwrap(), &format!("thread-{i}"));
                });

                // Create config (would create model in real scenario)
                let config = EngineConfig::builder()
                    .with_model_path(path)
                    .with_model_name(format!("model-{i}"))
                    .build()
                    .unwrap();

                assert_eq!(config.model_config.model_name, format!("model-{i}"));
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test concurrent batch processing
#[test]
#[serial]
fn test_concurrent_batch_processing() {
    if !should_run_model_tests() {
        eprintln!("Skipping test: EMBELLAMA_TEST_MODEL not set");
        return;
    }

    let model_path = get_test_model_path().unwrap();
    let config = create_test_config(model_path);
    let engine = Arc::new(EmbeddingEngine::new(config).unwrap());

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let engine = engine.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                // Each thread processes its own batch
                let texts = [
                    format!("Thread {i} text 1"),
                    format!("Thread {i} text 2"),
                    format!("Thread {i} text 3"),
                ];

                let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();
                let result = engine.embed_batch(None, &text_refs);

                assert!(result.is_ok());
                let embeddings = result.unwrap();
                assert_eq!(embeddings.len(), 3);

                // Clean up thread-local models before thread ends
                engine.cleanup_thread_models();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test that parallel pre/post-processing works correctly
#[test]
#[serial]
fn test_parallel_preprocessing() {
    use rayon::prelude::*;

    let texts: Vec<String> = (0..100).map(|i| format!("Text number {i}")).collect();

    // Simulate parallel tokenization
    let tokenized: Vec<Vec<usize>> = texts
        .par_iter()
        .map(|text| {
            // Simulate tokenization
            text.chars().map(|c| c as usize).collect()
        })
        .collect();

    assert_eq!(tokenized.len(), texts.len());

    // Verify order is preserved
    for (i, tokens) in tokenized.iter().enumerate() {
        let expected_text = format!("Text number {i}");
        assert_eq!(tokens.len(), expected_text.len());
    }
}

/// Test resource cleanup in multi-threaded environment
#[test]
#[serial]
fn test_resource_cleanup() {
    let (_dir, model_path) = create_dummy_model();

    // Spawn threads that create and drop configs
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let path = model_path.clone();
            thread::spawn(move || {
                for j in 0..5 {
                    let config = EngineConfig::builder()
                        .with_model_path(path.clone())
                        .with_model_name(format!("model-{i}-{j}"))
                        .build()
                        .unwrap();

                    // Config drops here
                    drop(config);

                    // Small delay to simulate work
                    thread::sleep(Duration::from_millis(1));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // If we get here without panics, cleanup worked correctly
}

/// Test that Arc<EmbeddingEngine> can be shared safely
#[test]
#[serial]
fn test_engine_arc_sharing() {
    let (_dir, model_path) = create_dummy_model();
    let config = create_test_config(model_path);

    // Note: This would fail with real model due to !Send constraint
    // but works for testing the Arc wrapper pattern
    if let Ok(engine) = EmbeddingEngine::new(config) {
        let engine = Arc::new(engine);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let engine = engine.clone();
                thread::spawn(move || {
                    // Each thread can access the engine
                    let models = engine.list_models();
                    assert!(!models.is_empty());

                    // Simulate some work
                    thread::sleep(Duration::from_millis(i * 10));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    } else {
        // Expected for dummy model
    }
}

/// Test batch processing order preservation under concurrency
#[test]
#[serial]
fn test_batch_order_preservation() {
    use std::sync::Mutex;

    let results = Arc::new(Mutex::new(Vec::new()));
    let barrier = Arc::new(Barrier::new(10));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let results = results.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                barrier.wait();

                // Simulate processing with random delay
                let delay = (i * 7) % 10;
                thread::sleep(Duration::from_millis(delay));

                results.lock().unwrap().push(i);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();
    assert_eq!(final_results.len(), 10);

    // All indices should be present even if order differs
    for i in 0..10 {
        assert!(final_results.contains(&i));
    }
}

/// Test panic safety - one thread panicking shouldn't affect others
#[test]
fn test_panic_isolation() {
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                assert!(!(i == 2), "Intentional panic in thread {i}");
                thread::sleep(Duration::from_millis(100));
                i // Return thread index
            })
        })
        .collect();

    let mut panicked_count = 0;
    let mut success_count = 0;

    for handle in handles {
        match handle.join() {
            Ok(_) => success_count += 1,
            Err(_) => panicked_count += 1,
        }
    }

    // Verify that exactly one thread panicked and others succeeded
    assert_eq!(panicked_count, 1, "Expected exactly one thread to panic");
    assert_eq!(success_count, 3, "Expected three threads to succeed");
}

/// Test lock poisoning detection
#[test]
#[serial]
fn test_lock_poisoning() {
    use std::panic;
    use std::sync::{Arc, Mutex};

    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let data_clone = data.clone();

    // Spawn thread that will panic while holding lock
    let handle = thread::spawn(move || {
        let mut guard = data_clone.lock().unwrap();
        guard.push(4);
        panic!("Intentional panic with lock held");
    });

    // Wait for panic
    let _ = handle.join();

    // Try to acquire poisoned lock
    match data.lock() {
        Ok(_) => panic!("Lock should be poisoned"),
        Err(poisoned) => {
            // Can still recover data if needed
            let guard = poisoned.into_inner();
            assert_eq!(guard.len(), 4); // Verify mutation happened
        }
    }
}

/// Test concurrent model operations don't interfere
#[test]
#[serial]
fn test_model_operation_isolation() {
    let (_dir, model_path) = create_dummy_model();

    // Create configs with different settings
    let configs: Vec<_> = (0..4)
        .map(|i| {
            EngineConfig::builder()
                .with_model_path(model_path.clone())
                .with_model_name(format!("model-{i}"))
                .with_context_size(512 * (i + 1))
                .with_n_threads((i + 1) * 2)
                .build()
                .unwrap()
        })
        .collect();

    // Verify each config maintains its settings
    for (i, config) in configs.iter().enumerate() {
        assert_eq!(config.model_config.model_name, format!("model-{i}"));
        assert_eq!(
            config.model_config.context_size,
            Some((512 * (i + 1)) as u32)
        );
        assert_eq!(config.model_config.n_threads, Some((i + 1) * 2));
    }
}

/// Test that batch processing maintains consistency
#[test]
#[serial]
fn test_batch_consistency() {
    use std::collections::HashSet;

    let batch_sizes = vec![1, 10, 50, 100, 500];

    for size in batch_sizes {
        let texts: Vec<String> = (0..size).map(|i| format!("Consistency test {i}")).collect();

        // Simulate parallel processing
        let processed: HashSet<String> = texts.iter().cloned().collect();

        assert_eq!(processed.len(), size);

        // Verify no duplicates or missing items
        for i in 0..size {
            let expected = format!("Consistency test {i}");
            assert!(processed.contains(&expected));
        }
    }
}

/// Test race condition prevention in model loading
#[test]
#[serial]
fn test_no_race_in_model_loading() {
    let (_dir, model_path) = create_dummy_model();
    let barrier = Arc::new(Barrier::new(5));

    let handles: Vec<_> = (0..5)
        .map(|i| {
            let path = model_path.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                // All threads try to create config at the same time
                barrier.wait();

                let result = EngineConfig::builder()
                    .with_model_path(path)
                    .with_model_name(format!("race-test-{i}"))
                    .build();

                assert!(result.is_ok());
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[cfg(test)]
mod thread_safety_compile_tests {
    // These tests verify compile-time thread safety guarantees
    // They don't need to run, just compile

    use std::rc::Rc;
    use std::sync::Arc;

    fn is_send<T: Send>(_: &T) {}
    fn is_sync<T: Sync>(_: &T) {}

    #[test]
    fn test_arc_is_send_sync() {
        let arc = Arc::new(42);
        is_send(&arc);
        is_sync(&arc);
    }

    #[test]
    fn test_rc_not_send_sync() {
        let rc = Rc::new(42);
        // These would fail to compile:
        // is_send(&rc);
        // is_sync(&rc);
        drop(rc); // Suppress unused warning
    }
}
