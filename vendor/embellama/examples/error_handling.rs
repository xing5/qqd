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

//! Example of robust error handling patterns

use embellama::{EmbeddingEngine, EngineConfig, Error};
use std::env;
use std::path::PathBuf;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("embellama=debug")
        .init();

    // Demonstrate various error handling patterns
    if let Err(e) = run_examples() {
        eprintln!("Example failed: {e}");
        std::process::exit(1);
    }
}

fn run_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("Error Handling Examples");
    println!("=======================\n");

    // Example 1: Handle missing model file
    handle_missing_model();

    // Example 2: Handle invalid configuration
    handle_invalid_config()?;

    // Example 3: Handle embedding errors with retry
    handle_embedding_errors()?;

    // Example 4: Handle batch processing errors
    handle_batch_errors()?;

    Ok(())
}

/// Example 1: Gracefully handle missing model file
fn handle_missing_model() {
    println!("1. Handling Missing Model File:");
    println!("-------------------------------");

    // Try to load a non-existent model
    let result = EngineConfig::builder()
        .with_model_path("/non/existent/model.gguf")
        .with_model_name("test")
        .build();

    match result {
        Ok(_) => {
            println!("  Unexpected: Config created with non-existent model");
        }
        Err(e) => {
            println!("  Expected error caught: {e}");

            // Check if it's a configuration error
            if e.is_configuration_error() {
                println!("  -> This is a configuration error (as expected)");
            }
        }
    }

    println!();
}

/// Example 2: Handle invalid configuration
fn handle_invalid_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Handling Invalid Configuration:");
    println!("----------------------------------");

    // Create a temporary model file for testing
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("test.gguf");
    std::fs::write(&model_path, b"dummy model")?;

    // Test various invalid configurations
    let invalid_configs = vec![
        (
            "Empty name",
            EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name("")
                .build(),
        ),
        (
            "Zero threads",
            EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name("test")
                .with_n_threads(0)
                .build(),
        ),
        (
            "Zero context",
            EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name("test")
                .with_context_size(0)
                .build(),
        ),
    ];

    for (desc, result) in invalid_configs {
        match result {
            Ok(_) => println!("  {desc}: Unexpected success"),
            Err(e) => println!("  {desc}: Caught error - {e}"),
        }
    }

    println!();
    Ok(())
}

/// Example 3: Handle embedding errors with retry logic
fn handle_embedding_errors() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Handling Embedding Errors with Retry:");
    println!("----------------------------------------");

    // Get model path from environment or skip
    let model_path = if let Ok(path) = env::var("EMBELLAMA_MODEL") {
        PathBuf::from(path)
    } else {
        println!("  Skipping: Set EMBELLAMA_MODEL to run this example");
        println!();
        return Ok(());
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("retry-example")
        .build()?;

    let engine = EmbeddingEngine::new(config)?;

    // Simulate retryable operations
    let texts = vec![
        "", // Empty text - will fail
        "Valid text for embedding",
    ];

    for text in texts {
        println!("  Attempting to embed: \"{text}\"");

        let mut retries = 3;
        loop {
            match engine.embed(None, text) {
                Ok(embedding) => {
                    println!("    Success! Embedding size: {}", embedding.len());
                    break;
                }
                Err(e) => {
                    println!("    Error: {e}");

                    if e.is_retryable() && retries > 0 {
                        retries -= 1;
                        println!("    Retrying... ({retries} attempts left)");
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    } else {
                        println!("    Failed permanently");
                        break;
                    }
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Example 4: Handle batch processing errors
fn handle_batch_errors() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Handling Batch Processing Errors:");
    println!("------------------------------------");

    // Get model path from environment or skip
    let model_path = if let Ok(path) = env::var("EMBELLAMA_MODEL") {
        PathBuf::from(path)
    } else {
        println!("  Skipping: Set EMBELLAMA_MODEL to run this example");
        println!();
        return Ok(());
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("batch-error-example")
        .build()?;

    let engine = EmbeddingEngine::new(config)?;

    // Test batch with mixed valid/invalid inputs
    let texts = vec![
        "Valid document 1",
        "", // Invalid: empty
        "Valid document 2",
        "   ", // Invalid: whitespace only
        "Valid document 3",
    ];

    println!(
        "  Processing batch of {} texts (including invalid ones)...",
        texts.len()
    );

    match engine.embed_batch(None, &texts) {
        Ok(embeddings) => {
            println!(
                "  Batch succeeded! Generated {} embeddings",
                embeddings.len()
            );

            for (i, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
                println!(
                    "    Text {}: \"{}\" -> {} dimensions",
                    i,
                    text.trim(),
                    emb.len()
                );
            }
        }
        Err(Error::BatchError {
            message,
            failed_indices,
        }) => {
            println!("  Batch error: {message}");
            println!("  Failed indices: {failed_indices:?}");

            // Process only the valid texts
            let valid_texts: Vec<&str> = texts
                .iter()
                .enumerate()
                .filter(|(i, _)| !failed_indices.contains(i))
                .map(|(_, t)| &**t)
                .collect();

            if !valid_texts.is_empty() {
                println!("  Retrying with only valid texts...");
                match engine.embed_batch(None, &valid_texts) {
                    Ok(embeddings) => {
                        println!(
                            "  Retry succeeded! Generated {} embeddings",
                            embeddings.len()
                        );
                    }
                    Err(e) => {
                        println!("  Retry failed: {e}");
                    }
                }
            }
        }
        Err(e) => {
            println!("  Unexpected error: {e}");

            // Check error type for appropriate handling
            match e {
                Error::ModelNotFound { name } => {
                    println!("  -> Model '{name}' not found. Load it first.");
                }
                Error::InvalidInput { message } => {
                    println!("  -> Invalid input: {message}");
                }
                Error::Timeout { message } => {
                    println!("  -> Operation timed out: {message}");
                    println!("  -> This is retryable!");
                }
                _ => {
                    println!("  -> Error type: {e:?}");
                }
            }
        }
    }

    println!();
    println!("Error handling examples completed!");
    Ok(())
}
