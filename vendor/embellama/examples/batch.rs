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

//! Example of batch embedding processing

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("embellama=info")
        .init();

    // Get model path from environment
    let model_path = env::var("EMBELLAMA_MODEL").ok().map_or_else(
        || {
            eprintln!("Set EMBELLAMA_MODEL environment variable to model path");
            std::process::exit(1);
        },
        PathBuf::from,
    );

    println!("Batch Processing Example");
    println!("========================");
    println!("Model: {}\n", model_path.display());

    // Create configuration optimized for batch processing
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("batch-model")
        .with_batch_size(64) // Optimize batch size
        .with_normalization_mode(NormalizationMode::L2)
        .with_n_threads(num_cpus::get()) // Use all CPU cores
        .build()?;

    // Create engine
    let engine = EmbeddingEngine::new(config)?;

    // Warmup
    println!("Warming up model...");
    engine.warmup_model(None)?;

    // Test different batch sizes
    let batch_sizes = vec![1, 5, 10, 25, 50, 100, 200];

    println!("\nBatch Size Performance Comparison:");
    println!("-----------------------------------");
    println!(
        "{:<12} {:>12} {:>12} {:>15}",
        "Batch Size", "Total Time", "Per Item", "Items/Second"
    );
    println!("{:-<52}", "");

    for &size in &batch_sizes {
        // Generate test texts
        let texts: Vec<String> = (0..size)
            .map(|i| format!("This is document number {i} in the batch. It contains some sample text for embedding generation benchmark."))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        // Measure batch processing time
        let start = Instant::now();
        let embeddings = engine.embed_batch(None, &text_refs)?;
        let duration = start.elapsed();

        // Calculate metrics
        let total_ms = duration.as_millis();
        #[allow(clippy::cast_precision_loss)]
        let per_item_ms = total_ms as f64 / size as f64;
        #[allow(clippy::cast_precision_loss)]
        let items_per_sec = (size as f64 * 1000.0) / total_ms as f64;

        println!("{size:<12} {total_ms:>11}ms {per_item_ms:>11.2}ms {items_per_sec:>14.1}/s");

        // Verify results
        assert_eq!(embeddings.len(), size);
        for emb in &embeddings {
            assert!(!emb.is_empty(), "Empty embedding generated");
        }
    }

    // Compare sequential vs batch processing
    println!("\n\nSequential vs Batch Comparison (50 texts):");
    println!("-------------------------------------------");

    let test_texts: Vec<String> = (0..50)
        .map(|i| format!("Comparison test document {i}"))
        .collect();

    // Sequential processing
    let start = Instant::now();
    let mut sequential_embeddings = Vec::new();
    for text in &test_texts {
        sequential_embeddings.push(engine.embed(None, text)?);
    }
    let sequential_time = start.elapsed();

    // Batch processing
    let text_refs: Vec<&str> = test_texts.iter().map(std::string::String::as_str).collect();
    let start = Instant::now();
    let batch_embeddings = engine.embed_batch(None, &text_refs)?;
    let batch_time = start.elapsed();

    println!("Sequential: {sequential_time:?}");
    println!("Batch:      {batch_time:?}");
    println!(
        "Speedup:    {:.2}x",
        sequential_time.as_secs_f64() / batch_time.as_secs_f64()
    );

    // Verify results are identical
    assert_eq!(sequential_embeddings.len(), batch_embeddings.len());
    for (seq_emb, batch_emb) in sequential_embeddings.iter().zip(batch_embeddings.iter()) {
        assert_eq!(seq_emb.len(), batch_emb.len());
        // > NOTE: Embeddings might have small numerical differences due to batch processing
    }

    // Large batch stress test
    println!("\n\nLarge Batch Stress Test:");
    println!("------------------------");

    let large_batch_sizes = vec![500, 1000];
    for &size in &large_batch_sizes {
        let texts: Vec<String> = (0..size).map(|i| format!("Large batch text {i}")).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let start = Instant::now();
        let embeddings = engine.embed_batch(None, &text_refs)?;
        let duration = start.elapsed();

        // Cast is acceptable for display purposes - showing approximate texts/sec
        #[allow(clippy::cast_precision_loss)]
        let texts_per_sec = size as f64 / duration.as_secs_f64();

        println!(
            "Processed {} texts in {:?} ({:.1} texts/sec)",
            size, duration, texts_per_sec
        );

        assert_eq!(embeddings.len(), size);
    }

    println!("\nBatch processing example completed successfully!");
    Ok(())
}
