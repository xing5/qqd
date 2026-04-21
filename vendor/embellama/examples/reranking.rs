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

//! Example of reranking documents using a cross-encoder model.
//!
//! This example demonstrates two patterns:
//! 1. Auto-detection: just provide the model path, pooling is auto-detected from GGUF metadata
//! 2. Explicit configuration: manually set `PoolingStrategy::Rank`
//!
//! Run with:
//! ```bash
//! EMBELLAMA_RERANK_MODEL=/path/to/bge-reranker-v2-m3.gguf cargo run --example reranking
//! ```

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode, PoolingStrategy};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("embellama=info")
        .init();

    // Get model path from environment
    let model_path = env::var("EMBELLAMA_RERANK_MODEL").ok().map_or_else(
        || {
            eprintln!("Set EMBELLAMA_RERANK_MODEL environment variable to reranker model path");
            eprintln!(
                "Example: EMBELLAMA_RERANK_MODEL=/path/to/bge-reranker-v2-m3.gguf cargo run --example reranking"
            );
            std::process::exit(1);
        },
        PathBuf::from,
    );

    println!("Loading reranker model from: {}", model_path.display());

    // =========================================================================
    // Pattern 1: Auto-detection from GGUF metadata
    // =========================================================================
    // For models like bge-reranker-v2-m3 that have pooling_type=4 (Rank) in
    // their GGUF metadata, embellama will automatically detect the model as a
    // reranker and configure PoolingStrategy::Rank + NormalizationMode::None.
    println!("\n--- Pattern 1: Auto-detection ---");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker-auto")
        .with_n_seq_max(8)
        .build()?;

    let engine = EmbeddingEngine::new(config)?;

    let query = "What is the capital of France?";
    let documents = [
        "Paris is the capital and largest city of France.",
        "Berlin is the capital of Germany.",
        "The weather is nice today.",
        "France is a country in Western Europe.",
    ];

    println!("Query: \"{query}\"");
    println!("Documents: {}", documents.len());

    let results = engine.rerank(Some("reranker-auto"), query, &documents, None, true)?;

    println!("\nResults (sorted by relevance):");
    for result in &results {
        println!(
            "  [{:.4}] Document {}: \"{}\"",
            result.relevance_score, result.index, documents[result.index]
        );
    }

    // =========================================================================
    // Pattern 2: Explicit configuration
    // =========================================================================
    // You can also explicitly configure PoolingStrategy::Rank and
    // NormalizationMode::None for reranker models.
    println!("\n--- Pattern 2: Explicit configuration ---");

    let config = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("reranker-explicit")
        .with_pooling_strategy(PoolingStrategy::Rank)
        .with_normalization_mode(NormalizationMode::None)
        .with_n_seq_max(8)
        .build()?;

    let engine = EmbeddingEngine::new(config)?;

    // Get top 2 results only
    let results = engine.rerank(Some("reranker-explicit"), query, &documents, Some(2), true)?;

    println!("\nTop 2 results:");
    for result in &results {
        println!(
            "  [{:.4}] Document {}: \"{}\"",
            result.relevance_score, result.index, documents[result.index]
        );
    }

    // Raw scores (without sigmoid normalization)
    let raw_results = engine.rerank(Some("reranker-explicit"), query, &documents, None, false)?;

    println!("\nRaw scores (no sigmoid normalization):");
    for result in &raw_results {
        println!(
            "  [{:.4}] Document {}: \"{}\"",
            result.relevance_score, result.index, documents[result.index]
        );
    }

    println!("\nReranking example completed successfully!");
    Ok(())
}
