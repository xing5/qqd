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

//! Simple example of generating text embeddings

use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("embellama=debug")
        .init();

    // Get model path from environment or use default
    let model_path = env::var("EMBELLAMA_MODEL").ok().map_or_else(
        || {
            eprintln!("Set EMBELLAMA_MODEL environment variable to model path");
            eprintln!("Example: EMBELLAMA_MODEL=/path/to/model.gguf cargo run --example simple");
            std::process::exit(1);
        },
        PathBuf::from,
    );

    println!("Loading model from: {}", model_path.display());

    // Create configuration
    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("example-model")
        .with_normalization_mode(NormalizationMode::L2)
        .build()?;

    // Create embedding engine
    println!("Creating embedding engine...");
    let engine = EmbeddingEngine::new(config)?;

    // Warmup the model
    println!("Warming up model...");
    engine.warmup_model(None)?;

    // Generate embedding for a single text
    let text = "This is a simple example of generating text embeddings with embellama.";
    println!("\nGenerating embedding for: \"{text}\"");

    let embedding = engine.embed(None, text)?;

    println!("Embedding dimensions: {}", embedding.len());
    println!(
        "First 10 values: {:?}",
        &embedding[..10.min(embedding.len())]
    );

    // Calculate and display L2 norm (should be ~1.0 if normalized)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("L2 norm: {norm:.6}");

    // Generate embeddings for multiple texts
    println!("\n--- Batch Embedding Example ---");
    let texts = vec![
        "First document about technology",
        "Second document about science",
        "Third document about mathematics",
    ];

    println!("Generating embeddings for {} texts...", texts.len());
    let embeddings = engine.embed_batch(None, &texts)?;

    for (i, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
        println!("\nText {}: \"{}\"", i + 1, text);
        println!("  Dimensions: {}", emb.len());
        println!("  First 5 values: {:?}", &emb[..5.min(emb.len())]);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  L2 norm: {norm:.6}");
    }

    // Calculate cosine similarity between embeddings
    println!("\n--- Cosine Similarity ---");
    for i in 0..embeddings.len() {
        for j in i + 1..embeddings.len() {
            let similarity = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("Text {} <-> Text {}: {:.4}", i + 1, j + 1, similarity);
        }
    }

    println!("\nExample completed successfully!");
    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot_product / (norm_a * norm_b)
}
