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

//! Performance benchmarks for embellama

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use embellama::{EmbeddingEngine, EngineConfig, NormalizationMode, PoolingStrategy};
use std::time::Duration;

/// Helper to get model path from environment
fn get_benchmark_model_path() -> Option<std::path::PathBuf> {
    std::env::var("EMBELLAMA_BENCH_MODEL")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

/// Benchmark single embedding generation
fn bench_single_embedding(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("bench-model")
        .with_normalization_mode(NormalizationMode::L2)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Warmup
    engine.warmup_model(None).expect("Warmup failed");

    let test_texts = vec![
        "Short text",
        "Medium length text with more words to process",
        "Long text with many words that will require more processing time and potentially more tokens to handle in the embedding generation process",
    ];

    let mut group = c.benchmark_group("single_embedding");
    group.measurement_time(Duration::from_secs(10));

    for text in &test_texts {
        let text_len = text.len();
        group.throughput(Throughput::Bytes(text_len as u64));
        group.bench_with_input(
            BenchmarkId::new("text_length", text_len),
            text,
            |b, text| {
                b.iter(|| {
                    let _ = engine
                        .embed(None, black_box(text))
                        .expect("Embedding failed");
                });
            },
        );
    }
    group.finish();
}

/// Benchmark batch embedding processing
fn bench_batch_embeddings(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("bench-model")
        .with_batch_size(64)
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    // Warmup
    engine.warmup_model(None).expect("Warmup failed");

    let mut group = c.benchmark_group("batch_embeddings");
    group.measurement_time(Duration::from_secs(15));

    let batch_sizes = vec![1, 5, 10, 25, 50, 100];

    for size in batch_sizes {
        let texts: Vec<String> = (0..size)
            .map(|i| format!("Sample text number {i} for batch processing benchmark"))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        group.throughput(Throughput::Elements(u64::try_from(size).unwrap()));
        group.bench_with_input(
            BenchmarkId::new("batch_size", size),
            &text_refs,
            |b, texts| {
                b.iter(|| {
                    let _ = engine
                        .embed_batch(None, black_box(texts))
                        .expect("Batch embedding failed");
                });
            },
        );
    }
    group.finish();
}

/// Benchmark different pooling strategies
fn bench_pooling_strategies(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let strategies = vec![
        PoolingStrategy::Mean,
        PoolingStrategy::Cls,
        PoolingStrategy::Max,
        PoolingStrategy::MeanSqrt,
    ];

    let text = "Benchmark text for testing different pooling strategies and their performance characteristics";

    let mut group = c.benchmark_group("pooling_strategies");
    group.measurement_time(Duration::from_secs(10));

    for strategy in strategies {
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name(format!("bench-{strategy:?}"))
            .with_pooling_strategy(strategy)
            .build()
            .expect("Failed to create config");

        let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
        engine.warmup_model(None).expect("Warmup failed");

        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{strategy:?}")),
            &text,
            |b, text| {
                b.iter(|| {
                    let _ = engine
                        .embed(None, black_box(text))
                        .expect("Embedding failed");
                });
            },
        );
    }
    group.finish();
}

/// Benchmark normalization overhead
fn bench_normalization(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let text = "Test text for normalization benchmark";

    let mut group = c.benchmark_group("normalization");
    group.measurement_time(Duration::from_secs(10));

    // Without normalization
    let config_no_norm = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("bench-no-norm")
        .with_normalization_mode(NormalizationMode::None)
        .build()
        .expect("Failed to create config");

    let engine_no_norm = EmbeddingEngine::new(config_no_norm).expect("Failed to create engine");
    engine_no_norm.warmup_model(None).expect("Warmup failed");

    group.bench_function("without_normalization", |b| {
        b.iter(|| {
            let _ = engine_no_norm
                .embed(None, black_box(text))
                .expect("Embedding failed");
        });
    });

    // With normalization
    let config_norm = EngineConfig::builder()
        .with_model_path(&model_path)
        .with_model_name("bench-norm")
        .with_normalization_mode(NormalizationMode::L2)
        .build()
        .expect("Failed to create config");

    let engine_norm = EmbeddingEngine::new(config_norm).expect("Failed to create engine");
    engine_norm.warmup_model(None).expect("Warmup failed");

    group.bench_function("with_normalization", |b| {
        b.iter(|| {
            let _ = engine_norm
                .embed(None, black_box(text))
                .expect("Embedding failed");
        });
    });

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("bench-memory")
        .build()
        .expect("Failed to create config");

    let engine = EmbeddingEngine::new(config).expect("Failed to create engine");

    let mut group = c.benchmark_group("memory_patterns");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark repeated allocations
    group.bench_function("repeated_small_batches", |b| {
        b.iter(|| {
            for i in 0..10 {
                let text = format!("Text {i}");
                let _ = engine
                    .embed(None, black_box(&text))
                    .expect("Embedding failed");
            }
        });
    });

    // Benchmark large batch allocation
    group.bench_function("single_large_batch", |b| {
        let texts: Vec<String> = (0..100).map(|i| format!("Text {i}")).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        b.iter(|| {
            let _ = engine
                .embed_batch(None, black_box(text_refs.as_slice()))
                .expect("Batch embedding failed");
        });
    });

    group.finish();
}

/// Benchmark model loading time
fn bench_model_loading(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let mut group = c.benchmark_group("model_loading");
    group.sample_size(10); // Reduce sample size for expensive operation
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("load_and_init", |b| {
        b.iter(|| {
            let config = EngineConfig::builder()
                .with_model_path(&model_path)
                .with_model_name("bench-load")
                .build()
                .expect("Failed to create config");

            let _engine = EmbeddingEngine::new(config).expect("Failed to create engine");
            // Engine drops here
        });
    });

    group.finish();
}

/// Benchmark thread scaling
fn bench_thread_scaling(c: &mut Criterion) {
    let Some(model_path) = get_benchmark_model_path() else {
        eprintln!("Skipping benchmark: Set EMBELLAMA_BENCH_MODEL to a valid model path");
        return;
    };

    let thread_counts = vec![1, 2, 4, 8, 16];
    let text = "Benchmark text for thread scaling test";

    let mut group = c.benchmark_group("thread_scaling");
    group.measurement_time(Duration::from_secs(10));

    for threads in thread_counts {
        let config = EngineConfig::builder()
            .with_model_path(&model_path)
            .with_model_name(format!("bench-threads-{threads}"))
            .with_n_threads(threads)
            .build()
            .expect("Failed to create config");

        let engine = EmbeddingEngine::new(config).expect("Failed to create engine");
        engine.warmup_model(None).expect("Warmup failed");

        group.bench_with_input(BenchmarkId::new("threads", threads), &text, |b, text| {
            b.iter(|| {
                let _ = engine
                    .embed(None, black_box(text))
                    .expect("Embedding failed");
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_embedding,
    bench_batch_embeddings,
    bench_pooling_strategies,
    bench_normalization,
    bench_memory_patterns,
    bench_model_loading,
    bench_thread_scaling
);
criterion_main!(benches);
