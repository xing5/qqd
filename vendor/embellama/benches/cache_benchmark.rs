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

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use embellama::cache::CacheStore;
use embellama::cache::embedding_cache::EmbeddingCache;
use embellama::{NormalizationMode, PoolingStrategy};
use std::sync::Arc;
use std::thread;

fn benchmark_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");

    // Create cache for benchmarking
    let cache = Arc::new(EmbeddingCache::new(10000, 3600));

    // Pre-populate cache with some data
    for i in 0..1000 {
        let key = format!("key_{}", i);
        let embedding = vec![i as f32; 384]; // Typical embedding size
        cache.insert(key, embedding);
    }

    // Benchmark cache hits
    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let key = format!("key_{}", 500);
            black_box(cache.get(&key))
        })
    });

    // Benchmark cache misses
    group.bench_function("cache_miss", |b| {
        let mut counter = 10000;
        b.iter(|| {
            let key = format!("missing_key_{}", counter);
            counter += 1;
            black_box(cache.get(&key))
        })
    });

    // Benchmark cache insertions
    group.bench_function("cache_insert", |b| {
        let mut counter = 20000;
        b.iter(|| {
            let key = format!("new_key_{}", counter);
            let embedding = vec![counter as f32; 384];
            counter += 1;
            cache.insert(key, embedding)
        })
    });

    // Benchmark key computation
    group.bench_function("compute_key", |b| {
        b.iter(|| {
            EmbeddingCache::compute_key(
                black_box("This is a sample text for embedding"),
                black_box("model-name"),
                black_box(PoolingStrategy::Mean),
                black_box(NormalizationMode::L2),
            )
        })
    });

    group.finish();
}

fn benchmark_cache_with_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_embedding_sizes");

    let sizes = vec![128, 384, 768, 1024, 2048];

    for size in sizes {
        let cache = EmbeddingCache::new(1000, 3600);

        // Benchmark insertion with different embedding sizes
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("insert", size), &size, |b, &size| {
            let mut counter = 0;
            b.iter(|| {
                let key = format!("key_{}", counter);
                let embedding = vec![0.1_f32; size];
                counter += 1;
                cache.insert(key, embedding);
            });
        });

        // Pre-populate for retrieval benchmark
        for i in 0..100 {
            let key = format!("test_key_{}", i);
            let embedding = vec![i as f32; size];
            cache.insert(key, embedding);
        }

        // Benchmark retrieval with different embedding sizes
        group.bench_with_input(BenchmarkId::new("get", size), &size, |b, _| {
            b.iter(|| {
                let key = format!("test_key_{}", 50);
                black_box(cache.get(&key))
            });
        });
    }

    group.finish();
}

fn benchmark_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_cache_access");

    let thread_counts = vec![1, 2, 4, 8];

    for num_threads in thread_counts {
        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", num_threads),
            &num_threads,
            |b, &num_threads| {
                let cache = Arc::new(EmbeddingCache::new(10000, 3600));

                // Pre-populate cache
                for i in 0..1000 {
                    let key = format!("key_{}", i);
                    let embedding = vec![i as f32; 384];
                    cache.insert(key, embedding);
                }

                b.iter(|| {
                    let mut handles = vec![];

                    for thread_id in 0..num_threads {
                        let cache_clone = Arc::clone(&cache);
                        let handle = thread::spawn(move || {
                            for i in 0..100 {
                                let key = format!("key_{}", (thread_id * 100 + i) % 1000);
                                black_box(cache_clone.get(&key));
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("concurrent_writes", num_threads),
            &num_threads,
            |b, &num_threads| {
                let cache = Arc::new(EmbeddingCache::new(10000, 3600));

                b.iter(|| {
                    let mut handles = vec![];

                    for thread_id in 0..num_threads {
                        let cache_clone = Arc::clone(&cache);
                        let handle = thread::spawn(move || {
                            for i in 0..100 {
                                let key = format!("thread_{}_key_{}", thread_id, i);
                                let embedding = vec![i as f32; 384];
                                cache_clone.insert(key, embedding);
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effectiveness");

    // Simulate workload with different hit rates
    let hit_rates = vec![0.0, 0.25, 0.50, 0.75, 0.95, 1.0];

    for hit_rate in hit_rates {
        group.bench_with_input(
            BenchmarkId::new("mixed_workload", format!("{:.0}%", hit_rate * 100.0)),
            &hit_rate,
            |b, &hit_rate| {
                let cache = EmbeddingCache::new(1000, 3600);

                // Pre-populate cache with 100 entries
                for i in 0..100 {
                    let key = format!("cached_{}", i);
                    let embedding = vec![i as f32; 384];
                    cache.insert(key, embedding);
                }

                let mut counter = 0;
                b.iter(|| {
                    let should_hit = (counter as f64 % 100.0) / 100.0 < hit_rate;
                    let key = if should_hit {
                        format!("cached_{}", counter % 100)
                    } else {
                        format!("uncached_{}", counter)
                    };
                    counter += 1;

                    if let Some(embedding) = cache.get(&key) {
                        black_box(embedding);
                    } else {
                        // Simulate cache miss by creating and inserting new embedding
                        let embedding = vec![counter as f32; 384];
                        cache.insert(key, embedding.clone());
                        black_box(embedding);
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_memory");

    // Benchmark memory tracking overhead
    group.bench_function("memory_tracking", |b| {
        let cache = EmbeddingCache::new(10000, 3600);

        // Pre-populate
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let embedding = vec![i as f32; 384];
            cache.insert(key, embedding);
        }

        b.iter(|| black_box(cache.stats()));
    });

    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let batch_sizes = vec![1, 10, 50, 100, 500];

    for batch_size in batch_sizes {
        let cache = Arc::new(EmbeddingCache::new(10000, 3600));

        // Pre-populate cache
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let embedding = vec![i as f32; 384];
            cache.insert(key, embedding);
        }

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_lookup", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let key = format!("key_{}", i % 1000);
                        results.push(cache.get(&key));
                    }
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_cache_operations,
    benchmark_cache_with_different_sizes,
    benchmark_concurrent_access,
    benchmark_cache_effectiveness,
    benchmark_memory_overhead,
    benchmark_batch_operations
);
criterion_main!(benches);
