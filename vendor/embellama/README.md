# Embellama

[![Crates.io](https://img.shields.io/crates/v/embellama.svg)](https://crates.io/crates/embellama)
[![Documentation](https://docs.rs/embellama/badge.svg)](https://docs.rs/embellama)
[![License](https://img.shields.io/crates/l/embellama.svg)](https://github.com/darjus/embellama/blob/master/LICENSE)
[![CI](https://github.com/darjus/embellama/actions/workflows/ci.yml/badge.svg)](https://github.com/darjus/embellama/actions/workflows/ci.yml)

High-performance Rust library for generating text embeddings using llama-cpp.

## Features

- **High Performance**: Optimized for speed with parallel pre/post-processing
- **Thread Safety**: Compile-time guarantees for safe concurrent usage
- **Multiple Models**: Support for managing multiple embedding models
- **Batch Processing**: Efficient batch embedding generation
- **Flexible Configuration**: Extensive configuration options for model tuning
- **Multiple Pooling Strategies**: Mean, CLS, Max, and `MeanSqrt` pooling
- **Hardware Acceleration**: Support for Metal (macOS), CUDA (NVIDIA), Vulkan, and optimized CPU backends

## Quick Start

```rust,no_run
# fn main() -> anyhow::Result<()> {
use embellama::{ModelConfig, EngineConfig, EmbeddingEngine, NormalizationMode};

// Build model configuration
let model_config = ModelConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .with_normalization_mode(NormalizationMode::L2)
    .build()?;

// Build engine configuration
let engine_config = EngineConfig::builder()
    .with_model_config(model_config)
    .build()?;

// Create engine
let engine = EmbeddingEngine::new(engine_config)?;

// Generate single embedding
let text = "Hello, world!";
let embedding = engine.embed(None, text)?;

// Generate batch embeddings
let texts = vec!["Text 1", "Text 2", "Text 3"];
let embeddings = engine.embed_batch(None, &texts)?;
# Ok(())
# }
```

### Singleton Pattern (Advanced)

The engine can optionally use a singleton pattern for shared access across your application. The singleton methods return `Arc<Mutex<EmbeddingEngine>>` for thread-safe access:

```rust,no_run
# fn main() -> anyhow::Result<()> {
# use embellama::{ModelConfig, EngineConfig, EmbeddingEngine};
# let model_config = ModelConfig::builder()
#     .with_model_path("/path/to/model.gguf")
#     .with_model_name("my-model")
#     .build()?;
# let config = EngineConfig::builder()
#     .with_model_config(model_config)
#     .build()?;
// Get or initialize singleton instance (returns Arc<Mutex<EmbeddingEngine>>)
let engine = EmbeddingEngine::get_or_init(config)?;

// Access the singleton from anywhere in your application
let engine_clone = EmbeddingEngine::instance()
    .expect("Engine not initialized");

// Use the engine (requires locking the mutex)
let embedding = {
    let engine_guard = engine.lock().unwrap();
    engine_guard.embed(None, "text")?
};
# Ok(())
# }
```

## Tested Models

The library has been tested with the following GGUF models:
- **MiniLM-L6-v2** (`Q4_K_M`): ~15MB, 384-dimensional embeddings - used for integration tests
- **Jina Embeddings v2 Base Code** (`Q4_K_M`): ~110MB, 768-dimensional embeddings - used for benchmarks
- **BAAI/bge-reranker-v2-m3** (`Q4_K_M`): Cross-encoder reranking model - auto-detected from GGUF metadata

Both BERT-style and LLaMA-style embedding models are supported, as well as cross-encoder reranking models.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
embellama = "0.10.1"
```

### Backend Features

Platform-specific GPU acceleration and other optional features are available via Cargo features. See the `[features]` section in [`Cargo.toml`](Cargo.toml) for the full list.

## Configuration

### Basic Configuration

```rust,no_run
# fn main() -> anyhow::Result<()> {
use embellama::{ModelConfig, EngineConfig};

let model_config = ModelConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .build()?;

let config = EngineConfig::builder()
    .with_model_config(model_config)
    .build()?;
# Ok(())
# }
```

### Advanced Configuration

```rust,no_run
# fn main() -> anyhow::Result<()> {
use embellama::{ModelConfig, EngineConfig, PoolingStrategy, NormalizationMode};

let model_config = ModelConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .with_context_size(2048)
    .with_n_threads(8)
    .with_n_gpu_layers(32)
    .with_normalization_mode(NormalizationMode::L2)
    .with_pooling_strategy(PoolingStrategy::Mean)
    .build()?;

let config = EngineConfig::builder()
    .with_model_config(model_config)
    .with_use_gpu(true)
    .with_batch_size(64)
    .build()?;
# Ok(())
# }
```

### Backend Auto-Detection

The library can automatically detect and use the best available backend:

```rust,no_run
# fn main() -> anyhow::Result<()> {
use embellama::{EngineConfig, detect_best_backend, BackendInfo};

// Automatic backend detection
let config = EngineConfig::with_backend_detection()
    .with_model_path("/path/to/model.gguf")
    .with_model_name("my-model")
    .build()?;

// Check which backend was selected
let backend_info = BackendInfo::new();
println!("Using backend: {}", backend_info.backend);
println!("Available features: {:?}", backend_info.available_features);
# Ok(())
# }
```

## Reranking

Embellama supports cross-encoder reranking models like `bge-reranker-v2-m3`. Reranker models
are auto-detected from GGUF metadata (`pooling_type = 4`), so you can just load the model
without any special configuration:

```rust,no_run
# fn main() -> anyhow::Result<()> {
use embellama::{EngineConfig, EmbeddingEngine};

let config = EngineConfig::builder()
    .with_model_path("/path/to/bge-reranker-v2-m3.gguf")
    .with_model_name("reranker")
    .build()?;

let engine = EmbeddingEngine::new(config)?;

let results = engine.rerank(
    None,
    "What is the capital of France?",
    &["Paris is the capital of France.", "Berlin is in Germany."],
    None,  // return all results
    true,  // apply sigmoid normalization
)?;

for r in &results {
    println!("[{:.4}] Document {}", r.relevance_score, r.index);
}
# Ok(())
# }
```

## Pooling Strategies

- **Mean**: Average pooling across all tokens (default for encoder models)
- **CLS**: Use the CLS token embedding
- **Max**: Maximum pooling across dimensions
- **`MeanSqrt`**: Mean pooling with square root of sequence length normalization
- **Last**: Use the last token embedding (auto-selected for decoder models)
- **Rank**: Cross-encoder reranking (auto-detected from GGUF metadata)

## Thread Safety

The `LlamaContext` from llama-cpp is `!Send` and `!Sync`, which means:

- Models cannot be moved between threads
- Models cannot be shared using `Arc` alone
- Each thread must own its model instance

The library is designed with these constraints in mind:

- Use thread-local storage for model instances
- Batch processing uses parallel pre/post-processing with sequential inference
- The singleton pattern provides `Arc<Mutex<EmbeddingEngine>>` for cross-thread coordination

## API Reference

### Model Management

The library provides granular control over model lifecycle:

#### Registration vs Loading

- **Registration**: Model configuration stored in registry
- **Loading**: Model actually loaded in thread-local memory

```rust,no_run
# fn main() -> anyhow::Result<()> {
# use embellama::{ModelConfig, EngineConfig, EmbeddingEngine};
# let model_config = ModelConfig::builder()
#     .with_model_path("/path/to/model.gguf")
#     .with_model_name("my-model")
#     .build()?;
# let config = EngineConfig::builder()
#     .with_model_config(model_config)
#     .build()?;
# let engine = EmbeddingEngine::new(config)?;
// Check if model is registered (has configuration)
if engine.is_model_registered("my-model") {
    println!("Model configuration exists");
}

// Check if model is loaded in current thread
if engine.is_model_loaded_in_thread("my-model") {
    println!("Model is ready to use in this thread");
}
# Ok(())
# }
```

#### Granular Unload Operations

```rust,no_run
# fn main() -> anyhow::Result<()> {
# use embellama::{ModelConfig, EngineConfig, EmbeddingEngine};
# let model_config = ModelConfig::builder()
#     .with_model_path("/path/to/model.gguf")
#     .with_model_name("my-model")
#     .build()?;
# let config = EngineConfig::builder()
#     .with_model_config(model_config)
#     .build()?;
# let mut engine = EmbeddingEngine::new(config)?;
// Remove only from current thread (keeps registration)
engine.drop_model_from_thread("my-model")?;
// Model can be reloaded on next use

// Remove only from registry (prevents future loads)
engine.unregister_model("my-model")?;
// Existing thread-local instances continue working

// Full unload - removes from both registry and thread
engine.unload_model("my-model")?;
// Completely removes the model
# Ok(())
# }
```

### Model Loading Behavior

- **Initial model** (via `EmbeddingEngine::new()`): Loaded immediately in current thread
- **Additional models** (via `load_model()`): Lazy-loaded on first use

```rust,no_run
# fn main() -> anyhow::Result<()> {
# use embellama::{ModelConfig, EngineConfig, EmbeddingEngine};
# let model_config = ModelConfig::builder()
#     .with_model_path("/path/to/model.gguf")
#     .with_model_name("model1")
#     .build()?;
# let config = EngineConfig::builder()
#     .with_model_config(model_config)
#     .build()?;
# let model_config2 = ModelConfig::builder()
#     .with_model_path("/path/to/model2.gguf")
#     .with_model_name("model2")
#     .build()?;
# let config2 = EngineConfig::builder()
#     .with_model_config(model_config2)
#     .build()?;
// First model - loaded immediately
let mut engine = EmbeddingEngine::new(config)?;
assert!(engine.is_model_loaded_in_thread("model1"));

// Additional model - lazy loaded
engine.load_model(config2)?;
assert!(engine.is_model_registered("model2"));
assert!(!engine.is_model_loaded_in_thread("model2"));  // Not yet loaded

// Triggers actual loading in thread
engine.embed(Some("model2"), "text")?;
assert!(engine.is_model_loaded_in_thread("model2"));  // Now loaded
# Ok(())
# }
```

## Performance

The library is optimized for high performance:

- Parallel tokenization for batch processing
- Efficient memory management
- Configurable thread counts
- GPU acceleration support

### Benchmarks

Run benchmarks with:

```bash
EMBELLAMA_BENCH_MODEL=/path/to/model.gguf cargo bench
```

### Performance Tips

1. **Batch Processing**: Use `embed_batch()` for multiple texts
2. **Thread Configuration**: Set `n_threads` based on CPU cores
3. **GPU Acceleration**: Enable GPU for larger models
4. **Warmup**: Call `warmup_model()` before processing

## Development

For development setup, testing, and contributing guidelines, please see [DEVELOPMENT.md](DEVELOPMENT.md).

### Changelog

This project uses [git-cliff](https://git-cliff.org/) to generate changelogs from [conventional commits](https://www.conventionalcommits.org/). Install with `cargo install git-cliff`, then:

```bash
just changelog              # Regenerate CHANGELOG.md
just changelog-unreleased   # Preview unreleased changes
```

## Examples

See the `examples/` directory for more examples:

- `simple.rs` - Basic embedding generation
- `batch.rs` - Batch processing example
- `multi_model.rs` - Using multiple models
- `config.rs` - Configuration examples
- `error_handling.rs` - Error handling patterns
- `reranking.rs` - Cross-encoder reranking

Run examples with:

```bash
cargo run --example simple
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for development setup and contribution guidelines.

## Support

For issues and questions, please use the GitHub issue tracker.
