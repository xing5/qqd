# Architecture Proposal for Embellama

This document outlines the architectural design for `embellama`, a Rust crate and server for generating text embeddings using `llama-cpp-2`.

## 1. Overview

Embellama will consist of two main components:

1.  A **core library** that provides a robust and ergonomic Rust API for interacting with `llama.cpp` to generate embeddings.
2.  An **API server**, available as a feature flag, that exposes the library's functionality through an OpenAI-compatible REST API.

The primary goal is to create a high-performance, easy-to-use tool for developers who need to integrate local embedding models into their Rust applications.

## 2. Goals and Non-Goals

### Goals

*   Provide a simple and intuitive Rust API for embedding generation.
*   Support for loading/unloading models, and both single-text and batch embedding generation.
*   Offer an optional, feature-flagged `axum`-based server with an OpenAI-compatible API (`/v1/embeddings`).
*   Prioritize both low-latency single requests and high-throughput batch processing.
*   Enable configuration of the library via a programmatic builder pattern and the server via CLI arguments.

### Non-Goals

*   The library will **not** handle the downloading of models. Users are responsible for providing their own GGUF-formatted model files.
*   The initial version will only support the `llama.cpp` backend via the `llama-cpp-2` crate. Other backends are out of scope for now.
*   The server will not handle authentication or authorization. It is expected to run in a trusted environment.

## 3. Core Concepts

### `EmbeddingModel`

A struct that represents a loaded `llama.cpp` model. It will encapsulate the `llama_cpp_2::LlamaModel` and `llama_cpp_2::LlamaContext` and handle the logic for generating embeddings.

### `EmbeddingEngine`

The main entry point for the library. It will manage the lifecycle of `EmbeddingModel` instances (loading, unloading) and provide the public-facing API for generating embeddings. It will be configurable using a builder pattern.

### `AppState` (for Server)

An `axum` state struct that holds communication channels to the worker pool. Since `EmbeddingEngine` contains `!Send` types, it cannot be directly shared. Instead, `AppState` contains:
- A sender channel (`tokio::sync::mpsc::Sender`) for dispatching requests to workers
- Configuration data and metrics that are `Send + Sync`

## 4. Threading Model & Concurrency

### Critical Constraint: `!Send` LlamaContext

The `llama-cpp-2` library's `LlamaContext` contains `NonNull` pointers and other thread-local data, making it `!Send` and `!Sync`. This means:
- The context **cannot** be safely moved between threads
- The context **cannot** be shared between threads using `Arc`
- Each context instance must remain on the thread that created it

This fundamental constraint drives the entire concurrency architecture of both the library and server components.

### Library Threading Model

For the core library, embedding operations are inherently single-threaded per model instance:

```rust
// This will NOT work - compilation error
let model = Arc::new(EmbeddingModel::new(...)?); // ❌ EmbeddingModel is !Send

// Instead, each thread needs its own instance
let model = EmbeddingModel::new(...)?; // ✅ Thread-local instance
```

The library provides thread-safe batch processing through careful orchestration:
- **Pre-processing** (parallel): Text tokenization, validation - uses `rayon` for CPU parallelism
- **Model inference** (sequential): Single model instance processes tokens sequentially
- **Post-processing** (parallel): Result normalization, formatting - uses `rayon`

### Server Threading Architecture

The server uses a **message-passing worker pool** architecture to handle the `!Send` constraint:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Axum Handler  │────▶│    Dispatcher    │────▶│  Worker Pool    │
│   (async)       │◀────│  (channel-based) │◀────│  (thread-local  │
└─────────────────┘     └──────────────────┘     │   models)       │
                                                  └─────────────────┘
```

#### Key Components:

1. **Inference Workers**: Dedicated threads that each own a `LlamaContext` instance
2. **Message Channels**: `tokio::sync::mpsc` for request routing
3. **Response Channels**: One-shot channels for request/response pattern
4. **Dispatcher**: Routes requests to available workers

#### Request Flow:

1. Axum handler receives HTTP request
2. Creates a one-shot response channel
3. Sends request + response channel to dispatcher via mpsc
4. Dispatcher forwards to next available worker
5. Worker processes with its thread-local model
6. Worker sends result back through one-shot channel
7. Handler awaits response and returns HTTP response

This architecture ensures:
- `LlamaContext` never crosses thread boundaries
- True parallel inference with multiple workers
- Non-blocking async HTTP handling
- Predictable performance under load

## 5. Library (`embellama`) Design

### Module Structure

```
src/
├── lib.rs         # Main library file, feature flags
├── engine.rs      # EmbeddingEngine implementation and builder
├── model.rs       # EmbeddingModel implementation
├── batch.rs       # Batch processing logic
├── config.rs      # Configuration structs for the engine
├── error.rs       # Custom error types
└── server/        # Server-specific modules (feature-gated)
    ├── worker.rs      # Inference worker thread implementation
    ├── dispatcher.rs  # Request routing to workers
    ├── channel.rs     # Channel types and message definitions
    └── state.rs       # AppState and server configuration
```

### Public API & Usage

The library will be configured using a builder pattern for `EmbeddingEngine`.

**Example Usage:**

```rust
use embellama::{ModelConfig, EngineConfig, EmbeddingEngine};

// 1. Build model configuration
let model_config = ModelConfig::builder()
    .with_model_path("path/to/your/model.gguf")
    .with_model_name("my-embedding-model")
    .build()?;

// 2. Build engine configuration
let engine_config = EngineConfig::builder()
    .with_model_config(model_config)
    .build()?;

// 3. Create the engine
let engine = EmbeddingEngine::new(engine_config)?;

// 4. Generate a single embedding
let embedding = engine.embed("my-embedding-model", "Hello, world!")?;

// 5. Generate embeddings in a batch
let texts = vec!["First text", "Second text"];
let embeddings = engine.embed_batch("my-embedding-model", texts)?;

// 6. Unload the model when no longer needed
engine.unload_model("my-embedding-model")?;
```

### Error Handling

A custom `Error` enum will be defined in `src/error.rs` to handle all possible failures, from model loading to embedding generation. It will implement `std::error::Error` and provide conversions from underlying errors like those from `llama-cpp-2`.

## 6. Server (`server` feature) Design

The server will be enabled with a `server` feature flag.

### Dependencies

*   `axum`: For the web framework.
*   `tokio`: For the async runtime with multi-threaded runtime.
*   `clap`: For parsing CLI arguments.
*   `serde`: For JSON serialization/deserialization.
*   `tracing`: For logging.
*   `crossbeam-channel`: For robust channel implementations (optional, can use tokio channels).

### CLI Arguments

The server will be configured via CLI arguments using `clap`.

```bash
embellama-server --model-path /path/to/model.gguf --model-name my-model --port 8080 --workers 4
```

### `main.rs` (under `src/bin/server.rs` or similar)

The server's entry point will:
1.  Parse CLI arguments using `clap`.
2.  Spawn worker pool threads, each with its own `EmbeddingEngine` instance.
3.  Create channels for communication between axum handlers and workers.
4.  Create an `axum` `Router` with `AppState` containing the sender channel.
5.  Define the `/v1/embeddings` and `/health` endpoints.
6.  Start the server with proper graceful shutdown handling.

### OpenAI-Compatible API

**Endpoint:** `POST /v1/embeddings`

**Request Body (`EmbeddingsRequest`):**

```json
{
  "model": "my-model",
  "input": "A single string"
}
```

or

```json
{
  "model": "my-model",
  "input": ["An array", "of strings"]
}
```

**Response Body (`EmbeddingsResponse`):**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "model": "my-model",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

### API Request Flow

The complete flow of an embedding request through the server:

```rust
// 1. Axum handler receives request
async fn embeddings_handler(
    State(app_state): State<AppState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, Error> {
    // 2. Create one-shot channel for response
    let (tx, rx) = tokio::sync::oneshot::channel();

    // 3. Send request to dispatcher
    app_state.dispatcher_tx
        .send(WorkerRequest {
            model: request.model,
            input: request.input,
            response_tx: tx,
        })
        .await?;

    // 4. Await response from worker
    let embeddings = rx.await??;

    // 5. Format and return response
    Ok(Json(format_response(embeddings)))
}

// Worker thread processing
fn worker_thread(receiver: Receiver<WorkerRequest>) {
    // Own thread-local model instance
    let model = EmbeddingModel::new(config).unwrap();

    while let Ok(request) = receiver.recv() {
        // Process with thread-local model
        let result = model.generate_embedding(&request.input);

        // Send back through one-shot channel
        let _ = request.response_tx.send(result);
    }
}
```

This architecture ensures:
- Models never cross thread boundaries
- Async handlers remain non-blocking
- Parallel processing with multiple workers
- Clean error propagation

## 7. Concurrency & Scaling

### Worker Pool Configuration

The server's concurrency model is based on a configurable worker pool:

```rust
struct WorkerPoolConfig {
    num_workers: usize,        // Number of parallel inference workers
    queue_size: usize,          // Max pending requests per worker
    timeout_ms: u64,            // Request timeout
    max_batch_size: usize,      // Max batch size per worker
}
```

### Performance Characteristics

#### Single Worker Performance
- **Latency**: ~50ms for single embedding (CPU)
- **Throughput**: Sequential processing of requests
- **Memory**: Model size + context buffer (~500MB-2GB per model)

#### Multi-Worker Scaling
- **Linear scaling** up to number of CPU cores for CPU inference
- **GPU considerations**: Limited by VRAM; typically 1-2 workers per GPU
- **Memory usage**: `num_workers × model_memory_footprint`

### Resource Management

#### Memory Management
- Each worker maintains its own model instance in memory
- VRAM allocation for GPU workers must fit within GPU memory limits
- Automatic cleanup on worker shutdown

#### Backpressure Handling
When all workers are busy:
1. Requests queue up to `queue_size` limit
2. Beyond queue limit, server returns 503 Service Unavailable
3. Clients should implement exponential backoff

### Scaling Strategies

#### Horizontal Scaling (Multiple Servers)
- Deploy multiple server instances behind a load balancer
- Each instance manages its own worker pool
- Stateless design enables easy scaling

#### Vertical Scaling (More Workers)
- Increase `--workers` parameter up to available cores
- Monitor memory usage to avoid OOM
- Profile to find optimal worker count

#### Batch Optimization
- Workers can process batches for better throughput
- Batch requests are processed atomically by single worker
- Trade-off between latency and throughput

### Monitoring & Metrics

Key metrics to track:
- **Worker utilization**: % time workers are busy
- **Queue depth**: Number of pending requests
- **Request latency**: P50, P95, P99 percentiles
- **Throughput**: Requests/second, embeddings/second
- **Memory usage**: Per worker and total

## 8. Project Structure

```
embellama/
├── .gitignore
├── Cargo.toml
├── ARCHITECTURE.md
└── src/
    ├── lib.rs
    ├── engine.rs
    ├── model.rs
    ├── batch.rs
    ├── config.rs
    ├── error.rs
    ├── server/              # Server modules (feature-gated)
    │   ├── mod.rs
    │   ├── worker.rs
    │   ├── dispatcher.rs
    │   ├── channel.rs
    │   └── state.rs
    └── bin/
        └── server.rs  # Compiled only when "server" feature is enabled
```

In `Cargo.toml`:

```toml
[package]
name = "embellama"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core library dependencies
llama-cpp-2 = "0.1.117"
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
serde = { version = "1.0", features = ["derive"] }
rayon = "1.8"  # For library batch processing only

# Server-only dependencies
axum = { version = "0.8", optional = true }
tokio = { version = "1.35", features = ["full"], optional = true }
clap = { version = "4.4", features = ["derive"], optional = true }
tower = { version = "0.4", optional = true }
tower-http = { version = "0.6", features = ["cors", "trace"], optional = true }

[features]
default = []
server = ["dep:axum", "dep:tokio", "dep:clap", "dep:tower", "dep:tower-http"]

[[bin]]
name = "embellama-server"
required-features = ["server"]
path = "src/bin/server.rs"
```

## 9. Testing Strategy

*   **Unit Tests:** Each module in the library will have unit tests to verify its logic in isolation.
*   **Integration Tests:** An `integration` test module will be created. These tests will require embedding models to be present for testing the full flow. We will specifically test against GGUF-converted versions of `sentence-transformers/all-MiniLM-L6-v2` and `jinaai/jina-embeddings-v2-base-code`. A build script or a helper script can be provided to download these models for testing purposes.
*   **Server E2E Tests:** A separate test suite will make HTTP requests to a running instance of the server to verify API compliance and behavior, using the same test models.
*   **Concurrency Tests:** Specific tests for the worker pool to verify thread safety, proper message passing, and concurrent request handling.

## 10. Thread Safety Guarantees

### Why LlamaContext is !Send

The `LlamaContext` from `llama-cpp-2` is marked as `!Send` (cannot be sent between threads) for several reasons:

1. **Raw Pointers**: Contains `NonNull` pointers to C++ objects that are not thread-safe
2. **FFI Boundary**: Interfaces with C++ code that assumes single-threaded access
3. **Internal State**: Maintains thread-local state that would be corrupted if accessed from multiple threads
4. **CUDA/GPU Context**: GPU operations are often tied to specific thread contexts

### Safety Guarantees

Our architecture provides the following guarantees:

#### For Library Users
```rust
// ✅ Safe: Each thread creates its own model
std::thread::spawn(|| {
    let model = EmbeddingModel::new(config)?;
    model.generate_embedding("text")
});

// ❌ Won't compile: Cannot share model between threads
let model = Arc::new(EmbeddingModel::new(config)?);  // Compilation error
```

#### For Server Operations
- **Guarantee 1**: Each worker thread owns exactly one model instance
- **Guarantee 2**: Models never move between threads (enforced at compile time)
- **Guarantee 3**: All communication uses message passing, not shared memory
- **Guarantee 4**: Request/response channels ensure proper synchronization

### Performance Trade-offs

The threading constraints lead to specific trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| **Worker Pool** (chosen) | True parallelism, predictable performance, safe | Higher memory usage (model per worker) |
| Single Shared Model | Lower memory usage | Sequential processing, lock contention |
| Model per Request | Maximum parallelism | High model loading overhead |

### Verification

The Rust compiler enforces these guarantees at compile time:
- Attempting to wrap `EmbeddingModel` in `Arc` results in compilation error
- Attempting to move models between threads results in compilation error
- The type system ensures all threading constraints are satisfied
