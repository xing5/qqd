# Agent Guidelines for Embellama

> **Note**: This document contains agent-specific guidelines and architectural constraints. For general development setup, testing, and contribution guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Critical Threading Constraints

**IMPORTANT**: The `LlamaContext` from `llama-cpp-2` is `!Send` and `!Sync`, meaning it cannot be shared between threads. This fundamental constraint drives the entire architecture.

### Key Points:
- **Never use `Arc<EmbeddingModel>`** - This will not compile due to `!Send` constraint
- **Each thread must own its model instance** - Models are thread-local
- **Use message passing, not shared state** - Channel-based communication between threads
- **Worker pool architecture for server** - Each worker thread owns a model instance

## Model Architecture Differences

### BERT vs LLaMA Models
The library supports both BERT-style and LLaMA-style embedding models, which require different handling:

#### BERT Models (e.g., MiniLM, Jina)
- Have `pooling_type` metadata indicating pre-pooled embeddings
- Use `embeddings_seq_ith(0)` to extract the pooled embedding
- May not normalize embeddings by default (check model-specific behavior)
- llama.cpp automatically calls `encode()` internally when `decode()` is called

#### LLaMA Models
- Generate token-level embeddings
- Use `embeddings_ith(i)` to extract embeddings for each token
- Require manual pooling (mean, CLS, max, etc.)
- Generally normalize embeddings by default

#### Detection and Handling
```rust
// Check if model returns pre-pooled embeddings
if let Ok(seq_embeddings) = ctx.embeddings_seq_ith(0) {
    // BERT model with pooling - single embedding for sequence
    Ok(vec![seq_embeddings.to_vec()])
} else {
    // LLaMA model - token-wise embeddings
    let mut token_embeddings = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        let embeddings = ctx.embeddings_ith(i as i32)?;
        token_embeddings.push(embeddings.to_vec());
    }
    Ok(token_embeddings)
}
```



## Project Overview

Embellama is a high-performance Rust crate and server for generating text embeddings using `llama-cpp-2`. The project provides:

1. **Core Library**: A robust and ergonomic Rust API for interacting with `llama.cpp` to generate embeddings
2. **API Server**: An optional OpenAI-compatible REST API server (available via `server` feature flag)

### Primary Goals
- Simple and intuitive Rust API for embedding generation
- Support for model loading/unloading and batch processing
- High performance for both low-latency single requests and high-throughput batch operations
- OpenAI API compatibility (`/v1/embeddings` endpoint)
- Clean separation between library and server concerns

## Rust Development Standards

### Code Quality Requirements

**MANDATORY**: Before ANY commit, you MUST:

1. **Format Code**: Run `cargo fmt` to ensure consistent formatting
2. **Lint Code**: Run `cargo clippy` and address ALL warnings
3. **Code Review**: Use the rust code review agent to validate changes
4. **Test**: Run `cargo test` to ensure all tests pass
5. **Update Changelog**: Use `git-cliff` CLI to update the changelog

### Error Handling

#### Use `thiserror` for Error Types
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbellamaError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model from {path}")]
    ModelLoadFailed { path: String, #[source] source: llama_cpp_2::Error },

    #[error("Embedding generation failed")]
    EmbeddingFailed(#[from] llama_cpp_2::Error),
}
```

#### Return `Result` Types
- All fallible operations MUST return `Result<T, E>`
- Use specific error types from `thiserror`
- Ensure compatibility with `anyhow` for application-level error handling

#### Example Usage
```rust
// Library code with specific errors
pub fn load_model(path: &str) -> Result<Model, EmbellamaError> {
    // Implementation
}

// Application code can use anyhow
use anyhow::Result;
fn main() -> Result<()> {
    let model = embellama::load_model("model.gguf")?;
    Ok(())
}
```

### Logging with `tracing`

#### Setup
```rust
use tracing::{debug, error, info, warn, trace};
```

#### Logging Guidelines
- **TRACE**: Very detailed information, hot path details
- **DEBUG**: Useful debugging information (model loading steps, batch sizes)
- **INFO**: Important state changes (model loaded, server started)
- **WARN**: Recoverable issues (fallback behavior, deprecated usage)
- **ERROR**: Unrecoverable errors with context

#### Example
```rust
#[tracing::instrument(skip(model_data))]
pub fn load_model(path: &str, model_data: &[u8]) -> Result<Model, EmbellamaError> {
    info!(path = %path, "Loading embedding model");
    debug!(size = model_data.len(), "Model data size");

    match internal_load(model_data) {
        Ok(model) => {
            info!("Model loaded successfully");
            Ok(model)
        }
        Err(e) => {
            error!(error = %e, "Failed to load model");
            Err(EmbellamaError::ModelLoadFailed { path: path.to_string(), source: e })
        }
    }
}
```

### Visibility and Encapsulation

#### Keep Implementation Details Private
```rust
// Use module-level privacy
mod internal {
    pub(crate) fn helper_function() { }
}

// Or crate-level visibility
pub(crate) struct InternalState { }

// Only expose necessary public interface
pub struct EmbeddingEngine {
    inner: Box<EngineInner>, // Private implementation (NOT Arc due to !Send)
}
```

#### Threading Architecture Pattern
```rust
// WRONG: Won't compile - LlamaContext is !Send
pub struct EmbeddingEngine {
    model: Arc<RwLock<EmbeddingModel>>, // ❌ Compilation error
}

// CORRECT: Use channels for thread communication
pub struct EmbeddingEngine {
    sender: mpsc::Sender<WorkerRequest>,  // ✅ Send requests to worker
}

// Worker owns model on its thread
fn worker_thread(receiver: mpsc::Receiver<WorkerRequest>) {
    let model = EmbeddingModel::new(...);  // Thread-local ownership
    while let Ok(req) = receiver.recv() {
        // Process with thread-local model
    }
}
```

#### Public API Design
- Minimize public surface area
- Use builder patterns for complex configurations
- Document all public items with rustdoc
- Implement `Debug` for ALL public types

### Debug Trait Implementation

**MANDATORY**: All public types MUST implement `Debug`

```rust
#[derive(Debug)]
pub struct EmbeddingEngine {
    // fields
}

// For types with sensitive data
impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("path", &self.path)
            .field("name", &self.name)
            .finish_non_exhaustive() // Hide internal details
    }
}
```

## License Compliance Requirements

**MANDATORY**: All source files MUST include the Apache 2.0 license header:

```rust
// Copyright 2024 Embellama Contributors
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
```

### When to Add License Headers

1. **New Files**: ALWAYS add the license header when creating any new `.rs` source file
2. **Modified Files**: Ensure existing files have the header before making changes
3. **Binary Files**: Do not add headers to binary files or generated content
4. **Test Files**: Include headers in test files as well

## Commit and Changelog Management

### Commit Messages

**CRITICAL RULES**:
1. **NEVER** mention AI tools, assistants, or automation in commit messages
2. Focus on WHAT changed and WHY, not HOW you arrived at the solution
3. Use conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation only
   - `style:` Formatting, missing semicolons, etc.
   - `refactor:` Code change that neither fixes a bug nor adds a feature
   - `perf:` Performance improvement
   - `test:` Adding missing tests
   - `chore:` Maintenance tasks

### Using `git-cliff` CLI

Before committing, update the changelog:

```bash
# Install git-cliff if not present
cargo install git-cliff

# Generate/update changelog
git-cliff -o CHANGELOG.md

# For specific version/tag range
git-cliff --tag v0.1.0 -o CHANGELOG.md

# Generate unreleased changes only
git-cliff --unreleased -o CHANGELOG.md
```

### Pre-Commit Checklist

1. ✅ Run `cargo fmt`
2. ✅ Run `cargo clippy` and fix ALL warnings
3. ✅ Run `cargo test`
4. ✅ Use rust code review agent
5. ✅ Update changelog with `git-cliff`
6. ✅ Ensure no secrets or API keys
7. ✅ Verify commit message follows guidelines

## Problem-Solving Tools and MCPs

### When to Use Each Tool

#### Web Search (Fetch and Brave MCPs)
Use for:
- Looking up Rust documentation and crate usage
- Finding solutions to specific error messages
- Researching llama.cpp integration details
- Checking best practices and patterns
- **Understanding !Send and !Sync constraints**

Example scenarios:
- "How to use llama-cpp-2 crate effectively"
- "Rust async performance optimization techniques"
- "OpenAI embedding API specification"
- "Working with !Send types in async Rust"

#### zen:thinkdeeper
Use when:
- Designing new architectural components
- Evaluating complex trade-offs
- Solving intricate algorithmic problems
- Making critical design decisions
- **Designing thread-safe architectures with !Send constraints**

Example scenarios:
- "Should we use channels or shared state for batch processing?" (Answer: Channels, due to !Send)
- "How to optimize memory usage for large embedding batches?"
- "Architecture for dynamic model loading/unloading"
- "Worker pool design for !Send types"

#### zen:debug
Use when:
- Standard debugging hasn't revealed root cause
- Dealing with complex async/concurrent issues
- Investigating memory leaks or performance problems
- Encountering mysterious segfaults or panics

Example scenarios:
- "Why does the model crash only under high concurrency?"
- "Memory usage grows unbounded during batch processing"
- "Deadlock occurring in specific request patterns"

### Decision Framework

```
Start with standard debugging
    ↓
If unclear → Use web search for similar issues
    ↓
If complex design question → Use zen:thinkdeeper
    ↓
If persistent bug → Use zen:debug
```

## Project Structure

```
embellama/
├── Cargo.toml          # Dependencies and features
├── CHANGELOG.md        # Maintained with git-cliff
├── AGENTS.md          # This file
├── ARCHITECTURE.md    # Design documentation
└── src/
    ├── lib.rs         # Public API, feature flags
    ├── engine.rs      # EmbeddingEngine (public interface)
    ├── model.rs       # EmbeddingModel (internal, !Send)
    ├── batch.rs       # Batch processing (internal)
    ├── config.rs      # Configuration types (public)
    ├── error.rs       # Error types with thiserror
    ├── server/        # Server-specific modules (feature-gated)
    │   ├── mod.rs
    │   ├── worker.rs      # Inference worker threads
    │   ├── dispatcher.rs  # Request routing
    │   ├── channel.rs     # Message types
    │   └── state.rs       # AppState (channels, not models)
    └── bin/
        └── server.rs  # Server binary (server feature)
```

## Testing Requirements

### Test Infrastructure
The project uses `just` for test automation. Key commands:
- `just test` - Run all tests
- `just test-unit` - Unit tests only
- `just test-integration` - Integration tests with real models
- `just test-concurrency` - Concurrency and thread safety tests
- `just bench` - Performance benchmarks
- `just bench-quick` - Quick benchmark subset

### Unit Tests
- Test each module in isolation
- Mock external dependencies
- Test error conditions thoroughly
- Use `#[tracing_test::traced_test]` for tests with logging

### Integration Tests
- **MUST use real GGUF models** - No mocks or fallbacks
- Tests should fail loudly if models aren't available
- Supported models:
  - MiniLM-L6-v2 (Q4_K_M) - For integration tests (~15MB)
  - Jina Embeddings v2 Base Code (Q4_K_M) - For benchmarks (~110MB)
- **Model-specific handling**: BERT models need `embeddings_seq_ith()` instead of `embeddings_ith()`
- Verify batch processing correctness
- Test model loading/unloading cycles

### Concurrency Tests (Critical)
- **Test worker pool message passing**
- **Verify models stay on their threads**
- **Test concurrent request handling**
- **Validate channel-based communication**
- **Ensure no data races or deadlocks**
- Use `serial_test::serial` for tests that need exclusive access

### Performance Tests
- Benchmark single embedding generation
- Benchmark batch processing with various sizes
- Test different pooling strategies
- Memory usage under load (per worker)
- Concurrent request handling
- Worker pool scaling characteristics
- Thread scaling benchmarks

## Security Considerations

- **Never** commit secrets, API keys, or credentials
- Validate all inputs in the server component
- Use `secrecy` crate for sensitive data if needed
- Run in trusted environments only (no built-in auth)
- Sanitize error messages to avoid information leakage

## Performance Goals

- Single embedding: < 50ms latency (model-dependent)
- Batch processing: > 1000 embeddings/second
- Memory efficiency: < 2x model size overhead
- Concurrent requests: Scale linearly with cores

### Optimization Checklist
- ✅ Profile with `cargo flamegraph`
- ✅ Minimize allocations in hot paths
- ✅ Use `Arc` for shared immutable data
- ✅ Prefer borrowing over cloning
- ✅ Use `SmallVec` for small collections
- ✅ Consider `parking_lot` for better mutex performance

## Documentation Standards

### Rustdoc Requirements
```rust
/// Generates embeddings for the given text.
///
/// # Arguments
/// * `model_name` - The name of the model to use
/// * `text` - The text to generate embeddings for
///
/// # Returns
/// A vector of floating-point embeddings
///
/// # Errors
/// Returns `EmbellamaError::ModelNotFound` if the model doesn't exist
///
/// # Example
/// ```
/// let embeddings = engine.embed("my-model", "Hello, world!")?;
/// ```
pub fn embed(&self, model_name: &str, text: &str) -> Result<Vec<f32>, EmbellamaError> {
    // Implementation
}
```

### Documentation Priorities
1. All public APIs must have rustdoc comments
2. Include usage examples for complex APIs
3. Document error conditions clearly
4. Keep internal documentation focused and technical
5. Update ARCHITECTURE.md for significant changes
6. Development setup and guidelines belong in DEVELOPMENT.md

## Development Workflow

**Note**: For detailed development setup, testing procedures, and `just` command reference, see [DEVELOPMENT.md](DEVELOPMENT.md).

### Quick Development Loop
```bash
just dev        # Run fix, fmt, clippy, and unit tests
just test       # Run all test suites
just pre-commit # Full validation before committing
```

### Standard Workflow
1. **Planning**: Use zen:thinkdeeper for design decisions
2. **Research**: Use Fetch/Brave for documentation lookup
3. **Implementation**: Follow Rust best practices
4. **Testing**:
   - Run `just test-unit` for quick feedback
   - Run `just test-integration` for real model testing
   - Run `just test-concurrency` for thread safety
5. **Debugging**: Use zen:debug for complex issues
6. **Review**: Run rust code review agent
7. **Pre-commit**: Run `just pre-commit`
8. **Commit**: Update changelog with git-cliff, write clear message
9. **Document**: Update rustdoc and ARCHITECTURE.md if needed

### Key Testing Philosophy
- **Test with real models** - No mocks for integration tests
- **Fail loudly** - Tests should clearly indicate what's wrong
- **Automated downloads** - Models are cached automatically via justfile
- **Fast feedback** - Use `just dev` for rapid iteration
