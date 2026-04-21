# Development Guide

Welcome to the Embellama development guide! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Documentation](#documentation)
- [Debugging](#debugging)
- [Contributing](#contributing)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Rust 1.70.0 or later (MSRV)
- CMake (for building llama.cpp)
- A C++ compiler (gcc, clang, or MSVC)
- Git

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/embellama/embellama.git
cd embellama
```

2. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. Add required components:
```bash
rustup component add rustfmt clippy
```

4. Build the project:
```bash
cargo build
```

5. Run tests:
```bash
cargo test
```

## Development Setup

### IDE Configuration

#### VS Code

Install the following extensions:
- rust-analyzer
- CodeLLDB (for debugging)
- Even Better TOML

Recommended settings (`.vscode/settings.json`):
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

#### IntelliJ IDEA / CLion

Install the Rust plugin and configure it to use `cargo clippy` for on-save checks.

### Development Commands with `just`

This project uses [just](https://github.com/casey/just) for task automation.

#### Available Commands

```bash
just               # Show all available commands
just test          # Run all tests (unit + integration + concurrency)
just test-unit     # Run unit tests only
just test-integration # Run integration tests with real model
just test-concurrency # Run concurrency tests
just bench         # Run full benchmarks
just bench-quick   # Run quick benchmark subset
just dev           # Run fix, fmt, clippy, and unit tests
just pre-commit    # Run all checks before committing
just clean-all     # Clean build artifacts and models
```

#### Model Management

Test models are automatically downloaded and cached:
- **Test model** (MiniLM-L6-v2): ~15MB, for integration tests
- **Benchmark model** (Jina Embeddings v2): ~110MB, for performance testing

```bash
just download-test-model   # Download test model
just download-bench-model  # Download benchmark model
just models-status        # Check cached models
```

#### Environment Variables

- `EMBELLAMA_TEST_MODEL`: Path to test model (auto-set by justfile)
- `EMBELLAMA_BENCH_MODEL`: Path to benchmark model (auto-set by justfile)
- `EMBELLAMA_MODEL`: Path to model for examples

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
# Create the hooks directory
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
set -e

# Format code
cargo fmt -- --check

# Run clippy
cargo clippy --all-features -- -D warnings

# Run tests
cargo test --quiet

echo "Pre-commit checks passed!"
EOF

# Make it executable
chmod +x .git/hooks/pre-commit
```

## Code Style

We use `rustfmt` for automatic code formatting and `clippy` for linting.

### Formatting

Always format your code before committing:
```bash
cargo fmt
```

Check formatting without modifying files:
```bash
cargo fmt -- --check
```

### Linting

Run clippy with all features enabled:
```bash
cargo clippy --all-features -- -D warnings
```

### Best Practices

1. **Error Handling**: Use the custom `Error` type from `error.rs` for all error handling
2. **Documentation**: All public APIs must have documentation comments
3. **Tests**: Write unit tests for all new functionality
4. **Safety**: Minimize use of `unsafe` code; document safety invariants when necessary
5. **Performance**: Profile before optimizing; use benchmarks to validate improvements

### Naming Conventions

- **Modules**: snake_case (e.g., `embedding_engine`)
- **Types**: PascalCase (e.g., `EmbeddingEngine`)
- **Functions/Methods**: snake_case (e.g., `generate_embedding`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)
- **Type Parameters**: Single capital letter or PascalCase (e.g., `T` or `ModelType`)

## Testing

### Running Tests

Run all tests:
```bash
cargo test
```

Run tests with output:
```bash
cargo test -- --nocapture
```

Run specific test:
```bash
cargo test test_name
```

Run tests in release mode:
```bash
cargo test --release
```

### Test Organization

- Unit tests: In the same file as the code being tested, in a `#[cfg(test)]` module
- Integration tests: In the `tests/` directory
- Doc tests: In documentation comments using ` ```rust ` blocks

### Test Models

For integration tests, you'll need GGUF model files. The project includes comprehensive test suites:

#### Unit Tests
```bash
just test-unit
```

#### Integration Tests
Tests with real GGUF models (downloads MiniLM automatically):
```bash
just test-integration
```

#### Concurrency Tests
Tests thread safety and parallel processing:
```bash
just test-concurrency
```

#### Testing Considerations

**Important**: Integration tests use the `serial_test` crate to ensure tests run sequentially. This is necessary because:
- The `LlamaBackend` can only be initialized once per process
- Each `EmbeddingEngine` owns its backend instance
- Tests must run serially to avoid backend initialization conflicts

When writing tests that create multiple engines, use a single engine with `load_model()` for different configurations:

```rust
#[test]
#[serial]  // Required for all integration tests
fn test_multiple_configurations() {
    let mut engine = EmbeddingEngine::new(initial_config)?;

    // Load additional models instead of creating new engines
    engine.load_model(config2)?;
    engine.load_model(config3)?;
}
```

### Writing Tests

Example unit test:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // Arrange
        let input = "test";

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

## Benchmarking

### Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark:
```bash
cargo bench -- benchmark_name
```

### Writing Benchmarks

Benchmarks are located in `benches/` and use the `criterion` crate:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_embedding(c: &mut Criterion) {
    c.bench_function("single_embedding", |b| {
        b.iter(|| {
            // Code to benchmark
            generate_embedding(black_box("test input"))
        });
    });
}

criterion_group!(benches, benchmark_embedding);
criterion_main!(benches);
```

### Performance Profiling

Use `cargo-flamegraph` for performance profiling:

```bash
cargo install flamegraph
cargo flamegraph --bench embeddings
```

## Documentation

### Building Documentation

Build and open documentation:
```bash
cargo doc --open
```

Build documentation for all dependencies:
```bash
cargo doc --open --all-features
```

### Writing Documentation

All public items must have documentation:

```rust
/// Brief description of what this does.
///
/// # Arguments
///
/// * `input` - Description of the input parameter
///
/// # Returns
///
/// Description of the return value
///
/// # Examples
///
/// ```rust
/// let result = function(input);
/// assert_eq!(result, expected);
/// ```
///
/// # Errors
///
/// Returns `Error::InvalidInput` if the input is invalid
pub fn function(input: &str) -> Result<String> {
    // Implementation
}
```

## Debugging

### Logging

Use the `tracing` crate for logging:

```rust
use tracing::{debug, info, warn, error};

info!("Loading model from {}", path.display());
debug!("Model metadata: {:?}", metadata);
warn!("Using fallback configuration");
error!("Failed to load model: {}", err);
```

Enable debug logging:
```bash
RUST_LOG=debug cargo run
```

Enable trace-level logging for specific modules:
```bash
RUST_LOG=embellama::engine=trace cargo run
```

### Using the Debugger

#### VS Code with CodeLLDB

1. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug unit tests",
            "cargo": {
                "args": ["test", "--no-run"],
                "filter": {
                    "name": "embellama",
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

2. Set breakpoints and press F5 to debug

#### Command Line with rust-gdb

```bash
cargo build
rust-gdb target/debug/embellama-server
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `cargo test`
6. Format your code: `cargo fmt`
7. Run clippy: `cargo clippy -- -D warnings`
8. Update documentation as needed
9. Commit with a descriptive message
10. Push to your fork
11. Create a pull request

### Commit Message Format

Follow the conventional commits specification:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `build`: Build system changes
- `ci`: CI configuration changes
- `chore`: Other changes

### Code Review Guidelines

- Be constructive and respectful
- Explain the "why" behind suggestions
- Consider performance implications
- Verify test coverage
- Check for breaking changes

## Release Process

### Version Bumping

Update version in `Cargo.toml`:
```toml
[package]
version = "0.2.0"
```

### Creating a Release

The project uses [git-cliff](https://git-cliff.org/) for automated changelog generation from conventional commits. Install with `cargo install git-cliff`.

The recommended way to create a release is via the justfile:

```bash
just release 0.9.0
```

This automates: version bumping, changelog generation, tagging, pushing, and creating a GitHub release.

Manual steps (if needed):

1. Generate changelog: `git-cliff --tag v0.9.0 -o CHANGELOG.md`
2. Bump version in `Cargo.toml`
3. Create a git tag: `git tag -a v0.9.0 -m "Release v0.9.0"`
4. Push tags: `git push origin --tags`
5. Create GitHub release
6. Publish to crates.io: `cargo publish`

### Checking before Publishing

Dry run to verify:
```bash
cargo publish --dry-run
```

Check package contents:
```bash
cargo package --list
```

## Architecture

### Backend and Engine Management

The library manages the LlamaBackend lifecycle:

- Each `EmbeddingEngine` owns its `LlamaBackend` instance
- Backend is initialized when the engine is created
- Backend is dropped when the engine is dropped
- Singleton pattern available for shared engine access

### Model Management

The library uses a thread-local architecture due to llama-cpp's `!Send` constraint:

- Each thread maintains its own model instance
- Models cannot be shared between threads
- Use message passing for concurrent operations

### Batch Processing Pipeline

1. **Parallel Pre-processing**: Tokenization in parallel using Rayon
2. **Sequential Inference**: Model inference on single thread
3. **Parallel Post-processing**: Normalization and formatting in parallel

## Error Handling

The library provides comprehensive error handling:

```rust
use embellama::Error;

match engine.embed(None, text) {
    Ok(embedding) => process_embedding(embedding),
    Err(Error::ModelNotFound { name }) => {
        println!("Model {} not found", name);
    }
    Err(Error::InvalidInput { message }) => {
        println!("Invalid input: {}", message);
    }
    Err(e) if e.is_retryable() => {
        // Retry logic for transient errors
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Troubleshooting

### Common Issues

#### llama-cpp-2 build failures

Ensure you have CMake and a C++ compiler installed:
```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential

# macOS
brew install cmake

# Windows
# Install Visual Studio with C++ development tools
```

#### Out of memory during model loading

Reduce the context size or use a smaller model:
```rust
let model_config = ModelConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_context_size(512)  // Smaller context
    .with_use_mmap(true)     // Enable memory mapping
    .build()?;

let config = EngineConfig::builder()
    .with_model_config(model_config)
    .build()?;
```

#### Slow inference on CPU

Enable multi-threading and optimize thread count:
```rust
let model_config = ModelConfig::builder()
    .with_model_path("/path/to/model.gguf")
    .with_n_threads(num_cpus::get())
    .build()?;

let config = EngineConfig::builder()
    .with_model_config(model_config)
    .build()?;
```

### Getting Help

- Check existing [GitHub Issues](https://github.com/embellama/embellama/issues)
- Join our [Discord server](https://discord.gg/embellama)
- Read the [API documentation](https://docs.rs/embellama)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

By contributing to Embellama, you agree that your contributions will be licensed under the Apache License 2.0.
