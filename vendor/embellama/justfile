# Embellama Build & Test Automation
# Run `just` to see available commands

# Default recipe - show help
default:
    @just --list

# Model cache directory
models_dir := "models"
test_model_dir := models_dir + "/test"
bench_model_dir := models_dir + "/bench"

# Platform-specific feature selection
# Use appropriate backend based on OS to avoid conflicts
platform_features := if os() == "macos" {
    "server,metal"
} else if os() == "linux" {
    "server,openmp"
} else {
    "server,openmp"
}

# Backend-only features (no server)
backend_features := if os() == "macos" {
    "metal"
} else if os() == "linux" {
    "openmp"
} else {
    "openmp"
}

# Model URLs
test_model_url := "https://huggingface.co/Jarbas/all-MiniLM-L6-v2-Q4_K_M-GGUF/resolve/main/all-minilm-l6-v2-q4_k_m.gguf"
bench_model_url := "https://huggingface.co/gaianet/jina-embeddings-v2-base-code-GGUF/resolve/main/jina-embeddings-v2-base-code-Q4_K_M.gguf"
decoder_test_model_url := "https://huggingface.co/jinaai/jina-code-embeddings-0.5b-GGUF/resolve/main/jina-code-embeddings-0.5b-IQ4_NL.gguf"

# Model filenames
test_model_file := test_model_dir + "/all-minilm-l6-v2-q4_k_m.gguf"
decoder_test_model_file := test_model_dir + "/jina-code-embeddings-0.5b-IQ4_NL.gguf"
bench_model_file := bench_model_dir + "/jina-embeddings-v2-base-code-Q4_K_M.gguf"

# Download small test model (MiniLM ~15MB) for integration tests
download-test-model:
    @echo "Setting up test model..."
    @mkdir -p {{test_model_dir}}
    @if [ ! -f {{test_model_file}} ]; then \
        echo "Downloading MiniLM-L6-v2 model (~15MB)..."; \
        curl -L --progress-bar -o {{test_model_file}} {{test_model_url}}; \
        echo "✓ Test model (MiniLM) downloaded successfully"; \
    else \
        echo "✓ Test model (MiniLM) already cached"; \
    fi

# Download decoder test model (Jina Code ~353MB) based on Qwen2.5 Coder for integration tests
download-decoder-model:
    @echo "Setting up decoder test model..."
    @mkdir -p {{test_model_dir}}
    @if [ ! -f {{decoder_test_model_file}} ]; then \
        echo "Downloading Jina Code decoder model (~353MB)..."; \
        curl -L --progress-bar -o {{decoder_test_model_file}} {{decoder_test_model_url}}; \
        echo "✓ Decoder test model (Jina Code) downloaded successfully"; \
    else \
        echo "✓ Decoder test model (Jina Code) already cached"; \
    fi

# Download benchmark model (Jina ~110MB) for performance testing
download-bench-model:
    @echo "Setting up benchmark model..."
    @mkdir -p {{bench_model_dir}}
    @if [ ! -f {{bench_model_file}} ]; then \
        echo "Downloading Jina Embeddings v2 Base Code model (~110MB)..."; \
        curl -L --progress-bar -o {{bench_model_file}} {{bench_model_url}}; \
        echo "✓ Benchmark model (Jina) downloaded successfully"; \
    else \
        echo "✓ Benchmark model (Jina) already cached"; \
    fi

# Download all models
download-all: download-test-model download-decoder-model download-bench-model
    @echo "✓ All models ready"

# Run unit tests (no models required)
test-unit:
    @echo "Running unit tests with backend features ({{backend_features}})..."
    cargo test --lib --features "{{backend_features}}"

# Run integration tests with encoder model (MiniLM)
test-integration-encoder: download-test-model
    @echo "Running integration tests with encoder model ({{backend_features}})..."
    RUST_BACKTRACE=1 RUST_LOG=DEBUG EMBELLAMA_TEST_MODEL={{test_model_file}} \
    cargo test --test integration_tests --features "{{backend_features}}" -- --nocapture

# Run integration tests with decoder model (Jina Code)
test-integration-decoder: download-decoder-model
    @echo "Running integration tests with decoder model ({{backend_features}})..."
    RUST_BACKTRACE=1 RUST_LOG=DEBUG EMBELLAMA_TEST_MODEL={{decoder_test_model_file}} \
    cargo test --test integration_tests --features "{{backend_features}}" -- --nocapture

# Run integration tests with both encoder and decoder models
test-integration: test-integration-encoder test-integration-decoder
    @echo "✓ Integration tests completed with both models"

# Run concurrency tests with encoder model (MiniLM)
test-concurrency-encoder: download-test-model
    @echo "Running concurrency tests with encoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{test_model_file}} \
    cargo test --test concurrency_tests --features "{{backend_features}}" -- --nocapture

# Run concurrency tests with decoder model (Jina Code)
test-concurrency-decoder: download-decoder-model
    @echo "Running concurrency tests with decoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{decoder_test_model_file}} \
    cargo test --test concurrency_tests --features "{{backend_features}}" -- --nocapture

# Run concurrency tests with both encoder and decoder models
test-concurrency: test-concurrency-encoder test-concurrency-decoder
    @echo "✓ Concurrency tests completed with both models"

# Run property-based tests with encoder model (MiniLM)
test-property-encoder: download-test-model
    @echo "Running property-based tests with encoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{test_model_file}} \
    cargo test --test property_tests --features "{{backend_features}}" -- --nocapture

# Run property-based tests with decoder model (Jina Code)
test-property-decoder: download-decoder-model
    @echo "Running property-based tests with decoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{decoder_test_model_file}} \
    EMBELLAMA_TEST_CONTEXT_SIZE=8192 \
    cargo test --test property_tests --features "{{backend_features}}" -- --nocapture

# Run property-based tests with both encoder and decoder models
test-property: test-property-quick-encoder test-property-quick-decoder
    @echo "✓ Property tests completed with both models"

# Run property-based tests with fewer cases (faster) - encoder model
test-property-quick-encoder: download-test-model
    @echo "Running property-based tests (quick mode) with encoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{test_model_file}} \
    PROPTEST_CASES=10 \
    cargo test --test property_tests --features "{{backend_features}}" -- --nocapture

# Run property-based tests with fewer cases (faster) - decoder model
test-property-quick-decoder: download-decoder-model
    @echo "Running property-based tests (quick mode) with decoder model ({{backend_features}})..."
    EMBELLAMA_TEST_MODEL={{decoder_test_model_file}} \
    EMBELLAMA_TEST_CONTEXT_SIZE=8192 \
    PROPTEST_CASES=10 \
    cargo test --test property_tests --features "{{backend_features}}" -- --nocapture

# Run property-based tests (quick mode) with both encoder and decoder models
test-property-quick: test-property-quick-encoder test-property-quick-decoder
    @echo "✓ Property tests (quick mode) completed with both models"

# Run all tests
test: test-unit test-integration test-concurrency test-property
    @echo "✓ All tests completed"

# Run benchmarks with real model
bench: download-bench-model
    @echo "Running benchmarks with real model..."
    EMBELLAMA_BENCH_MODEL={{bench_model_file}} \
    cargo bench

# Quick benchmark (subset of benchmarks for faster testing)
bench-quick: download-bench-model
    @echo "Running quick benchmarks (subset only)..."
    EMBELLAMA_BENCH_MODEL={{bench_model_file}} \
    cargo bench -- "single_embedding/text_length/11$|batch_embeddings/batch_size/1$|thread_scaling/threads/1$"

# Run example with model
example NAME="simple": download-test-model
    @echo "Running example: {{NAME}}..."
    EMBELLAMA_MODEL={{test_model_file}} \
    cargo run --example {{NAME}}

# Check for compilation warnings
check:
    @echo "Checking for warnings with platform features ({{platform_features}})..."
    cargo check --all-targets --features "{{platform_features}}"

# Check with all features (may fail due to backend conflicts)
check-all-features:
    @echo "Checking with all features (may fail on some platforms)..."
    cargo check --all-targets --all-features

# Fix common issues
fix:
    @echo "Running cargo fix..."
    cargo fix --all-targets --allow-dirty --allow-staged
    cargo clippy --fix --all-targets --allow-dirty --allow-staged

# Format code
fmt:
    @echo "Formatting code..."
    cargo fmt

# Check formatting for code
fmtcheck:
    @echo "Formatting code..."
    cargo fmt -- --check

# Check different backend features compilation
check-backends:
    @echo "Checking backend features compilation..."
    @echo "✓ Checking OpenMP backend..."
    @cargo check --no-default-features --features openmp
    @echo "✓ Checking Native backend..."
    @cargo check --no-default-features --features native
    @if [ "$(uname)" = "Darwin" ]; then \
        echo "✓ Checking Metal backend (macOS)..."; \
        cargo check --no-default-features --features metal; \
    fi
    @if [ "$(uname)" = "Linux" ]; then \
        echo "✓ Checking CUDA backend (compile only)..."; \
        cargo check --no-default-features --features cuda; \
        echo "✓ Checking Vulkan backend (compile only)..."; \
        cargo check --no-default-features --features vulkan; \
    fi
    @echo "✓ All backend features compile successfully"

# Test with platform-specific backend
test-backend: download-test-model
    @echo "Testing with platform-specific backend..."
    @if [ "$(uname)" = "Darwin" ]; then \
        echo "Testing with Metal backend on macOS..."; \
        EMBELLAMA_TEST_MODEL={{test_model_file}} \
        cargo test --features metal -- --nocapture; \
    elif [ "$(uname)" = "Linux" ]; then \
        echo "Testing with OpenMP backend on Linux..."; \
        EMBELLAMA_TEST_MODEL={{test_model_file}} \
        cargo test --features openmp -- --nocapture; \
    else \
        echo "Testing with OpenMP backend..."; \
        EMBELLAMA_TEST_MODEL={{test_model_file}} \
        cargo test --features openmp -- --nocapture; \
    fi

# Run clippy with platform-appropriate features
clippy:
    @echo "Running clippy with platform features ({{platform_features}})..."
    @echo "Library and binaries (strict)..."
    cargo clippy --lib --bins --features "{{platform_features}}" -- -D warnings -D clippy::pedantic
    @echo "Tests, examples, and benches (lenient)..."
    cargo clippy --tests --examples --benches --features "{{platform_features}}" -- -W warnings -W clippy::pedantic

# Run clippy with all features (may fail due to backend conflicts)
clippy-all-features:
    @echo "Running clippy with all features (may fail on some platforms)..."
    cargo clippy --lib --bins --all-features -- -D warnings -D clippy::pedantic
    cargo clippy --tests --examples --benches --all-features -- -W warnings -W clippy::pedantic

# Compile tests without running them (for pre-commit)
test-compile:
    @echo "Compiling tests with platform features ({{platform_features}})..."
    cargo test --no-run --lib --features "{{platform_features}}"

# Run doc tests (compiles README and doc examples)
test-doc:
    @echo "Running doc tests with platform features ({{platform_features}})..."
    cargo test --doc --features "{{platform_features}}"

# Build documentation with platform features
doc:
    @echo "Building documentation with platform features ({{platform_features}})..."
    cargo doc --no-deps --features "{{platform_features}}"

# Build documentation with all features (for docs.rs)
doc-all-features:
    @echo "Building documentation with all features..."
    cargo doc --no-deps --all-features

# Clean build artifacts
clean:
    @echo "Cleaning build artifacts..."
    cargo clean

# Clean downloaded models
clean-models:
    @echo "Removing cached models..."
    rm -rf {{models_dir}}

# Clean everything
clean-all: clean clean-models
    @echo "✓ All cleaned"

# Show model cache status
models-status:
    @echo "Model cache status:"
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @if [ -f {{test_model_file}} ]; then \
        size=`du -h {{test_model_file}} | cut -f1`; \
        echo "✓ Encoder test model: {{test_model_file}} ($$size)"; \
    else \
        echo "✗ Encoder test model: Not downloaded"; \
    fi
    @if [ -f {{decoder_test_model_file}} ]; then \
        size=`du -h {{decoder_test_model_file}} | cut -f1`; \
        echo "✓ Decoder test model: {{decoder_test_model_file}} ($$size)"; \
    else \
        echo "✗ Decoder test model: Not downloaded"; \
    fi
    @if [ -f {{bench_model_file}} ]; then \
        size=`du -h {{bench_model_file}} | cut -f1`; \
        echo "✓ Bench model: {{bench_model_file}} ($$size)"; \
    else \
        echo "✗ Bench model: Not downloaded"; \
    fi
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @if [ -d {{models_dir}} ]; then \
        total=`du -sh {{models_dir}} | cut -f1`; \
        echo "Total cache size: $$total"; \
    fi

# Development workflow - fix issues and test
dev: fix fmt clippy test-unit
    @echo "✓ Ready for integration testing"

# Install pre-commit hooks
install-hooks:
    @echo "Installing pre-commit hooks..."
    @if command -v uvx >/dev/null 2>&1; then \
        uvx pre-commit install; \
        echo "✓ Pre-commit hooks installed using uvx"; \
    elif command -v pipx >/dev/null 2>&1; then \
        pipx run pre-commit install; \
        echo "✓ Pre-commit hooks installed using pipx"; \
    elif command -v pre-commit >/dev/null 2>&1; then \
        pre-commit install; \
        echo "✓ Pre-commit hooks installed"; \
    else \
        echo "pre-commit not found. Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
        echo "Or install pipx with: pip install --user pipx"; \
        exit 1; \
    fi

# Run pre-commit hooks on all files
pre-commit-all:
    @echo "Running pre-commit on all files..."
    @if command -v uvx >/dev/null 2>&1; then \
        uvx pre-commit run --all-files; \
    elif command -v pipx >/dev/null 2>&1; then \
        pipx run pre-commit run --all-files; \
    elif command -v pre-commit >/dev/null 2>&1; then \
        pre-commit run --all-files; \
    else \
        echo "pre-commit not found. Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
        echo "Or install pipx with: pip install --user pipx"; \
        exit 1; \
    fi

# Run pre-commit hooks on staged files
pre-commit:
    @echo "Running pre-commit on staged files..."
    @if command -v uvx >/dev/null 2>&1; then \
        uvx pre-commit run; \
    elif command -v pipx >/dev/null 2>&1; then \
        pipx run pre-commit run; \
    elif command -v pre-commit >/dev/null 2>&1; then \
        pre-commit run; \
    else \
        echo "pre-commit not found. Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
        echo "Or install pipx with: pip install --user pipx"; \
        exit 1; \
    fi

# Update pre-commit hooks to latest versions
update-hooks:
    @echo "Updating pre-commit hooks..."
    @if command -v uvx >/dev/null 2>&1; then \
        uvx pre-commit autoupdate; \
        echo "✓ Pre-commit hooks updated using uvx"; \
    elif command -v pipx >/dev/null 2>&1; then \
        pipx run pre-commit autoupdate; \
        echo "✓ Pre-commit hooks updated using pipx"; \
    elif command -v pre-commit >/dev/null 2>&1; then \
        pre-commit autoupdate; \
        echo "✓ Pre-commit hooks updated"; \
    else \
        echo "pre-commit not found. Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
        echo "Or install pipx with: pip install --user pipx"; \
        exit 1; \
    fi

# Generate changelog (requires git-cliff: cargo install git-cliff)
changelog:
    @echo "Generating changelog..."
    git-cliff -o CHANGELOG.md
    @echo "✓ CHANGELOG.md updated"

# Generate changelog for unreleased changes only
changelog-unreleased:
    @echo "Unreleased changes:"
    @git-cliff --unreleased

# Full CI simulation
ci: clean check clippy test bench
    @echo "✓ CI checks completed successfully"

# Build the server binary
build-server:
    @echo "Building server binary..."
    cargo build --features server --bin embellama-server
    @echo "✓ Server binary built"

# Run the server with test model
run-server: download-test-model build-server
    @echo "Starting server with test model..."
    cargo run --features server --bin embellama-server -- \
        --model-path {{test_model_file}} \
        --model-name test-model \
        --workers 2 \
        --log-level info

# Run server in background for testing (returns immediately)
start-server: download-test-model build-server
    @echo "Starting server in background..."
    @cargo run --features server --bin embellama-server -- \
        --model-path {{test_model_file}} \
        --model-name test-model \
        --workers 2 \
        --log-level info > server.log 2>&1 & \
        echo $$! > server.pid
    @sleep 2
    @echo "✓ Server started (PID: `cat server.pid`)"

# Stop the background server
stop-server:
    @if [ -f server.pid ]; then \
        echo "Stopping server (PID: `cat server.pid`)..."; \
        kill `cat server.pid` 2>/dev/null || true; \
        rm -f server.pid; \
        echo "✓ Server stopped"; \
    else \
        echo "No server running"; \
    fi

# Test server API endpoints
test-server-api:
    @echo "Testing server API endpoints..."
    @echo "================================"
    @echo
    @echo "1. Testing /health endpoint:"
    @curl -s "http://localhost:8080/health" | jq . || echo "Failed - is server running?"
    @echo
    @echo "2. Testing /v1/models endpoint:"
    @curl -s "http://localhost:8080/v1/models" | jq . || echo "Failed - is server running?"
    @echo
    @echo "3. Testing /v1/embeddings with single text:"
    @curl -s -X POST "http://localhost:8080/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"model": "test-model", "input": "Hello, world!"}' \
        | jq '.object, .model, .usage' || echo "Failed - is server running?"
    @echo
    @echo "4. Testing /v1/embeddings with batch:"
    @curl -s -X POST "http://localhost:8080/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d '{"model": "test-model", "input": ["Hello", "World"]}' \
        | jq '.object, .model, (.data | length)' || echo "Failed - is server running?"
    @echo
    @echo "================================"
    @echo "✓ API tests complete"

# Run server integration tests
test-server-integration: download-test-model
    @echo "Running server integration tests..."
    cargo test --features server --test server_api_tests -- --nocapture

# Run OpenAI compatibility tests
test-server-compat: download-test-model
    @echo "Running OpenAI compatibility tests..."
    cargo test --features server --test openai_compat_tests -- --nocapture

# Run server load tests (excluding slow tests)
test-server-load: download-test-model
    @echo "Running server load tests..."
    cargo test --features server --test server_load_tests -- --nocapture

# Run ALL server load tests (including slow/ignored tests)
test-server-load-all: download-test-model
    @echo "Running all server load tests (including slow tests)..."
    cargo test --features server --test server_load_tests -- --nocapture --ignored

# Run all server tests
test-server-all: test-server-integration test-server-compat test-server-load
    @echo "✓ All server tests completed"

# Test with Python OpenAI SDK
test-server-python: start-server
    @echo "Testing with Python OpenAI SDK..."
    @python3 scripts/test-openai-python.py || echo "Python SDK test failed - is openai package installed?"
    @just stop-server

# Test with JavaScript OpenAI SDK
test-server-js: start-server
    @echo "Testing with JavaScript OpenAI SDK..."
    @node scripts/test-openai-js.mjs || echo "JS SDK test failed - is openai package installed?"
    @just stop-server

# Full server test workflow (old compatibility)
test-server: start-server test-server-api stop-server
    @echo "✓ Server tests completed"

# Check server compilation
check-server:
    @echo "Checking server compilation..."
    cargo check --features server --bin embellama-server
    cargo check --features server --tests

# Clean server artifacts
clean-server: stop-server
    @echo "Cleaning server artifacts..."
    @rm -f server.log server.pid
    @echo "✓ Server artifacts cleaned"

# Create a new release
release VERSION:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "🚀 Creating release v{{VERSION}}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check prerequisites
    echo "📋 Checking prerequisites..."

    # Check if git-cliff is installed
    if ! command -v git-cliff &> /dev/null; then
        echo "❌ git-cliff is not installed. Install with: cargo install git-cliff"
        exit 1
    fi

    # Check if gh CLI is installed
    if ! command -v gh &> /dev/null; then
        echo "❌ GitHub CLI (gh) is not installed. Install from: https://cli.github.com"
        exit 1
    fi

    # Ensure working directory is clean
    if [ -n "$(git status --porcelain)" ]; then
        echo "❌ Working directory is not clean. Please commit or stash changes."
        exit 1
    fi

    # Get the default branch (usually main or master)
    default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "master")
    current_branch=$(git branch --show-current)

    # Check we're on the default branch
    if [ "$current_branch" != "$default_branch" ]; then
        echo "❌ Not on $default_branch branch (currently on $current_branch)"
        echo "   Run: git checkout $default_branch"
        exit 1
    fi

    # Fetch latest from remote
    echo "📡 Fetching latest from remote..."
    git fetch origin

    # Check we're up to date with remote
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u})
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "❌ Branch is not up to date with remote"
        echo "   Run: git pull origin $default_branch"
        exit 1
    fi

    # Check if tag already exists
    if git rev-parse "v{{VERSION}}" >/dev/null 2>&1; then
        echo "❌ Tag v{{VERSION}} already exists"
        exit 1
    fi

    echo "✅ All prerequisites passed"
    echo

    # Run quality checks
    echo "🧪 Running quality checks..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Format check
    echo "📝 Checking code formatting..."
    just fmtcheck

    # Clippy
    echo "🔍 Running clippy..."
    just clippy

    # Tests
    echo "🧪 Running tests..."
    just test-unit

    # Doc tests (README examples)
    echo "📖 Running doc tests..."
    just test-doc

    echo "✅ All quality checks passed"
    echo

    # Update version
    echo "📦 Updating version to {{VERSION}}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Update version in Cargo.toml
    sed -i.bak 's/^version = ".*"/version = "{{VERSION}}"/' Cargo.toml
    rm Cargo.toml.bak

    # Update version in README.md (use -E for extended regex, portable across macOS/Linux)
    sed -i.bak -E 's/embellama = "[0-9]+\.[0-9]+\.[0-9]+"/embellama = "{{VERSION}}"/g' README.md
    sed -i.bak -E 's/(version = ")[0-9]+\.[0-9]+\.[0-9]+(")/\1{{VERSION}}\2/g' README.md
    rm -f README.md.bak

    # Verify README was actually updated
    if grep -q 'embellama = "{{VERSION}}"' README.md; then
        echo "✅ README.md version updated to {{VERSION}}"
    else
        echo "❌ Failed to update version in README.md"
        exit 1
    fi

    # Update Cargo.lock
    cargo update -p embellama

    # Generate changelog
    echo "📋 Generating changelog..."
    git-cliff --tag v{{VERSION}} -o CHANGELOG.md

    # Commit release changes
    echo "💾 Committing release changes..."
    git add Cargo.toml Cargo.lock CHANGELOG.md README.md
    git commit -m "chore(release): prepare for v{{VERSION}}"

    # Create git tag
    echo "🏷️  Creating git tag v{{VERSION}}..."
    git tag -a v{{VERSION}} -m "Release v{{VERSION}}"

    # Push to GitHub
    echo "🚀 Pushing to GitHub..."
    git push origin $default_branch --tags

    # Extract release notes for this version from CHANGELOG.md
    echo "📝 Extracting release notes..."

    # Get content between this version and the next version header
    release_notes=$(awk '/^## \[{{VERSION}}\]/,/^## \[/' CHANGELOG.md | sed '$ d' | tail -n +3)

    # Create GitHub release
    echo "🎉 Creating GitHub release..."
    gh release create v{{VERSION}} \
        --title "v{{VERSION}}" \
        --notes "$release_notes"

    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Release v{{VERSION}} created successfully!"
    echo
    echo "📦 GitHub Release: https://github.com/darjus/embellama/releases/tag/v{{VERSION}}"
    echo
    echo "📚 To publish to crates.io, run:"
    echo "   cargo publish --dry-run  # Test first"
    echo "   cargo publish            # Publish for real"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
