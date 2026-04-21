# Contributing to Embellama

Thank you for your interest in contributing to Embellama! This guide details how to contribute to the project in a variety of ways.

## How to Contribute

We welcome many forms of contributions, including code, documentation, bug reports, and feature requests.

### Reporting Bugs

High-quality bug reports are valuable. If you encounter a bug, please open an issue on our GitHub issue tracker. Before submitting, please search existing issues to see if it has already been reported.

When filing a bug report, please provide:

- A clear and descriptive title.
- A detailed description of the problem, including what you expected to happen.
- The simplest possible steps to reproduce the bug.
- Your system environment (OS, Rust version).
- Any relevant logs or error messages.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement, we'd love to hear it. Please open an issue to start a discussion.

In your proposal, please include:

- A clear description of the proposed enhancement.
- The problem or use case it solves ("Why is this needed?").
- Any alternative solutions or features you've considered.
- (Optional) A sketch of the proposed API or user interface.

### Contributing Code

If you're ready to contribute code, please follow the setup and workflow guides below.

## Setting Up Your Development Environment

### Prerequisites

1. **Rust**: Install via [rustup](https://rustup.rs/).
2. **uv** or **pipx**: For isolated Python tools (recommended).
   ```bash
   # Install uv (preferred)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # OR install pipx
   pip install --user pipx
   ```
3. **Just**: A command runner for development tasks.
   ```bash
   cargo install just
   ```

### Initial Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/embellama.git
   cd embellama
   ```

2. Install pre-commit hooks:
   ```bash
   # Install the git hooks (uses uvx/pipx automatically)
   just install-hooks
   ```
   This uses `uvx` (or `pipx` as a fallback) to run pre-commit in an isolated environment, avoiding global Python package pollution.

3. Download test models (optional, required for running all tests):
   ```bash
   just download-test-model
   ```

## Development Workflow

### Available Commands

Run `just` to see all available commands. Key commands include:

- `just fmt`: Format code with `rustfmt`.
- `just clippy`: Run clippy lints.
- `just test`: Run all tests.
- `just check`: Check for compilation errors.
- `just dev`: Run fix, fmt, clippy, and unit tests.
- `just pre-commit`: Run pre-commit hooks on staged files.
- `just pre-commit-all`: Run pre-commit hooks on all files.

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. The hooks will automatically run when you commit, checking for:

- Code formatting (`rustfmt`)
- Linting (`clippy`)
- Compilation checks
- File hygiene (trailing whitespace, file endings, etc.)

If a hook fails, your commit will be blocked. Fix the reported issues and try committing again.

To manually run hooks:
```bash
# On staged files only
just pre-commit

# On all files
just pre-commit-all
```

To skip hooks temporarily (use with caution):
```bash
git commit --no-verify -m "your message"
```

### Code Style

- Follow standard Rust naming conventions.
- Use `rustfmt` for formatting (enforced by pre-commit).
- Address all `clippy` warnings (pedantic level is enforced).
- Write tests for new functionality.
- Document all public APIs.

### Testing

Run tests before submitting a pull request:

```bash
# Unit tests (fast, no model required)
just test-unit

# Integration tests (requires test model)
just test-integration

# All tests
just test
```

### Benchmarking

To run performance benchmarks:

```bash
# Download benchmark model first
just download-bench-model

# Run benchmarks
just bench

# Quick benchmark subset
just bench-quick
```

## Submitting Your Changes

### Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Make your changes.
3. Ensure all tests pass (`just test`).
4. Ensure pre-commit hooks pass on all files (`just pre-commit-all`).
5. Commit your changes, following the guidelines below.
6. Push your branch to your fork.
7. Open a Pull Request against the `main` branch of the original repository.

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. The format is:

```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

**Common Types:**
- `feat`: A new feature.
- `fix`: A bug fix.
- `docs`: Documentation changes.
- `style`: Code style changes (formatting, etc.).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `test`: Adding missing tests or correcting existing ones.
- `chore`: Changes to the build process or auxiliary tools.
- `perf`: A code change that improves performance.

## Communication

- **Discussions**: For questions and general discussion, please use GitHub Discussions.
- **Issues**: For bug reports and feature requests, use GitHub Issues.
- Please check existing discussions and issues before creating a new one.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
