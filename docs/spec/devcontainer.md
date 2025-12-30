# Development Container Specification

**Status:** Draft
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

This specification defines the VS Code DevContainer configuration for Hologram development. It provides a consistent, reproducible development environment across all platforms.

## DevContainer Philosophy

- **Reproducible** - Same environment for all developers
- **Complete** - All tools pre-installed and configured
- **Fast** - Quick startup with cached layers
- **Isolated** - No conflicts with host system
- **GPU-Ready** - Optional GPU support for backend development

## DevContainer Configuration

### File Structure

```
.devcontainer/
├── devcontainer.json        # Main configuration
├── Dockerfile               # Custom image (optional)
├── docker-compose.yml       # Multi-container setup (optional)
└── post-create.sh          # Post-creation script
```

### devcontainer.json

**File:** `.devcontainer/devcontainer.json`

```json
{
  "name": "Hologram Development",
  "image": "mcr.microsoft.com/devcontainers/rust:1-bookworm",

  "features": {
    "ghcr.io/devcontainers/features/common-utils:2": {
      "installZsh": true,
      "installOhMyZsh": true,
      "upgradePackages": true
    },
    "ghcr.io/devcontainers/features/git:1": {
      "version": "latest",
      "ppa": true
    },
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/node:1": {
      "version": "lts",
      "nodeGypDependencies": true
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12",
      "installTools": true
    }
  },

  "customizations": {
    "vscode": {
      "settings": {
        // Rust settings
        "rust-analyzer.checkOnSave.command": "clippy",
        "rust-analyzer.checkOnSave.allTargets": true,
        "rust-analyzer.cargo.allFeatures": true,
        "rust-analyzer.procMacro.enable": true,
        "rust-analyzer.inlayHints.enable": true,

        // Editor settings
        "editor.formatOnSave": true,
        "editor.rulers": [100],
        "files.trimTrailingWhitespace": true,
        "files.insertFinalNewline": true,

        // Terminal settings
        "terminal.integrated.defaultProfile.linux": "zsh",

        // Git settings
        "git.autofetch": true,
        "git.confirmSync": false,

        // Testing settings
        "rust-analyzer.runnables.extraArgs": ["--", "--show-output"]
      },

      "extensions": [
        // Rust development
        "rust-lang.rust-analyzer",
        "vadimcn.vscode-lldb",
        "serayuzgur.crates",

        // TOML support
        "tamasfe.even-better-toml",

        // Git
        "eamodio.gitlens",

        // Testing
        "hbenl.vscode-test-explorer",
        "swellaby.vscode-rust-test-adapter",

        // Markdown
        "yzhang.markdown-all-in-one",
        "DavidAnson.vscode-markdownlint",

        // General utilities
        "streetsidesoftware.code-spell-checker",
        "EditorConfig.EditorConfig",
        "GitHub.copilot"
      ]
    }
  },

  "postCreateCommand": "bash .devcontainer/post-create.sh",

  "remoteUser": "vscode",

  "mounts": [
    // Cache cargo registry
    "source=hologram-cargo-registry,target=/usr/local/cargo/registry,type=volume",
    // Cache cargo git
    "source=hologram-cargo-git,target=/usr/local/cargo/git,type=volume",
    // Cache target directory
    "source=hologram-target,target=${containerWorkspaceFolder}/target,type=volume"
  ],

  "runArgs": [
    "--cap-add=SYS_PTRACE",
    "--security-opt",
    "seccomp=unconfined"
  ],

  "containerEnv": {
    "RUST_BACKTRACE": "1",
    "CARGO_INCREMENTAL": "1",
    "CARGO_TARGET_DIR": "/workspace/target"
  }
}
```

### GPU-Enabled DevContainer

**File:** `.devcontainer/devcontainer-gpu.json`

For CUDA/GPU backend development:

```json
{
  "name": "Hologram Development (GPU)",
  "build": {
    "dockerfile": "Dockerfile.gpu",
    "context": ".."
  },

  "runArgs": [
    "--gpus", "all",
    "--cap-add=SYS_PTRACE",
    "--security-opt", "seccomp=unconfined"
  ],

  "containerEnv": {
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
  },

  // ... rest of configuration same as devcontainer.json
}
```

### GPU Dockerfile

**File:** `.devcontainer/Dockerfile.gpu`

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3 \
    python3-pip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust components
RUN rustup component add rustfmt clippy rust-analyzer

# Install cargo tools
RUN cargo install cargo-edit cargo-watch cargo-tarpaulin

WORKDIR /workspace
```

### Post-Create Script

**File:** `.devcontainer/post-create.sh`

```bash
#!/bin/bash
set -e

echo "🚀 Setting up Hologram development environment..."

# Install Rust components
echo "📦 Installing Rust components..."
rustup component add rustfmt clippy rust-analyzer
rustup target add wasm32-unknown-unknown

# Install cargo tools
echo "🔧 Installing cargo tools..."
cargo install cargo-edit --quiet
cargo install cargo-watch --quiet
cargo install cargo-tarpaulin --quiet
cargo install cargo-benchcmp --quiet
cargo install uniffi-bindgen --quiet

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --user --upgrade \
    pytest \
    black \
    mypy \
    ruff

# Setup git hooks
echo "🪝 Setting up git hooks..."
if [ -d ".githooks" ]; then
    git config core.hooksPath .githooks
    chmod +x .githooks/*
fi

# Build project to populate cache
echo "🏗️  Building project (this may take a while)..."
cargo build --workspace --all-features

# Run tests to verify setup
echo "🧪 Running tests..."
cargo test --workspace --all-features -- --test-threads=1

echo "✅ Development environment ready!"
echo ""
echo "Quick start:"
echo "  cargo build          # Build the project"
echo "  cargo test           # Run tests"
echo "  cargo clippy         # Run linter"
echo "  cargo fmt            # Format code"
echo "  cargo run --example basic_operations  # Run example"
```

## Tool Configuration

### Rust Toolchain

**File:** `rust-toolchain.toml`

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy", "rust-analyzer"]
targets = ["wasm32-unknown-unknown"]
profile = "default"
```

### Editor Config

**File:** `.editorconfig`

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.rs]
indent_style = space
indent_size = 4
max_line_length = 100

[*.toml]
indent_style = space
indent_size = 2

[*.md]
indent_style = space
indent_size = 2
trim_trailing_whitespace = false

[*.json]
indent_style = space
indent_size = 2

[*.yml]
indent_style = space
indent_size = 2
```

### VS Code Workspace Settings

**File:** `.vscode/settings.json`

```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.allTargets": true,
  "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"],
  "rust-analyzer.cargo.allFeatures": true,
  "rust-analyzer.procMacro.enable": true,
  "rust-analyzer.cargo.buildScripts.enable": true,

  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "editor.codeActionsOnSave": {
    "source.fixAll": true,
    "source.organizeImports": true
  },

  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.watcherExclude": {
    "**/target/**": true
  },

  "search.exclude": {
    "**/target": true,
    "**/Cargo.lock": true
  },

  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.formatOnSave": true
  },

  "[toml]": {
    "editor.defaultFormatter": "tamasfe.even-better-toml"
  },

  "[markdown]": {
    "editor.defaultFormatter": "yzhang.markdown-all-in-one",
    "files.trimTrailingWhitespace": false
  }
}
```

### VS Code Tasks

**File:** `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "cargo build",
      "type": "shell",
      "command": "cargo build --workspace --all-features",
      "group": "build",
      "problemMatcher": ["$rustc"]
    },
    {
      "label": "cargo test",
      "type": "shell",
      "command": "cargo test --workspace --all-features",
      "group": "test",
      "problemMatcher": ["$rustc"]
    },
    {
      "label": "cargo clippy",
      "type": "shell",
      "command": "cargo clippy --workspace --all-targets --all-features -- -D warnings",
      "group": "test",
      "problemMatcher": ["$rustc"]
    },
    {
      "label": "cargo fmt",
      "type": "shell",
      "command": "cargo fmt --all",
      "group": "build"
    },
    {
      "label": "cargo check",
      "type": "shell",
      "command": "cargo check --workspace --all-features",
      "group": "build",
      "problemMatcher": ["$rustc"],
      "isBackground": false
    },
    {
      "label": "cargo run example",
      "type": "shell",
      "command": "cargo run --example ${input:exampleName}",
      "group": "test"
    }
  ],
  "inputs": [
    {
      "id": "exampleName",
      "type": "promptString",
      "description": "Example name",
      "default": "basic_operations"
    }
  ]
}
```

### VS Code Launch Configuration

**File:** `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug unit tests in library",
      "cargo": {
        "args": [
          "test",
          "--no-run",
          "--lib",
          "--package=hologram-core"
        ],
        "filter": {
          "name": "hologram-core",
          "kind": "lib"
        }
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    },
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug example",
      "cargo": {
        "args": [
          "build",
          "--example=${input:exampleName}"
        ]
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    },
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug binary",
      "cargo": {
        "args": [
          "build",
          "--bin=hologram-compile"
        ]
      },
      "args": [],
      "cwd": "${workspaceFolder}"
    }
  ],
  "inputs": [
    {
      "id": "exampleName",
      "type": "promptString",
      "description": "Example name",
      "default": "basic_operations"
    }
  ]
}
```

## Environment Variables

### Development .env

**File:** `.env.development`

```bash
# Backend configuration
HOLOGRAM_BACKEND=cpu
HOLOGRAM_DEVICE_ID=0

# Memory configuration
HOLOGRAM_POOL_SIZE=1073741824  # 1 GB
HOLOGRAM_MAX_BUFFERS=1024

# Logging
RUST_LOG=hologram=debug,hologram_core=trace
RUST_BACKTRACE=1

# Performance
HOLOGRAM_ENABLE_PROFILING=true
HOLOGRAM_CACHE_SIZE=536870912  # 512 MB

# Testing
HOLOGRAM_TEST_TIMEOUT=30
```

## Docker Compose (Multi-Container Setup)

**File:** `.devcontainer/docker-compose.yml`

For complex setups with databases, Redis, etc.:

```yaml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    volumes:
      - ..:/workspace:cached
      - hologram-cargo-registry:/usr/local/cargo/registry
      - hologram-cargo-git:/usr/local/cargo/git
      - hologram-target:/workspace/target
    command: sleep infinity
    environment:
      - RUST_BACKTRACE=1
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp:unconfined

volumes:
  hologram-cargo-registry:
  hologram-cargo-git:
  hologram-target:
```

## Performance Optimization

### Volume Caching

**Benefits:**
- Cargo registry cached: 5× faster dependency downloads
- Target directory cached: 10× faster rebuilds
- Git cache: Faster git operations

### Build Cache Size

**Monitoring:**
```bash
# Check cache size
docker system df -v

# Clean old caches
docker volume prune
```

### Recommended Docker Settings

**Docker Desktop:**
- Memory: 8 GB minimum (16 GB recommended)
- CPUs: 4 minimum (8 recommended)
- Disk: 100 GB minimum

## Common Tasks

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/hologram.git
cd hologram

# Open in VS Code
code .

# Reopen in container
# Command Palette (Ctrl+Shift+P) → "Dev Containers: Reopen in Container"
```

### Daily Development

```bash
# Build project
cargo build --workspace

# Run tests
cargo test --workspace

# Run specific test
cargo test test_name

# Watch mode (auto-rebuild on changes)
cargo watch -x build

# Format code
cargo fmt --all

# Check for errors
cargo clippy --workspace -- -D warnings

# Run example
cargo run --example basic_operations
```

### Debugging

```bash
# Debug with lldb
# Set breakpoint in VS Code, then F5 to start debugging

# Run with backtrace
RUST_BACKTRACE=1 cargo test failing_test

# Run with logging
RUST_LOG=trace cargo test
```

## Troubleshooting

### DevContainer Won't Start

**Problem:** Container fails to start

**Solutions:**
```bash
# Rebuild container
Command Palette → "Dev Containers: Rebuild Container"

# Clear Docker cache
docker system prune -a --volumes

# Check Docker resources (increase memory/CPU if needed)
```

### Slow Builds

**Problem:** Rebuilds are very slow

**Solutions:**
```bash
# Enable sccache
cargo install sccache
export RUSTC_WRAPPER=sccache

# Use cargo-chef for Docker layer caching
# Add to Dockerfile:
FROM rust:latest as planner
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json
```

### Rust Analyzer Issues

**Problem:** Rust analyzer not working

**Solutions:**
```bash
# Restart rust-analyzer
Command Palette → "Rust Analyzer: Restart Server"

# Rebuild project index
rm -rf target/debug/.fingerprint
cargo check
```

### Permission Errors

**Problem:** Permission denied errors

**Solutions:**
```bash
# Fix ownership (run in container)
sudo chown -R vscode:vscode /workspace

# Or rebuild with correct user
docker-compose down
docker-compose up --build
```

## Best Practices

### 1. Use Volume Mounts for Caches

Always mount cargo registry and target directory to volumes for performance.

### 2. Keep Container Lightweight

Only install tools you actually need in the container.

### 3. Use Post-Create Script

Automate environment setup in `post-create.sh` for consistency.

### 4. Pin Tool Versions

Specify exact versions in `rust-toolchain.toml` for reproducibility.

### 5. Regular Updates

Update base image and tools monthly:
```bash
docker pull mcr.microsoft.com/devcontainers/rust:1-bookworm
Command Palette → "Dev Containers: Rebuild Container"
```

## Security Considerations

### 1. Secrets Management

Never commit secrets to `.devcontainer/`:
```bash
# Add to .gitignore
.devcontainer/.env
.devcontainer/secrets/
```

### 2. Minimal Permissions

Run container as non-root user:
```json
"remoteUser": "vscode"
```

### 3. Network Isolation

Limit network access when possible:
```json
"runArgs": ["--network", "none"]  // For offline development
```

## References

- [VS Code DevContainers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [DevContainer Features](https://containers.dev/features)
- [Rust DevContainer Images](https://mcr.microsoft.com/en-us/product/devcontainers/rust/about)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
