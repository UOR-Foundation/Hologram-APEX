# Installation Guide

This guide explains how to install and use Hologram crates in your Rust projects.

## Overview

Hologram is distributed as a private GitHub repository. Since GitHub Packages does not currently support native Cargo/Rust registries, we use **git dependencies** for crate distribution.

## Prerequisites

- Rust 1.70 or later (install via [rustup](https://rustup.rs/))
- Git
- GitHub account with access to the repository

## Authentication

### For Private Repository Access

If the repository is private, you'll need to authenticate with GitHub. There are two methods:

#### Method 1: SSH (Recommended)

1. Set up SSH keys with GitHub: [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
2. Use SSH URLs in your `Cargo.toml`:

```toml
[dependencies]
hologram = { git = "ssh://git@github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }
```

#### Method 2: HTTPS with Personal Access Token

1. Create a GitHub Personal Access Token (PAT):
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a new token with `repo` scope
   - Copy the token

2. Configure Git credentials:

```bash
# Option A: Using Git credential helper (recommended)
git config --global credential.helper store
# Then git will prompt for your token on first use

# Option B: Using .cargo/config.toml
# Create or edit ~/.cargo/config.toml
[net]
git-fetch-with-cli = true
```

3. Use HTTPS URLs:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }
```

## Adding Hologram to Your Project

### Full Installation (All Crates)

To use all Hologram features, add to your `Cargo.toml`:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }
```

### Specific Crates

You can also depend on specific crates individually:

```toml
[dependencies]
# Core mathematical foundation
hologram-core = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }

# Compiler
hologram-compiler = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }

# Backends
hologram-backends = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }

# Common utilities
hologram-common = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }
```

### Using Specific Branches

For development, you can use a specific branch:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", branch = "develop" }
```

### Using Specific Commits

For reproducible builds, you can pin to a specific commit:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", rev = "abc1234" }
```

## Features

Hologram supports optional features:

```toml
[dependencies]
hologram = {
    git = "https://github.com/UOR-Foundation/hologramapp",
    tag = "v0.1.0",
    features = ["ffi", "cuda", "webgpu"]
}
```

Available features:
- `ffi` - Enable FFI bindings for Python, Swift, Kotlin
- `cuda` - CUDA backend support
- `metal` - Metal backend support (macOS)
- `webgpu` - WebGPU backend support

## Installing the CLI Tool

### Download Pre-built Binary

1. Go to [Releases](https://github.com/UOR-Foundation/hologramapp/releases)
2. Download the appropriate binary for your platform:
   - Linux: `hologram-compile-linux-amd64.tar.gz`
   - macOS (Intel): `hologram-compile-macos-amd64.tar.gz`
   - macOS (Apple Silicon): `hologram-compile-macos-arm64.tar.gz`
   - Windows: `hologram-compile-windows-amd64.exe.zip`
3. Extract and add to your PATH

### Build from Source

```bash
# Clone the repository
git clone https://github.com/UOR-Foundation/hologramapp
cd hologramapp

# Build the CLI tool
cargo build --release --bin hologram-compile

# Install to ~/.cargo/bin
cargo install --path bins/hologram-compile
```

## Verifying Installation

Create a simple test project:

```rust
// src/main.rs
use hologram_core::Tensor;

fn main() {
    let tensor = Tensor::zeros(&[2, 3]);
    println!("Created tensor: {:?}", tensor);
}
```

Build and run:

```bash
cargo build
cargo run
```

## Updating Hologram

To update to the latest version:

1. Check for new releases: [GitHub Releases](https://github.com/UOR-Foundation/hologramapp/releases)
2. Update the tag in your `Cargo.toml`:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.2.0" }
```

3. Update dependencies:

```bash
cargo update
```

## Troubleshooting

### Authentication Errors

**Error:** `failed to authenticate when downloading repository`

**Solution:**
- Ensure you have access to the repository
- Verify your SSH key or PAT is configured correctly
- Try using `git-fetch-with-cli = true` in `.cargo/config.toml`

### Dependency Resolution Errors

**Error:** `failed to select a version`

**Solution:**
- Ensure the tag exists in the repository
- Try using `cargo clean` and rebuild
- Check that all dependencies use compatible versions

### Build Errors

**Error:** Compilation failures

**Solution:**
- Ensure you're using Rust 1.70 or later: `rustc --version`
- Update Rust: `rustup update`
- Check that required system dependencies are installed

## CI/CD Integration

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Configure Git for private repos
        run: |
          git config --global url."https://${{ secrets.GH_TOKEN }}@github.com/".insteadOf "https://github.com/"

      - name: Build
        run: cargo build --release
```

Add `GH_TOKEN` secret to your repository settings with a PAT.

### GitLab CI

```yaml
build:
  image: rust:latest
  before_script:
    - git config --global url."https://oauth2:${CI_JOB_TOKEN}@github.com/".insteadOf "https://github.com/"
  script:
    - cargo build --release
```

## Advanced Configuration

### Vendoring Dependencies

For offline builds or maximum reproducibility:

```bash
# Download all dependencies including git repos
cargo vendor

# Configure Cargo to use vendored dependencies
# Add to .cargo/config.toml:
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"
```

### Workspace Integration

If using Hologram in a workspace:

```toml
# Workspace Cargo.toml
[workspace]
members = ["your-crate"]

[workspace.dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologramapp", tag = "v0.1.0" }

# In your-crate/Cargo.toml
[dependencies]
hologram.workspace = true
```

## Support

- **Documentation:** [docs/](https://github.com/UOR-Foundation/hologramapp/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/UOR-Foundation/hologramapp/issues)
- **Releases:** [GitHub Releases](https://github.com/UOR-Foundation/hologramapp/releases)

## Next Steps

- Read the [API Documentation](../api/)
- Check out [Examples](../../examples/)
- Review the [Architecture Overview](../architecture/overview.md)
