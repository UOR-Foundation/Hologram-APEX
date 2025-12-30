# GitHub Packages Publishing Specification

**Status:** Draft
**Version:** 0.1.0

## Overview

This specification defines the publishing strategy, workflows, and configuration for publishing Hologram crates to GitHub Packages.

## Publishing Strategy

### Goals

1. **Automated Publishing** - Trigger on version tags
2. **Quality Gates** - Tests must pass before publishing
3. **Version Management** - Workspace-wide semantic versioning
4. **Easy Installation** - Simple for users to consume packages

### Package Registry

- **Primary:** GitHub Packages (GitHub Container Registry)
- **Fallback:** Git dependency for development

### Published Crates

| Crate | Publish | Notes |
|-------|---------|-------|
| hologram-core | ✅ Yes | Core functionality |
| hologram-compiler | ✅ Yes | Compiler |
| hologram-backends | ✅ Yes | Backends |
| hologram-config | ✅ Yes | Configuration |
| hologram-ffi | ✅ Yes | FFI bindings (feature-gated) |
| hologram (main) | ✅ Yes | Top-level API |

## Cargo.toml Configuration

### Workspace Manifest

```toml
[workspace]
members = [
    "crates/core",
    "crates/compiler",
    "crates/backends",
    "crates/config",
    "crates/ffi",
    "bins/hologram-compile",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Hologram Contributors"]
repository = "https://github.com/OWNER/hologram"
license = "MIT OR Apache-2.0"

[workspace.dependencies]
# Shared dependencies
thiserror = "1.0"
bytemuck = "1.14"
```

### Main Crate Cargo.toml

```toml
[package]
name = "hologram"
version.workspace = true
edition.workspace = true
authors.workspace = true
description = "High-performance compute acceleration via geometric canonicalization"
repository.workspace = true
homepage = "https://hologram.dev"
documentation = "https://docs.rs/hologram"
readme = "README.md"
license.workspace = true
keywords = ["compute", "acceleration", "quantum", "canonicalization", "high-performance"]
categories = ["science", "mathematics", "hardware-support"]

# Publishing metadata
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[dependencies]
hologram-core = { path = "crates/core", version = "0.1.0" }
hologram-compiler = { path = "crates/compiler", version = "0.1.0" }
hologram-backends = { path = "crates/backends", version = "0.1.0" }
hologram-config = { path = "crates/config", version = "0.1.0" }

[features]
default = []
ffi = ["hologram-ffi"]
cuda = ["hologram-backends/cuda"]
metal = ["hologram-backends/metal"]
webgpu = ["hologram-backends/webgpu"]
```

### Sub-crate Example

```toml
[package]
name = "hologram-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
description = "Hologram core mathematical foundation and runtime"
repository.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
bytemuck.workspace = true
```

## GitHub Packages Configuration

### .cargo/config.toml

```toml
[registries.github]
index = "sparse+https://ghcr.io/OWNER/hologram/"

[net]
git-fetch-with-cli = true
```

### GitHub Actions Secret

- **GITHUB_TOKEN** - Automatically provided by GitHub Actions (no manual setup)

## Publishing Workflow

### File: `.github/workflows/publish.yml`

```yaml
name: Publish to GitHub Packages

on:
  push:
    tags:
      - 'v*.*.*'  # Trigger on version tags (e.g., v0.1.0)

env:
  CARGO_TERM_COLOR: always

jobs:
  publish:
    name: Publish Crates
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Cache dependencies
        uses: Swatinem/rust-cache@v2

      - name: Verify version matches tag
        run: |
          TAG_VERSION=${GITHUB_REF#refs/tags/v}
          CARGO_VERSION=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "hologram") | .version')
          if [ "$TAG_VERSION" != "$CARGO_VERSION" ]; then
            echo "Error: Tag version ($TAG_VERSION) doesn't match Cargo version ($CARGO_VERSION)"
            exit 1
          fi

      - name: Run tests
        run: cargo test --workspace --all-features

      - name: Build release
        run: cargo build --workspace --release --all-features

      - name: Publish hologram-core
        working-directory: crates/core
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
        continue-on-error: false

      - name: Wait for crates.io propagation
        run: sleep 30

      - name: Publish hologram-compiler
        working-directory: crates/compiler
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
        continue-on-error: false

      - name: Wait for crates.io propagation
        run: sleep 30

      - name: Publish hologram-backends
        working-directory: crates/backends
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
        continue-on-error: false

      - name: Wait for crates.io propagation
        run: sleep 30

      - name: Publish hologram-config
        working-directory: crates/config
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
        continue-on-error: false

      - name: Wait for crates.io propagation
        run: sleep 30

      - name: Publish hologram-ffi
        working-directory: crates/ffi
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }} --features ffi
        continue-on-error: false

      - name: Wait for crates.io propagation
        run: sleep 30

      - name: Publish hologram (main crate)
        run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
          files: |
            target/release/hologram-compile
```

**Note:** For GitHub Packages, use `--registry github` flag. For crates.io, use `--token` as shown above.

## Version Management

### Versioning Strategy

- **Semantic Versioning (SemVer):** `MAJOR.MINOR.PATCH`
- **Pre-releases:** `0.x.y` until 1.0.0 release
- **Workspace-wide:** All crates share the same version
- **Update location:** Workspace `Cargo.toml` `[workspace.package]` section

### Version Bump Workflow

**File: `.github/workflows/version-bump.yml`**

```yaml
name: Version Bump

on:
  workflow_dispatch:
    inputs:
      version_type:
        description: 'Version bump type'
        required: true
        type: choice
        options:
          - patch
          - minor
          - major

jobs:
  bump-version:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-edit
        run: cargo install cargo-edit

      - name: Bump version
        run: |
          cargo set-version --workspace --bump ${{ inputs.version_type }}

      - name: Get new version
        id: version
        run: |
          VERSION=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "hologram") | .version')
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Update CHANGELOG.md
        run: |
          echo "## [${{ steps.version.outputs.version }}] - $(date +%Y-%m-%d)" >> CHANGELOG.tmp
          echo "" >> CHANGELOG.tmp
          cat CHANGELOG.md >> CHANGELOG.tmp
          mv CHANGELOG.tmp CHANGELOG.md

      - name: Commit and tag
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "chore: bump version to v${{ steps.version.outputs.version }}"
          git tag v${{ steps.version.outputs.version }}
          git push origin main
          git push origin v${{ steps.version.outputs.version }}
```

## Release Process

### Manual Release Steps

1. **Prepare release**
   - Ensure all tests pass: `cargo test --workspace`
   - Ensure no warnings: `cargo clippy --workspace -- -D warnings`
   - Update `CHANGELOG.md` with release notes

2. **Bump version** (via GitHub Actions)
   - Go to Actions → Version Bump
   - Select version type (patch/minor/major)
   - Click "Run workflow"
   - Wait for workflow to complete

3. **Verify tag and publish**
   - Tag automatically created: `v0.x.y`
   - Publishing workflow automatically triggered
   - Monitor GitHub Actions for publish status

4. **Verify publication**
   - Check GitHub Packages page
   - Verify GitHub Release created
   - Test installation from published package

### Automated Release (CI)

1. **Push version tag manually** (if not using version bump workflow)
   ```bash
   # Update version in Cargo.toml
   cargo set-version --workspace 0.2.0

   # Commit and tag
   git add -A
   git commit -m "chore: bump version to v0.2.0"
   git tag v0.2.0
   git push origin main
   git push origin v0.2.0
   ```

2. **GitHub Actions automatically:**
   - Runs full test suite
   - Builds release artifacts
   - Publishes all crates in dependency order
   - Creates GitHub Release

## Installation Guide

### For End Users

#### From crates.io (once published)

```toml
[dependencies]
hologram = "0.1"

# With optional features
hologram = { version = "0.1", features = ["ffi", "cuda"] }
```

#### From GitHub Packages

```toml
[dependencies]
hologram = { version = "0.1", registry = "github" }
```

**Configuration (.cargo/config.toml in project root):**

```toml
[registries.github]
index = "sparse+https://ghcr.io/OWNER/hologram/"
```

#### From Git (development)

```toml
[dependencies]
hologram = { git = "https://github.com/OWNER/hologram", tag = "v0.1.0" }
```

## Testing Publishing

### Dry Run

Before actual release, test publishing:

```bash
# Test main crate
cargo publish --dry-run

# Test all crates
cd crates/core && cargo publish --dry-run
cd ../compiler && cargo publish --dry-run
cd ../backends && cargo publish --dry-run
cd ../config && cargo publish --dry-run
cd ../ffi && cargo publish --dry-run --features ffi
```

### Local Package Testing

After publishing, test installation:

```bash
# Create test project
cargo new --bin test-hologram
cd test-hologram

# Add dependency
echo 'hologram = "0.1"' >> Cargo.toml

# Build to verify
cargo build
```

## Troubleshooting

### Version Mismatch

**Error:** "Tag version doesn't match Cargo version"

**Solution:**
- Ensure workspace `Cargo.toml` version matches git tag
- Use version bump workflow to avoid mismatches

### Publish Failure

**Error:** "crate already exists"

**Solution:**
- Cannot republish same version
- Bump version and create new tag

### Dependency Resolution

**Error:** "failed to select a version for dependency `hologram-core`"

**Solution:**
- Wait 30 seconds between publishing dependent crates (already in workflow)
- crates.io needs time to propagate new versions

### GitHub Token Permissions

**Error:** "authentication required"

**Solution:**
- Ensure workflow has `packages: write` permission
- GitHub automatically provides GITHUB_TOKEN with correct permissions

## Quality Gates

Before publishing, the following must pass:

1. ✅ All tests pass: `cargo test --workspace --all-features`
2. ✅ Zero warnings: `cargo clippy --workspace -- -D warnings`
3. ✅ Code formatted: `cargo fmt --check`
4. ✅ Documentation builds: `cargo doc --workspace --no-deps`
5. ✅ Version matches tag
6. ✅ CHANGELOG.md updated

## Future Enhancements

- [ ] Publish pre-release versions (alpha, beta, rc)
- [ ] Automated CHANGELOG generation
- [ ] Signed releases with GPG
- [ ] Binary distribution via GitHub Releases
- [ ] Docker image publishing to GHCR
- [ ] Homebrew formula automation

## References

- [Cargo Publishing Guide](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [GitHub Packages for Rust](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-cargo-registry)
- [Semantic Versioning](https://semver.org/)
- [cargo-edit](https://github.com/killercup/cargo-edit)
