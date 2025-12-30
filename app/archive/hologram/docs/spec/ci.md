# Continuous Integration Specification

**Status:** Draft
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

This specification defines the complete CI/CD pipeline for Hologram using GitHub Actions. It covers automated testing, linting, building, publishing, and release management.

## CI/CD Philosophy

- **Test Everything** - Every commit must pass full test suite
- **Fast Feedback** - CI results within 10 minutes
- **Zero Warnings** - No compiler or clippy warnings allowed
- **Automated Publishing** - Tags trigger automatic releases
- **Multi-Platform** - Test on Linux, macOS, Windows
- **Caching** - Aggressive caching for fast builds

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Trigger:** Every push and pull request

**Purpose:** Validate code quality, run tests, check formatting

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  # Job 1: Check formatting
  format:
    name: Format Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt

      - name: Check formatting
        run: cargo fmt --all --check

  # Job 2: Clippy linting
  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy

      - name: Cache cargo registry
        uses: actions/cache@v3
        with:
          path: ~/.cargo/registry
          key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache cargo index
        uses: actions/cache@v3
        with:
          path: ~/.cargo/git
          key: ${{ runner.os }}-cargo-index-${{ hashFiles('**/Cargo.lock') }}

      - name: Cache cargo build
        uses: actions/cache@v3
        with:
          path: target
          key: ${{ runner.os }}-cargo-build-target-${{ hashFiles('**/Cargo.lock') }}

      - name: Run clippy
        run: cargo clippy --workspace --all-targets --all-features -- -D warnings

  # Job 3: Build
  build:
    name: Build
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Build workspace
        run: cargo build --workspace --all-targets --all-features

      - name: Check for warnings
        run: cargo build --workspace --all-targets --all-features 2>&1 | tee build.log && ! grep -i "warning" build.log

  # Job 4: Test
  test:
    name: Test
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-test-${{ hashFiles('**/Cargo.lock') }}

      - name: Run tests
        run: cargo test --workspace --all-features

      - name: Run doc tests
        run: cargo test --workspace --doc

  # Job 5: Integration tests
  integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-integration-${{ hashFiles('**/Cargo.lock') }}

      - name: Run integration tests
        run: cargo test --workspace --test '*'

  # Job 6: Documentation
  docs:
    name: Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-doc-${{ hashFiles('**/Cargo.lock') }}

      - name: Build documentation
        run: cargo doc --workspace --no-deps --all-features

      - name: Check for broken links
        run: cargo doc --workspace --no-deps --all-features 2>&1 | tee doc.log && ! grep -i "warning" doc.log

  # Job 7: FFI bindings
  ffi:
    name: FFI Bindings
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install UniFFI tools
        run: cargo install uniffi-bindgen

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-ffi-${{ hashFiles('**/Cargo.lock') }}

      - name: Build FFI bindings
        run: cargo build --features ffi --package hologram-ffi

      - name: Test FFI bindings
        run: cargo test --features ffi --package hologram-ffi

      - name: Generate bindings
        run: |
          cd crates/ffi
          uniffi-bindgen generate hologram.udl --language python
          uniffi-bindgen generate hologram.udl --language swift
          uniffi-bindgen generate hologram.udl --language kotlin

  # Job 8: Examples
  examples:
    name: Examples
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-examples-${{ hashFiles('**/Cargo.lock') }}

      - name: Build examples
        run: cargo build --examples --workspace

      - name: Run basic_operations example
        run: cargo run --example basic_operations

      - name: Run tensor_operations example
        run: cargo run --example tensor_operations

  # Job 9: Security audit
  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Security audit
        uses: actions-rs/audit-check@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  # Job 10: Code coverage
  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate coverage
        run: cargo tarpaulin --workspace --all-features --out Xml

      - name: Upload to codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./cobertura.xml
          fail_ci_if_error: true
```

### 2. Publish Workflow (`.github/workflows/publish.yml`)

**Trigger:** Version tags (v*.*.*)

**Purpose:** Publish crates to GitHub Packages and create GitHub Release

```yaml
name: Publish

on:
  push:
    tags:
      - 'v*.*.*'

env:
  CARGO_TERM_COLOR: always
  CARGO_REGISTRIES_GITHUB_INDEX: https://github.com/your-org/hologram/_git/

jobs:
  # Verify version matches tag
  verify:
    name: Verify Version
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Extract version from tag
        id: tag
        run: echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Extract version from Cargo.toml
        id: cargo
        run: echo "version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)" >> $GITHUB_OUTPUT

      - name: Verify versions match
        run: |
          if [ "${{ steps.tag.outputs.version }}" != "${{ steps.cargo.outputs.version }}" ]; then
            echo "Tag version (${{ steps.tag.outputs.version }}) does not match Cargo.toml version (${{ steps.cargo.outputs.version }})"
            exit 1
          fi

  # Run full test suite
  test:
    name: Test Before Publish
    needs: verify
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run tests
        run: cargo test --workspace --all-features

      - name: Run clippy
        run: cargo clippy --workspace --all-targets --all-features -- -D warnings

  # Build release artifacts
  build:
    name: Build Release
    needs: test
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: macos-latest
            target: aarch64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Build release binary
        run: cargo build --release --bin hologram-compile --target ${{ matrix.target }}

      - name: Package binary
        shell: bash
        run: |
          cd target/${{ matrix.target }}/release
          if [ "${{ matrix.os }}" == "windows-latest" ]; then
            7z a ../../../hologram-compile-${{ matrix.target }}.zip hologram-compile.exe
          else
            tar czf ../../../hologram-compile-${{ matrix.target }}.tar.gz hologram-compile
          fi

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: hologram-compile-${{ matrix.target }}
          path: hologram-compile-${{ matrix.target }}.*

  # Publish crates to GitHub Packages
  publish:
    name: Publish Crates
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Login to GitHub Packages
        run: cargo login ${{ secrets.GITHUB_TOKEN }}

      - name: Publish hologram-core
        run: cargo publish --package hologram-core --registry github
        continue-on-error: false

      - name: Wait for hologram-core to be available
        run: sleep 30

      - name: Publish hologram-compiler
        run: cargo publish --package hologram-compiler --registry github
        continue-on-error: false

      - name: Wait for hologram-compiler to be available
        run: sleep 30

      - name: Publish hologram-backends
        run: cargo publish --package hologram-backends --registry github
        continue-on-error: false

      - name: Wait for hologram-backends to be available
        run: sleep 30

      - name: Publish hologram-config
        run: cargo publish --package hologram-config --registry github
        continue-on-error: false

      - name: Wait for hologram-config to be available
        run: sleep 30

      - name: Publish hologram-ffi
        run: cargo publish --package hologram-ffi --registry github --features ffi
        continue-on-error: false

      - name: Wait for hologram-ffi to be available
        run: sleep 30

      - name: Publish hologram (main crate)
        run: cargo publish --package hologram --registry github
        continue-on-error: false

  # Create GitHub Release
  release:
    name: Create GitHub Release
    needs: publish
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Download artifacts
        uses: actions/download-artifact@v3
        with:
          path: artifacts

      - name: Generate release notes
        id: release_notes
        run: |
          VERSION=${GITHUB_REF#refs/tags/v}
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")

          echo "## Changes in v${VERSION}" > release_notes.md
          echo "" >> release_notes.md

          if [ -n "$PREV_TAG" ]; then
            echo "### Commits since ${PREV_TAG}" >> release_notes.md
            git log ${PREV_TAG}..HEAD --pretty=format:"- %s (%h)" >> release_notes.md
          else
            echo "Initial release" >> release_notes.md
          fi

          echo "" >> release_notes.md
          echo "## Installation" >> release_notes.md
          echo "" >> release_notes.md
          echo "Add to your \`Cargo.toml\`:" >> release_notes.md
          echo "\`\`\`toml" >> release_notes.md
          echo "[dependencies]" >> release_notes.md
          echo "hologram = \"${VERSION}\"" >> release_notes.md
          echo "\`\`\`" >> release_notes.md

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: artifacts/**/*
          body_path: release_notes.md
          draft: false
          prerelease: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 3. Version Bump Workflow (`.github/workflows/version-bump.yml`)

**Trigger:** Manual workflow dispatch

**Purpose:** Automated version bumping

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
  bump:
    name: Bump Version
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-edit
        run: cargo install cargo-edit

      - name: Bump version
        run: cargo set-version --workspace --bump ${{ github.event.inputs.version_type }}

      - name: Get new version
        id: version
        run: echo "version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)" >> $GITHUB_OUTPUT

      - name: Update CHANGELOG.md
        run: |
          DATE=$(date +%Y-%m-%d)
          VERSION=${{ steps.version.outputs.version }}

          # Insert new version header after [Unreleased]
          sed -i "/## \[Unreleased\]/a\\
          \\
          ## [${VERSION}] - ${DATE}" CHANGELOG.md

      - name: Commit changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add Cargo.toml Cargo.lock CHANGELOG.md crates/*/Cargo.toml
          git commit -m "chore: bump version to v${{ steps.version.outputs.version }}"

      - name: Create tag
        run: |
          git tag -a v${{ steps.version.outputs.version }} -m "Release version ${{ steps.version.outputs.version }}"

      - name: Push changes
        run: |
          git push origin main
          git push origin v${{ steps.version.outputs.version }}
```

### 4. Benchmark Workflow (`.github/workflows/benchmark.yml`)

**Trigger:** Push to main, manual dispatch

**Purpose:** Track performance over time

```yaml
name: Benchmark

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  benchmark:
    name: Run Benchmarks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-benchmark-${{ hashFiles('**/Cargo.lock') }}

      - name: Run benchmarks
        run: cargo bench --workspace --all-features

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/benchmarks.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

## Quality Gates

### Pre-Merge Requirements

All pull requests must pass:

1. ✅ Format check (`cargo fmt --check`)
2. ✅ Clippy with zero warnings (`cargo clippy -- -D warnings`)
3. ✅ Build on Linux, macOS, Windows
4. ✅ All tests pass (`cargo test --workspace`)
5. ✅ Integration tests pass
6. ✅ Documentation builds
7. ✅ Examples compile and run
8. ✅ FFI bindings generate successfully
9. ✅ Security audit passes
10. ✅ Code coverage >= 80%

### Pre-Publish Requirements

Before publishing crates:

1. ✅ Version matches git tag
2. ✅ CHANGELOG.md updated
3. ✅ All tests pass
4. ✅ Zero clippy warnings
5. ✅ Documentation builds
6. ✅ Examples work
7. ✅ FFI bindings generate
8. ✅ Release binaries build

## Caching Strategy

### Cargo Cache

```yaml
- name: Cache cargo
  uses: actions/cache@v3
  with:
    path: |
      ~/.cargo/registry
      ~/.cargo/git
      target
    key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    restore-keys: |
      ${{ runner.os }}-cargo-
```

**Benefits:**
- Reduces build time from 15 minutes to 3 minutes
- Saves bandwidth on dependency downloads
- Faster feedback on PRs

### Incremental Compilation

```yaml
env:
  CARGO_INCREMENTAL: 1
```

### sccache (Shared Compilation Cache)

```yaml
- name: Setup sccache
  uses: mozilla-actions/sccache-action@v0.0.3

- name: Configure sccache
  run: |
    echo "RUSTC_WRAPPER=sccache" >> $GITHUB_ENV
    echo "SCCACHE_GHA_ENABLED=true" >> $GITHUB_ENV
```

## Branch Protection

### Main Branch

```yaml
required_status_checks:
  strict: true
  contexts:
    - "Format Check"
    - "Clippy"
    - "Build (ubuntu-latest, stable)"
    - "Build (macos-latest, stable)"
    - "Build (windows-latest, stable)"
    - "Test (ubuntu-latest, stable)"
    - "Test (macos-latest, stable)"
    - "Test (windows-latest, stable)"
    - "Integration Tests"
    - "Documentation"
    - "FFI Bindings"
    - "Examples"
    - "Security Audit"
    - "Code Coverage"

required_pull_request_reviews:
  required_approving_review_count: 1
  dismiss_stale_reviews: true

enforce_admins: true
```

## Secrets Configuration

### Required Secrets

In GitHub repository settings → Secrets and variables → Actions:

| Secret | Purpose |
|--------|---------|
| `GITHUB_TOKEN` | Automatically provided, used for releases and package publishing |
| `CODECOV_TOKEN` | Upload coverage to Codecov (optional) |

## Performance Targets

### CI Pipeline Speed

- **Format check**: < 30 seconds
- **Clippy**: < 2 minutes (with cache)
- **Build**: < 5 minutes per platform (with cache)
- **Test**: < 3 minutes per platform
- **Total CI time**: < 10 minutes (parallel jobs)

### Cache Hit Rate

- **Target**: > 90% cache hit rate
- **Monitoring**: Track cache effectiveness in workflow logs

## Monitoring and Alerts

### Failed Build Notifications

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'CI build failed on ${{ github.ref }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Benchmark Regression Detection

```yaml
- name: Check for performance regression
  run: |
    if [ $(cargo bench --bench criterion_benchmarks | grep -c "regressed") -gt 0 ]; then
      echo "Performance regression detected!"
      exit 1
    fi
```

## Troubleshooting

### Cache Issues

**Problem:** Stale cache causing build failures

**Solution:**
```yaml
# Clear cache by changing cache key
key: ${{ runner.os }}-cargo-v2-${{ hashFiles('**/Cargo.lock') }}
#                             ^^^ increment version
```

### Flaky Tests

**Problem:** Intermittent test failures

**Solution:**
```yaml
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: cargo test --workspace
```

### Platform-Specific Failures

**Problem:** Tests pass on Linux but fail on Windows

**Solution:**
- Use conditional steps:
```yaml
- name: Windows-specific setup
  if: runner.os == 'Windows'
  run: |
    # Windows-specific commands
```

## Best Practices

### 1. Fail Fast

```yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
```

### 2. Parallel Execution

Run independent jobs in parallel for faster feedback:
- Format, Clippy, Build, Test all run concurrently

### 3. Minimal Checkout

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 1  # Shallow clone for speed
```

### 4. Artifact Management

```yaml
- name: Upload test results
  if: always()  # Upload even if tests fail
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: target/test-results/
    retention-days: 7
```

## Local CI Testing

### Act (Run GitHub Actions Locally)

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI workflow locally
act -j test

# Run specific job
act -j clippy

# Run with specific platform
act -P ubuntu-latest=ubuntu:22.04
```

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [cargo-edit Documentation](https://github.com/killercup/cargo-edit)
- [Codecov GitHub Action](https://github.com/codecov/codecov-action)
- [Release Process Specification](release.md)
- [Publishing Specification](publishing.md)
