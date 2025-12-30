# Development Scripts

This directory contains scripts for maintaining the hologramapp project.

## FFI Rebuild Script (Unified)

### `rebuild_ffi.sh` ⭐ Recommended

**Comprehensive FFI rebuild script** that handles everything in one command.

**Usage:**
```bash
# Standard rebuild (after making FFI changes)
./scripts/rebuild_ffi.sh

# Clean rebuild (fixes UniFFI checksum errors)
./scripts/rebuild_ffi.sh --clean
```

**What it does:**
1. Cleans hologram-ffi (optional, with `--clean` flag)
2. Builds hologram-ffi in release mode
3. Copies library to all language binding directories
4. Reinstalls PyTorch extension (if present)

**When to use:**

| Use Case | Command | Why |
|----------|---------|-----|
| Made FFI changes | `./scripts/rebuild_ffi.sh` | Updates all bindings |
| UniFFI checksum error | `./scripts/rebuild_ffi.sh --clean` | Fixes checksum mismatch |
| After git pull | `./scripts/rebuild_ffi.sh` | Synchronizes bindings |
| Daily development | `./scripts/rebuild_ffi.sh` | Fast rebuild + update |

**Platform support:** Auto-detects Linux/macOS/Windows and uses correct library extension.

---

### Deprecated Scripts

The following scripts have been replaced by `rebuild_ffi.sh`:
- ~~`update_ffi_bindings.sh`~~ → Use `rebuild_ffi.sh`
- ~~`fix_uniffi_checksum.sh`~~ → Use `rebuild_ffi.sh --clean`

Backup copies available as `.bak` files.

---

## Model Management

### `download_models.sh`

Downloads small test models for Candle inference validation.

**Usage:**
```bash
./scripts/download_models.sh
```

**What it downloads:**
- `distilbert-base-uncased` (~250MB) - For embedding tests
- `distilgpt2` (~82MB) - For completion tests
- **Total:** ~330MB

**Output location:** `./models/`

**When to use:**
- When running tests with `--features candle`
- When validating real ML inference
- Before deploying with actual models

**When NOT needed:**
- Regular development (uses fast fallback tests)
- CI (unless explicitly testing real models)
- Building/compiling code

**Example workflow:**
```bash
# Download models once
./scripts/download_models.sh

# Run real model tests
cargo test -p hologram-model-server --features candle -- --include-ignored
```

See [hologram-models/TESTING.md](../hologram-sdk/rust/hologram-models/TESTING.md) for complete testing guide.

---

## WASM Testing Scripts

### `test-wasm.sh` ⭐ Run before committing WASM changes

Comprehensive local WASM testing script that validates WASM builds before pushing to CI.

**Usage:**
```bash
./scripts/test-wasm.sh
```

**What it does:**
1. ✅ Checks prerequisites (Rust, WASM target, wasm-pack)
2. ✅ Runs workspace tests
3. ✅ Builds WASM (debug, release, with WebGPU)
4. ✅ Checks binary size (warns if >10MB)
5. ✅ Runs browser tests in headless Chrome (if available)

**Prerequisites:**
```bash
# Install WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack (optional, for browser tests)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**When to use:**
- Before committing WASM-related changes
- After modifying `hologram-backends/src/backends/wasm/`
- Before creating a PR that touches WASM code

---

### `benchmark-webgpu.sh`

Runs CPU baseline benchmarks and detects performance regressions.

**Usage:**
```bash
# Run benchmarks and compare against baseline
./scripts/benchmark-webgpu.sh

# Run with custom baseline
./scripts/benchmark-webgpu.sh path/to/baseline.txt

# Quick test mode (runs only 2 benchmarks, ~1 minute)
./scripts/benchmark-webgpu.sh --quick
```

**What it does:**
1. ✅ Runs CPU baseline benchmarks
2. ✅ Compares against baseline (if exists)
3. ✅ Generates performance report
4. ✅ Exits with error if >10% regression detected

**Files generated:**
- `benchmarks/webgpu-current.txt` - Current results
- `benchmarks/webgpu-baseline.txt` - Baseline for comparison
- `benchmarks/webgpu-report.md` - Performance report

**When to use:**
- Before committing performance optimizations
- To verify changes don't regress performance
- When updating WASM/WebGPU kernels

---

## CI/CD Integration

### GitHub Actions Workflows

**wasm-ci.yml** - Multi-platform WASM testing
- Tests on macOS ARM64, macOS x86-64, Linux, Windows
- Runs browser tests in headless Chrome
- Collects performance benchmarks
- Compares WASM binary sizes across platforms

**ci.yml** - Main CI workflow (updated)
- Now includes WASM build check
- Ensures every PR verifies WASM compatibility

---

## Troubleshooting

### FFI Issues
See [docs/UNIFFI_CHECKSUM_FIX.md](../docs/UNIFFI_CHECKSUM_FIX.md) for FFI troubleshooting.

### WASM Issues

**"WASM target not installed"**
```bash
rustup target add wasm32-unknown-unknown
```

**"wasm-pack not found"**
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**"Chrome not found"**
- macOS: `brew install --cask google-chrome`
- Ubuntu: `sudo apt-get install chromium-browser`
- Or skip browser tests (they're optional)

**Browser tests fail with "WebGPU not available"**
- Expected on systems without WebGPU support
- Fallback WASM-only tests should still pass
- Use Chrome 113+ or Edge 113+ for WebGPU support

---

## Related Documentation

- [WASM Deployment Guide](../docs/wasm/WASM_DEPLOYMENT_GUIDE.md)
- [Cross-Platform Testing](../docs/wasm/CROSS_PLATFORM_TESTING.md)
- [WASM Next Steps](../docs/wasm/NEXT_STEPS.md)
- [WebGPU Implementation](../docs/webgpu/WEBGPU_IMPLEMENTATION.md)
