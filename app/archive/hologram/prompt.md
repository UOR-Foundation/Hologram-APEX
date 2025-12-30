# Hologram Refactoring Prompt

## Objective

Refactor the current Hologram codebase (`/workspace/`) into a unified, simplified architecture under `/workspace/hologram/` according to the specifications in `/workspace/hologram/docs/spec/`.

## Context

The current codebase (`/workspace/`) is a sprawling collection of crates with:
- Duplicate code across crates
- Deprecated functionality that should be removed
- Inconsistent architecture
- Missing documentation

The desired end-state is defined in `/workspace/hologram/docs/spec/`. Your task is to reconcile the current state with the desired state.

## Instructions

### Phase 1: Setup & Analysis (Days 1-2)

#### 1.1 Read Specifications Thoroughly

1. **Start with the master index**
   - Read `/workspace/hologram/docs/spec/INDEX.md`
   - Understand overall architecture and principles

2. **Read all crate specifications**
   - `/workspace/hologram/docs/spec/crates/core.md`
   - `/workspace/hologram/docs/spec/crates/compiler.md`
   - `/workspace/hologram/docs/spec/crates/backends.md`
   - `/workspace/hologram/docs/spec/crates/config.md`
   - `/workspace/hologram/docs/spec/crates/ffi.md`

3. **Read infrastructure specifications**
   - `/workspace/hologram/docs/spec/binaries/hologram-compile.md`
   - `/workspace/hologram/docs/spec/ci.md`
   - `/workspace/hologram/docs/spec/publishing.md`
   - `/workspace/hologram/docs/spec/devcontainer.md`
   - `/workspace/hologram/docs/spec/testing.md`
   - `/workspace/hologram/docs/spec/benchmarking.md`

#### 1.2 Analyze Current Codebase

1. **Inventory existing code**
   ```bash
   # List all current crates
   ls -la /workspace/crates/

   # Check dependencies
   cargo tree --workspace

   # Review test coverage
   cargo test --workspace -- --list
   ```

2. **Map current to desired**
   - Create mapping document: `MIGRATION_MAP.md`
   - List what to port from each current crate
   - List what to delete (deprecated code)
   - List what needs to be implemented fresh

3. **Identify dependencies**
   - External crate dependencies
   - Internal crate dependencies
   - Build-time dependencies

#### 1.3 Create Project Structure

```bash
cd /workspace/hologram

# Create workspace structure
mkdir -p crates/{core,compiler,backends,config,ffi}/src
mkdir -p bins/hologram-compile/src
mkdir -p kernels/{stdlib,onnx}
mkdir -p examples tests benches
mkdir -p docs/{architecture,guides,api}
mkdir -p .devcontainer .githooks .github/workflows .cargo
```

**Create workspace Cargo.toml:**

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
thiserror = "1.0"
bytemuck = "1.14"
parking_lot = "0.12"
dashmap = "5.5"
rayon = "1.8"
```

### Phase 2: Port Core Crate (Days 3-5)

#### 2.1 Setup hologram-core Structure

Create directory structure per `/workspace/hologram/docs/spec/crates/core.md`:

```bash
cd /workspace/hologram/crates/core/src
mkdir -p atlas torus monster algebra runtime ops interop
```

**Create Cargo.toml:**

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
bytemuck.workspace = true
thiserror.workspace = true
num-bigint = "0.4"
dashmap.workspace = true
parking_lot.workspace = true

hologram-backends = { path = "../backends" }
hologram-config = { path = "../config" }

[dev-dependencies]
proptest = "1.4"
criterion = "0.5"
```

#### 2.2 Port Atlas Module

**Source:** `/workspace/crates/atlas-core/`
**Destination:** `/workspace/hologram/crates/core/src/atlas/`

1. Port `src/lib.rs` → `atlas/graph.rs`
2. Port `src/invariants.rs` → `atlas/invariants.rs`
3. Port `src/constants.rs` → `atlas/constants.rs`
4. Create `atlas/class.rs` for resonance classes
5. Create `atlas/mod.rs` with public API

**Simplifications during port:**
- Remove deprecated functions
- Inline small utilities used once
- Ensure files < 1K lines
- Add comprehensive tests

#### 2.3 Port Torus Module

**Source:** `/workspace/crates/hrm-spec/src/torus/`
**Destination:** `/workspace/hologram/crates/core/src/torus/`

1. Port projection logic
2. Port lifting logic
3. Implement 48×256 lattice operations
4. Add property-based tests for coherence

#### 2.4 Port Monster Module

**Source:** `/workspace/crates/hrm-spec/src/monster/` + `/workspace/crates/hologram-hrm/src/griess/`
**Destination:** `/workspace/hologram/crates/core/src/monster/`

1. Port 196,884-dimensional representation
2. Port conjugacy class logic
3. Port O(1) routing algorithms
4. Add tests for Monster group properties

#### 2.5 Port Algebra Module

**Source:** `/workspace/crates/hrm-spec/src/algebra/`
**Destination:** `/workspace/hologram/crates/core/src/algebra/`

1. Implement ⊕, ⊗, ⊙ generators
2. Implement coherence proofs
3. Implement derived operations (MatMul, Conv, etc.)
4. Property-based tests for algebraic laws

#### 2.6 Port Runtime Module

**Source:** `/workspace/crates/hologram-core/src/`
**Destination:** `/workspace/hologram/crates/core/src/runtime/`

1. Port `executor.rs`
2. Port `buffer.rs`
3. Port `tensor.rs`
4. Port `address.rs` (96-class addressing)
5. Add integration tests

#### 2.7 Port Operations Module

**Source:** `/workspace/crates/hologram-core/src/ops/`
**Destination:** `/workspace/hologram/crates/core/src/ops/`

1. Port `math.rs` (element-wise operations)
2. Port `activation.rs` (neural network activations)
3. Port `reduce.rs` (reductions)
4. Port `loss.rs` (loss functions)
5. Port `linalg.rs` (linear algebra)
6. Port `memory.rs` (memory operations)
7. Add tests for each operation

#### 2.8 Port Interop Module

**Source:** `/workspace/crates/hologram-core/src/interop/`
**Destination:** `/workspace/hologram/crates/core/src/interop/`

1. Port `dlpack.rs` (DLPack tensor exchange)
2. Add zero-copy interop tests

#### 2.9 Verify Core Crate

```bash
cd /workspace/hologram/crates/core

# Build
cargo build

# Test
cargo test

# Clippy (zero warnings)
cargo clippy -- -D warnings

# Format check
cargo fmt --check
```

### Phase 3: Port Compiler Crate (Days 6-8)

#### 3.1 Setup hologram-compiler Structure

**Source:** `/workspace/crates/hologram-compiler/`
**Destination:** `/workspace/hologram/crates/compiler/`

Create structure per `/workspace/hologram/docs/spec/crates/compiler.md`.

#### 3.2 Port Components

1. **Quantum module** - Port 768-cycle quantum computing
2. **Canonical module** - Port pattern-based canonicalization
3. **Class module** - Port 96-class system
4. **Generators module** - Port 7 fundamental generators
5. **IR module** - Port intermediate representation
6. **Lang module** - Port circuit language (AST, lexer, parser)
7. **Compile module** - Port compilation pipeline

#### 3.3 Verify Compiler Crate

```bash
cd /workspace/hologram/crates/compiler

cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

### Phase 4: Port Backends Crate (Days 9-11)

#### 4.1 Setup hologram-backends Structure

**CRITICAL:** All backend implementations must be in `src/backends/` subdirectory.

```bash
cd /workspace/hologram/crates/backends/src
mkdir -p isa backends/{cpu,cuda,metal,wasm,webgpu} translators
```

#### 4.2 Port Components

1. **ISA module** - Port complete ISA (50+ instructions)
2. **Backend trait** - Port Backend trait definition
3. **CPU backend** - Port to `backends/cpu/`
4. **GPU backends** - Port to `backends/{cuda,metal,webgpu}/` (feature-gated)
5. **WASM backend** - Port to `backends/wasm/`
6. **Translators** - Port circuit_to_isa and json_to_isa
7. **Pool storage** - Port O(1) space streaming
8. **Program caching** - Port Blake3-based caching

#### 4.3 Verify Backends Crate

```bash
cd /workspace/hologram/crates/backends

cargo build
cargo build --all-features
cargo test
cargo clippy -- -D warnings
```

### Phase 5: Create Config Crate (Day 12)

#### 5.1 Create hologram-config

**NEW CRATE** - Implement per `/workspace/hologram/docs/spec/crates/config.md`.

```rust
// src/env.rs - Environment variable support
// src/file.rs - TOML config file parsing
// src/runtime.rs - Runtime configuration
// src/error.rs - Config errors
```

Features:
- `.env` file support (dotenvy crate)
- TOML config parsing
- Environment variable overrides
- Backend selection

### Phase 6: Create FFI Crate (Days 13-14)

#### 6.1 Setup hologram-ffi with UniFFI

**Implement per `/workspace/hologram/docs/spec/crates/ffi.md`.**

1. Create `hologram.udl` (UniFFI interface definition)
2. Implement Rust FFI wrappers (Executor, Buffer, Tensor)
3. Configure UniFFI build system (build.rs)
4. Generate C header with cbindgen
5. Create comprehensive README.md with:
   - How to add new languages
   - Workflow for updating bindings
   - Examples for each language

#### 6.2 FFI README.md

Create `/workspace/hologram/crates/ffi/README.md` with:
- Adding new language guide
- Updating workflow when core changes
- Language support matrix
- Examples (Python, Swift, TypeScript, C++)

### Phase 7: Create hologram-compile Binary (Day 15)

#### 7.1 Implement CLI Binary

**Implement per `/workspace/hologram/docs/spec/binaries/hologram-compile.md`.**

```rust
// bins/hologram-compile/src/main.rs - CLI entry point
// bins/hologram-compile/src/cli.rs - clap argument parsing
// bins/hologram-compile/src/frontends/ - Language frontends
// bins/hologram-compile/src/pipeline.rs - Compilation pipeline
```

**CLI Interface:**

```bash
hologram-compile path/to/kernel.py -o output.json
hologram-compile kernels path/to/kernels/ -o output_dir/
hologram-compile kernels path/ --verbose --lang python
```

### Phase 8: Create Top-Level API (Day 16)

#### 8.1 Implement hologram/src/lib.rs

```rust
// Re-export all public types
pub use hologram_core::{Executor, Buffer, Tensor, ops};
pub use hologram_compiler::{Circuit, Compiler, Canonicalizer};
pub use hologram_backends::{Backend, backends::{CpuBackend}};
pub use hologram_config::Config;

#[cfg(feature = "ffi")]
pub use hologram_ffi;

// Convenience functions
pub fn create_executor(backend: BackendType) -> Result<Executor>;
pub fn compile_circuit(source: &str) -> Result<Circuit>;
```

### Phase 9: Infrastructure (Days 17-18)

#### 9.1 DevContainer

Create `.devcontainer/` per `/workspace/hologram/docs/spec/devcontainer.md`:

```json
// .devcontainer/devcontainer.json
{
  "name": "Hologram Development",
  "dockerFile": "Dockerfile",
  "extensions": [
    "rust-lang.rust-analyzer",
    "ms-python.python"
  ],
  "postCreateCommand": "cargo build --workspace"
}
```

```dockerfile
# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/rust:1-bullseye
RUN apt-get update && apt-get install -y python3.11 python3-pip clang llvm
RUN cargo install uniffi-bindgen-cs cbindgen
```

#### 9.2 Git Hooks

```bash
# .githooks/pre-commit
#!/bin/bash
echo "Running pre-commit checks..."
cargo test --workspace || exit 1
cargo fmt --check || exit 1
cargo clippy --workspace -- -D warnings || exit 1
cargo test --features ffi --package hologram-ffi || exit 1
echo "All checks passed!"
```

```bash
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

#### 9.3 GitHub Actions

Create workflows per `/workspace/hologram/docs/spec/ci.md` and `publishing.md`:

- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/publish.yml` - Publishing workflow
- `.github/workflows/version-bump.yml` - Version management
- `.github/workflows/benchmark.yml` - Benchmarking

#### 9.4 Cargo Registry Config

```toml
# .cargo/config.toml
[registries.github]
index = "sparse+https://ghcr.io/OWNER/hologram/"

[net]
git-fetch-with-cli = true
```

### Phase 10: Documentation & Examples (Days 19-20)

#### 10.1 Create Examples

Create focused examples (NO ONNX):

```
examples/
├── basic_operations.rs         # Simple vector operations
├── tensor_operations.rs        # Multi-dimensional tensors
├── custom_kernel.rs            # Define custom kernel via Python
├── backend_selection.rs        # Backend switching
├── streaming_computation.rs   # Pool storage O(1) space
└── circuit_compilation.rs      # Full compilation pipeline
```

#### 10.2 Write User Guides

Create in `/workspace/hologram/docs/guides/`:
- Getting started guide
- Architecture overview
- Performance tuning guide
- FFI usage guide

#### 10.3 Generate API Documentation

```bash
cargo doc --workspace --no-deps --open
```

### Phase 11: Testing & Validation (Days 21-22)

#### 11.1 Integration Tests

Create in `/workspace/hologram/tests/`:
- `integration_tests.rs` - End-to-end tests
- `backend_compatibility.rs` - Multi-backend tests
- `kernel_compilation.rs` - Kernel compilation pipeline tests
- `ffi_tests.rs` - FFI binding tests

#### 11.2 Benchmarks

Create in `/workspace/hologram/benches/`:
- `operations_bench.rs` - Operation benchmarks
- `compilation_bench.rs` - Compilation speed
- `backend_bench.rs` - Backend comparison

#### 11.3 Run Full Test Suite

```bash
cd /workspace/hologram

# All tests
cargo test --workspace
cargo test --workspace --all-features
cargo test --workspace --release

# Zero warnings
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check

# Benchmarks
cargo bench --workspace

# FFI
cargo build --features ffi
cargo test --features ffi --package hologram-ffi

# Dry-run publishing
cargo publish --dry-run
cd crates/core && cargo publish --dry-run
cd ../compiler && cargo publish --dry-run
cd ../backends && cargo publish --dry-run
cd ../config && cargo publish --dry-run
cd ../ffi && cargo publish --dry-run --features ffi
```

### Phase 12: Final Reconciliation (Day 23)

#### 12.1 Specification Compliance Check

For each specification document, verify:

1. **hologram-core** (`crates/core.md`)
   - ✅ All specified APIs implemented
   - ✅ Module structure matches spec
   - ✅ Performance requirements met
   - ✅ Test coverage ≥80%

2. **hologram-compiler** (`crates/compiler.md`)
   - ✅ All components present
   - ✅ Canonicalization rules implemented
   - ✅ Property tests for idempotence

3. **hologram-backends** (`crates/backends.md`)
   - ✅ All backends in `backends/` subdirectory
   - ✅ Complete ISA implemented
   - ✅ Backend trait matches spec

4. **hologram-config** (`crates/config.md`)
   - ✅ .env support working
   - ✅ TOML parsing functional

5. **hologram-ffi** (`crates/ffi.md`)
   - ✅ UniFFI bindings generate
   - ✅ README.md comprehensive
   - ✅ Examples for each language

6. **hologram-compile** (`binaries/hologram-compile.md`)
   - ✅ CLI functional
   - ✅ Python frontend working

7. **CI/CD** (`ci.md`, `publishing.md`)
   - ✅ All workflows created
   - ✅ Publishing workflow tested (dry-run)
   - ✅ Version bump workflow created

8. **Testing** (`testing.md`)
   - ✅ Unit tests comprehensive
   - ✅ Integration tests present
   - ✅ Benchmarks functional

#### 12.2 Final Cleanup

1. **Remove all TODOs**
   ```bash
   rg "TODO|FIXME|XXX" --type rust
   # Should return nothing
   ```

2. **Ensure files < 1K lines**
   ```bash
   find . -name "*.rs" -exec wc -l {} + | awk '$1 > 1000 {print}'
   # Should return nothing (excluding tests)
   ```

3. **Remove dead code**
   - Run `cargo clean`
   - Remove unused dependencies
   - Remove commented-out code

4. **Final test run**
   ```bash
   cargo clean
   cargo build --workspace --all-targets
   cargo test --workspace --all-features
   cargo clippy --workspace --all-targets -- -D warnings
   cargo fmt --check
   cargo doc --workspace --no-deps
   ```

#### 12.3 Update Spec if Needed

If implementation revealed spec issues:
1. Update affected specification documents
2. Document rationale for changes
3. Ensure consistency across related specs

## Key Principles

1. **Specification is source of truth** - When in doubt, follow the spec
2. **No TODOs or stubs** - Implement everything fully
3. **Ruthless simplicity** - Delete unnecessary complexity
4. **Performance-first** - O(1), zero-copy, parallel, compile-time
5. **Complete tasks fully** - Finish what you start
6. **Zero warnings** - Compiler + clippy warnings must be zero

## Validation Checklist

Before considering refactoring complete:

- [ ] All crate specifications implemented
- [ ] All binary specifications implemented
- [ ] All tests passing (`cargo test --workspace`)
- [ ] Zero compiler warnings (`cargo build --workspace`)
- [ ] Zero clippy warnings (`cargo clippy --workspace -- -D warnings`)
- [ ] Code formatted (`cargo fmt --check`)
- [ ] Documentation builds (`cargo doc --workspace --no-deps`)
- [ ] All examples run successfully
- [ ] FFI bindings generate and test successfully
- [ ] DevContainer builds and functions
- [ ] Git hooks execute successfully
- [ ] GitHub Actions workflows created
- [ ] Publishing workflow tested (dry-run)
- [ ] Benchmarks run successfully
- [ ] All files < 1K lines (excluding tests)
- [ ] No TODO comments in code
- [ ] Specification compliance verified

## Success Criteria

✅ **Architecture**
- Unified `hologram/` crate structure
- Clear separation of concerns
- All backends in `backends/` subdirectory

✅ **Code Quality**
- Zero warnings (compiler + clippy)
- All tests passing
- No TODOs or stubs
- Files < 1K lines

✅ **Documentation**
- All public APIs documented
- Examples for major features
- User guides complete
- FFI README comprehensive

✅ **Infrastructure**
- DevContainer functional
- Git hooks working
- CI workflows created
- Publishing workflow ready

✅ **Compliance**
- Implementation matches specifications
- Performance requirements met
- Test coverage requirements met

## Timeline Estimate

- Phase 1 (Setup & Analysis): 2 days
- Phase 2 (Port Core): 3 days
- Phase 3 (Port Compiler): 3 days
- Phase 4 (Port Backends): 3 days
- Phase 5 (Create Config): 1 day
- Phase 6 (Create FFI): 2 days
- Phase 7 (Create hologram-compile): 1 day
- Phase 8 (Top-Level API): 1 day
- Phase 9 (Infrastructure): 2 days
- Phase 10 (Documentation & Examples): 2 days
- Phase 11 (Testing & Validation): 2 days
- Phase 12 (Final Reconciliation): 1 day

**Total: ~23 days**

## Getting Started

Run this prompt with Claude Code to begin the refactoring. Work through each phase systematically, verifying completion criteria before moving to the next phase.

**Remember: The specifications in `/workspace/hologram/docs/spec/` are the source of truth. Your implementation must match them exactly.**
