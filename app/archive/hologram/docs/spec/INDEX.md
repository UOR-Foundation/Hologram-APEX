# Hologram Specification Index

**Version:** 0.1.0
**Status:** Draft
**Last Updated:** 2025-01-18

## Overview

This directory contains the complete technical specification for the Hologram compute acceleration platform. These specifications define the desired end-state architecture and serve as the source of truth for development.

## Purpose

Hologram is a high-performance compute acceleration platform built on:
- **Two-torus lattice** (48 × 256 cells)
- **Monster group representation** (196,884 dimensions)
- **96-class geometric system**
- **MoonshineHRM algebraic framework** (⊕, ⊗, ⊙)
- **Pattern-based canonicalization**
- **O(1) routing** via modular arithmetic

## Core Principles

1. **O(1) constant-time complexity** - Prefer direct access over iteration
2. **Zero-copy operations** - Use views/slices instead of clones
3. **Parallel execution** - Design for data parallelism
4. **Compile-time computation** - Use `const`, generic parameters, types
5. **Exact arithmetic** - No floating-point where precision required
6. **No backwards compatibility** - Delete obsolete code

## Specification Documents

### Crate Specifications

| Crate | Specification | Purpose |
|-------|--------------|---------|
| **hologram-core** | [crates/core.md](crates/core.md) | Mathematical foundation + runtime |
| **hologram-compiler** | [crates/compiler.md](crates/compiler.md) | Circuit compilation + canonicalization |
| **hologram-backends** | [crates/backends.md](crates/backends.md) | ISA + backend implementations |
| **hologram-config** | [crates/config.md](crates/config.md) | Configuration + .env support |
| **hologram-ffi** | [crates/ffi.md](crates/ffi.md) | UniFFI-based language bindings |

### Binary Specifications

| Binary | Specification | Purpose |
|--------|--------------|---------|
| **hologram-compile** | [binaries/hologram-compile.md](binaries/hologram-compile.md) | Schema compilation CLI |

### Infrastructure Specifications

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **CI/CD** | [ci.md](ci.md) | GitHub Actions pipeline |
| **Publishing** | [publishing.md](publishing.md) | GitHub Packages publishing |
| **DevContainer** | [devcontainer.md](devcontainer.md) | Development environment |
| **Testing** | [testing.md](testing.md) | Testing strategy |
| **Benchmarking** | [benchmarking.md](benchmarking.md) | Performance benchmarking |

## Project Structure

```
hologram/
├── Cargo.toml (workspace)
├── CLAUDE.md                   # Development guide
├── README.md                    # Project overview
├── .devcontainer/              # See: devcontainer.md
├── .githooks/                   # Git hooks (pre-commit)
├── .github/workflows/          # See: ci.md, publishing.md
├── .cargo/                      # Cargo registry config
├── crates/
│   ├── core/                   # See: crates/core.md
│   ├── compiler/               # See: crates/compiler.md
│   ├── backends/               # See: crates/backends.md
│   ├── config/                 # See: crates/config.md
│   └── ffi/                    # See: crates/ffi.md
├── bins/
│   └── hologram-compile/       # See: binaries/hologram-compile.md
├── kernels/                    # Python/TypeScript/C kernel definitions
├── examples/                   # Usage examples
├── tests/                      # Integration tests (see: testing.md)
├── benches/                    # Benchmarks (see: benchmarking.md)
├── docs/
│   ├── spec/                   # THIS DIRECTORY
│   ├── architecture/           # Architecture documentation
│   ├── api/                    # API reference (generated)
│   └── guides/                 # User guides
└── src/
    └── lib.rs                  # Top-level unified API
```

## Development Workflow

### 1. Before Implementation
- Read relevant specification
- Understand requirements fully
- Check for dependencies on other crates

### 2. During Implementation
- Follow spec exactly
- Write tests alongside code
- Ensure zero warnings
- Keep files < 1K lines

### 3. After Implementation
- Verify spec compliance
- Run full test suite
- Update documentation if API changed
- Create PR with spec reference

## Specification Format

Each specification document follows this structure:

1. **Overview** - Purpose and scope
2. **Public API** - Types, traits, functions
3. **Internal Structure** - Module organization
4. **Dependencies** - External and internal dependencies
5. **Testing Requirements** - Required test coverage
6. **Performance Requirements** - Latency/throughput targets
7. **Examples** - Usage examples
8. **Migration Notes** - Porting from current codebase

## Versioning

Specifications follow semantic versioning:
- **Major**: Breaking API changes
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes, clarifications

## Status Indicators

- **Draft** - Under development
- **Review** - Ready for review
- **Approved** - Reviewed and approved
- **Implemented** - Code matches spec
- **Deprecated** - Replaced by newer spec

## Reconciliation

When code differs from specification:

1. Determine which is correct (spec or code)
2. Update the incorrect one
3. Document the change
4. Test thoroughly

**Default assumption: Specification is correct.**

## Contributing

To update specifications:

1. Create feature branch
2. Update relevant spec documents
3. Ensure consistency across related specs
4. Submit PR with rationale
5. Await review and approval

## Quick Links

- **Start Here:** [Architecture Overview](crates/core.md#architecture)
- **API Reference:** [Core API](crates/core.md#public-api)
- **FFI Bindings:** [FFI Spec](crates/ffi.md)
- **CLI Usage:** [hologram-compile](binaries/hologram-compile.md)
- **Testing:** [Testing Strategy](testing.md)
- **Publishing:** [GitHub Packages Publishing](publishing.md)

## License

Same as hologram (see root LICENSE file)
