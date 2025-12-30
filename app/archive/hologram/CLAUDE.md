# Hologram Development Guide

This document provides development guidelines for AI-assisted development (specifically Claude Code) when working on the Hologram project.

## Core Principle: Specification-Driven Development

**🚨 CRITICAL: All development MUST follow the specifications in `/docs/spec/`.**

The `/docs/spec/` directory contains the canonical source of truth for:
- Architecture and design
- Crate structure and APIs
- Testing requirements
- CI/CD pipelines
- Development workflows

### Workflow

1. **Before writing code**: Read the relevant specification in `/docs/spec/`
2. **During development**: Ensure your implementation matches the spec
3. **After implementation**: Verify compliance with the spec
4. **When uncertain**: Consult `/docs/spec/INDEX.md` for the complete specification index

### Specification Structure

```
docs/spec/
├── INDEX.md                    # Master index (START HERE)
├── crates/
│   ├── core.md                # hologram-core specification
│   ├── compiler.md            # hologram-compiler specification
│   ├── backends.md            # hologram-backends specification
│   ├── config.md              # hologram-config specification
│   └── ffi.md                 # hologram-ffi specification
├── binaries/
│   └── hologram-compile.md    # CLI binary specification
├── ci.md                       # GitHub Actions/CI specification
├── publishing.md               # GitHub Packages publishing
├── devcontainer.md             # Development environment
├── testing.md                  # Testing strategy
└── benchmarking.md             # Benchmarking requirements
```

## Core Development Principles

### 1. No TODOs, No Stubs, Ever

**🚨 CRITICAL: NEVER LEAVE PLACEHOLDERS OR TODOs.**

- ❌ NO `todo!()` macros
- ❌ NO `unimplemented!()` macros
- ❌ NO `// TODO:` comments
- ❌ NO stub functions
- ❌ NO placeholder implementations

✅ **ONLY** complete, production-ready, fully-tested implementations

**If you cannot implement a function completely RIGHT NOW, do NOT create it.**

### 2. Ruthless Simplicity

**Keep files and methods as simple as necessary.**

- Favor simplicity over cleverness
- Delete unnecessary abstractions
- Inline small functions used once
- Remove unused parameters
- Eliminate dead code paths

### 3. Performance-First Design

**Optimize for:**

1. **O(1) complexity** - Use direct access over iteration
2. **Zero-copy operations** - Use references, slices, views
3. **Parallel execution** - Design for data parallelism
4. **Compile-time computation** - Use `const`, generic parameters, types

### 4. Test-Driven Development

**No feature is complete without comprehensive tests.**

Every piece of code must include:
- Unit tests for individual functions
- Integration tests for component interactions
- Property-based tests (proptest) for mathematical invariants
- Benchmarks for performance-critical code

**Completion criteria:**
- `cargo test --workspace` passes
- `cargo clippy --workspace -- -D warnings` produces zero warnings
- `cargo build --workspace` produces zero warnings

### 5. Task Completion Discipline

**Complete every task fully. No shortcuts, no excuses.**

- Never stop before completion
- No excuses about constraints
- Do the work sequentially
- Finish what you start

### 6. No Backwards Compatibility

**We do NOT support backwards compatibility.**

- Delete old code, don't deprecate it
- Don't add feature gates for old APIs
- Clean removal over preservation

## Common Utilities

**Write common utility functions in the appropriate crate when they can be used across multiple crates.**

Avoid code duplication:
- Don't copy-paste utility functions across crates
- Make utilities reusable and generic
- Document thoroughly

## Documentation Standards

**All `.md` documentation files belong in `/docs/`**

Exceptions:
- `README.md` - Project overview at repository root
- `CLAUDE.md` - This development guide at repository root

### Documentation Organization

All documentation must be organized into subdirectories:
- `docs/spec/` - **Specifications (source of truth)**
- `docs/architecture/` - Architecture documentation
- `docs/api/` - API reference (generated from rustdoc)
- `docs/guides/` - User guides and tutorials

**NEVER** place documentation directly in `/docs/` root.

## Continuous Integration

Before committing code, ensure:

```bash
# Build everything (zero warnings required)
cargo build --workspace --all-targets

# Run all tests (must all pass)
cargo test --workspace

# Check formatting (must pass)
cargo fmt --check

# Run clippy (zero warnings required)
cargo clippy --workspace -- -D warnings
```

**🚨 CRITICAL: All commands must complete successfully with zero warnings.**

## Specification Compliance

When implementing any feature:

1. **Read the spec** - Find relevant specification in `/docs/spec/`
2. **Understand requirements** - Ensure you understand all requirements
3. **Implement completely** - Match the specification exactly
4. **Test thoroughly** - Verify spec compliance
5. **Update spec if needed** - If requirements change, update spec first

### Reconciliation Process

If current code differs from specification:

1. **Identify discrepancy** - What differs from spec?
2. **Determine correct state** - Is spec or code correct?
3. **Update accordingly**:
   - If spec is correct: Update code to match spec
   - If code reveals spec issue: Update spec, then code
4. **Test changes** - Ensure compliance
5. **Document** - Update any related documentation

## Getting Started

**New to the project?**

1. Read `/docs/spec/INDEX.md` - Understand the complete architecture
2. Review crate specifications in `/docs/spec/crates/`
3. Check out examples in `/examples/`
4. Run the test suite: `cargo test --workspace`
5. Try building: `cargo build --workspace`

## Contributing

When adding new functionality:

1. **Update specification first** - Modify `/docs/spec/` before coding
2. **Implement to spec** - Write code that matches updated spec
3. **Add tests** - Comprehensive test coverage
4. **Update documentation** - Ensure docs reflect changes
5. **Run CI checks** - All tests and lints must pass

## Resources

### Project Documentation

- [Specification Index](/docs/spec/INDEX.md) - **START HERE**
- [Architecture Overview](/docs/architecture/overview.md)
- [API Reference](/docs/api/) - Generated from rustdoc

### External Resources

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [UniFFI Book](https://mozilla.github.io/uniffi-rs/) - For FFI bindings

---

**Remember: The specification in `/docs/spec/` is the source of truth. When in doubt, consult the spec.**
