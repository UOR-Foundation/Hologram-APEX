# Hologramapp Documentation

This directory contains technical documentation for the hologramapp project.

## Architecture & Design

### [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md)

**Essential reading for backend development**

Comprehensive explanation of the backend architecture design, including:

- Why `instruction_ops` are free functions vs trait methods
- Performance analysis and tradeoffs
- Guidelines for implementing new backends (GPU, TPU, FPGA)
- Testing strategies
- Clear separation between backend-specific and shared operations

**Key takeaway**: The current design achieves optimal performance and maintainability by keeping shared instructions as free functions and only backend-specific operations as trait methods.

### [Backend Trait Architecture](./architecture/BACKEND_TRAIT_ARCHITECTURE.md)

Documentation of the two-trait architecture:

- `Backend` trait: Public API (buffer/pool management, program execution)
- `Executor` trait: Internal execution interface (instruction dispatch, synchronization)

Created during the refactoring that moved all execution logic into the `CpuExecutor` struct.

### [CPU Backend Tracing](./performance/CPU_BACKEND_TRACING.md)

Comprehensive guide to the tracing instrumentation added to the CPU backend:

- How to enable and configure tracing
- Performance impact analysis
- Example output and usage
- Environment variables for configuration
- Integration with `hologram-tracing` crate

## Circuit Compilation

### Circuit Compiler Guide

User guide for the canonical circuit compiler:

- Circuit compiler usage
- Pattern-based canonicalization
- 96-class geometric system
- Generator operations
- Transform algebra

Technical implementation details:

- AST structure
- Parser design
- Rewrite engine
- Canonicalization algorithms
- Performance characteristics

## Future Work

### [Future Prompts](./FUTURE_PROMPTS.md)

**Development roadmap and task queue**

Contains planned features and improvements:

- Move circuit compilation from runtime to compile-time
- Implement GPU backend
- Implement TPU backend
- Backend auto-selection
- And more...

This document serves as the task backlog for the project.

## Project Guidelines

See [../CLAUDE.md](../CLAUDE.md) for:

- Development workflow
- Code organization standards
- Testing requirements
- Documentation standards
- Common patterns and best practices

## Contributing

When adding new documentation:

1. Place `.md` files in the `/docs` directory
2. Add entry to this README with brief description
3. Update references in [CLAUDE.md](../CLAUDE.md)
4. Use clear headers and table of contents for long documents
5. Include code examples where appropriate

## Quick Links

### By Topic

**Backend Development**:
- [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md) - Design decisions
- [Backend Trait Architecture](./architecture/BACKEND_TRAIT_ARCHITECTURE.md) - Trait design
- [CPU Backend Tracing](./performance/CPU_BACKEND_TRACING.md) - Performance monitoring

**Circuit Compilation**:

- Circuit Compiler Guide - User guide
- Circuit compiler implementation - Technical details

**Planning**:

- [Future Prompts](./FUTURE_PROMPTS.md) - Roadmap and task queue

### By Audience

**New Contributors**:

1. Start with [CLAUDE.md](../CLAUDE.md) - Development guidelines
2. Read [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md) - System design
3. Review Circuit Compiler Guide - Core concepts

**Backend Implementers** (GPU, TPU, etc.):
1. [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md) - **Must read**
2. [Backend Trait Architecture](./architecture/BACKEND_TRAIT_ARCHITECTURE.md)
3. [CPU Backend Tracing](./performance/CPU_BACKEND_TRACING.md) - Add tracing to your backend

**Performance Engineers**:
1. [CPU Backend Tracing](./performance/CPU_BACKEND_TRACING.md) - Monitoring tools
2. [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md) - Performance analysis
3. Circuit Compiler Guide - Canonicalization optimization

## Document Status

| Document | Status | Last Updated | Maintenance |
|----------|--------|--------------|-------------|
| [Backend Architecture](./architecture/BACKEND_ARCHITECTURE.md) | ✅ Current | 2025-10-28 | Update when adding new backends |
| [Backend Trait Architecture](./architecture/BACKEND_TRAIT_ARCHITECTURE.md) | ✅ Current | 2024-12 | Update if trait design changes |
| [CPU Backend Tracing](./performance/CPU_BACKEND_TRACING.md) | ✅ Current | 2025-10-28 | Update when adding new metrics |
| Circuit Compiler Guide | ✅ Current | 2024 | Update with new features |
| Circuit compiler implementation | ✅ Current | 2024 | Update with major changes |
| [Future Prompts](./FUTURE_PROMPTS.md) | 🔄 Living Document | Ongoing | Update as tasks complete |

---

**Note**: All documentation follows the project standard that `.md` files belong in `/docs`, with two exceptions:

- `README.md` - Project overview at repository root
- `CLAUDE.md` - Development guide at repository root
