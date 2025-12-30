# Atlas (monorepo)

This repository (Atlas) is a monorepo composed from multiple projects. It contains code, docs, and CI configuration for the subprojects listed below.

## Repository layout

- app/ — (project)  
- archive/ — (archival material)  
- atlas/ — (core project)  
- dev/ — (development tooling)  
- docs/ — (consolidated documentation)
- embeddings/ — (project)  
- hologram/ — (project)  
- mcp/ — (project)  
- onnx/ — (project)  
- sigmatics/ — (project)  

## Documentation

All consolidated docs are in the `docs/` directory. Start with the `docs/index.md` TOC (or `docs/README.md`). Below are some pointers to the most commonly used documentation pages.

### Key documents and sections
- [Overview, quick start, and FFI docs](docs/index.md)
- [Architecture](ARCHITECTURE.md)
- [Getting started and build instructions](guides/getting-started.md)
- [API Reference](api/)
- [Concepts and Design](concepts/)
- [Examples and Tutorials](examples/)
- [Testing and CI](testing/)

If you're browsing on GitHub, `docs/index.md` or `docs/README.md` are a good place to start.

## CI and workflows

GitHub Actions workflows live in `.github/workflows/`. Workflows that were migrated from other repositories have been consolidated here — ensure per-workflow path filters are updated for this monorepo layout. Non-workflow GitHub configuration (issue templates, copilot instructions, changelog configuration, etc.) remain in `.github/`.

## Contributing

- See `docs/` for contribution guidance.
- For code style, testing, and local checks, refer to project-specific README files inside each subdirectory.

## Quickstart

1. Clone the repo
2. See the per-project README at `<project> / README.md` for build/run/test commands.
3. To run monorepo CI locally, use your preferred runner or verify workflows via GitHub Actions.

## Contact / Maintainers

- Maintained by the CitizenGardens-org organization.
