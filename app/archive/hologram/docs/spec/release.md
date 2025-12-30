# Release Specification

**Status:** Draft
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

This specification defines the complete release process for Hologram, including version management, testing requirements, publishing workflows, and release artifacts.

## Release Philosophy

- **Semantic Versioning (SemVer)** - MAJOR.MINOR.PATCH
- **Quality Gates** - All tests must pass before release
- **Automated Publishing** - GitHub Actions handles the publishing process
- **Reproducible Builds** - Releases are deterministic and verifiable
- **Comprehensive Artifacts** - Binaries, documentation, and packages

## Versioning Strategy

### Semantic Versioning

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
- 0.1.0          # Pre-1.0 development
- 1.0.0          # First stable release
- 1.2.3          # Patch release
- 2.0.0-alpha.1  # Pre-release
- 2.0.0-beta.2   # Beta release
- 2.0.0-rc.1     # Release candidate
```

**Version Increments:**

- **MAJOR**: Breaking API changes (incompatible changes)
- **MINOR**: New features (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

**Pre-1.0.0 Releases:**
- **0.x.y**: Development phase, API may change
- Breaking changes increment MINOR (0.1.0 → 0.2.0)
- New features increment PATCH (0.1.0 → 0.1.1)

**Post-1.0.0 Releases:**
- API stability guaranteed
- Follow strict SemVer

### Workspace Versioning

**All crates share the same version number:**

```toml
[workspace.package]
version = "0.1.0"  # Single source of truth
```

**Rationale:**
- Simplifies dependency management
- Ensures compatibility across crates
- Easier for users to understand
- Clearer release notes

### Version Locations

**Primary:** `Cargo.toml` (workspace)

```toml
[workspace.package]
version = "0.1.0"
```

**Secondary:** Individual crate manifests

```toml
[package]
version.workspace = true  # Inherits from workspace
```

## Release Process

### 1. Pre-Release Checks

Before starting a release, ensure:

```bash
# All tests pass
cargo test --workspace --all-features

# Zero warnings
cargo build --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings

# Code formatted
cargo fmt --check

# Documentation builds
cargo doc --workspace --no-deps

# Benchmarks run
cargo bench --workspace

# Examples work
cargo run --example basic_operations
cargo run --example tensor_operations
# ... test all examples

# FFI bindings generate
cargo build --features ffi
cargo test --features ffi --package hologram-ffi
```

### 2. Update CHANGELOG

Create entry in `CHANGELOG.md`:

```markdown
## [0.2.0] - 2025-01-20

### Added
- New tensor operation: `reshape()`
- Support for WebGPU backend
- Python FFI bindings via UniFFI

### Changed
- Improved SIMD performance for f32 operations (2× faster)
- Updated DLPack interop to version 0.8

### Fixed
- Buffer allocation race condition on multi-threaded executors
- Tensor stride calculation for non-contiguous arrays

### Breaking Changes
- Renamed `Executor::new_with_backend()` to `Executor::with_backend()`
- Removed deprecated `ops::legacy` module
```

### 3. Version Bump

**Option A: Manual Version Bump**

```bash
# Update version in Cargo.toml
cargo set-version --workspace 0.2.0

# Commit changes
git add -A
git commit -m "chore: bump version to v0.2.0"
```

**Option B: Automated via GitHub Actions**

1. Go to Actions → "Version Bump"
2. Click "Run workflow"
3. Select version type: patch/minor/major
4. Workflow automatically:
   - Bumps version
   - Updates CHANGELOG.md
   - Creates commit
   - Creates tag
   - Pushes to repository

### 4. Create Release Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag
git push origin v0.2.0
```

**Tag triggers automatic publishing workflow.**

### 5. Automated Publishing

GitHub Actions workflow (`.github/workflows/publish.yml`) automatically:

1. **Verifies version matches tag**
2. **Runs full test suite**
3. **Builds release artifacts**:
   - All crates in release mode
   - `hologram-compile` binary
   - FFI bindings
4. **Publishes crates** (in dependency order):
   - hologram-core
   - hologram-compiler
   - hologram-backends
   - hologram-config
   - hologram-ffi
   - hologram (main crate)
5. **Creates GitHub Release**:
   - Auto-generated release notes
   - Attached binaries
   - Links to documentation

### 6. Post-Release Verification

After publishing:

```bash
# Verify crates.io listing
# Check https://crates.io/crates/hologram

# Test installation
cargo new test-hologram
cd test-hologram
echo 'hologram = "0.2.0"' >> Cargo.toml
cargo build

# Verify documentation
# Check https://docs.rs/hologram/0.2.0

# Announce release
# Update website, social media, Discord, etc.
```

## Release Artifacts

### 1. Published Crates

**Primary Registry:** crates.io

| Crate | Description |
|-------|-------------|
| `hologram` | Main crate (re-exports all functionality) |
| `hologram-core` | Mathematical foundation + runtime |
| `hologram-compiler` | Circuit compilation + canonicalization |
| `hologram-backends` | ISA + backend implementations |
| `hologram-config` | Configuration management |
| `hologram-ffi` | FFI bindings (feature-gated) |

### 2. Binaries

**Platform:** Linux, macOS, Windows

| Binary | Description |
|--------|-------------|
| `hologram-compile` | Kernel compilation CLI |

**Distribution:**
- GitHub Releases (download directly)
- Future: Homebrew, Cargo install, package managers

### 3. Documentation

| Location | Content |
|----------|---------|
| docs.rs | API documentation (auto-generated from rustdoc) |
| GitHub Pages | User guides, tutorials, architecture docs |
| README.md | Quick start, installation, links |

### 4. FFI Bindings

**Languages:** Python, Swift, Kotlin, Ruby, TypeScript (Neon), C++ (CXX)

**Distribution:**
- Python: PyPI (future)
- Swift: Swift Package Manager (future)
- TypeScript: npm (future)
- C/C++: Header files in GitHub Release

## Quality Gates

### Pre-Release Quality Gates

**MUST PASS before creating release tag:**

- [ ] All workspace tests pass
- [ ] Zero compiler warnings
- [ ] Zero clippy warnings
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation builds
- [ ] All examples run successfully
- [ ] FFI bindings generate and test successfully
- [ ] Benchmarks run (no regressions)
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml

### Publishing Quality Gates

**MUST PASS during publish workflow:**

- [ ] Version matches tag
- [ ] Full test suite passes
- [ ] Release build succeeds
- [ ] All crates publish successfully
- [ ] GitHub Release created

## Release Types

### Patch Release (0.1.0 → 0.1.1)

**Scope:** Bug fixes only

**Process:**
1. Fix bugs on `main` branch
2. Update CHANGELOG.md (### Fixed)
3. Run pre-release checks
4. Version bump: `cargo set-version --workspace --bump patch`
5. Create tag: `git tag v0.1.1`
6. Push tag: `git push origin v0.1.1`
7. Automated publishing

**Timeline:** As needed (urgent fixes: same day, minor fixes: weekly)

### Minor Release (0.1.0 → 0.2.0)

**Scope:** New features (backwards-compatible)

**Process:**
1. Develop features on feature branches
2. Merge to `main` via pull requests
3. Update CHANGELOG.md (### Added, ### Changed)
4. Run pre-release checks
5. Version bump: `cargo set-version --workspace --bump minor`
6. Create tag: `git tag v0.2.0`
7. Push tag: `git push origin v0.2.0`
8. Automated publishing

**Timeline:** Monthly cadence (or when significant features ready)

### Major Release (0.9.0 → 1.0.0)

**Scope:** Breaking changes, API redesign

**Process:**
1. **Plan breaking changes** - Document in RFC
2. **Deprecation warnings** - Add warnings in previous minor release
3. **Migration guide** - Write comprehensive migration documentation
4. **Beta releases** - Tag as `1.0.0-beta.1`, `1.0.0-beta.2`, etc.
5. **Release candidates** - Tag as `1.0.0-rc.1`, `1.0.0-rc.2`, etc.
6. **Final release** - After RC tested in production
7. Update CHANGELOG.md (### Breaking Changes)
8. Version bump: `cargo set-version --workspace --bump major`
9. Create tag: `git tag v1.0.0`
10. Push tag: `git push origin v1.0.0`
11. Automated publishing
12. **Announce broadly** - Blog post, social media, mailing list

**Timeline:** As needed (major milestones, API stability)

### Pre-Release (2.0.0-alpha.1, 2.0.0-beta.1, 2.0.0-rc.1)

**Scope:** Testing unreleased features

**Types:**
- **Alpha** (α): Early testing, API may change significantly
- **Beta** (β): Feature-complete, API frozen, testing for bugs
- **RC** (Release Candidate): Production-ready candidate

**Process:**
1. Create pre-release tag: `git tag v2.0.0-alpha.1`
2. Push tag: `git push origin v2.0.0-alpha.1`
3. Publish to crates.io with `--allow-dirty` if needed
4. Mark GitHub Release as "Pre-release"
5. Gather feedback
6. Iterate (alpha.2, alpha.3, → beta.1, → rc.1)
7. Final release when RC stable

**Timeline:** As needed for major releases

## Rollback Procedure

### If Release Fails

**During publishing (automated workflow fails):**

1. **Identify failure point** - Check GitHub Actions logs
2. **Fix issue** - Correct code, tests, or configuration
3. **Delete tag** (if not published):
   ```bash
   git tag -d v0.2.0
   git push origin :refs/tags/v0.2.0
   ```
4. **Re-release** - Create tag again after fix

**After publishing (critical bug found):**

1. **Yank broken version from crates.io**:
   ```bash
   cargo yank --version 0.2.0 hologram
   ```
2. **Fix bug immediately**
3. **Release patch version** (0.2.1)
4. **Announce** - Notify users to upgrade

**Note:** Cannot delete published crates, only yank (prevents new downloads but allows existing users)

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Feature X
- Feature Y

### Changed
- Improvement Z

### Fixed
- Bug A

## [0.2.0] - 2025-01-20

### Added
- New `Tensor::reshape()` method
- WebGPU backend support

### Changed
- Improved SIMD performance (2× faster)

### Deprecated
- `Executor::new_with_backend()` (use `Executor::with_backend()` instead)

### Removed
- Legacy `ops::old_module` (deprecated since 0.1.0)

### Fixed
- Buffer allocation race condition
- Tensor stride calculation bug

### Security
- Updated dependency X to fix CVE-YYYY-NNNN

## [0.1.0] - 2025-01-01

Initial release.

[Unreleased]: https://github.com/OWNER/hologram/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OWNER/hologram/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OWNER/hologram/releases/tag/v0.1.0
```

## Release Checklist

### Pre-Release

- [ ] All features merged to `main`
- [ ] All tests passing
- [ ] Zero warnings (compiler + clippy)
- [ ] Code formatted
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with all changes
- [ ] Version bumped in Cargo.toml
- [ ] Examples tested
- [ ] Benchmarks run (no regressions)
- [ ] Migration guide written (if breaking changes)

### Release

- [ ] Create annotated git tag
- [ ] Push tag to GitHub
- [ ] Wait for publish workflow to complete
- [ ] Verify all crates published to crates.io
- [ ] Verify GitHub Release created
- [ ] Verify binaries attached to release

### Post-Release

- [ ] Test installation from crates.io
- [ ] Verify docs.rs built successfully
- [ ] Update website/blog with release announcement
- [ ] Announce on social media, Discord, mailing list
- [ ] Create GitHub Discussion for release
- [ ] Monitor for critical bugs (48-hour window)
- [ ] Start next development cycle (update milestones)

## Emergency Hotfix Procedure

**Critical bug in production release:**

1. **Create hotfix branch** from release tag:
   ```bash
   git checkout -b hotfix/0.2.1 v0.2.0
   ```

2. **Fix bug** (minimal changes only)
3. **Test thoroughly**
4. **Update CHANGELOG.md**
5. **Bump version**: `cargo set-version --workspace 0.2.1`
6. **Merge to main**:
   ```bash
   git checkout main
   git merge --no-ff hotfix/0.2.1
   ```
7. **Create tag**: `git tag v0.2.1`
8. **Push tag**: `git push origin v0.2.1`
9. **Yank broken version**: `cargo yank --version 0.2.0 hologram`
10. **Announce hotfix**

**Timeline:** Within 24 hours of critical bug discovery

## Documentation Requirements

### Release Notes

Each release must include:

1. **Summary** - What's new in this release
2. **Highlights** - 3-5 key improvements
3. **Breaking Changes** - Complete list with migration guide
4. **New Features** - All new functionality
5. **Bug Fixes** - Important bugs fixed
6. **Performance** - Benchmark improvements
7. **Deprecations** - APIs scheduled for removal
8. **Migration Guide** - How to upgrade (for breaking changes)

### API Documentation

- All public APIs documented with rustdoc
- Examples in doc comments
- README updated with new features
- Architecture docs updated if design changed

## Versioning Best Practices

### When to Increment MAJOR (Breaking Changes)

- Public API function signature changed
- Public type removed
- Required dependency major version changed
- Behavior change that breaks existing code
- Minimum Rust version (MSRV) increased significantly

### When to Increment MINOR (New Features)

- New public API added
- New optional feature added
- Performance improvement
- New backend support
- Deprecation warning added (not removal)

### When to Increment PATCH (Bug Fixes)

- Bug fix (behavior correction)
- Documentation improvement
- Internal refactoring (no API change)
- Dependency patch update

## Release Schedule

### Development Phase (0.x.y)

- **Patch releases**: As needed (bug fixes)
- **Minor releases**: Monthly (new features)
- **Major release (1.0.0)**: When API stable and production-ready

### Stable Phase (1.x.y+)

- **Patch releases**: As needed (critical bugs within days, minor bugs weekly)
- **Minor releases**: Every 6 weeks (feature release cycle)
- **Major releases**: Yearly or when necessary (breaking changes)

## Deprecation Policy

**Pre-1.0.0:** Can remove APIs with one minor version warning

**Post-1.0.0:** Deprecation process:

1. **Mark as deprecated** in version N.x.0:
   ```rust
   #[deprecated(since = "1.2.0", note = "Use new_api() instead")]
   pub fn old_api() { }
   ```

2. **Keep for at least one MAJOR version** (N.x.y)

3. **Remove in next MAJOR version** ((N+1).0.0)

**Example:**
- 1.2.0: Deprecate `old_api()`, add `new_api()`
- 1.3.0 - 1.9.0: Both exist (warnings for old_api)
- 2.0.0: Remove `old_api()`

## CI/CD Integration

See [ci.md](ci.md) and [publishing.md](publishing.md) for complete GitHub Actions workflows.

**Key workflows:**
- `.github/workflows/ci.yml` - Test every commit
- `.github/workflows/publish.yml` - Publish on tags
- `.github/workflows/version-bump.yml` - Automated version bumping

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Cargo Publishing Guide](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [Rust API Guidelines - Versioning](https://rust-lang.github.io/api-guidelines/compatibility.html)
