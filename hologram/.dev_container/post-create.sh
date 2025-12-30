#!/bin/bash
set -e

echo "🚀 Running post-create setup for Hologram workspace..."

# Ensure we're in the correct directory
cd /workspace

# Verify Rust toolchain
echo "🦀 Verifying Rust toolchain..."
rustc --version
cargo --version

# Build the workspace to cache dependencies
echo "📦 Building workspace and caching dependencies..."
cargo fetch --locked 2>/dev/null || cargo fetch

# Run a quick check to ensure everything compiles
echo "🔍 Verifying workspace compiles..."
cargo check --workspace --lib

# Install Python dependencies for FFI tests if they exist
if [ -f "crates/ffi/interfaces/python/requirements.txt" ]; then
    echo "🐍 Installing Python dependencies for FFI tests..."
    pip3 install --user -r crates/ffi/interfaces/python/requirements.txt
fi

# Pre-compile frequently used crates for faster subsequent builds
echo "🔨 Pre-compiling hologram-compiler..."
cargo build -p hologram-compiler --lib

echo "🔨 Pre-compiling hologram-core..."
cargo build -p hologram-core --lib

echo "🔨 Pre-compiling hologram-backends..."
cargo build -p hologram-backends --lib

# Display project info
echo ""
echo "✅ Post-create setup complete!"
echo ""
echo "📊 Workspace Structure:"
echo "  - crates/core       - Core operations & executor (HRM, tensors, ops)"
echo "  - crates/compiler   - Circuit compilation & canonicalization"
echo "  - crates/backends   - Multi-backend execution (CPU/CUDA/Metal/WebGPU)"
echo "  - crates/common     - Configuration, tracing, error handling"
echo "  - crates/ffi        - Multi-language FFI bindings (Python, Kotlin)"
echo "  - binaries/hologram-compile - CLI compilation tool"
echo ""
echo "🧪 Quick tests:"
echo "  cargo test -p hologram-compiler --lib  # Test compiler (92.60% coverage)"
echo "  cargo test -p hologram-common --lib    # Test config system"
echo "  cargo test --workspace --lib           # Test all libraries"
echo ""
echo "📊 Test coverage:"
echo "  ./scripts/coverage.sh                  # Generate HTML coverage report"
echo "  ./scripts/coverage.sh lcov             # Generate LCOV for CI"
echo ""
echo "📚 Documentation:"
echo "  docs/guides/getting-started.md         - Getting started guide"
echo "  docs/guides/multi-backend.md           - Multi-backend execution"
echo "  docs/guides/circuit-compilation.md     - Circuit compilation guide"
echo "  docs/testing/coverage.md               - Test coverage report"
echo "  MIGRATION_MAP.md                       - Migration status tracker"
echo ""
echo "🎯 Ready to develop!"
