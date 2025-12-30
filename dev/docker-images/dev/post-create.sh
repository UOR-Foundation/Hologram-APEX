#!/bin/bash
set -e

echo "🚀 Running post-create setup for Hologram workspace..."

# Ensure we're in the correct directory
cd /workspace

# Build the workspace to cache dependencies
echo "📦 Building workspace and caching dependencies..."
cargo fetch --locked 2>/dev/null || cargo fetch

# Run a quick check to ensure everything compiles
echo "🔍 Verifying workspace compiles..."
cargo check --workspace --lib

# Install Python dependencies if requirements.txt exists
if [ -f "../schemas/requirements.txt" ]; then
    echo "🐍 Installing Python dependencies..."
    pip3 install --user -r ../schemas/requirements.txt
fi

# Pre-compile for faster subsequent builds
echo "🔨 Pre-compiling hologram-codegen..."
cargo build -p hologram-codegen --lib

echo "🔨 Pre-compiling hologram-core..."
cargo build -p hologram-core --lib

# Display project info
echo ""
echo "✅ Post-create setup complete!"
echo ""
echo "📊 Workspace Structure:"
echo "  - crates/core       - Core operations library"
echo "  - crates/compiler   - Circuit canonicalization"
echo "  - crates/backends   - Multi-backend execution (CPU/CUDA/Metal/WebGPU)"
echo "  - crates/codegen    - Build-time kernel compilation"
echo "  - crates/config     - Configuration management"
echo "  - crates/ffi        - Multi-language FFI bindings"
echo "  - bins/hologram-compile - CLI compilation tool"
echo ""
echo "🧪 Quick test:"
echo "  cargo test -p hologram-core --lib"
echo ""
echo "📚 Documentation:"
echo "  REFACTORING_COMPLETE.md - Phase 2 summary"
echo "  V2_THREAD_SAFETY.md     - Thread safety implementation"
echo ""
echo "🎯 Ready to develop!"
