#!/usr/bin/env bash
set -euo pipefail

# Hologram FFI Rebuild Script
# Comprehensive script that:
# 1. Cleans hologram-ffi (optional, with --clean flag)
# 2. Rebuilds hologram-ffi in release mode
# 3. Copies library to language binding directories
# 4. Reinstalls PyTorch extension
#
# Usage:
#   ./scripts/rebuild_ffi.sh          # Build without cleaning
#   ./scripts/rebuild_ffi.sh --clean  # Clean and build (fixes UniFFI checksum errors)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
CLEAN_FIRST=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN_FIRST=true
fi

# Determine library extension based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    LIB_EXT="dll"
else
    LIB_EXT="so"
fi

echo "🔨 Hologram FFI Rebuild"
echo "======================="
echo ""

# Step 1: Clean (optional)
if [[ "$CLEAN_FIRST" == true ]]; then
    echo "🧹 Step 1: Cleaning hologram-ffi package..."
    cd "$WORKSPACE_ROOT"
    cargo clean -p hologram-ffi
    echo "✓ Cleaned hologram-ffi"
    echo ""
fi

# Step 2: Build hologram-ffi
echo "🔨 Step 2: Building hologram-ffi (release mode)..."
cd "$WORKSPACE_ROOT"
cargo build --release -p hologram-ffi
echo "✓ Built hologram-ffi"
echo ""

# Step 3: Copy to language binding directories
echo "📦 Step 3: Copying library to binding directories..."

RELEASE_LIB="$WORKSPACE_ROOT/target/release/libhologram_ffi.$LIB_EXT"
if [[ ! -f "$RELEASE_LIB" ]]; then
    echo "❌ Error: $RELEASE_LIB not found"
    exit 1
fi

# Copy to Python FFI bindings (if they exist)
PYTHON_FFI_DEST="$WORKSPACE_ROOT/crates/hologram-ffi/interfaces/python/hologram_ffi/libuniffi_hologram_ffi.$LIB_EXT"
if [[ -d "$(dirname "$PYTHON_FFI_DEST")" ]]; then
    cp -v "$RELEASE_LIB" "$PYTHON_FFI_DEST"
    echo "✓ Copied to Python FFI bindings"
else
    echo "⚠️  Skipped Python FFI bindings (directory not found)"
fi
echo ""

# Step 4: Reinstall PyTorch extension (if it exists)
PYTORCH_DIR="$WORKSPACE_ROOT/hologram-sdk/python/hologram-torch"
if [[ -d "$PYTORCH_DIR" ]]; then
    echo "🐍 Step 4: Reinstalling hologram-torch extension..."
    cd "$PYTORCH_DIR"
    pip install --force-reinstall --no-deps -e .
    echo "✓ Reinstalled hologram-torch"
    echo ""
else
    echo "⚠️  Step 4: Skipped (hologram-torch not found)"
    echo ""
fi

echo "✅ FFI rebuild complete!"
echo ""
echo "📝 What was updated:"
echo "   • hologram-ffi library (release mode)"
echo "   • Python FFI bindings (if present)"
echo "   • PyTorch extension (if present)"
echo ""
echo "💡 Usage notes:"
echo "   • Run with --clean flag to fix UniFFI checksum errors"
echo "   • Example: ./scripts/rebuild_ffi.sh --clean"
