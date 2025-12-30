#!/bin/bash
# Build script for hologram-onnx WASM demo

set -e

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "================================"
echo "Building Hologram-ONNX WASM Demo"
echo "================================"
echo ""

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found"
    echo "Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

echo "✓ wasm-pack found"

# Check if we're in the workspace root
if [ ! -d "$WORKSPACE_ROOT/hologram-sdk/rust/hologram-onnx" ]; then
    echo "❌ hologram-onnx directory not found at $WORKSPACE_ROOT/hologram-sdk/rust/hologram-onnx"
    exit 1
fi

echo "✓ In workspace root"
echo ""

# Navigate to hologram-onnx
cd "$WORKSPACE_ROOT/hologram-sdk/rust/hologram-onnx"

echo "Building hologram-onnx for WASM..."
echo "Target: web"
echo "Features: webgpu"
echo "Output: public/lib/onnx"
echo ""

# Define output directory (Next.js structure - matches @/lib/onnx import alias)
WASM_DIR="$WORKSPACE_ROOT/public/lib/onnx"

# Build with wasm-pack (output directly to lib directory)
wasm-pack build \
    --target web \
    --out-dir "$WASM_DIR" \
    --features webgpu

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ WASM build complete"
    echo ""

    # Add cache-busting to generated JS file
    echo "Adding cache-busting to hologram_onnx.js..."
    JS_FILE="$WASM_DIR/hologram_onnx.js"

    # Replace the WASM URL line with cache-busting version
    if [ -f "$JS_FILE" ]; then
        sed -i "s|module_or_path = new URL('hologram_onnx_bg.wasm', import.meta.url);|// Cache-busting: add timestamp query parameter to force reload\n        const wasmUrl = new URL('hologram_onnx_bg.wasm', import.meta.url);\n        wasmUrl.searchParams.set('v', Date.now().toString());\n        module_or_path = wasmUrl;|" "$JS_FILE"
        echo "✓ Cache-busting added"
    else
        echo "⚠ Warning: Could not find $JS_FILE to add cache-busting"
    fi

    echo ""
    echo "================================"
    echo "✅ Build successful!"
    echo "================================"
    echo ""
    echo "WASM output: $WASM_DIR"
    echo ""
    echo "Files generated:"
    ls -lh "$WASM_DIR"
    echo ""
    echo "To test the demo:"
    echo "  1. cd $WORKSPACE_ROOT/public"
    echo "  2. pnpm dev"
    echo "  3. Open http://localhost:3000/demos/stable-diffusion in Chrome 113+ with WebGPU enabled"
    echo ""
    echo "Note: Demo requires SD Turbo models in public/models/onnx/sd-turbo/"
    echo "Run ./scripts/download_models.sh to download models"
    echo ""
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi
