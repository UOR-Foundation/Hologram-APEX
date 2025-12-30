#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$WORKSPACE_ROOT"

echo "================================"
echo "Compiling SD Turbo ONNX Models"
echo "================================"
echo ""

# Build compiler in release mode
echo "📦 Building hologram-onnx-compiler..."
cargo build --release --package hologram-onnx-compiler
echo "✅ Compiler built"
echo ""

# Create output directory
MODELS_DIR="public/public/models"
ONNX_DIR="$MODELS_DIR/onnx/sd-turbo"
COMPILED_DIR="$MODELS_DIR/compiled"
mkdir -p "$COMPILED_DIR"

# SD Turbo models
MODELS=(
    "text_encoder:650MB"
    "unet:1.7GB"
    "vae_decoder:95MB"
)

echo "🔨 Compiling SD Turbo models..."
echo ""

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r model_name size <<< "$model_info"

    ONNX_PATH="$ONNX_DIR/$model_name/model.onnx"
    BIN_PATH="$COMPILED_DIR/${model_name}.bin"

    if [ ! -f "$ONNX_PATH" ]; then
        echo "⚠️  $model_name: ONNX not found, skipping..."
        continue
    fi

    if [ -f "$BIN_PATH" ]; then
        echo "✅ $model_name: Already compiled, skipping..."
        continue
    fi

    echo "⚡ Compiling $model_name ($size)..."
    ./target/release/hologram-onnx-compiler \
        --input "$ONNX_PATH" \
        --output "$BIN_PATH" \
        -O 2

    # Show file sizes
    if [ -f "$BIN_PATH" ]; then
        BIN_SIZE=$(du -h "$BIN_PATH" | cut -f1)
        WEIGHTS_PATH="${BIN_PATH%.bin}.safetensors"
        WEIGHTS_SIZE=$(du -h "$WEIGHTS_PATH" | cut -f1)
        echo "   Binary: $BIN_SIZE"
        echo "   Weights: $WEIGHTS_SIZE"
    fi
    echo ""
done

echo "================================"
echo "✅ Compilation Complete!"
echo "================================"
echo ""
echo "📊 Compiled models:"
ls -lh "$COMPILED_DIR"/*.bin 2>/dev/null || echo "No models compiled yet"
echo ""
echo "📁 Location: $COMPILED_DIR/"
echo ""
echo "Next steps:"
echo "  1. Update demo to use compiled models"
echo "  2. cd public && pnpm dev"
echo "  3. Open http://localhost:3000/demos/stable-diffusion"
echo ""
