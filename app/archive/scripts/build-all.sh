#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Hologram Build System - Automated Build Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="public/public/models"
BIN_DIR="$MODELS_DIR/bin/sd-turbo"
ONNX_DIR="$MODELS_DIR/onnx/sd-turbo"
WASM_PKG_DIR="public/public/pkg"
COMPILER_BIN="target/release/hologram-onnx-compiler"

# Track if anything was rebuilt
REBUILT_SOMETHING=false

# Helper functions
log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check and download models
echo ""
log_step "Step 1: Checking ONNX models..."
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "$ONNX_DIR/unet" ] && [ -d "$ONNX_DIR/text_encoder" ] && [ -d "$ONNX_DIR/vae_decoder" ]; then
    log_success "ONNX models already downloaded"
else
    log_step "Downloading models from HuggingFace..."
    ./scripts/download_models.sh
    REBUILT_SOMETHING=true
fi

# Step 2: Build ONNX compiler (always)
echo ""
log_step "Step 2: Building hologram-onnx-compiler..."
echo "────────────────────────────────────────────────────────────────────────────────"

log_step "Building compiler (release mode)..."
cargo build --release --bin hologram-onnx-compiler
log_success "Compiler built successfully"
REBUILT_SOMETHING=true

# Step 3: Compile ONNX models to .bin format
echo ""
log_step "Step 3: Compiling ONNX models to .bin format..."
echo "────────────────────────────────────────────────────────────────────────────────"

mkdir -p "$BIN_DIR"

COMPONENTS=("text_encoder" "unet" "vae_decoder")

for component in "${COMPONENTS[@]}"; do
    ONNX_FILE="$ONNX_DIR/$component/model.onnx"
    BIN_FILE="$BIN_DIR/$component.bin"
    WEIGHTS_FILE="$BIN_DIR/$component.safetensors"

    if [ ! -f "$ONNX_FILE" ]; then
        log_error "$component ONNX model not found at $ONNX_FILE"
        continue
    fi

    NEEDS_COMPILE=false

    # Check if binary files exist
    if [ ! -f "$BIN_FILE" ] || [ ! -f "$WEIGHTS_FILE" ]; then
        NEEDS_COMPILE=true
    # Check if ONNX model is newer than compiled binary
    elif is_newer "$ONNX_FILE" "$BIN_FILE"; then
        NEEDS_COMPILE=true
    # Check if compiler is newer than compiled binary
    elif is_newer "$COMPILER_BIN" "$BIN_FILE"; then
        NEEDS_COMPILE=true
    fi

    if [ "$NEEDS_COMPILE" = true ]; then
        log_step "Compiling $component..."
        if "$COMPILER_BIN" --input "$ONNX_FILE" --output "$BIN_FILE"; then
            BIN_SIZE=$(du -h "$BIN_FILE" | cut -f1)
            WEIGHTS_SIZE=$(du -h "$WEIGHTS_FILE" | cut -f1)
            log_success "$component compiled: $BIN_SIZE binary + $WEIGHTS_SIZE weights"
            REBUILT_SOMETHING=true
        else
            log_error "Failed to compile $component"
            exit 1
        fi
    else
        log_skip "$component.bin is up to date"
    fi
done

# Step 4: Build WASM package (always)
echo ""
log_step "Step 4: Building WASM package..."
echo "────────────────────────────────────────────────────────────────────────────────"

log_step "Building WASM package..."
./scripts/build-wasm-demo.sh
log_success "WASM package built successfully"
REBUILT_SOMETHING=true

# Step 5: Install npm dependencies (always)
echo ""
log_step "Step 5: Installing npm dependencies..."
echo "────────────────────────────────────────────────────────────────────────────────"

cd public
log_step "Running pnpm install..."
pnpm install
log_success "Dependencies installed"
cd "$WORKSPACE_ROOT"
REBUILT_SOMETHING=true

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Build Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

log_success "All components built successfully"

echo ""
echo "📊 System Status:"
echo "   Compiler: $COMPILER_BIN"
echo "   Models:   $BIN_DIR/"
ls -lh "$BIN_DIR/" 2>/dev/null | grep -E '\.(bin|safetensors)$' | awk '{printf "      - %s (%s)\n", $9, $5}'
echo "   WASM:     $WASM_PKG_DIR/"
echo ""

echo "🚀 Next Steps:"
echo ""
echo "   Start development server:"
echo "   $ cd public && pnpm dev"
echo ""
echo "   Open demo in browser:"
echo "   $ open http://localhost:3000/demos/stable-diffusion"
echo ""
echo "   Or run both in one command:"
echo "   $ ./scripts/start-demo.sh"
echo ""

# # Offer to start the server
# if [ -t 0 ]; then  # Check if running interactively
#     echo -n "Would you like to start the development server now? (y/N) "
#     read -r response
#     if [[ "$response" =~ ^[Yy]$ ]]; then
#         echo ""
#         log_step "Starting development server..."
#         cd public
#         exec pnpm dev
#     fi
# fi
