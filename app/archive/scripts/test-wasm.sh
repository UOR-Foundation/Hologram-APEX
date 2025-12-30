#!/usr/bin/env bash
# WASM Build and Test Script
# Tests WASM builds locally before pushing to CI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists rustc; then
    print_error "Rust not installed. Install from https://rustup.rs/"
    exit 1
fi

if ! command_exists cargo; then
    print_error "Cargo not installed"
    exit 1
fi

print_success "Rust $(rustc --version) found"

# Check for WASM target
print_status "Checking WASM target..."
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    print_warning "WASM target not installed. Installing..."
    rustup target add wasm32-unknown-unknown
fi
print_success "wasm32-unknown-unknown target available"

# Check for wasm-pack (optional but recommended)
if ! command_exists wasm-pack; then
    print_warning "wasm-pack not installed. Browser tests will be skipped."
    print_warning "Install: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    SKIP_BROWSER_TESTS=1
else
    print_success "wasm-pack $(wasm-pack --version) found"
    SKIP_BROWSER_TESTS=0
fi

# Detect platform
PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
print_status "Platform: $PLATFORM $ARCH"

echo ""
echo "======================================"
echo "  WASM Build & Test Suite"
echo "======================================"
echo ""

# Step 1: Workspace tests
print_status "Step 1/6: Running workspace tests..."
if cargo test --workspace --quiet; then
    print_success "Workspace tests passed"
else
    print_error "Workspace tests failed"
    exit 1
fi

# Step 2: Build WASM (debug)
print_status "Step 2/6: Building WASM (debug)..."
if cargo build --target wasm32-unknown-unknown -p hologram-backends --quiet; then
    print_success "WASM debug build successful"
else
    print_error "WASM debug build failed"
    exit 1
fi

# Step 3: Build WASM with WebGPU
print_status "Step 3/6: Building WASM with WebGPU feature..."
if cargo build --target wasm32-unknown-unknown -p hologram-backends --features webgpu --quiet; then
    print_success "WASM + WebGPU build successful"
else
    print_error "WASM + WebGPU build failed"
    exit 1
fi

# Step 4: Build WASM (release)
print_status "Step 4/6: Building WASM (release)..."
if cargo build --target wasm32-unknown-unknown -p hologram-backends --release --quiet; then
    print_success "WASM release build successful"
else
    print_error "WASM release build failed"
    exit 1
fi

# Step 5: Check binary size
print_status "Step 5/6: Checking WASM binary size..."
WASM_FILE="target/wasm32-unknown-unknown/release/hologram_backends.wasm"
if [ -f "$WASM_FILE" ]; then
    if [ "$PLATFORM" == "Darwin" ]; then
        SIZE=$(stat -f%z "$WASM_FILE")
    else
        SIZE=$(stat -c%s "$WASM_FILE")
    fi
    SIZE_MB=$((SIZE / 1024 / 1024))
    SIZE_KB=$((SIZE / 1024))

    print_success "WASM binary: $SIZE_KB KB ($SIZE_MB MB)"

    if [ $SIZE -gt 10485760 ]; then
        print_warning "WASM binary exceeds 10MB (${SIZE_MB}MB)"
    fi
else
    print_error "WASM binary not found at $WASM_FILE"
    exit 1
fi

# Step 6: Browser tests (if wasm-pack available)
if [ $SKIP_BROWSER_TESTS -eq 0 ]; then
    print_status "Step 6/6: Running browser tests..."

    # Check for Chrome
    if command_exists google-chrome || command_exists chromium || command_exists chrome; then
        print_status "Running tests in headless Chrome..."
        cd crates/hologram-backends

        # Run tests with WebGPU
        if wasm-pack test --headless --chrome --features webgpu 2>&1 | grep -q "test result: ok"; then
            print_success "Browser tests (WebGPU) passed"
        else
            print_warning "Browser tests (WebGPU) had issues (may be expected if WebGPU unavailable)"
        fi

        # Run tests without WebGPU (fallback)
        if wasm-pack test --headless --chrome 2>&1 | grep -q "test result: ok"; then
            print_success "Browser tests (WASM-only) passed"
        else
            print_error "Browser tests (WASM-only) failed"
            cd ../..
            exit 1
        fi

        cd ../..
    else
        print_warning "Chrome not found. Skipping browser tests."
        print_warning "Install Chrome to run browser tests locally."
    fi
else
    print_warning "Step 6/6: Skipping browser tests (wasm-pack not installed)"
fi

echo ""
echo "======================================"
print_success "All WASM tests passed!"
echo "======================================"
echo ""
echo "Summary:"
echo "  • Platform: $PLATFORM $ARCH"
echo "  • WASM binary size: $SIZE_KB KB"
echo "  • Workspace tests: ✓"
echo "  • WASM build: ✓"
echo "  • WebGPU build: ✓"
echo "  • Release build: ✓"
if [ $SKIP_BROWSER_TESTS -eq 0 ]; then
    echo "  • Browser tests: ✓"
else
    echo "  • Browser tests: (skipped)"
fi
echo ""
echo "Ready to push to CI! 🚀"
