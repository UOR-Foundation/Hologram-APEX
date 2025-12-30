#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Hologram Stable Diffusion Demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build everything first
echo "🔨 Building system (if needed)..."
echo ""
./scripts/build-all.sh

# Check if build was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed. Please fix errors and try again."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Starting Development Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Server will be available at:"
echo "   http://localhost:3000"
echo ""
echo "Demo page:"
echo "   http://localhost:3000/demos/stable-diffusion"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

cd public
exec pnpm dev
