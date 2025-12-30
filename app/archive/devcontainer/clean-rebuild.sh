#!/usr/bin/env bash
# Clean rebuild script for devcontainer
set -euo pipefail

echo "🧹 Cleaning up old devcontainer..."

# Stop and remove containers
echo "  → Stopping containers..."
docker compose -f .devcontainer/docker-compose.yml down -v 2>/dev/null || true

# Remove any containers with hologramapp in the name
echo "  → Removing old containers..."
docker ps -a --filter "name=hologramapp" --format "{{.ID}}" | xargs -r docker rm -f 2>/dev/null || true

# Remove images
echo "  → Removing old images..."
docker images --filter "reference=*hologramapp*" --format "{{.ID}}" | xargs -r docker rmi -f 2>/dev/null || true

# Clean build cache (optional - comment out if you want to keep cache)
# echo "  → Cleaning build cache..."
# docker builder prune -f

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Close VS Code completely"
echo "  2. Reopen VS Code in this workspace"
echo "  3. Run: 'Dev Containers: Rebuild Container'"
echo ""
