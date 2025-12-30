# Hologram Component Manager
# Simple Makefile for common operations

# Variables
SRC_DIR = src
DIST_DIR = src/dist

# Default target
.PHONY: help
help:
	@echo "Hologram Component Manager"
	@echo ""
	@echo "Common commands:"
	@echo "  make build          # Build TypeScript"
	@echo "  make test           # Run tests"
	@echo "  make validate       # Validate all components"
	@echo "  make clean          # Clean all build artifacts"
	@echo ""
	@echo "MCP Server:"
	@echo "  make mcp            # Start MCP server"
	@echo ""
	@echo "Maintenance:"
	@echo "  make gc             # Check for orphaned files"
	@echo "  make gc-clean       # Remove orphaned files"
	@echo ""
	@echo "All commands:"
	@echo "  make help           # Show this help"

# Install dependencies
.PHONY: install
install:
	@cd $(SRC_DIR) && npm install

# Build TypeScript
.PHONY: build
build: install
	@echo "🔨 Building TypeScript..."
	@cd $(SRC_DIR) && npm run build

# Clean everything
.PHONY: clean
clean:
	@echo "🧹 Cleaning..."
	@cd $(SRC_DIR) && npm run clean
	@rm -rf .artifacts

# Run tests
.PHONY: test
test: build
	@echo "🧪 Running tests..."
	@cd $(SRC_DIR) && npm test

# Run tests with coverage
.PHONY: test-coverage
test-coverage: build
	@echo "📊 Running tests with coverage..."
	@cd $(SRC_DIR) && npm run test:coverage

# Validate all components
.PHONY: validate
validate: build
	@echo "🔍 Validating components..."
	@cd $(SRC_DIR) && npm run validate

# Start MCP server
.PHONY: mcp
mcp: build
	@echo "🚀 Starting MCP server..."
	@cd $(SRC_DIR) && npm run mcp

# Check for orphaned files
.PHONY: gc
gc: build
	@cd $(SRC_DIR) && npm run gc

# Remove orphaned files
.PHONY: gc-clean
gc-clean: build
	@cd $(SRC_DIR) && npm run gc:clean

# Lint code
.PHONY: lint
lint:
	@echo "✨ Linting code..."
	@cd $(SRC_DIR) && npm run lint

# Format code
.PHONY: format
format:
	@echo "🎨 Formatting code..."
	@cd $(SRC_DIR) && npm run format

# Check formatting
.PHONY: format-check
format-check:
	@echo "📝 Checking code format..."
	@cd $(SRC_DIR) && npm run format:check

# Watch mode for development
.PHONY: watch
watch:
	@echo "👁️  Watching for changes..."
	@cd $(SRC_DIR) && npm run watch

.DEFAULT_GOAL := help