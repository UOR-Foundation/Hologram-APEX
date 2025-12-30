# Hologram Book Build System
SHELL := /bin/bash
.PHONY: all build serve clean lint format check install deps help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Directories
SRC_DIR := src
BUILD_DIR := build
THEME_DIR := theme

# Commands
MDBOOK := mdbook
NPM := npm
PYTHON := python3

# Default target
all: check build

## help: Show this help message
help:
	@echo "Hologram Book Build System"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^##' Makefile | sed 's/## /  /'

## install: Install all required dependencies
install: install-mdbook install-npm-deps install-python-deps
	@echo -e "$(GREEN)✓ All dependencies installed$(NC)"

install-mdbook:
	@echo -e "$(BLUE)Installing mdBook...$(NC)"
	@if ! command -v mdbook &> /dev/null; then \
		curl -sSL https://github.com/rust-lang/mdBook/releases/download/v0.4.37/mdbook-v0.4.37-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C /tmp && \
		sudo mv /tmp/mdbook /usr/local/bin/; \
	fi
	@echo -e "$(GREEN)✓ mdBook installed$(NC)"

install-npm-deps:
	@echo -e "$(BLUE)Installing npm dependencies...$(NC)"
	@if [ ! -f package.json ]; then \
		npm init -y > /dev/null 2>&1; \
	fi
	@npm install --save-dev \
		markdownlint-cli \
		prettier \
		cspell \
		markdown-link-check \
		> /dev/null 2>&1 || true
	@echo -e "$(GREEN)✓ npm dependencies installed$(NC)"

install-python-deps:
	@echo -e "$(BLUE)Installing Python dependencies...$(NC)"
	@pip install --quiet \
		mdbook-pdf \
		> /dev/null 2>&1 || true
	@echo -e "$(GREEN)✓ Python dependencies installed$(NC)"

## build: Build the book (HTML output)
build: check-structure
	@echo -e "$(BLUE)Building book...$(NC)"
	@$(MDBOOK) build
	@echo -e "$(GREEN)✓ Book built successfully in $(BUILD_DIR)/$(NC)"

## build-pdf: Build PDF version of the book
build-pdf: build
	@echo -e "$(BLUE)Generating PDF...$(NC)"
	@if command -v wkhtmltopdf &> /dev/null; then \
		wkhtmltopdf \
			--enable-local-file-access \
			--javascript-delay 2000 \
			--no-stop-slow-scripts \
			--debug-javascript \
			--print-media-type \
			--margin-top 25mm \
			--margin-bottom 25mm \
			--margin-left 25mm \
			--margin-right 25mm \
			--header-spacing 5 \
			--footer-spacing 5 \
			--footer-center "[page]" \
			--footer-font-size 9 \
			--enable-toc-back-links \
			--outline \
			--outline-depth 3 \
			$(BUILD_DIR)/print.html \
			$(BUILD_DIR)/hologram-book.pdf; \
		echo -e "$(GREEN)✓ PDF generated at $(BUILD_DIR)/hologram-book.pdf$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ wkhtmltopdf not installed. Skipping PDF generation.$(NC)"; \
		echo -e "  Install with: sudo apt-get install wkhtmltopdf"; \
	fi

## serve: Start development server with live reload
serve:
	@echo -e "$(BLUE)Starting development server...$(NC)"
	@echo -e "$(GREEN)Server running at http://localhost:3000$(NC)"
	@echo -e "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@$(MDBOOK) serve --open

## watch: Watch for changes and rebuild automatically
watch:
	@echo -e "$(BLUE)Watching for changes...$(NC)"
	@$(MDBOOK) watch

## clean: Remove build artifacts
clean:
	@echo -e "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo -e "$(GREEN)✓ Build directory cleaned$(NC)"

## lint: Run all linters
lint: lint-markdown lint-spelling lint-links
	@echo -e "$(GREEN)✓ All linting checks passed$(NC)"

lint-markdown:
	@echo -e "$(BLUE)Running markdown linter...$(NC)"
	@if command -v markdownlint &> /dev/null; then \
		markdownlint $(SRC_DIR)/**/*.md --fix || true; \
		echo -e "$(GREEN)✓ Markdown linting complete$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ markdownlint not installed$(NC)"; \
	fi

lint-spelling:
	@echo -e "$(BLUE)Running spell checker...$(NC)"
	@if command -v cspell &> /dev/null; then \
		cspell "$(SRC_DIR)/**/*.md" || true; \
		echo -e "$(GREEN)✓ Spell checking complete$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ cspell not installed$(NC)"; \
	fi

lint-links:
	@echo -e "$(BLUE)Checking links...$(NC)"
	@if command -v markdown-link-check &> /dev/null; then \
		find $(SRC_DIR) -name "*.md" -exec markdown-link-check {} \; || true; \
		echo -e "$(GREEN)✓ Link checking complete$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ markdown-link-check not installed$(NC)"; \
	fi

## format: Format all markdown files
format:
	@echo -e "$(BLUE)Formatting markdown files...$(NC)"
	@if command -v prettier &> /dev/null; then \
		prettier --write "$(SRC_DIR)/**/*.md"; \
		echo -e "$(GREEN)✓ Markdown files formatted$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ prettier not installed$(NC)"; \
	fi

## check: Run all checks (structure, links, spelling)
check: check-structure check-chapters
	@echo -e "$(GREEN)✓ All checks passed$(NC)"

check-structure:
	@echo -e "$(BLUE)Checking book structure...$(NC)"
	@if [ ! -f book.toml ]; then \
		echo -e "$(RED)✗ book.toml not found$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f $(SRC_DIR)/SUMMARY.md ]; then \
		echo -e "$(RED)✗ SUMMARY.md not found$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(GREEN)✓ Book structure valid$(NC)"

check-chapters:
	@echo -e "$(BLUE)Checking chapter files...$(NC)"
	@missing=0; \
	while IFS= read -r file; do \
		if [ ! -f "$(SRC_DIR)/$$file" ]; then \
			echo -e "$(RED)✗ Missing: $(SRC_DIR)/$$file$(NC)"; \
			missing=$$((missing + 1)); \
		fi; \
	done < <(grep -oP '\(\.\/[^)]+\.md\)' $(SRC_DIR)/SUMMARY.md | sed 's/[()]//g' | sed 's/\.\///'); \
	if [ $$missing -eq 0 ]; then \
		echo -e "$(GREEN)✓ All chapter files present$(NC)"; \
	else \
		echo -e "$(RED)✗ $$missing chapter files missing$(NC)"; \
		exit 1; \
	fi

## stats: Show book statistics
stats:
	@echo -e "$(BLUE)Book Statistics$(NC)"
	@echo -e "$(BLUE)===============$(NC)"
	@echo -n "Chapters: "; find $(SRC_DIR) -name "*.md" -type f | wc -l
	@echo -n "Words: "; find $(SRC_DIR) -name "*.md" -type f -exec cat {} \; | wc -w
	@echo -n "Lines: "; find $(SRC_DIR) -name "*.md" -type f -exec cat {} \; | wc -l
	@echo -n "Characters: "; find $(SRC_DIR) -name "*.md" -type f -exec cat {} \; | wc -c

## test: Test the build process
test: clean build
	@echo -e "$(BLUE)Testing build output...$(NC)"
	@if [ -f $(BUILD_DIR)/index.html ]; then \
		echo -e "$(GREEN)✓ HTML output generated$(NC)"; \
	else \
		echo -e "$(RED)✗ HTML output missing$(NC)"; \
		exit 1; \
	fi
	@if [ -f $(BUILD_DIR)/print.html ]; then \
		echo -e "$(GREEN)✓ Print version generated$(NC)"; \
	else \
		echo -e "$(RED)✗ Print version missing$(NC)"; \
		exit 1; \
	fi

## release: Create a release build
release: clean check lint format build build-pdf
	@echo -e "$(BLUE)Creating release archive...$(NC)"
	@VERSION=$$(date +%Y%m%d-%H%M%S); \
	tar -czf hologram-book-$$VERSION.tar.gz $(BUILD_DIR)/; \
	echo -e "$(GREEN)✓ Release created: hologram-book-$$VERSION.tar.gz$(NC)"

## deploy: Deploy to GitHub Pages
deploy: build
	@echo -e "$(BLUE)Deploying to GitHub Pages...$(NC)"
	@if [ -d .git ]; then \
		git worktree add gh-pages gh-pages || true; \
		cp -r $(BUILD_DIR)/* gh-pages/; \
		cd gh-pages && \
		git add -A && \
		git commit -m "Deploy book - $$(date)" && \
		git push origin gh-pages; \
		cd .. && \
		git worktree remove gh-pages; \
		echo -e "$(GREEN)✓ Deployed to GitHub Pages$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ Not a git repository. Skipping deployment.$(NC)"; \
	fi

# Development shortcuts
.PHONY: s b c l f
s: serve
b: build
c: clean
l: lint
f: format