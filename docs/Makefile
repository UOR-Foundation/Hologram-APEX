# Multi-Book Build System
SHELL := /bin/bash
.PHONY: all help list

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Find all book directories (those with book.toml)
BOOK_DIRS := $(dir $(wildcard */book.toml))
BOOKS := $(BOOK_DIRS:%/=%)

# Default target
all: help

## help: Show this help message
help:
	@echo "Multi-Book Build System"
	@echo "======================"
	@echo ""
	@echo "Available commands:"
	@echo "  make list                    - List all available books"
	@echo "  make build-all              - Build all books"
	@echo "  make clean-all              - Clean all book builds"
	@echo "  make serve BOOK=<name>      - Serve a specific book"
	@echo "  make build BOOK=<name>      - Build a specific book"
	@echo "  make clean BOOK=<name>      - Clean a specific book"
	@echo "  make new BOOK=<name>        - Create a new book from template"
	@echo ""
	@echo "Available books:"
	@for book in $(BOOKS); do \
		echo "  - $$book"; \
	done

## list: List all available books
list:
	@echo -e "$(BLUE)Available books:$(NC)"
	@for book in $(BOOKS); do \
		if [ -f $$book/book.toml ]; then \
			title=$$(grep "^title" $$book/book.toml | cut -d'"' -f2); \
			echo -e "  $(GREEN)$$book$(NC): $$title"; \
		fi; \
	done

## build-all: Build all books
build-all:
	@echo -e "$(BLUE)Building all books...$(NC)"
	@for book in $(BOOKS); do \
		echo -e "$(BLUE)Building $$book...$(NC)"; \
		$(MAKE) -C $$book build; \
	done
	@echo -e "$(GREEN)✓ All books built successfully$(NC)"

## clean-all: Clean all book builds
clean-all:
	@echo -e "$(BLUE)Cleaning all books...$(NC)"
	@for book in $(BOOKS); do \
		echo -e "$(BLUE)Cleaning $$book...$(NC)"; \
		$(MAKE) -C $$book clean; \
	done
	@echo -e "$(GREEN)✓ All books cleaned$(NC)"

## pdf-all: Build PDFs for all books
pdf-all:
	@echo -e "$(BLUE)Building PDFs for all books...$(NC)"
	@for book in $(BOOKS); do \
		echo -e "$(BLUE)Building PDF for $$book...$(NC)"; \
		$(MAKE) -C $$book build-pdf; \
	done
	@echo -e "$(GREEN)✓ All PDFs built successfully$(NC)"

## deploy-prep: Prepare combined build for deployment
deploy-prep: build-all
	@echo -e "$(BLUE)Preparing deployment directory...$(NC)"
	@rm -rf _site
	@mkdir -p _site
	@for book in $(BOOKS); do \
		echo -e "$(BLUE)Copying $$book to _site/$$book$(NC)"; \
		mkdir -p _site/$$book; \
		cp -r $$book/build/* _site/$$book/; \
	done
	@echo -e "$(GREEN)✓ Deployment directory ready at _site/$(NC)"

## serve: Serve a specific book (usage: make serve BOOK=book-name)
serve:
	@if [ -z "$(BOOK)" ]; then \
		echo -e "$(RED)Error: BOOK parameter required$(NC)"; \
		echo "Usage: make serve BOOK=<book-name>"; \
		exit 1; \
	fi
	@if [ ! -d "$(BOOK)" ]; then \
		echo -e "$(RED)Error: Book '$(BOOK)' not found$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)Serving $(BOOK)...$(NC)"
	@$(MAKE) -C $(BOOK) serve

## build: Build a specific book (usage: make build BOOK=book-name)
build:
	@if [ -z "$(BOOK)" ]; then \
		echo -e "$(RED)Error: BOOK parameter required$(NC)"; \
		echo "Usage: make build BOOK=<book-name>"; \
		exit 1; \
	fi
	@if [ ! -d "$(BOOK)" ]; then \
		echo -e "$(RED)Error: Book '$(BOOK)' not found$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)Building $(BOOK)...$(NC)"
	@$(MAKE) -C $(BOOK) build

## clean: Clean a specific book (usage: make clean BOOK=book-name)
clean:
	@if [ -z "$(BOOK)" ]; then \
		echo -e "$(RED)Error: BOOK parameter required$(NC)"; \
		echo "Usage: make clean BOOK=<book-name>"; \
		exit 1; \
	fi
	@if [ ! -d "$(BOOK)" ]; then \
		echo -e "$(RED)Error: Book '$(BOOK)' not found$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)Cleaning $(BOOK)...$(NC)"
	@$(MAKE) -C $(BOOK) clean

## new: Create a new book from template (usage: make new BOOK=book-name)
new:
	@if [ -z "$(BOOK)" ]; then \
		echo -e "$(RED)Error: BOOK parameter required$(NC)"; \
		echo "Usage: make new BOOK=<book-name>"; \
		exit 1; \
	fi
	@if [ -d "$(BOOK)" ]; then \
		echo -e "$(RED)Error: Book '$(BOOK)' already exists$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)Creating new book '$(BOOK)'...$(NC)"
	@mkdir -p $(BOOK)/src $(BOOK)/theme
	@cp shared/config/book.template.toml $(BOOK)/book.toml
	@cp shared/templates/Makefile.template $(BOOK)/Makefile
	@ln -s ../shared/theme/custom.css $(BOOK)/theme/custom.css
	@echo "# Summary" > $(BOOK)/src/SUMMARY.md
	@echo "" >> $(BOOK)/src/SUMMARY.md
	@echo "[Introduction](./introduction.md)" >> $(BOOK)/src/SUMMARY.md
	@echo "# Introduction" > $(BOOK)/src/introduction.md
	@echo "" >> $(BOOK)/src/introduction.md
	@echo "Welcome to your new book!" >> $(BOOK)/src/introduction.md
	@echo -e "$(GREEN)✓ Book '$(BOOK)' created successfully$(NC)"
	@echo -e "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit $(BOOK)/book.toml to set your book title and metadata"
	@echo "  2. Edit $(BOOK)/src/SUMMARY.md to define your table of contents"
	@echo "  3. Run 'make serve BOOK=$(BOOK)' to start developing"

# Shortcuts for common books
.PHONY: book hologram
book:
	@$(MAKE) serve BOOK=book

hologram:
	@$(MAKE) serve BOOK=book