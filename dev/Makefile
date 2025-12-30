# Docker Image Build and Publish Makefile
# Configurable variables for Docker Hub publishing

# Load environment variables from .env file if it exists
-include .env

# Configuration - Set these via .env file, environment variables, or override when calling make
DOCKER_USERNAME ?= yourorgname
DOCKER_PASSWORD ?=
IMAGE_NAME ?= hologram-devcontainer
VERSION ?= latest
PYTHON_VERSION ?= 3.12
NODE_VERSION ?= 20
RUST_VERSION ?= nightly

# Multi-architecture support
PLATFORMS ?= linux/amd64,linux/arm64
BUILDX_BUILDER ?= hologram-builder

# Computed variables
DEV_IMAGE := $(DOCKER_USERNAME)/$(IMAGE_NAME):$(VERSION)
PROD_IMAGE := $(DOCKER_USERNAME)/$(IMAGE_NAME):prod-$(VERSION)
DEV_LATEST := $(DOCKER_USERNAME)/$(IMAGE_NAME):latest
PROD_LATEST := $(DOCKER_USERNAME)/$(IMAGE_NAME):prod-latest

.PHONY: help
help: ## Show this help message
	@echo "Docker Image Build and Publish Commands"
	@echo "========================================"
	@echo ""
	@echo "Configuration (set via .env file, environment, or make arguments):"
	@echo "  DOCKER_USERNAME  - Docker Hub username/org (default: $(DOCKER_USERNAME))"
	@echo "  IMAGE_NAME       - Image name (default: $(IMAGE_NAME))"
	@echo "  VERSION          - Version tag (default: $(VERSION))"
	@echo "  PLATFORMS        - Target platforms (default: $(PLATFORMS))"
	@echo ""
	@echo "Examples:"
	@echo "  make build-dev DOCKER_USERNAME=myorg VERSION=v1.0.0"
	@echo "  make buildx-dev DOCKER_USERNAME=myorg PLATFORMS=linux/amd64,linux/arm64"
	@echo "  make publish-all-multiarch DOCKER_USERNAME=myorg VERSION=v1.0.0"
	@echo ""
	@echo "Available targets:"
	@grep -h -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /|/' | awk -F'|' '{printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

.PHONY: build-dev
build-dev: ## Build development image
	@echo "Building development image: $(DEV_IMAGE)"
	docker build \
		--build-arg PYTHON_VARIANT=1-$(PYTHON_VERSION)-bullseye \
		--build-arg NODE_VERSION=$(NODE_VERSION) \
		--build-arg RUST_VERSION=$(RUST_VERSION) \
		--target dev \
		-t $(DEV_IMAGE) \
		-t $(DEV_LATEST) \
		-f docker-images/dev/Dockerfile \
		docker-images
	@echo "✅ Development image built successfully"

.PHONY: build-prod
build-prod: ## Build production image
	@echo "Building production image: $(PROD_IMAGE)"
	docker build \
		--build-arg PYTHON_VARIANT=$(PYTHON_VERSION)-slim-bullseye \
		--build-arg NODE_VERSION=$(NODE_VERSION) \
		--build-arg RUST_VERSION=$(RUST_VERSION) \
		--target production \
		-t $(PROD_IMAGE) \
		-t $(PROD_LATEST) \
		-f docker-images/prod/Dockerfile \
		docker-images
	@echo "✅ Production image built successfully"

.PHONY: build-all
build-all: build-dev build-prod ## Build both dev and prod images

# Multi-architecture builds using buildx
.PHONY: buildx-setup
buildx-setup: ## Setup Docker buildx for multi-arch builds
	@echo "Setting up Docker buildx..."
	@docker buildx inspect $(BUILDX_BUILDER) > /dev/null 2>&1 || \
		docker buildx create --name $(BUILDX_BUILDER) --driver docker-container --bootstrap --use
	@echo "✅ Buildx builder '$(BUILDX_BUILDER)' ready"

.PHONY: buildx-dev
buildx-dev: buildx-setup ## Build multi-arch development image
	@echo "Building multi-arch development image: $(DEV_IMAGE)"
	@echo "Platforms: $(PLATFORMS)"
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg PYTHON_VARIANT=1-$(PYTHON_VERSION)-bullseye \
		--build-arg NODE_VERSION=$(NODE_VERSION) \
		--build-arg RUST_VERSION=$(RUST_VERSION) \
		--target dev \
		-t $(DEV_IMAGE) \
		-t $(DEV_LATEST) \
		-f docker-images/dev/Dockerfile \
		--push \
		docker-images
	@echo "✅ Multi-arch development image built and pushed"

.PHONY: buildx-prod
buildx-prod: buildx-setup ## Build multi-arch production image
	@echo "Building multi-arch production image: $(PROD_IMAGE)"
	@echo "Platforms: $(PLATFORMS)"
	docker buildx build \
		--platform $(PLATFORMS) \
		--build-arg PYTHON_VARIANT=$(PYTHON_VERSION)-slim-bullseye \
		--build-arg NODE_VERSION=$(NODE_VERSION) \
		--build-arg RUST_VERSION=$(RUST_VERSION) \
		--target production \
		-t $(PROD_IMAGE) \
		-t $(PROD_LATEST) \
		-f docker-images/prod/Dockerfile \
		--push \
		docker-images
	@echo "✅ Multi-arch production image built and pushed"

.PHONY: buildx-all
buildx-all: buildx-dev buildx-prod ## Build and push all multi-arch images

.PHONY: publish-all-multiarch
publish-all-multiarch: login buildx-all ## Login and publish all multi-arch images

.PHONY: test-dev
test-dev: ## Test development image
	@echo "Testing development image..."
	docker run --rm $(DEV_IMAGE) bash -c " \
		echo '🧪 Running development image tests...' && \
		rustc --version && \
		cargo --version && \
		python3 --version && \
		node --version && \
		pnpm --version && \
		sqlite3 --version && \
		zig version && \
		echo '✅ All tools verified successfully!'"

.PHONY: test-prod
test-prod: ## Test production image
	@echo "Testing production image..."
	docker run --rm $(PROD_IMAGE) bash -c " \
		echo '🧪 Running production image tests...' && \
		rustc --version && \
		cargo --version && \
		python3 --version && \
		node --version && \
		pnpm --version && \
		sqlite3 --version && \
		zig version && \
		echo '✅ All tools verified successfully!'"

.PHONY: test-all
test-all: test-dev test-prod ## Test both images

.PHONY: login
login: ## Login to Docker Hub
	@echo "Logging in to Docker Hub..."
	@if [ -n "$(DOCKER_PASSWORD)" ]; then \
		echo "Using credentials from environment..."; \
		echo "$(DOCKER_PASSWORD)" | docker login -u "$(DOCKER_USERNAME)" --password-stdin; \
	else \
		echo "No DOCKER_PASSWORD in .env, prompting for credentials..."; \
		docker login; \
	fi

.PHONY: push-dev
push-dev: ## Push development image to Docker Hub
	@echo "Pushing development image to Docker Hub..."
	docker push $(DEV_IMAGE)
	docker push $(DEV_LATEST)
	@echo "✅ Development image pushed: $(DEV_IMAGE)"
	@echo "✅ Development image pushed: $(DEV_LATEST)"

.PHONY: push-prod
push-prod: ## Push production image to Docker Hub
	@echo "Pushing production image to Docker Hub..."
	docker push $(PROD_IMAGE)
	docker push $(PROD_LATEST)
	@echo "✅ Production image pushed: $(PROD_IMAGE)"
	@echo "✅ Production image pushed: $(PROD_LATEST)"

.PHONY: push-all
push-all: push-dev push-prod ## Push both images to Docker Hub

.PHONY: publish-dev
publish-dev: build-dev test-dev push-dev ## Build, test, and push development image

.PHONY: publish-prod
publish-prod: build-prod test-prod push-prod ## Build, test, and push production image

.PHONY: publish-all
publish-all: build-all test-all push-all ## Build, test, and push all images

.PHONY: clean
clean: ## Remove local images
	@echo "Removing local images..."
	-docker rmi $(DEV_IMAGE) $(DEV_LATEST) $(PROD_IMAGE) $(PROD_LATEST)
	@echo "✅ Local images removed"

.PHONY: info
info: ## Show configuration info
	@echo "Current Configuration:"
	@echo "====================="
	@echo "Docker Username: $(DOCKER_USERNAME)"
	@echo "Image Name:      $(IMAGE_NAME)"
	@echo "Version:         $(VERSION)"
	@echo "Platforms:       $(PLATFORMS)"
	@echo ""
	@echo "Image Tags:"
	@echo "  Dev (versioned):  $(DEV_IMAGE)"
	@echo "  Dev (latest):     $(DEV_LATEST)"
	@echo "  Prod (versioned): $(PROD_IMAGE)"
	@echo "  Prod (latest):    $(PROD_LATEST)"
	@echo ""
	@echo "Build Args:"
	@echo "  Python: $(PYTHON_VERSION)"
	@echo "  Node:   $(NODE_VERSION)"
	@echo "  Rust:   $(RUST_VERSION)"
