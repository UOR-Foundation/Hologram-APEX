# Hologram Development Environment

A comprehensive development environment setup for the Hologram unified workspace, featuring multi-architecture Docker images, VS Code DevContainer configuration, and automated CI/CD workflows.

## Overview

This repository contains:

- **[docker-images/](docker-images/)** - Production-ready Docker images for development and production environments
- **[.devcontainer/](.devcontainer/)** - VS Code DevContainer configuration
- **[Makefile](Makefile)** - Build and publish automation for Docker images

## Quick Start

### Using the DevContainer (Recommended)

The fastest way to get started:

1. **Install VS Code** and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open this project in VS Code**

3. **Reopen in Container** when prompted (or `Ctrl+Shift+P` → "Dev Containers: Reopen in Container")

The devcontainer will automatically pull the pre-built development image and start your environment with all tools ready:
- Rust nightly with full toolchain
- Python 3.12
- Node.js 20 with pnpm
- LLVM 11, SQLite3, Zig
- Docker-in-Docker support
- Oh My Zsh with plugins

See [.devcontainer/README.md](.devcontainer/README.md) for more details.

### Building Docker Images Locally

If you want to build or customize the Docker images:

1. **Configure your credentials** (one-time setup):
   ```bash
   cp .env.example .env
   # Edit .env and set DOCKER_USERNAME=yourorg
   ```

2. **Build images:**
   ```bash
   # Build development image
   make build-dev

   # Build production image
   make build-prod

   # Build both
   make build-all

   # Build multi-architecture images (amd64 + arm64)
   make buildx-all

   # Show all available commands
   make help
   ```

You can also override settings via command-line arguments:
```bash
make build-dev DOCKER_USERNAME=yourorg VERSION=v2.0.0
```

See [docker-images/README.md](docker-images/README.md) for comprehensive documentation.

## Project Structure

```
hologram-dev/
├── .devcontainer/              # VS Code DevContainer configuration
│   ├── devcontainer.json       # Container config (uses published images)
│   └── README.md               # DevContainer usage guide
│
├── docker-images/              # Docker image source and build configs
│   ├── dev/                    # Development image (full toolchain)
│   │   ├── Dockerfile          # Multi-stage build for dev environment
│   │   ├── devcontainer.json   # Dev-specific VS Code config
│   │   ├── docker-compose.yml  # Docker Compose for development
│   │   └── dotfiles/           # Developer dotfiles (.zshrc, .gitconfig)
│   │
│   ├── prod/                   # Production image (optimized)
│   │   ├── Dockerfile          # Minimal production build
│   │   └── docker-compose.yml  # Production compose config
│   │
│   ├── .github/workflows/      # CI/CD automation
│   │   └── publish-images.yml  # Multi-arch build and publish workflow
│   │
│   ├── scripts/                # Helper scripts and examples
│   │   ├── publish-images-native-arm64.yml  # Alternative workflow
│   │   ├── setup-oracle-arm64-runner.md     # Free ARM64 runner setup
│   │   └── README.md           # Scripts documentation
│   │
│   ├── README.md               # Comprehensive Docker images docs
│   └── QUICK_START.md          # Quick start guide
│
├── Makefile                    # Build and publish automation
└── README.md                   # This file
```

## What's Included

### Development Environment

The development Docker image includes:

**Languages & Runtimes:**
- Rust nightly (rustfmt, clippy, rust-src)
- Python 3.12
- Node.js 20 with pnpm
- LLVM 11
- SQLite3
- Zig 0.11.0

**Development Tools:**
- cargo-watch, cargo-expand, cargo-edit
- wasm-pack for WebAssembly
- Docker CLI (Docker-in-Docker)
- Oh My Zsh with plugins
- Git, vim, curl, jq

**VS Code Extensions:**
- rust-analyzer
- Python + Pylance + Black
- Claude Code
- ESLint, Prettier
- LLDB debugger

### Production Environment

The production image is optimized for deployment:
- Minimal base image
- Same language runtimes without dev tools
- Smaller image size
- Security-focused configuration

## Multi-Architecture Support

All Docker images support both **AMD64 (x86_64)** and **ARM64** architectures out of the box:

- **Intel/AMD machines**: Pull and run natively
- **Apple Silicon (M1/M2/M3)**: Run natively without Rosetta
- **ARM servers**: AWS Graviton, Oracle Cloud, etc.

Docker automatically pulls the correct architecture for your platform.

### Building for Multiple Architectures

```bash
# Build for both amd64 and arm64
make buildx-all DOCKER_USERNAME=yourorg

# Build for specific platforms
make buildx-dev DOCKER_USERNAME=yourorg PLATFORMS=linux/arm64

# See all multi-arch options
make help
```

For faster ARM64 builds, see [docker-images/README.md#arm64-build-alternatives](docker-images/README.md#arm64-build-alternatives).

## Publishing Images to Docker Hub

### Prerequisites

1. **Create a Docker Hub account** at https://hub.docker.com

2. **For local publishing** - Configure credentials via `.env`:
   ```bash
   # Copy and edit .env file
   cp .env.example .env

   # Add your credentials:
   # DOCKER_USERNAME=yourorg
   # DOCKER_PASSWORD=dckr_pat_your_token_here

   # Create a Personal Access Token at:
   # https://hub.docker.com/settings/security
   ```

3. **For automated CI/CD** - Add secrets to your GitHub repository:
   - `DOCKER_USERNAME` - Your Docker Hub username
   - `DOCKER_PASSWORD` - Your Docker Hub password or access token

### Manual Publishing

```bash
# Login to Docker Hub
make login

# Build, test, and publish all images
make publish-all DOCKER_USERNAME=yourorg VERSION=v1.0.0

# Or publish multi-architecture images
make publish-all-multiarch DOCKER_USERNAME=yourorg VERSION=v1.0.0
```

### Automated Publishing (GitHub Actions)

The project includes automated CI/CD:

1. **Push to main branch** - Builds and publishes images with `latest` tag
2. **Create git tag** (e.g., `v1.0.0`) - Publishes versioned images
3. **Manual workflow dispatch** - Trigger builds manually from GitHub UI

Images are automatically published to Docker Hub with multi-architecture support.

See [docker-images/.github/workflows/](docker-images/.github/workflows/) for workflow details.

## Using Published Images

Once published, use your images across multiple projects:

**In any project's `.devcontainer/devcontainer.json`:**
```json
{
  "name": "My Project",
  "image": "yourorgname/hologram-devcontainer:dev-latest",
  "workspaceFolder": "/workspace",
  "remoteUser": "vscode"
}
```

**In Docker Compose:**
```yaml
services:
  app:
    image: yourorgname/hologram-devcontainer:prod-latest
    volumes:
      - ./:/app
```

**Direct Docker run:**
```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  yourorgname/hologram-devcontainer:dev-latest \
  bash
```

## Configuration

### Using .env File (Recommended)

The easiest way to configure your local environment:

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env with your Docker Hub credentials:**
   ```bash
   # .env file
   DOCKER_USERNAME=myorg
   DOCKER_PASSWORD=dckr_pat_your_token_here
   IMAGE_NAME=hologram-devcontainer
   VERSION=v1.0.0
   PLATFORMS=linux/amd64,linux/arm64
   ```

   **To create a Docker Hub Personal Access Token:**
   - Go to https://hub.docker.com/settings/security
   - Click "New Access Token"
   - Give it a description (e.g., "hologram-dev builds")
   - Copy the token and paste it as `DOCKER_PASSWORD` in your `.env` file

3. **Build and publish:**
   ```bash
   make publish-all-multiarch
   ```

The `.env` file is automatically loaded by the Makefile and is ignored by git (your credentials stay local and secure).

### Makefile Variables

You can also customize builds via environment variables or command-line arguments:

```bash
make build-all \
  DOCKER_USERNAME=myorg \
  VERSION=v2.0.0 \
  PYTHON_VERSION=3.11 \
  NODE_VERSION=18 \
  RUST_VERSION=stable \
  PLATFORMS=linux/amd64,linux/arm64
```

**Priority order:** Command-line arguments > Environment variables > .env file > Makefile defaults

See `make help` for all options and `make info` for current configuration.

## Common Tasks

### Update to Latest Images

```bash
# Pull latest images
docker pull yourorgname/hologram-devcontainer:dev-latest

# Rebuild devcontainer in VS Code
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### Customize Images

1. Fork or modify Dockerfiles in [docker-images/dev/](docker-images/dev/) or [docker-images/prod/](docker-images/prod/)
2. Build locally: `make build-dev`
3. Test: `make test-dev`
4. Publish: `make publish-dev DOCKER_USERNAME=yourorg`

### Add Custom Tools

Edit the appropriate Dockerfile:

```dockerfile
# In docker-images/dev/Dockerfile, add before the final USER command:
RUN cargo install your-custom-tool
```

Then rebuild and publish.

## Documentation

- **[docker-images/README.md](docker-images/README.md)** - Comprehensive Docker images documentation
  - Multi-architecture builds
  - ARM64 build alternatives
  - GitHub Actions CI/CD
  - Cloud build services

- **[docker-images/QUICK_START.md](docker-images/QUICK_START.md)** - Quick start guide for DevContainers

- **[docker-images/scripts/README.md](docker-images/scripts/README.md)** - Alternative workflows and setup guides
  - Native ARM64 builds
  - Oracle Cloud free tier setup
  - Docker Build Cloud

- **[.devcontainer/README.md](.devcontainer/README.md)** - DevContainer usage and configuration

## Troubleshooting

### DevContainer won't start

```bash
# Pull latest image
docker pull yourorgname/hologram-devcontainer:dev-latest

# Rebuild container
# In VS Code: Ctrl+Shift+P → "Dev Containers: Rebuild Container Without Cache"
```

### Docker build fails

```bash
# Clean rebuild
make clean
make build-all DOCKER_USERNAME=yourorg
```

### ARM64 builds are slow

See [docker-images/README.md#arm64-build-alternatives](docker-images/README.md#arm64-build-alternatives) for faster options:
- Use Oracle Cloud free ARM64 runner (free!)
- Use Docker Build Cloud
- Build on Apple Silicon locally

### Permission issues

```bash
# Fix file permissions in workspace
sudo chown -R $(id -u):$(id -g) .
```

## Support

- **Documentation**: See linked READMEs above
- **Issues**: Report issues in this repository
- **VS Code DevContainers**: [Official Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- **Docker Multi-Arch**: [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)

## Contributing

Contributions welcome! Areas to improve:

- Add more language runtimes
- Optimize build times
- Add more VS Code extensions
- Improve documentation
- Add helper scripts

## License

[Add your license here]

## Version History

- **v1.0.0** - Initial release
  - Multi-architecture support (amd64 + arm64)
  - Development and production images
  - Automated CI/CD with GitHub Actions
  - Rust nightly, Python 3.12, Node.js 20
  - LLVM 11, SQLite3, Zig 0.11.0
