# Hologram CLI

A command-line interface for running AI models locally, mirroring Ollama's command structure but using `hologram-onnx` for model conversion/compilation and the Hologram runtime for O(1) inference.

## Installation

```bash
cargo install --path bins/hologram
```

Or build from source:

```bash
cargo build --release -p hologram
```

## Quick Start

```bash
# Start the server
hologram serve

# Pull a model
hologram pull deepseek-ai/DeepSeek-OCR

# Run a model
hologram run deepseek-ocr --prompt "Describe this image"

# List models
hologram list
```

## Commands

### Server Commands

#### `hologram serve`

Start the Hologram server. All other commands communicate with this server.

```bash
hologram serve [OPTIONS]

Options:
  --host <HOST>           Host to bind to [default: 127.0.0.1]
  --port <PORT>           Port to listen on [default: 11434]
  --memory-budget <MB>    Maximum memory for loaded models [default: 8192]
  --idle-timeout <SECS>   Unload models after idle seconds [default: 300]
```

#### `hologram stop <MODEL>`

Stop a running model, unloading it from memory.

```bash
hologram stop sd-turbo
```

#### `hologram ps`

List currently running models.

```bash
hologram ps

# Output:
# NAME          SIZE      PROCESSOR    UNTIL
# sd-turbo      4.2 GB    100% GPU     4 minutes from now
# bert-base     440 MB    100% CPU     Forever
```

### Model Management

#### `hologram list`

List all locally available models.

```bash
hologram list

# Output:
# NAME                    ID              SIZE      MODIFIED
# sd-turbo:latest         a1b2c3d4e5f6    4.2 GB    2 days ago
# bert-base:latest        f6e5d4c3b2a1    440 MB    5 days ago
```

#### `hologram show <MODEL>`

Show detailed information about a model.

```bash
hologram show sd-turbo

# Output:
# Model: sd-turbo:latest
# Architecture: stable-diffusion
# Parameters: 860M
# Quantization: fp16
# Format: .holo (compiled)
#
# Input Shapes:
#   prompt: [1, 77]
#
# Output Shapes:
#   image: [1, 3, 512, 512]
```

#### `hologram pull <MODEL>`

Pull a model from a registry (HuggingFace or Hologram registry).

```bash
# Pull from HuggingFace
hologram pull stabilityai/sd-turbo

# Pull specific file
hologram pull stabilityai/sd-turbo:unet

# Pull with progress
hologram pull deepseek-ai/DeepSeek-OCR --verbose
```

#### `hologram push <MODEL>`

Push a model to the Hologram registry (requires authentication).

```bash
hologram push my-model:latest
```

#### `hologram create <NAME>`

Create a model from a Holofile (similar to Ollama's Modelfile).

```bash
hologram create my-model -f Holofile

# Or from ONNX directly
hologram create my-model --from model.onnx
```

**Holofile format:**

```dockerfile
FROM stabilityai/sd-turbo

# Set input shapes
INPUT prompt 1,77
INPUT negative_prompt 1,77

# Compilation settings
MEMORY_BUDGET 8192
PARALLEL true

# Metadata
DESCRIPTION "My fine-tuned SD model"
```

#### `hologram cp <SOURCE> <DEST>`

Copy a model to a new name.

```bash
hologram cp sd-turbo:latest my-sd:v1
```

#### `hologram rm <MODEL>`

Remove a model from local storage.

```bash
hologram rm sd-turbo

# Remove multiple
hologram rm sd-turbo bert-base

# Force remove running model
hologram rm -f sd-turbo
```

### Inference

#### `hologram run <MODEL>`

Run a model interactively or with a prompt.

```bash
# Interactive mode
hologram run sd-turbo

# With prompt
hologram run sd-turbo --prompt "A sunset over mountains"

# With input file
hologram run bert-base --input data.json --output results.json

# Streaming output
hologram run llama --prompt "Hello" --stream
```

### Authentication

#### `hologram signin`

Sign in to the Hologram registry.

```bash
hologram signin

# Or with token
hologram signin --token <TOKEN>
```

#### `hologram signout`

Sign out from the Hologram registry.

```bash
hologram signout
```

## Configuration

### Configuration File

Configuration is loaded from `~/.hologram/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 11434
memory_budget_mb = 8192
idle_timeout_secs = 300

[models]
path = "~/.hologram/models"

[registry]
default = "https://registry.hologram.dev"

[logging]
level = "info"  # trace, debug, info, warn, error
```

### Environment Variables

All configuration options can be overridden with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOLOGRAM_HOST` | Server host | `127.0.0.1` |
| `HOLOGRAM_PORT` | Server port | `11434` |
| `HOLOGRAM_MODELS_PATH` | Model storage directory | `~/.hologram/models` |
| `HOLOGRAM_MEMORY_BUDGET` | Max memory (MB) | `8192` |
| `HOLOGRAM_IDLE_TIMEOUT` | Model idle timeout (secs) | `300` |
| `HOLOGRAM_REGISTRY` | Default registry URL | `https://registry.hologram.dev` |
| `HOLOGRAM_LOG_LEVEL` | Log level | `info` |

### Precedence

Configuration is loaded in this order (later overrides earlier):

1. Built-in defaults
2. `~/.hologram/config.toml`
3. Environment variables
4. Command-line flags

## API Endpoints

When running `hologram serve`, the following REST API is available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate inference output |
| `/api/tags` | GET | List local models |
| `/api/show` | POST | Show model info |
| `/api/create` | POST | Create model from Holofile |
| `/api/pull` | POST | Pull model from registry |
| `/api/push` | POST | Push model to registry |
| `/api/copy` | POST | Copy model |
| `/api/delete` | DELETE | Delete model |
| `/api/ps` | GET | List running models |
| `/` | GET | Health check |

### Example API Usage

```bash
# Generate
curl http://localhost:11434/api/generate -d '{
  "model": "sd-turbo",
  "prompt": "A sunset over mountains"
}'

# List models
curl http://localhost:11434/api/tags

# Pull model
curl http://localhost:11434/api/pull -d '{
  "name": "stabilityai/sd-turbo"
}'
```

## Architecture

### Crate Structure

```
hologram-app/
├── crates/
│   └── hologram-core/        # Core library (used by CLI and Tauri)
│       ├── model.rs          # ModelRef, ModelInfo types
│       ├── registry.rs       # Local model storage
│       ├── server.rs         # HTTP server (axum)
│       ├── config.rs         # Configuration
│       └── error.rs          # Error types
├── bins/
│   └── hologram/             # CLI binary
│       ├── cli.rs            # clap command definitions
│       └── commands/         # Command handlers
└── src-tauri/                # Tauri desktop app
    └── (consumes hologram-core)
```

### Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      hologram serve                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ REST API    │  │ Model       │  │ Registry            │ │
│  │ (axum)      │  │ Manager     │  │ (local + remote)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼─────────────────▼─────────────────────▼─────────┐ │
│  │                    hologram-onnx                        │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │ │
│  │  │ Compiler   │  │ Downloader │  │ Converter  │       │ │
│  │  └────────────┘  └────────────┘  └────────────┘       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Model Lifecycle

1. **Pull**: Download from HuggingFace → Store in `~/.hologram/models/`
2. **Create**: Compile ONNX → `.holo` format (O(1) lookup tables)
3. **Load**: Server loads `.holo` into memory on first `run`
4. **Run**: Execute inference via Hologram runtime
5. **Idle**: Model unloaded after `idle_timeout` seconds
6. **Stop**: Explicitly unload model from memory

## Command Mapping to hologram-onnx

| CLI Command | hologram-onnx Function |
|-------------|------------------------|
| `pull` | `download_model_with_options()` |
| `create` | `Compiler::compile()` |
| `show` | Model manifest + `info` command |
| `run` | `RuntimeExecutor::execute()` |

## Implementation Phases

### Phase 1: Foundation
- Workspace structure
- `hologram-core` crate with ModelRef, ModelInfo, LocalRegistry
- Configuration loading (file + env)
- CLI binary with clap

### Phase 2: Core Commands (Model Management)
- `list` - scan local models directory
- `show` - display model info
- `pull` - download from HuggingFace
- `create` - compile model
- `rm` - delete model
- `cp` - copy/alias model

### Phase 3: Runtime Commands
- `run` - execute model inference
- `serve` - start HTTP server
- `stop` - stop running model
- `ps` - list running models

### Phase 4: Registry Commands
- `signin` - store auth token
- `signout` - clear auth token
- `push` - upload to registry

### Phase 5: Tauri Integration
- Add hologram-core dependency to src-tauri
- Create Tauri commands
- Progress events for long operations

## Dependencies

```toml
# hologram-core
hologram-onnx = { path = "../../../hologram-onnx" }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
thiserror = "1"
tracing = "0.1"
directories = "5"
toml = "0.8"

# hologram CLI
hologram-core = { path = "../../crates/hologram-core" }
clap = { version = "4.5", features = ["derive", "env"] }
```

## Critical Files Reference

Before implementation, review these existing files:

1. `/hologram-onnx/src/lib.rs` - Public API exports
2. `/hologram-onnx/src/main.rs` - CLI patterns and command implementations
3. `/hologram-onnx/crates/downloader/src/lib.rs` - Download API
4. `/hologram-onnx/crates/compiler/src/lib.rs` - Compiler API
5. `/hologram-app/src-tauri/src/lib.rs` - Tauri command patterns
