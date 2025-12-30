# Hologram Model Server

Pure Rust OpenAI-compatible model server powered by hologram-memory-manager and hologram-core.

## Features

- **OpenAI API Compatibility** - Works with any OpenAI client
- **Processor Integration** - All inputs processed with gauge construction
- **Domain Head Architecture** - Model serving as processor domain heads
- **Pure Rust** - Single binary, no FFI or Python bridge

## Supported Endpoints

- `POST /v1/embeddings` - Generate text embeddings
- `POST /v1/completions` - Generate text completions
- `GET /v1/models` - List available models

## Quick Start

```bash
# Build and run
cargo build --release
cargo run --release

# Server starts on http://0.0.0.0:8000
```

## Testing with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required
)

# Generate embedding
response = client.embeddings.create(
    input="Hello, world!",
    model="hologram-embedding-v1"
)
print(response.data[0].embedding)

# Generate completion
response = client.completions.create(
    model="hologram-completion-v1",
    prompt="Once upon a time",
    max_tokens=50
)
print(response.choices[0].text)
```

## Testing with curl

```bash
# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "hologram-embedding-v1"}'

# Completions
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hologram-completion-v1", "prompt": "Once upon a time", "max_tokens": 50}'

# Models
curl http://localhost:8000/v1/models
```

## Available Models

- `hologram-embedding-v1` - Text embedding model (128-dim)
- `hologram-completion-v1` - Text completion model

## Architecture

```
OpenAI Client → HTTP API → InferenceEngine → Domain Heads → Processor → hologram-core
```

- **HTTP API**: Axum server with OpenAI-compatible endpoints
- **InferenceEngine**: Model weights management and inference
- **Domain Heads**: Text embedding and completion extraction
- **Processor**: Universal memory pool with gauge construction
- **hologram-core**: Low-level compute operations (GEMM, softmax, etc.)

## Development

```bash
# Run tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Build for release
cargo build --release
```

## Implementation Plan

See [docs/MODEL_SERVER_IMPLEMENTATION_PLAN.md](/workspaces/hologramapp/docs/MODEL_SERVER_IMPLEMENTATION_PLAN.md) for complete implementation details.
