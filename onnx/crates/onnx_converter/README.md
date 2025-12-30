# hologram-onnx-converter

PyTorch to ONNX converter with automatic Python environment management.

## Features

### 🔄 PyTorch to ONNX Conversion
- **Multiple model types**: Stable Diffusion, BERT, GPT, Vision Transformers
- **Automatic model detection**: Analyzes config.json to determine architecture
- **Flexible configuration**: Opset version, dynamic axes, optimization, external data format
- **Rust wrapper**: Type-safe Rust API wrapping Python conversion tools

### 🐍 Automatic Python Environment Management
- **Virtual environment creation**: Automatically creates isolated Python venv
- **Package installation**: Auto-installs torch, onnx, transformers, diffusers, etc.
- **Smart caching**: Reuses venv across conversions (~/.cache/hologram-onnx/python)
- **Zero manual setup**: No need to manually install Python packages

### ✨ Additional Features
- **External data format**: Optionally convert to ONNX external data (weights in .bin files)
- **Graph optimization**: Optional ONNX graph optimization and simplification
- **Verbose logging**: Detailed progress output for debugging
- **Cross-platform**: Works on Windows, Linux, and macOS

## Quick Start

```rust
use hologram_onnx_converter::{convert_pytorch_to_onnx, ConversionConfig, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Simple conversion
    let config = ConversionConfig {
        model_dir: PathBuf::from("models/my-pytorch-model"),
        output_onnx: PathBuf::from("output/model.onnx"),
        model_type: ModelType::Auto,
        ..Default::default()
    };

    let onnx_path = convert_pytorch_to_onnx(&config).await?;
    println!("ONNX model saved to: {}", onnx_path.display());

    Ok(())
}
```

## Configuration Options

```rust
pub struct ConversionConfig {
    /// Input model directory (contains config.json, model.safetensors, etc.)
    pub model_dir: PathBuf,

    /// Output ONNX file path
    pub output_onnx: PathBuf,

    /// Model architecture type
    pub model_type: ModelType,

    /// Opset version for ONNX (default: 17)
    pub opset_version: u32,

    /// Export with dynamic axes (variable batch/sequence length)
    pub dynamic_axes: bool,

    /// Optimize the ONNX graph after export
    pub optimize: bool,

    /// Simplify the ONNX graph (removes redundant nodes)
    pub simplify: bool,

    /// Use external data format (weights stored separately)
    pub external_data: bool,

    /// Verbose output
    pub verbose: bool,
}
```

## Supported Model Types

```rust
pub enum ModelType {
    StableDiffusion,  // SD components (UNet, VAE, TextEncoder)
    Bert,             // BERT-style transformers
    Gpt,              // GPT-style transformers
    Vit,              // Vision transformers
    Auto,             // Auto-detect from config.json
}
```

## Python Environment Management

The converter automatically manages a Python virtual environment:

1. **First run**: Creates venv at `~/.cache/hologram-onnx/python/venv`
2. **Package installation**: Installs required packages:
   - `torch`
   - `onnx`
   - `onnxruntime`
   - `transformers`
   - `diffusers`
   - `safetensors`
3. **Subsequent runs**: Reuses existing venv (no reinstallation)

You can manually control the Python environment:

```rust
use hologram_onnx_converter::{
    setup_python_environment,
    check_missing_packages,
    get_python_executable
};

// Setup Python environment (create venv + install packages)
let python_exe = setup_python_environment(true).await?;

// Check which packages are missing
let missing = check_missing_packages(&["torch", "onnx"]).await?;
println!("Missing packages: {:?}", missing);

// Get Python executable path (with optional auto-setup)
let python = get_python_executable(true, true).await?;
```

## Architecture

```
crates/onnx_converter/
├── src/
│   ├── lib.rs                 # Main crate entry point
│   ├── pytorch_converter.rs   # PyTorch → ONNX conversion logic
│   └── python_env.rs          # Python venv management
├── scripts/
│   └── pytorch_to_onnx.py    # Python conversion script (embedded)
├── Cargo.toml
└── README.md
```

## Requirements

- **Rust**: 1.70+
- **Python 3**: 3.8+ (for conversion only)
- **Disk space**: ~2-3 GB for Python packages (on first run)

## How It Works

1. **Rust API Call**: You call `convert_pytorch_to_onnx()` with a config
2. **Venv Setup**: Automatically creates Python venv and installs packages
3. **Python Script**: Writes embedded Python script to temp file
4. **Conversion**: Executes Python script with configured options
5. **Verification**: Checks output ONNX file was created successfully
6. **Cleanup**: Removes temp files (venv is kept for reuse)

## Examples

### Convert Stable Diffusion Model

```rust
let config = ConversionConfig {
    model_dir: PathBuf::from("models/stable-diffusion-v1-5/unet"),
    output_onnx: PathBuf::from("unet.onnx"),
    model_type: ModelType::StableDiffusion,
    opset_version: 17,
    dynamic_axes: true,
    optimize: true,
    external_data: true,  // Store weights separately
    verbose: true,
};

convert_pytorch_to_onnx(&config).await?;
```

### Convert Transformer Model

```rust
let config = ConversionConfig {
    model_dir: PathBuf::from("models/bert-base-uncased"),
    output_onnx: PathBuf::from("bert.onnx"),
    model_type: ModelType::Bert,
    opset_version: 14,
    dynamic_axes: true,
    optimize: true,
    ..Default::default()
};

convert_pytorch_to_onnx(&config).await?;
```

### Auto-Detect Model Type

```rust
let config = ConversionConfig {
    model_dir: PathBuf::from("models/some-model"),
    output_onnx: PathBuf::from("model.onnx"),
    model_type: ModelType::Auto,  // Will detect from config.json
    ..Default::default()
};

convert_pytorch_to_onnx(&config).await?;
```

## Error Handling

The converter provides detailed error messages:

```rust
match convert_pytorch_to_onnx(&config).await {
    Ok(path) => println!("Success: {}", path.display()),
    Err(e) => {
        eprintln!("Conversion failed: {}", e);
        // Common errors:
        // - Model directory doesn't exist
        // - Python packages failed to install
        // - ONNX export failed (incompatible ops, etc.)
        // - Output file not created
    }
}
```

## Testing

```bash
# Run all tests
cargo test --package hologram-onnx-converter

# Run with verbose output
cargo test --package hologram-onnx-converter -- --nocapture
```

## License

MIT OR Apache-2.0
