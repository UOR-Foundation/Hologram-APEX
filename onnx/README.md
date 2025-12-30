# Hologram ONNX Toolchain

**Complete toolchain for converting PyTorch/ONNX models to Hologram's O(1) lookup-based inference format.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

Hologram ONNX provides a complete end-to-end pipeline for neural network deployment:

1. **Download** models from HuggingFace Hub
2. **Convert** PyTorch models to ONNX format (with automatic Python environment setup)
3. **Compile** ONNX to `.holo` format for O(1) inference

### Key Features

- ✅ **Auto-Installing Python Environment** - No manual `pip install` needed
- ✅ **SafeTensors Support** - Load weights separately from ONNX graph
- ✅ **PyTorch → ONNX Converter** - Embedded Python conversion script
- ✅ **Enhanced HuggingFace Downloader** - Resumable downloads with filtering
- ✅ **Auto-Convert Command** - Download + convert in one step
- ✅ **Unified CLI** - Four commands: `download`, `convert`, `auto-convert`, `compile`

## 🚀 Quick Start

### Installation

```bash
cargo install --path .
```

### Simplified Workflow (Auto-Convert)

```bash
# Download and convert in one step
hologram-onnx auto-convert IDKiro/sdxs-512-0.9 \
  --component unet \
  --output unet.onnx \
  --model-type stable-diffusion \
  --dynamic-axes --optimize --verbose

# Compile to .holo format
hologram-onnx compile \
  --input unet.onnx \
  --output unet.holo \
  --parallel --verbose
```

### Traditional Workflow (Manual Steps)

```bash
# 1. Download PyTorch model from HuggingFace
hologram-onnx download IDKiro/sdxs-512-0.9 --output ./models -v

# 2. Convert to ONNX (auto-installs Python packages on first run)
hologram-onnx convert \
  --model-dir ./models/IDKiro/sdxs-512-0.9/unet \
  --output unet.onnx \
  --model-type stable-diffusion \
  --dynamic-axes --optimize --verbose

# 3. Compile to .holo
hologram-onnx compile \
  --input unet.onnx \
  --output unet.holo \
  --parallel --verbose
```

## 📦 Architecture

This project consists of three main crates:

### 1. [hologram-onnx-compiler](crates/compiler/)
Compiles ONNX models to Hologram's binary format with:
- Perfect hash tables for O(1) pattern lookup
- Pre-computed operation results
- Zero-copy runtime execution
- SafeTensors weight loading

### 2. [hologram-onnx-downloader](crates/downloader/)
Downloads models from HuggingFace with:
- Resumable downloads (HTTP Range headers)
- Advanced filtering (file types, patterns, subdirectories)
- External data format conversion

### 3. [hologram-onnx-converter](crates/onnx_converter/)
Converts PyTorch models to ONNX with:
- Automatic Python virtual environment creation
- Auto-installs required packages (torch, onnx, transformers, etc.)
- Supports Stable Diffusion, BERT, GPT, and other architectures
- Embedded Python conversion script

## 🎨 Features in Detail

### Auto-Installing Python Environment

**Zero manual setup required!** The converter automatically:
- Creates an isolated Python virtual environment
- Downloads and installs all required packages
- Caches everything for instant subsequent runs
- Falls back gracefully if any step fails

**First run:**
```
Setting up Python environment...
  Creating Python virtual environment...
  ✓ Virtual environment created
  Installing 6 packages...
  This may take a few minutes on first run...
  ✓ Packages installed successfully
```

**Subsequent runs:**
```
✓ All packages already installed  (< 1 second)
```

### Auto-Convert Command

Streamlines the workflow by combining download and conversion:

```bash
hologram-onnx auto-convert IDKiro/sdxs-512-0.9 \
  --component unet \
  --output unet.onnx \
  --optimize --dynamic-axes
```

**Features:**
- One-step workflow (download + convert)
- Component selection (e.g., `unet`, `vae`, `text_encoder`)
- Auto-cleanup (optionally delete downloads to save space)
- Same auto-installing Python environment

### SafeTensors Weight Loading

Load ONNX graph structure separately from weights:

```bash
hologram-onnx compile \
  --input model.onnx \
  --weights encoder.safetensors \
  --weights decoder.safetensors \
  --output model.holo
```

## 📋 Command Reference

### `hologram-onnx download`

Download models from HuggingFace Hub.

```bash
hologram-onnx download <REPO> --output <DIR>
```

### `hologram-onnx convert`

Convert PyTorch models to ONNX format.

```bash
hologram-onnx convert \
  --model-dir <DIR> \
  --output <FILE> \
  --model-type <TYPE> \
  --optimize --dynamic-axes
```

**Model types:** `auto`, `stable-diffusion`, `bert`, `gpt`, `vit`

### `hologram-onnx auto-convert`

Download and convert in one step.

```bash
hologram-onnx auto-convert <REPO> \
  --component <COMPONENT> \
  --output <FILE> \
  --optimize
```

### `hologram-onnx compile`

Compile ONNX to .holo format.

```bash
hologram-onnx compile \
  --input <FILE> \
  --output <FILE> \
  --parallel \
  [--weights <FILE>...]
```

## 🔧 Requirements

### Rust
- Rust 1.75 or later

### Python (Auto-Managed)
The converter automatically manages Python dependencies:
- Python 3.8+ (system installation required)
- Virtual environment created automatically
- Packages (torch, onnx, transformers, etc.) installed automatically

**No manual `pip install` needed!**

## 🏗️ Building from Source

```bash
# Clone the repository
git clone https://github.com/hologramapp/hologram-onnx
cd hologram-onnx

# Build release binary
cargo build --release

# Run
./target/release/hologram-onnx --help
```

## 📊 Performance

| Stage | Traditional ONNX | Hologram (.holo) |
|-------|-----------------|------------------|
| **Loading** | ~10s | <100µs |
| **Inference** | ~100ms | ~35ns |
| **Memory** | Dynamic allocation | Zero-copy mmap |

## 🎓 Advanced Usage

### Custom Input Shapes (100% Pre-Compilation)

```bash
hologram-onnx compile \
  --input model.onnx \
  --input-shape "input_ids:1,77" \
  --input-shape "attention_mask:1,77" \
  --output model.holo \
  --parallel
```

### Skip Auto-Installation

For users who already have Python packages:

```bash
hologram-onnx convert \
  --model-dir ./models/unet \
  --output unet.onnx \
  --no-auto-install
```

### Clean Python Cache

```bash
# Remove cached environment (force reinstall)
rm -rf ~/.cache/hologram-onnx/python

# Next run will reinstall everything
hologram-onnx convert --model-dir ./models/unet --output unet.onnx
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Test specific crate
cargo test --package hologram-onnx-downloader
cargo test --package hologram-onnx-compiler
cargo test --package hologram-onnx-converter
```

## 📚 Documentation

- [Compiler Crate](crates/compiler/README.md) - ONNX compilation details
- [Downloader Crate](crates/downloader/README.md) - Download features and API
- [Converter Crate](crates/onnx_converter/README.md) - PyTorch to ONNX conversion

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- [ONNX](https://onnx.ai/) - Neural network model format
- [HuggingFace](https://huggingface.co/) - Model repository
- [Hologram](https://github.com/hologramapp/hologram) - O(1) inference engine

---

**Built with ❤️ using a hybrid Rust + Python approach**

*Zero manual setup. Maximum performance. Pure magic.* ✨
