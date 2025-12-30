# Model Setup for Hologram Demo

This directory contains ONNX models for the Hologram demo.

## Required Models

The demo requires Stable Diffusion Turbo models in ONNX format with external data:

```
models/onnx/sd-turbo-external/
├── text_encoder/
│   ├── model.onnx (0.1MB - graph structure)
│   └── model.bin (649MB - weights)
├── unet/
│   ├── model.onnx (0.4MB - graph structure)
│   └── model.bin (1.7GB - weights)
└── vae_decoder/
    ├── model.onnx (0.1MB - graph structure)
    └── model.bin (94MB - weights)
```

**Total size**: ~2.4GB

## Setup Instructions

### Option 1: Download Pre-converted Models

1. Download the SD Turbo ONNX models from HuggingFace:
   ```bash
   # Text Encoder
   cd models/onnx/sd-turbo-external/text_encoder/
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.onnx
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.bin

   # U-Net
   cd ../unet/
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/unet/model.onnx
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/unet/model.bin

   # VAE Decoder
   cd ../vae_decoder/
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/vae_decoder/model.onnx
   wget https://huggingface.co/stabilityai/sd-turbo/resolve/main/vae_decoder/model.bin
   ```

### Option 2: Convert from Standard ONNX

If you have standard ONNX models (with embedded weights), convert them to external data format:

1. Install dependencies:
   ```bash
   pip install onnx
   ```

2. Run the conversion script:
   ```bash
   python3 scripts/convert-onnx-external-data.py
   ```

This will convert models from `models/onnx/sd-turbo/` to `models/onnx/sd-turbo-external/`.

## Why External Data Format?

The external data format separates the model graph (protobuf) from the weights (binary):

**Benefits**:
- **Streaming Support**: Only small .onnx file loads to WASM (~0.4MB)
- **No Memory Limits**: Weights stream directly to GPU memory
- **Faster Loading**: Parallel streaming of multiple initializers
- **Lower Memory Usage**: No 2GB WASM memory limit

**Standard Format** (embedded weights):
```
model.onnx: 1.7GB → ❌ Crashes in WASM
```

**External Format** (streaming):
```
model.onnx: 0.4MB  → ✅ Loads to WASM
model.bin:  1.7GB  → ✅ Streams to GPU
```

## Verification

After setup, verify the files exist:

```bash
ls -lh models/onnx/sd-turbo-external/text_encoder/
ls -lh models/onnx/sd-turbo-external/unet/
ls -lh models/onnx/sd-turbo-external/vae_decoder/
```

You should see both `.onnx` and `.bin` files in each directory.

## Next Steps

1. Ensure models are downloaded
2. Build WASM module: `./scripts/build-wasm-demo.sh`
3. Start dev server: `cd public && pnpm dev`
4. Open demo: `http://localhost:3000/demo`
5. Click "Initialize Pipeline" to load models
6. Enter a prompt and click "Generate Image"

## Troubleshooting

### "Failed to fetch model"
- Check that both `.onnx` and `.bin` files exist
- Verify dev server is running
- Check browser console for CORS errors

### "WebGPU not available"
- Use Chrome 113+ or Edge 113+
- Enable WebGPU: `chrome://flags/#enable-unsafe-webgpu`

### "Out of memory"
- Close other browser tabs
- Ensure GPU has at least 2.3GB available memory
- Check GPU memory usage in task manager

## Additional Documentation

- [SDXS Setup Guide](../../docs/webgpu/SDXS_SETUP.md)
- [Demo Implementation](../../docs/webgpu/DEMO_UPDATED.md)
- [Streaming Architecture](../../docs/webgpu/STREAMING_ARCHITECTURE.md)
