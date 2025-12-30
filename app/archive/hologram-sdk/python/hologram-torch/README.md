# Hologram Torch - Native torch.device('hologram') Support

Native PyTorch backend integration for Hologram, enabling seamless use of `torch.device('hologram')` with full autograd support.

## Features

✅ **Native PyTorch device**: Use `torch.device('hologram')` directly
✅ **All backends supported**: CPU, Metal (macOS), CUDA
✅ **Full autograd**: `loss.backward()` works automatically
✅ **30+ operations**: Math, activations, reductions, linear algebra, loss functions
✅ **Zero reimplementation**: Delegates to hologram-ffi operations

## Installation

```bash
# Ensure hologram-ffi is built
cd /workspace
cargo build --release --lib -p hologram-ffi

# Install hologram-torch
cd hologram-sdk/python/hologram-torch
pip install -e .
```

## Quick Start

```python
import torch
import hologram_torch

# Create tensors on Hologram device
x = torch.randn(10, 10, device='hologram')
y = torch.randn(10, 10, device='hologram')

# Operations run on Hologram backend
z = x + y
loss = z.sum()

# Autograd works!
x.requires_grad = True
loss = (x * y).sum()
loss.backward()
print(x.grad)  # Gradients computed on Hologram
```

## Supported Operations

### Element-wise Operations
- Binary: `add`, `sub`, `mul`, `div`, `min`, `max`
- Unary: `abs`, `neg`, `relu`, `clamp`
- Scalar: `add(scalar)`, `mul(scalar)`

### Activations
- `sigmoid`, `tanh`, `gelu`, `softmax`

### Reductions
- `sum`, `min`, `max`, `mean`

### Linear Algebra
- `matmul` (matrix multiplication)

### Loss Functions
- `mse_loss` (mean squared error)

## Backend Management

```python
import hologram_torch

# Check available backends
print(hologram_torch.list_available_backends())
# Output: ['cpu', 'metal']  # On macOS with Metal

# Get current backend
print(hologram_torch.get_backend())
# Output: 'metal'

# Switch backend
hologram_torch.set_backend('cpu')
```

## Example: Training a Neural Network

```python
import torch
import torch.nn as nn
import hologram_torch

# Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to('hologram')

# Training loop
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    # Data on Hologram device
    data = torch.randn(32, 784, device='hologram')
    target = torch.randint(0, 10, (32,), device='hologram')

    # Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    # Backward pass (autograd on Hologram)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Architecture

- **C++ Extension**: Implements PyTorch's PrivateUse1 device interface
- **Hologram FFI**: Calls existing hologram-ffi operations (Rust library)
- **No Reimplementation**: All operations delegate to hologram-core
- **Hybrid Approach**: CPU uses raw pointers, GPU uses DLPack (planned)

## Current Status

**v0.1.0 - Initial Release**

✅ CPU backend working
✅ 30+ operations implemented
✅ Autograd support
⚠️ GPU backends (Metal/CUDA) use CPU fallback (DLPack integration pending)

## Development

```bash
# Build from source
cargo build --release --lib -p hologram-ffi
cd hologram-sdk/python/hologram-torch
pip install -e .

# Run tests
python -m pytest tests/
```

## See Also

- [Hologram Core](../../README.md) - Main Hologram project
- [PyTorch Device API Example](/workspace/examples/pytorch_device_api.py) - Detailed examples
- [DLPack Integration](/workspace/examples/README_PYTORCH_DEVICE.md) - Alternative approach

## License

MIT License (same as Hologram project)
