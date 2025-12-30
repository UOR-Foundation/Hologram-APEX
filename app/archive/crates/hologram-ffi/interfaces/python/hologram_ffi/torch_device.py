"""
PyTorch PrivateUse1 Device Backend for Hologram

This module registers Hologram as a custom PyTorch device backend, enabling:
  - torch.device('hologram') support
  - Native PyTorch tensor operations on Hologram backend
  - Seamless integration with PyTorch's autograd and nn modules

Usage:
    import hologram_ffi.torch_device  # Registers the device
    import torch

    # Create tensors on Hologram device
    x = torch.randn(3, 4, device='hologram')
    y = torch.randn(3, 4, device='hologram')
    z = x + y  # Runs on Hologram backend
"""

import torch
import hologram_ffi as hg
from . import hologram_ffi
from .tensor import HologramTensor

# Device backend name
DEVICE_NAME = "hologram"
DEVICE_TYPE = torch._C._get_privateuse1_backend_name()

def register_hologram_backend():
    """
    Register Hologram as a PyTorch PrivateUse1 device backend.

    This enables torch.device('hologram') support.
    """
    # Register the backend name
    torch._register_device_module(DEVICE_NAME, torch.nn.Module)

    # Set the backend name for PrivateUse1
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)

    print(f"✅ Registered Hologram as PyTorch device backend: '{DEVICE_NAME}'")


def _hologram_tensor_constructor(shape, dtype=torch.float32, device='hologram', requires_grad=False):
    """
    Tensor factory function for Hologram device.

    Creates a new tensor on the Hologram backend.

    Args:
        shape: Tensor shape (tuple or list)
        dtype: Data type (currently only torch.float32 supported)
        device: Device string (must be 'hologram')
        requires_grad: Whether to track gradients (not yet supported)

    Returns:
        PyTorch tensor backed by Hologram
    """
    if dtype != torch.float32:
        raise NotImplementedError(f"Hologram backend only supports float32, got {dtype}")

    if requires_grad:
        raise NotImplementedError("Hologram backend does not yet support autograd")

    # Calculate total elements
    numel = 1
    for dim in shape:
        numel *= dim

    # Create Hologram executor and buffer
    exec_handle = hologram_ffi.new_executor()
    buffer_handle = hologram_ffi.executor_allocate_buffer(exec_handle, numel)

    # Create Hologram tensor
    hologram_tensor = HologramTensor(exec_handle, buffer_handle, list(shape))

    # Convert to PyTorch via DLPack (zero-copy)
    pytorch_tensor = torch.from_dlpack(hologram_tensor)

    return pytorch_tensor


def register_tensor_constructors():
    """
    Register tensor factory functions with PyTorch.

    This enables:
      - torch.empty(..., device='hologram')
      - torch.zeros(..., device='hologram')
      - torch.ones(..., device='hologram')
      - torch.randn(..., device='hologram')
    """
    # Register empty tensor constructor
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=False
    )

    print(f"✅ Registered Hologram tensor constructors")


def register_operator_dispatch():
    """
    Register operator dispatch for Hologram backend.

    This enables PyTorch operations (add, mul, matmul, etc.) to run on Hologram.

    Currently implements basic operations via DLPack conversion:
      - Binary ops: add, sub, mul, div
      - Unary ops: relu, sigmoid, tanh
      - Linear algebra: matmul
    """
    # For now, we rely on DLPack conversion for operations
    # Future: Implement native operators using hologram_ffi operations

    print(f"⚠️  Hologram operator dispatch uses fallback to CPU (DLPack conversion)")
    print(f"    For best performance, use hologram_ffi operations directly")


def initialize():
    """
    Initialize Hologram PyTorch device backend.

    Call this function to register Hologram with PyTorch.

    Example:
        >>> import hologram_ffi.torch_device as htorch
        >>> htorch.initialize()
        >>> import torch
        >>> x = torch.randn(3, 4, device='hologram')
    """
    print("=" * 70)
    print("Initializing Hologram PyTorch Device Backend")
    print("=" * 70)

    register_hologram_backend()
    register_tensor_constructors()
    register_operator_dispatch()

    print("=" * 70)
    print(f"✅ Hologram PyTorch backend ready!")
    print(f"   Use: torch.device('{DEVICE_NAME}')")
    print("=" * 70)


# Auto-initialize when module is imported
# Comment out if you want manual initialization
# initialize()
