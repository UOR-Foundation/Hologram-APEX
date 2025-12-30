"""
Hologram Torch - Native torch.device('hologram') support

This package provides native PyTorch integration for Hologram's canonical
compilation engine, enabling seamless use of torch.device('hologram').

Example:
    >>> import torch
    >>> import hologram_torch
    >>> x = torch.randn(10, 10, device='hologram')
    >>> y = torch.randn(10, 10, device='hologram')
    >>> z = x + y  # Runs on Hologram backend
    >>> loss = z.sum()
    >>> loss.backward()  # Autograd works!
"""

import sys
import torch

# Import C++ extension
try:
    from . import _hologram_torch
except ImportError as e:
    raise ImportError(
        "Failed to import hologram_torch C++ extension. "
        "Please ensure the extension is built: "
        "cd hologram-sdk/python/hologram-torch && pip install -e ."
    ) from e

# Initialize Hologram backend FIRST
_hologram_torch.init_hologram()

# Register 'hologram' as the PrivateUse1 device name
# Try both old and new APIs for compatibility
try:
    # PyTorch 2.0+ API
    torch.utils.rename_privateuse1_backend('hologram')
except Exception as e:
    print(f"Warning: Could not rename PrivateUse1 backend: {e}")

try:
    # PyTorch 2.8+ might need this
    torch._register_device_module('hologram', sys.modules[__name__])
except Exception as e:
    print(f"Note: _register_device_module not available: {e}")

# Register this module as torch.hologram to support device string parsing
sys.modules['torch.hologram'] = sys.modules[__name__]


def is_available():
    """Check if Hologram backend is available.

    Returns:
        bool: True if Hologram backend is initialized
    """
    return _hologram_torch.is_available()


def device(index=0):
    """Get Hologram device.

    Args:
        index (int): Device index (default: 0)

    Returns:
        torch.device: Hologram device object

    Example:
        >>> dev = hologram_torch.device()
        >>> x = torch.randn(10, device=dev)
    """
    return torch.device('hologram', index)


def get_backend():
    """Get current backend type.

    Returns:
        str: Backend name ('cpu', 'metal', or 'cuda')

    Example:
        >>> hologram_torch.get_backend()
        'cpu'
    """
    return _hologram_torch.get_backend()


def set_backend(backend):
    """Set active backend.

    Args:
        backend (str): Backend name ('cpu', 'metal', or 'cuda')

    Raises:
        RuntimeError: If backend is not available

    Example:
        >>> hologram_torch.set_backend('metal')  # macOS only
    """
    if backend not in list_available_backends():
        raise RuntimeError(
            f"Backend '{backend}' not available. "
            f"Available: {list_available_backends()}"
        )
    _hologram_torch.set_backend(backend)


def list_available_backends():
    """List all available backends.

    Returns:
        list[str]: Available backend names

    Example:
        >>> hologram_torch.list_available_backends()
        ['cpu', 'metal']
    """
    return _hologram_torch.list_backends()


def get_executor_handle():
    """Get current executor handle (for advanced use).

    Returns:
        int: Executor handle (u64)
    """
    return _hologram_torch.get_executor_handle()


# Print initialization message
print("✅ Hologram Torch backend initialized")
print(f"   Backend: {get_backend()}")
print(f"   Available backends: {', '.join(list_available_backends())}")
print("   Use: torch.device('hologram')")


__all__ = [
    'is_available',
    'device',
    'get_backend',
    'set_backend',
    'list_available_backends',
    'get_executor_handle',
]

__version__ = '0.1.0'
