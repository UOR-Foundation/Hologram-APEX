"""DLPack PyCapsule utilities for zero-copy tensor exchange.

This module provides low-level utilities for creating and managing DLPack
PyCapsules, enabling zero-copy tensor exchange between Hologram and frameworks
like PyTorch, JAX, and TensorFlow.
"""

import ctypes
from typing import Tuple

# DLPack device type constants (from DLPack specification)
DLPACK_DEVICE_CPU = 1
DLPACK_DEVICE_CUDA = 2
DLPACK_DEVICE_CUDA_HOST = 3
DLPACK_DEVICE_OPENCL = 4
DLPACK_DEVICE_VULKAN = 7
DLPACK_DEVICE_METAL = 8
DLPACK_DEVICE_ROCM = 10


def create_dlpack_capsule(dlpack_ptr: int) -> object:
    """
    Create a PyCapsule from DLPack pointer for zero-copy tensor exchange.

    This function wraps a raw DLManagedTensor pointer (returned from the FFI)
    in a Python PyCapsule object that can be consumed by torch.from_dlpack(),
    jax.dlpack.from_dlpack(), and other DLPack-compatible frameworks.

    Args:
        dlpack_ptr: Raw pointer to DLManagedTensor (as u64 from FFI)

    Returns:
        PyCapsule object with name "dltensor"

    Raises:
        ValueError: If dlpack_ptr is null (0)

    Notes:
        - The capsule does NOT have a Python-side destructor
        - The deleter is already set in the DLManagedTensor structure (Rust side)
        - The consuming framework will call the deleter when done

    Example:
        >>> ptr = hologram_ffi.tensor_to_dlpack(exec_handle, tensor_handle)
        >>> capsule = create_dlpack_capsule(ptr)
        >>> pytorch_tensor = torch.from_dlpack(capsule)  # Zero-copy!
    """
    if dlpack_ptr == 0:
        raise ValueError("Invalid DLPack pointer (null)")

    # Configure ctypes for PyCapsule_New
    # Set argtypes and restype for proper calling convention
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,  # pointer
        ctypes.c_char_p,  # name
        ctypes.c_void_p   # destructor (NULL in our case)
    ]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

    # Create PyCapsule with name "dltensor" as required by DLPack spec
    # No destructor is needed - the DLManagedTensor already has a deleter
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.c_void_p(dlpack_ptr),
        b"dltensor",
        None  # No destructor - handled by DLManagedTensor.deleter
    )

    return capsule


def get_device_tuple(device_type: int, device_id: int) -> Tuple[int, int]:
    """
    Format device info as DLPack device tuple.

    Args:
        device_type: DLPack device type code
            - 1: CPU
            - 2: CUDA
            - 3: CUDA Host (pinned memory)
            - 4: OpenCL
            - 7: Vulkan
            - 8: Metal
            - 10: ROCm
        device_id: Device index (0 for CPU, GPU index for CUDA)

    Returns:
        Tuple of (device_type, device_id)

    Example:
        >>> device = get_device_tuple(DLPACK_DEVICE_CUDA, 0)
        >>> print(device)  # (2, 0) - CUDA GPU 0
    """
    return (device_type, device_id)


def device_type_name(device_type: int) -> str:
    """
    Get human-readable name for DLPack device type.

    Args:
        device_type: DLPack device type code

    Returns:
        Device type name string

    Example:
        >>> name = device_type_name(DLPACK_DEVICE_CUDA)
        >>> print(name)  # "CUDA"
    """
    names = {
        DLPACK_DEVICE_CPU: "CPU",
        DLPACK_DEVICE_CUDA: "CUDA",
        DLPACK_DEVICE_CUDA_HOST: "CUDA_HOST",
        DLPACK_DEVICE_OPENCL: "OpenCL",
        DLPACK_DEVICE_VULKAN: "Vulkan",
        DLPACK_DEVICE_METAL: "Metal",
        DLPACK_DEVICE_ROCM: "ROCm",
    }
    return names.get(device_type, f"Unknown({device_type})")
