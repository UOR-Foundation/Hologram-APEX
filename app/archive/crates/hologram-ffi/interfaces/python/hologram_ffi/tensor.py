"""Tensor wrapper with DLPack protocol support for zero-copy interoperability.

This module provides a high-level HologramTensor class that implements the DLPack
protocol, enabling seamless zero-copy tensor exchange with PyTorch, JAX, TensorFlow,
and other frameworks.
"""

import json
from typing import List, Optional
from . import hologram_ffi
from .dlpack_utils import create_dlpack_capsule, get_device_tuple, device_type_name


class HologramTensor:
    """
    Hologram tensor with PyTorch-compatible DLPack interface.

    This class wraps a Hologram tensor and implements the DLPack protocol
    (__dlpack__ and __dlpack_device__), enabling zero-copy tensor exchange
    with PyTorch, JAX, and TensorFlow.

    The tensor shares memory with the underlying Hologram buffer, so no data
    is copied during conversion to/from other frameworks.

    Attributes:
        executor_handle: Handle to the Hologram executor
        buffer_handle: Handle to the underlying buffer
        shape: Tensor dimensions (e.g., [3, 4] for 3×4 matrix)
        tensor_handle: Internal handle to the Hologram tensor

    Example:
        >>> import hologram_ffi as hg
        >>> import torch
        >>>
        >>> # Create Hologram tensor
        >>> exec_handle = hg.new_executor()
        >>> buffer_handle = hg.executor_allocate_buffer(exec_handle, 12)
        >>> tensor = HologramTensor(exec_handle, buffer_handle, shape=[3, 4])
        >>>
        >>> # Zero-copy conversion to PyTorch
        >>> pytorch_tensor = torch.from_dlpack(tensor)
        >>> print(pytorch_tensor.shape)  # torch.Size([3, 4])
    """

    def __init__(
        self,
        executor_handle: int,
        buffer_handle: int,
        shape: List[int],
    ):
        """
        Create a Hologram tensor from a buffer and shape.

        Args:
            executor_handle: Handle to the Hologram executor
            buffer_handle: Handle to the underlying buffer
            shape: Tensor dimensions (e.g., [3, 4] for 3×4 matrix)

        Raises:
            ValueError: If tensor creation fails (e.g., buffer too small)

        Example:
            >>> exec = hg.new_executor()
            >>> buf = hg.executor_allocate_buffer(exec, 24)
            >>> tensor = HologramTensor(exec, buf, shape=[4, 6])
        """
        self.executor_handle = executor_handle
        self.buffer_handle = buffer_handle
        self.shape = shape

        # Create tensor via FFI
        shape_json = json.dumps(shape)
        self.tensor_handle = hologram_ffi.tensor_from_buffer(
            buffer_handle,
            shape_json
        )

    @classmethod
    def from_dlpack(cls, external_tensor, executor_handle=None):
        """
        Import tensor from any DLPack-compatible framework (PyTorch, JAX, TensorFlow, etc.)

        Creates a Hologram tensor from external framework tensor. Data is copied to Hologram memory.

        Args:
            external_tensor: Any object with __dlpack__() method (PyTorch/JAX/TF/CuPy tensor)
            executor_handle: Optional executor handle (creates new if not provided)

        Returns:
            HologramTensor with data from external tensor

        Raises:
            TypeError: If external_tensor doesn't support DLPack protocol
            RuntimeError: If import fails

        Example:
            >>> import torch
            >>> import hologram_ffi as hg
            >>>
            >>> # Create PyTorch tensor
            >>> pytorch_tensor = torch.randn(3, 4, dtype=torch.float32)
            >>>
            >>> # Import to Hologram (data copied)
            >>> hologram_tensor = hg.HologramTensor.from_dlpack(pytorch_tensor)
            >>>
            >>> # Run Hologram operations
            >>> # ...
            >>>
            >>> # Export back to PyTorch (zero-copy)
            >>> result = torch.from_dlpack(hologram_tensor)

        Notes:
            - Import is copy-based for safety with Hologram's class-based memory
            - Export (via __dlpack__) is zero-copy for maximum performance
            - Supports f32 tensors (most common ML use case)
            - Works with PyTorch, JAX, TensorFlow, CuPy, and any DLPack-compatible framework
        """
        import ctypes
        from . import hologram_ffi

        # Validate external tensor has DLPack protocol
        if not hasattr(external_tensor, '__dlpack__'):
            raise TypeError(
                f"Object of type {type(external_tensor)} does not support DLPack protocol. "
                f"Expected __dlpack__() method."
            )

        # Create executor if not provided
        if executor_handle is None:
            executor_handle = hologram_ffi.new_executor()

        # Get DLPack capsule from external tensor
        capsule = external_tensor.__dlpack__()

        # Extract pointer from PyCapsule
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p

        try:
            dlpack_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        except Exception as e:
            raise RuntimeError(f"Failed to extract DLPack pointer from capsule: {e}")

        if dlpack_ptr == 0 or dlpack_ptr is None:
            raise RuntimeError("DLPack pointer is null")

        # Import via FFI (f32 only for now)
        result_json = hologram_ffi.tensor_from_dlpack_capsule(executor_handle, dlpack_ptr)

        if not result_json:
            raise RuntimeError(
                "Failed to import tensor from DLPack. "
                "Ensure tensor is f32 dtype and on CPU or CUDA device."
            )

        # Parse result JSON
        result = json.loads(result_json)
        tensor_handle = result["tensor_handle"]
        buffer_handle = result["buffer_handle"]

        # Get shape from original tensor
        shape = list(external_tensor.shape)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.executor_handle = executor_handle
        instance.tensor_handle = tensor_handle
        instance.buffer_handle = buffer_handle
        instance.shape = shape

        return instance

    def __dlpack__(self, stream=None):
        """
        Export tensor to DLPack format (implements DLPack protocol).

        This method is called by torch.from_dlpack(), jax.dlpack.from_dlpack(),
        and other frameworks supporting the DLPack protocol. It returns a PyCapsule
        containing a pointer to the tensor data with no data copying.

        Args:
            stream: Optional CUDA stream for synchronization (currently unused)

        Returns:
            PyCapsule containing DLManagedTensor pointer

        Raises:
            ValueError: If tensor is not contiguous
            RuntimeError: If DLPack export fails

        Example:
            >>> import torch
            >>> pytorch_tensor = torch.from_dlpack(hologram_tensor)
            >>> # pytorch_tensor shares memory with hologram_tensor (zero-copy)

        Notes:
            - The tensor must be contiguous (call .contiguous() if needed)
            - The returned PyCapsule transfers ownership to the consumer
            - The deleter will be called automatically when the consumer is done
        """
        # Ensure tensor is contiguous (required by DLPack)
        if not hologram_ffi.tensor_is_contiguous(self.tensor_handle):
            raise ValueError(
                "DLPack export requires contiguous tensor. "
                "Call .contiguous() first."
            )

        # Get DLPack pointer from FFI
        dlpack_ptr = hologram_ffi.tensor_to_dlpack(
            self.executor_handle,
            self.tensor_handle
        )

        if dlpack_ptr == 0:
            raise RuntimeError(
                "Failed to export tensor to DLPack. "
                "Check that the executor and tensor are valid."
            )

        # Wrap in PyCapsule for consumption by other frameworks
        return create_dlpack_capsule(dlpack_ptr)

    def __dlpack_device__(self):
        """
        Query which device this tensor resides on (implements DLPack protocol).

        This method is called by frameworks to determine where the tensor data
        is located before importing it via __dlpack__().

        Returns:
            Tuple of (device_type, device_id)
                device_type: 1=CPU, 2=CUDA, 3=CUDA_HOST, etc.
                device_id: GPU index for CUDA (0, 1, 2, ...), 0 for CPU

        Example:
            >>> device_type, device_id = tensor.__dlpack_device__()
            >>> if device_type == 2:  # CUDA
            >>>     print(f"Tensor on GPU {device_id}")
            >>> else:
            >>>     print("Tensor on CPU")

        Notes:
            - CPU tensors return (1, 0)
            - CUDA tensors return (2, gpu_index)
            - This is called automatically by torch.from_dlpack()
        """
        device_type = hologram_ffi.tensor_dlpack_device_type(
            self.executor_handle
        )
        device_id = hologram_ffi.tensor_dlpack_device_id(
            self.executor_handle
        )

        return get_device_tuple(device_type, device_id)

    def contiguous(self):
        """
        Create a contiguous copy of this tensor if needed.

        DLPack requires tensors to be contiguous (row-major layout). If this
        tensor is not contiguous, this method creates a contiguous copy.

        Returns:
            HologramTensor (contiguous)

        Example:
            >>> # Ensure tensor is contiguous before export
            >>> tensor = tensor.contiguous()
            >>> pt_tensor = torch.from_dlpack(tensor)
        """
        if hologram_ffi.tensor_is_contiguous(self.tensor_handle):
            # Already contiguous, return self
            return self

        # Create contiguous copy via FFI
        new_tensor_handle = hologram_ffi.tensor_contiguous(
            self.executor_handle,
            self.tensor_handle
        )

        # Wrap in new HologramTensor (avoid re-initialization)
        result = HologramTensor.__new__(HologramTensor)
        result.executor_handle = self.executor_handle
        result.buffer_handle = self.buffer_handle  # Same underlying buffer
        result.shape = self.shape
        result.tensor_handle = new_tensor_handle
        return result

    def is_contiguous(self) -> bool:
        """
        Check if tensor has contiguous memory layout.

        Returns:
            True if contiguous (row-major), False otherwise

        Example:
            >>> if not tensor.is_contiguous():
            >>>     tensor = tensor.contiguous()
        """
        return bool(hologram_ffi.tensor_is_contiguous(self.tensor_handle))

    def device_info(self) -> str:
        """
        Get human-readable device information.

        Returns:
            String describing device (e.g., "CPU", "CUDA GPU 0")

        Example:
            >>> print(tensor.device_info())  # "CPU" or "CUDA GPU 0"
        """
        device_type, device_id = self.__dlpack_device__()
        type_name = device_type_name(device_type)

        if device_type == 1:  # CPU
            return "CPU"
        else:
            return f"{type_name} GPU {device_id}"

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return hologram_ffi.tensor_ndim(self.tensor_handle)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return hologram_ffi.tensor_numel(self.tensor_handle)

    def __repr__(self) -> str:
        """String representation."""
        device = self.device_info()
        contiguous = "contiguous" if self.is_contiguous() else "non-contiguous"
        return (
            f"HologramTensor(shape={self.shape}, "
            f"device={device}, {contiguous})"
        )

    def __del__(self):
        """Cleanup tensor handle when Python object is destroyed."""
        if hasattr(self, 'tensor_handle'):
            try:
                hologram_ffi.tensor_cleanup(self.tensor_handle)
            except Exception:
                # Ignore errors during cleanup (executor may already be destroyed)
                pass
