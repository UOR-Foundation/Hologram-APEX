"""
Hologram - Python bindings for Atlas virtual GPU runtime

Provides a PyTorch-like API for executing operations on the Atlas computational
memory space.
"""

from hologram._hologram import (
    Executor,
    Buffer,
)

# Import all operations under ops namespace
from hologram import _hologram
import types

# Create ops module dynamically
ops = types.SimpleNamespace()

# Math operations
ops.vector_add = _hologram.vector_add
ops.vector_mul = _hologram.vector_mul
ops.vector_div = _hologram.vector_div
ops.neg = _hologram.neg
ops.abs = _hologram.abs
ops.min = _hologram.min
ops.max = _hologram.max

# Activation functions
ops.relu = _hologram.relu
ops.sigmoid = _hologram.sigmoid
ops.tanh = _hologram.tanh
ops.softmax = _hologram.softmax

# Transcendental functions
ops.exp = _hologram.exp
ops.log = _hologram.log
ops.sqrt = _hologram.sqrt
ops.pow = _hologram.pow

# Reduction operations
ops.sum = _hologram.sum
ops.max_reduce = _hologram.max_reduce
ops.min_reduce = _hologram.min_reduce

# Linear algebra
ops.gemm = _hologram.gemm

# Loss functions
ops.mse_loss = _hologram.mse_loss
ops.cross_entropy_loss = _hologram.cross_entropy_loss

# Memory operations
ops.copy = _hologram.copy
ops.fill = _hologram.fill

__version__ = "0.1.0"

__all__ = [
    "Executor",
    "Buffer",
    "ops",
    "__version__",
]
