"""
Tanh Activation - Element-wise hyperbolic tangent

Operation: c[i] = tanh(a[i]) = (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i]))

This kernel performs element-wise tanh activation using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Output range: (-1, 1)
- Odd function: tanh(-x) = -tanh(x)
- Derivative: tanh'(x) = 1 - tanh²(x)

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def tanh_activation(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise tanh: c = tanh(a)"""
    idx = get_global_id()
    if idx < n:
        c[idx] = tanh(a[idx])
