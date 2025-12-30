"""
Sigmoid Activation - Element-wise sigmoid function

Operation: c[i] = 1 / (1 + exp(-a[i]))

This kernel performs element-wise sigmoid activation using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Output range: (0, 1)
- Smooth S-shaped curve
- Derivative: σ'(x) = σ(x) * (1 - σ(x))

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def sigmoid(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise sigmoid: c = 1 / (1 + exp(-a))"""
    idx = get_global_id()
    if idx < n:
        c[idx] = 1.0 / (1.0 + exp(-a[idx]))
