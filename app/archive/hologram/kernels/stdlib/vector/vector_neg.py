"""
Vector Negation - Element-wise negation

Operation: c[i] = -a[i]

This kernel performs element-wise negation using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Self-inverse: -(-x) = x
- Distributes over addition: -(x + y) = -x + -y
- Identity for zero: -0 = 0

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_neg(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise negation: c = -a"""
    idx = get_global_id()
    if idx < n:
        c[idx] = -a[idx]
