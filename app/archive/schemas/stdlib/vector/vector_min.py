"""
Vector Minimum - Element-wise minimum of two vectors

Operation: c[i] = min(a[i], b[i])

This kernel performs element-wise vector minimum using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Commutative: min(a, b) = min(b, a)
- Associative: min(min(a, b), c) = min(a, min(b, c))
- Idempotent: min(a, a) = a

Performance:
- Memory bandwidth bound (3 vectors: 2 reads, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_min(a: DeviceArray[f32], b: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise minimum: c = min(a, b)"""
    idx = get_global_id()
    if idx < n:
        c[idx] = min(a[idx], b[idx])
