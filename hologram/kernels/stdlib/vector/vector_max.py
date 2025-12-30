"""
Vector Maximum - Element-wise maximum of two vectors

Operation: c[i] = max(a[i], b[i])

This kernel performs element-wise vector maximum using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Commutative: max(a, b) = max(b, a)
- Associative: max(max(a, b), c) = max(a, max(b, c))
- Idempotent: max(a, a) = a

Performance:
- Memory bandwidth bound (3 vectors: 2 reads, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_max(a: DeviceArray[f32], b: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise maximum: c = max(a, b)"""
    idx = get_global_id()
    if idx < n:
        c[idx] = max(a[idx], b[idx])
