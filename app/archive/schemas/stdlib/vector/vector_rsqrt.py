"""
Vector Reciprocal Square Root - Element-wise reciprocal square root

Operation: c[i] = 1 / sqrt(a[i])

This kernel performs element-wise reciprocal square root using Atlas geometric folding.
Commonly used in normalization operations (e.g., LayerNorm, GroupNorm).

Mathematical Properties:
- Only defined for positive inputs: rsqrt(x) requires x > 0
- Equivalent to: 1 / sqrt(x)
- rsqrt(1) = 1
- More efficient than computing sqrt then dividing

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
- Typically faster than separate sqrt + div
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_rsqrt(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise reciprocal square root: c = 1 / sqrt(a)"""
    idx = get_global_id()
    if idx < n:
        c[idx] = rsqrt(a[idx])
