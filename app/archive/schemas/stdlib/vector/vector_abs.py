"""
Vector Absolute Value - Element-wise absolute value

Operation: c[i] = |a[i]|

This kernel performs element-wise absolute value using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Always non-negative: |x| >= 0
- Idempotent on non-negative: |x| = x for x >= 0
- Symmetric: |-x| = |x|

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_abs(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise absolute value: c = |a|"""
    idx = get_global_id()
    if idx < n:
        c[idx] = abs(a[idx])
