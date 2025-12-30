"""
Vector Division - Element-wise division of two vectors

Operation: c[i] = a[i] / b[i]

This kernel performs element-wise vector division using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Non-commutative: a / b ≠ b / a
- Right identity: a / 1 = a
- Division by zero undefined (NaN or Inf)

Performance:
- Memory bandwidth bound (3 vectors: 2 reads, 1 write)
- O(n) time complexity
- Fully parallelizable
- Expected: ~GB/s on CPU with SIMD
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_div(a: DeviceArray[f32], b: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Divide two vectors element-wise: c = a / b"""
    idx = get_global_id()
    if idx < n:
        c[idx] = a[idx] / b[idx]
