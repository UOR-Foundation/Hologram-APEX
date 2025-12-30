"""
Vector Square Root - Element-wise square root

Operation: c[i] = sqrt(a[i])

This kernel performs element-wise square root using Atlas geometric folding
with compile-time class indices for optimal cache layout.

Mathematical Properties:
- Only defined for non-negative inputs: sqrt(x) requires x >= 0
- Inverse of squaring: sqrt(x²) = |x|
- sqrt(0) = 0
- sqrt(1) = 1

Performance:
- Memory bandwidth bound (2 vectors: 1 read, 1 write)
- O(n) time complexity
- Fully parallelizable
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_sqrt(a: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise square root: c = sqrt(a)"""
    idx = get_global_id()
    if idx < n:
        c[idx] = sqrt(a[idx])
