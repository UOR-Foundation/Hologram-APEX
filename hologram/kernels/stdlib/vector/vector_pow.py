"""
Vector Power - Element-wise power operation

Operation: c[i] = pow(a[i], b[i])

This kernel performs element-wise power computation using Atlas geometric folding.

Mathematical Properties:
- a^0 = 1 (for any a != 0)
- a^1 = a
- a^(-1) = 1/a
- a^(b+c) = a^b * a^c
- (a*b)^c = a^c * b^c

Performance:
- Memory bandwidth bound (3 vectors: 2 reads, 1 write)
- O(n) time complexity
- Fully parallelizable
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_pow(a: DeviceArray[f32], b: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    """Compute element-wise power: c = a^b"""
    idx = get_global_id()
    if idx < n:
        c[idx] = pow(a[idx], b[idx])
