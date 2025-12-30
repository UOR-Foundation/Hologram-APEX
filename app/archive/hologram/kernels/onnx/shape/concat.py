"""
ONNX Concat Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Concat.html

Concatenates input tensors along a specified axis.

Simplified Implementation:
  - Concatenates 2 tensors along axis 0 (batch dimension)
  - Assumes same shape except for concat axis

Shapes:
  - A: (M, N) - First input
  - B: (K, N) - Second input
  - Y: (M+K, N) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def concat(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    Y: DeviceArray[f32],
    M: u32,
    K: u32,
    N: u32
):
    """
    ONNX Concat: Concatenate along axis 0

    Parameters:
    - A: First input (M × N)
    - B: Second input (K × N)
    - Y: Output ((M+K) × N)
    - M: Rows in A
    - K: Rows in B
    - N: Columns (same for both)
    """
    idx = get_global_id()

    total = (M + K) * N

    if idx < total:
        # Copy from A or B depending on row index
        row = idx // N
        col = idx % N

        if row < M:
            # Copy from A
            Y[idx] = A[row * N + col]
        else:
            # Copy from B
            b_row = row - M
            Y[idx] = B[b_row * N + col]
