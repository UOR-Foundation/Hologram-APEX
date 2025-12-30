"""
ONNX Transpose Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Transpose.html

Transposes the input tensor similar to numpy.transpose.

Simplified Implementation:
  - 2D transpose only (matrix transpose)
  - Swaps dimensions: (M, N) → (N, M)

Shapes:
  - X: (M, N) - Input
  - Y: (N, M) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def transpose(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    M: u32,
    N: u32
):
    """
    ONNX Transpose: Transpose 2D matrix

    Parameters:
    - X: Input matrix (M × N)
    - Y: Output matrix (N × M)
    - M: Number of rows in X
    - N: Number of columns in X
    """
    idx = get_global_id()

    total = M * N

    if idx < total:
        # Position in input: (i, j)
        i = idx // N
        j = idx % N

        # Position in output: (j, i)
        output_idx = j * M + i

        Y[output_idx] = X[idx]
