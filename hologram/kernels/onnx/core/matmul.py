"""
ONNX MatMul Operation - Matrix Multiplication

ONNX Spec: https://onnx.ai/onnx/operators/onnx__MatMul.html

Computes matrix product of two tensors.

Behavior:
  - If both 2D: standard matrix multiply (M, K) × (K, N) → (M, N)
  - If ND: batched matrix multiply with broadcasting

Simplified Atlas Implementation:
  - 2D matrix multiplication only
  - For batched operations, use loop over batch dimension in ONNX runtime

Shapes:
  - A: (M, K)
  - B: (K, N)
  - Y: (M, N)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def matmul(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    Y: DeviceArray[f32],
    M: u32,
    N: u32,
    K: u32
):
    """
    ONNX MatMul: Y = A × B

    Parameters:
    - A: Input matrix A (M×K)
    - B: Input matrix B (K×N)
    - Y: Output matrix Y (M×N)
    - M, N, K: Matrix dimensions
    """
    idx = get_global_id()

    total = M * N

    if idx < total:
        i = idx // N  # Row in A
        j = idx % N   # Column in B

        # Dot product of row i of A with column j of B
        sum_val = 0.0
        for k in range(K):
            sum_val += A[i * K + k] * B[k * N + j]

        Y[idx] = sum_val
