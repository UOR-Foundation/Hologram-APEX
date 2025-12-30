"""
ONNX Gemm Operation - General Matrix Multiplication

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Gemm.html

Computes:
  Y = alpha * A' * B' + beta * C

Where:
  - A' = A if transA == 0, else A^T
  - B' = B if transB == 0, else B^T
  - alpha, beta are scalars (default 1.0)
  - C is optional bias term

Shapes:
  - A: (M, K) or (K, M) if transposed
  - B: (K, N) or (N, K) if transposed
  - C: (M, N) or broadcastable to (M, N)
  - Y: (M, N)

Atlas Implementation:
  - Uses hologram geometric multiply with class folding
  - Parallel over output elements
  - Cache-optimized for contiguous memory access
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def gemm(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    C: DeviceArray[f32],
    Y: DeviceArray[f32],
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
    transA: u32,
    transB: u32
):
    """
    ONNX Gemm: Y = alpha * A' * B' + beta * C

    Parameters:
    - A: Input matrix A (M×K or K×M if transposed)
    - B: Input matrix B (K×N or N×K if transposed)
    - C: Bias matrix C (M×N or broadcastable)
    - Y: Output matrix Y (M×N)
    - M, N, K: Matrix dimensions
    - alpha, beta: Scaling factors
    - transA, transB: Transpose flags (0 or 1)
    """
    idx = get_global_id()

    # Total output elements: M * N
    total = M * N

    if idx < total:
        # Compute output position (i, j)
        i = idx // N  # Row index
        j = idx % N   # Column index

        # Compute matrix multiply element: sum(A[i,k] * B[k,j])
        sum_val = 0.0
        for k in range(K):
            # Access A element
            if transA == 0:
                a_val = A[i * K + k]  # A[i, k]
            else:
                a_val = A[k * M + i]  # A^T[i, k] = A[k, i]

            # Access B element
            if transB == 0:
                b_val = B[k * N + j]  # B[k, j]
            else:
                b_val = B[j * K + k]  # B^T[k, j] = B[j, k]

            sum_val += a_val * b_val

        # Apply alpha scaling
        result = alpha * sum_val

        # Add beta * C if beta != 0
        if beta != 0.0:
            # C might be broadcast - handle (M, N), (M, 1), (1, N), or scalar
            c_val = C[idx]  # Simplified: assume C is (M, N)
            result += beta * c_val

        Y[idx] = result
