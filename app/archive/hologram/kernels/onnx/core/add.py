"""
ONNX Add Operation - Element-wise Addition

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Add.html

Performs element-wise binary addition with NumPy-style broadcasting.

Simplified Atlas Implementation:
  - Assumes inputs have same shape (no broadcasting)
  - For broadcasting, handle in ONNX runtime layer

Shapes:
  - A: (N,) or (M, N) or any shape
  - B: (N,) or (M, N) or broadcastable
  - C: (N,) or (M, N) - same as broadcasted shape
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def add(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    C: DeviceArray[f32],
    n: u32
):
    """
    ONNX Add: C = A + B (element-wise)

    Parameters:
    - A: First input tensor
    - B: Second input tensor
    - C: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        C[idx] = A[idx] + B[idx]
