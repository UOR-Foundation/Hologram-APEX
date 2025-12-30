"""
ONNX Sub Operation - Element-wise Subtraction

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Sub.html

Performs element-wise binary subtraction with NumPy-style broadcasting.

Shapes:
  - A, B: Any shape (broadcastable)
  - C: Broadcasted shape
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def sub(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    C: DeviceArray[f32],
    n: u32
):
    """
    ONNX Sub: C = A - B (element-wise)

    Parameters:
    - A: First input tensor
    - B: Second input tensor
    - C: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        C[idx] = A[idx] - B[idx]
