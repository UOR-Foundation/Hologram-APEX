"""
ONNX Div Operation - Element-wise Division

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Div.html

Performs element-wise binary division with NumPy-style broadcasting.

Shapes:
  - A, B: Any shape (broadcastable)
  - C: Broadcasted shape
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def div(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    C: DeviceArray[f32],
    n: u32
):
    """
    ONNX Div: C = A / B (element-wise)

    Parameters:
    - A: Dividend tensor
    - B: Divisor tensor
    - C: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        C[idx] = A[idx] / B[idx]
