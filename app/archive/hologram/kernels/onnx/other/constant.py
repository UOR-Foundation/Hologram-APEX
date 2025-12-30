"""
ONNX Constant Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Constant.html

Produces a constant tensor.

Simplified Implementation:
  - Fills output with a single constant value
  - For complex constant tensors, handle in ONNX runtime layer

Shapes:
  - Y: Any shape
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def constant_fill(
    Y: DeviceArray[f32],
    value: f32,
    n: u32
):
    """
    ONNX Constant: Y = [value, value, ...]

    Parameters:
    - Y: Output tensor
    - value: Constant value to fill
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = value
