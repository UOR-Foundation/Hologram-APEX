"""
ONNX Relu Operation - Rectified Linear Unit

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Relu.html

Applies Relu function element-wise:
  Relu(x) = max(0, x)

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def relu(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Relu: Y = max(0, X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        val = X[idx]
        if val > 0.0:
            Y[idx] = val
        else:
            Y[idx] = 0.0
