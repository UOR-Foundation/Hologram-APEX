"""
ONNX LeakyRelu Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__LeakyRelu.html

Applies LeakyRelu function element-wise:
  LeakyRelu(x) = x if x >= 0 else alpha * x

where alpha is a small constant (typically 0.01).

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def leakyrelu(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32,
    alpha: f32
):
    """
    ONNX LeakyRelu: Y = x if x >= 0 else alpha * x

    Parameters:
    - X: Input tensor
    - Y: Output tensor
    - n: Total number of elements
    - alpha: Slope for negative values (default 0.01)
    """
    idx = get_global_id()

    if idx < n:
        x = X[idx]
        if x >= 0.0:
            Y[idx] = x
        else:
            Y[idx] = alpha * x
