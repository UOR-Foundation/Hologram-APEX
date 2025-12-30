"""
ONNX Tanh Operation - Hyperbolic Tangent

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Tanh.html

Applies Tanh function element-wise:
  Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def tanh_activation(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Tanh: Y = tanh(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = tanh(X[idx])
