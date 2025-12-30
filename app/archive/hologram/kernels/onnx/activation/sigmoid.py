"""
ONNX Sigmoid Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Sigmoid.html

Applies Sigmoid function element-wise:
  Sigmoid(x) = 1 / (1 + exp(-x))

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def sigmoid(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Sigmoid: Y = 1 / (1 + exp(-X))

    Parameters:
    - X: Input tensor
    - Y: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = 1.0 / (1.0 + exp(-X[idx]))
