"""
ONNX Softmax Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Softmax.html

Applies Softmax function along an axis:
  Softmax(x) = exp(x) / sum(exp(x))

Simplified Implementation:
  - Element-wise exp() only (normalization handled separately)
  - Full softmax requires multiple passes (handled in runtime)

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def softmax_exp(
    X: DeviceArray[f32],
    MaxVal: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Softmax (exp pass): Y = exp(X - max)

    Parameters:
    - X: Input tensor
    - MaxVal: Maximum value (1 element)
    - Y: Output tensor
    - n: Total number of elements

    Note: This is the exp pass. Normalization pass handled separately.
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = exp(X[idx] - MaxVal[0])
