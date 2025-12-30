"""
ONNX Cast Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Cast.html

Casts the elements of a given input tensor to a data type specified by the 'to' argument.

Simplified Implementation:
  - f32 to f32 (identity)
  - For other type conversions, handle in ONNX runtime layer

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def cast_f32_to_f32(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Cast: Y = cast(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor (casted)
    - n: Total number of elements

    Note: This is identity for f32→f32. Runtime handles other types.
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = X[idx]
