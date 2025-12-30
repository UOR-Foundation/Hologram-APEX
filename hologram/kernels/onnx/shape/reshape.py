"""
ONNX Reshape Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Reshape.html

Reshapes the input tensor to the specified shape.

Simplified Implementation:
  - Memory layout remains same (row-major)
  - Only metadata (shape/strides) changes
  - This is a simple copy operation

Shapes:
  - X: Any shape
  - Y: Target shape (same total elements)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def reshape(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Reshape: Y = reshape(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor (same data, different shape)
    - n: Total number of elements

    Note: Shape metadata handled by runtime, this is identity copy
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = X[idx]
