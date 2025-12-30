"""
ONNX Flatten Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Flatten.html

Flattens the input tensor into a 2D matrix.

Simplified Implementation:
  - Flattens to 1D (axis=0)
  - Memory layout remains same (row-major)
  - This is a simple copy operation

Shapes:
  - X: Any shape
  - Y: Flattened shape
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def flatten(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Flatten: Y = flatten(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor (flattened)
    - n: Total number of elements

    Note: Shape metadata handled by runtime, this is identity copy
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = X[idx]
