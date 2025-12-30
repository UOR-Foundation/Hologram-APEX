"""
ONNX Squeeze Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Squeeze.html

Removes dimensions of size 1 from the shape of a tensor.

Simplified Implementation:
  - Memory layout remains same (row-major)
  - Only metadata (shape) changes
  - This is a simple copy operation

Shapes:
  - X: Any shape with some dimensions = 1
  - Y: Shape with size-1 dimensions removed
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def squeeze(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Squeeze: Y = squeeze(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor (squeezed)
    - n: Total number of elements

    Note: Shape metadata handled by runtime, this is identity copy
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = X[idx]
