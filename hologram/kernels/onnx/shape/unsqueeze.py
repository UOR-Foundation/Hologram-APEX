"""
ONNX Unsqueeze Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html

Inserts dimensions of size 1 into the shape of a tensor.

Simplified Implementation:
  - Memory layout remains same (row-major)
  - Only metadata (shape) changes
  - This is a simple copy operation

Shapes:
  - X: Any shape
  - Y: Shape with size-1 dimensions inserted
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def unsqueeze(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Unsqueeze: Y = unsqueeze(X)

    Parameters:
    - X: Input tensor
    - Y: Output tensor (unsqueezed)
    - n: Total number of elements

    Note: Shape metadata handled by runtime, this is identity copy
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = X[idx]
