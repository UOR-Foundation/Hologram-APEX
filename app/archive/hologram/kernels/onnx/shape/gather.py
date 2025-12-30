"""
ONNX Gather Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Gather.html

Gathers entries of the input tensor along the specified axis using indices.

Simplified Implementation:
  - 1D gather only
  - Gathers elements based on index array

Shapes:
  - X: (N,) - Input data
  - Indices: (M,) - Index array
  - Y: (M,) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def gather(
    X: DeviceArray[f32],
    Indices: DeviceArray[u32],
    Y: DeviceArray[f32],
    m: u32
):
    """
    ONNX Gather: Y[i] = X[Indices[i]]

    Parameters:
    - X: Input tensor
    - Indices: Index array
    - Y: Output tensor
    - m: Number of indices
    """
    idx = get_global_id()

    if idx < m:
        index = Indices[idx]
        Y[idx] = X[index]
