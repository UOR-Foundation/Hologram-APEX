"""
ONNX Slice Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Slice.html

Produces a slice of the input tensor along multiple axes.

Simplified Implementation:
  - 1D slice only
  - Extracts elements from start to end (exclusive)

Shapes:
  - X: (N,) - Input
  - Y: (end - start,) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def slice_1d(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    start: u32,
    end: u32
):
    """
    ONNX Slice: Y = X[start:end]

    Parameters:
    - X: Input tensor
    - Y: Output tensor (sliced)
    - start: Start index
    - end: End index (exclusive)
    """
    idx = get_global_id()

    length = end - start

    if idx < length:
        Y[idx] = X[start + idx]
