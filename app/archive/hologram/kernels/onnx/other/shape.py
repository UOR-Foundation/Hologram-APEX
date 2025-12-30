"""
ONNX Shape Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Shape.html

Returns the shape of the input tensor as a 1D tensor.

Simplified Implementation:
  - Shape is typically known at compile time
  - This operation is usually handled at the metadata level
  - Runtime implementation just copies shape information

Shapes:
  - X: Any shape
  - Y: (rank,) - Shape dimensions as tensor
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def shape_to_tensor(
    Shape: DeviceArray[u32],
    Y: DeviceArray[u32],
    rank: u32
):
    """
    ONNX Shape: Y = shape(X)

    Parameters:
    - Shape: Input shape array (precomputed)
    - Y: Output tensor containing shape values
    - rank: Number of dimensions
    """
    idx = get_global_id()

    if idx < rank:
        Y[idx] = Shape[idx]
