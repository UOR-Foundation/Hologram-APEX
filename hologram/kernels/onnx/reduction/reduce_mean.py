"""
ONNX ReduceMean Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__ReduceMean.html

Computes the mean of input tensor's elements along specified axes.

Simplified Implementation:
  - Reduces over last axis
  - keepdims = 0 (removes reduced dimension)

Shapes:
  - X: (batch_size, features) - Input
  - Y: (batch_size,) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def reduce_mean(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    batch_size: u32,
    features: u32
):
    """
    ONNX ReduceMean: Compute mean over features dimension

    Parameters:
    - X: Input tensor (batch_size × features)
    - Y: Output tensor (batch_size,)
    - batch_size: Outer dimension size
    - features: Dimension to reduce over
    """
    batch_idx = get_global_id()

    if batch_idx < batch_size:
        offset = batch_idx * features

        sum_val = 0.0
        for i in range(features):
            sum_val += X[offset + i]

        Y[batch_idx] = sum_val / features
