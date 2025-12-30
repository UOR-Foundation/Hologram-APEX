"""
ONNX ArgMax Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__ArgMax.html

Computes the indices of the max elements of the input tensor's elements along the provided axis.

Simplified Implementation:
  - 1D only: finds index of maximum value
  - Returns single index

Shapes:
  - X: (N,) - Input
  - Y: Scalar - Output index
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def argmax(
    X: DeviceArray[f32],
    Y: DeviceArray[u32],
    n: u32
):
    """
    ONNX ArgMax: Y = argmax(X)

    Parameters:
    - X: Input tensor
    - Y: Output index (u32)
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx == 0:
        max_val = X[0]
        max_idx = 0

        for i in range(1, n):
            if X[i] > max_val:
                max_val = X[i]
                max_idx = i

        Y[0] = max_idx
