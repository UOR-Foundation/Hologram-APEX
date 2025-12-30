"""
ONNX Range Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Range.html

Generates a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive).

Similar to Python's range() or NumPy's arange().

Shapes:
  - Output: ((limit - start) / delta,)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def range_op(
    Y: DeviceArray[f32],
    start: f32,
    delta: f32,
    n: u32
):
    """
    ONNX Range: Y = [start, start+delta, start+2*delta, ...]

    Parameters:
    - Y: Output tensor
    - start: Starting value
    - delta: Step size
    - n: Number of elements
    """
    idx = get_global_id()

    if idx < n:
        Y[idx] = start + f32(idx) * delta
