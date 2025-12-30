"""
Scalar Multiplication Operation

Multiplies each element of a vector by a scalar value.

Output[i] = Input[i] * scalar

Shapes:
  - Input: (N,) - Input vector
  - Output: (N,) - Output vector
  - scalar: scalar value passed as parameter
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def scalar_mul(
    Input: DeviceArray[f32],
    Output: DeviceArray[f32],
    n: u32,
    scalar: f32
):
    """
    Scalar Multiplication: Output = Input * scalar

    Parameters:
    - Input: Input tensor
    - Output: Output tensor
    - n: Total number of elements
    - scalar: Scalar multiplier
    """
    idx = get_global_id()

    if idx < n:
        Output[idx] = Input[idx] * scalar
