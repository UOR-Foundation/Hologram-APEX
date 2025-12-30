"""
ONNX Gelu Operation - Gaussian Error Linear Unit

ONNX Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu

Applies GELU activation function element-wise.

Approximate formula (tanh-based):
  GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

Shapes:
  - X: Any shape
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def gelu(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX Gelu: Y = 0.5 * x * (1 + tanh(...))

    Parameters:
    - X: Input tensor
    - Y: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        x = X[idx]
        # Constants for GELU approximation
        sqrt_2_over_pi = 0.7978845608  # sqrt(2/π)
        coeff = 0.044715

        # Compute: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + coeff * x_cubed)
        Y[idx] = 0.5 * x * (1.0 + tanh(inner))
