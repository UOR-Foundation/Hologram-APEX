"""
ONNX BiasGelu Operation (FastGelu)

Custom operation combining bias addition and GELU activation.
Common in transformer models (BERT, GPT).

GELU (Gaussian Error Linear Unit):
  GELU(x) = x * Φ(x)
  where Φ(x) is the cumulative distribution function of standard normal distribution

Fast approximation:
  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

Shapes:
  - X: Any shape
  - Bias: Broadcastable to X
  - Y: Same shape as X
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def bias_gelu(
    X: DeviceArray[f32],
    Bias: DeviceArray[f32],
    Y: DeviceArray[f32],
    n: u32
):
    """
    ONNX BiasGelu: Y = GELU(X + Bias)

    Parameters:
    - X: Input tensor
    - Bias: Bias tensor
    - Y: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        x = X[idx] + Bias[idx]

        # Fast GELU approximation
        # 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        x_cubed = x * x * x
        inner = 0.797885 * (x + 0.044715 * x_cubed)  # √(2/π) ≈ 0.797885
        tanh_val = tanh(inner)

        Y[idx] = 0.5 * x * (1.0 + tanh_val)
