"""
ONNX BatchNormalization Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html

Applies Batch Normalization over a 4D input (NCHW format).

Formula:
  y = scale * (x - mean) / sqrt(variance + epsilon) + bias

where:
  - mean, variance: per-channel statistics
  - scale, bias: learned per-channel parameters
  - epsilon: small constant for numerical stability

Shapes:
  - X: (N, C, H, W) - Input
  - scale: (C,) - Per-channel scale
  - bias: (C,) - Per-channel bias
  - mean: (C,) - Per-channel mean
  - var: (C,) - Per-channel variance
  - Y: (N, C, H, W) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def batch_normalization(
    X: DeviceArray[f32],
    scale: DeviceArray[f32],
    bias: DeviceArray[f32],
    mean: DeviceArray[f32],
    var: DeviceArray[f32],
    Y: DeviceArray[f32],
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    epsilon: f32
):
    """
    ONNX BatchNormalization: Normalize across batch dimension

    Parameters:
    - X: Input tensor (N×C×H×W)
    - scale: Per-channel scale factors (C,)
    - bias: Per-channel bias terms (C,)
    - mean: Per-channel mean values (C,)
    - var: Per-channel variance values (C,)
    - Y: Output tensor (N×C×H×W)
    - N: Batch size
    - C: Number of channels
    - H: Height
    - W: Width
    - epsilon: Small constant for numerical stability
    """
    idx = get_global_id()

    total = N * C * H * W

    if idx < total:
        # Compute indices: idx = n*C*H*W + c*H*W + h*W + w
        spatial_size = H * W
        channel_size = C * spatial_size

        n = idx // channel_size
        remainder = idx % channel_size
        c = remainder // spatial_size

        # Normalize: (x - mean) / sqrt(var + eps)
        x = X[idx]
        normalized = (x - mean[c]) / sqrt(var[c] + epsilon)

        # Scale and shift: scale * normalized + bias
        Y[idx] = scale[c] * normalized + bias[c]
