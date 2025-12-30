"""
ONNX LayerNormalization Operation

ONNX Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization

Applies Layer Normalization over the last dimensions.

Formula:
  y = scale * (x - mean) / sqrt(variance + epsilon) + bias

where mean and variance are computed over the normalized_shape dimensions.

Simplified Implementation:
  - Normalizes over last axis (features)
  - Assumes 2D input: (batch_size, features)

Shapes:
  - X: (batch_size, features)
  - scale: (features,)
  - bias: (features,)
  - Y: (batch_size, features)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def layer_norm(
    X: DeviceArray[f32],
    scale: DeviceArray[f32],
    bias: DeviceArray[f32],
    Y: DeviceArray[f32],
    batch_size: u32,
    features: u32,
    epsilon: f32
):
    """
    ONNX LayerNormalization: Normalize over feature dimension

    Parameters:
    - X: Input tensor (batch_size × features)
    - scale: Per-feature scale factors (features,)
    - bias: Per-feature bias terms (features,)
    - Y: Output tensor (batch_size × features)
    - batch_size: Number of samples
    - features: Number of features per sample
    - epsilon: Small constant for numerical stability
    """
    batch_idx = get_global_id()

    if batch_idx < batch_size:
        offset = batch_idx * features

        # Compute mean
        sum_val = 0.0
        for i in range(features):
            sum_val += X[offset + i]
        mean = sum_val / features

        # Compute variance
        sum_sq = 0.0
        for i in range(features):
            diff = X[offset + i] - mean
            sum_sq += diff * diff
        variance = sum_sq / features

        # Normalize and scale
        std_inv = 1.0 / sqrt(variance + epsilon)
        for i in range(features):
            normalized = (X[offset + i] - mean) * std_inv
            Y[offset + i] = scale[i] * normalized + bias[i]
