"""
ONNX InstanceNormalization Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html

Applies Instance Normalization over each channel in each sample.

Formula:
  y = scale * (x - mean) / sqrt(variance + epsilon) + bias

where mean and variance are computed per instance, per channel.

Shapes:
  - X: (N, C, H, W) - Input
  - scale: (C,) - Per-channel scale
  - bias: (C,) - Per-channel bias
  - Y: (N, C, H, W) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def instance_normalization(
    X: DeviceArray[f32],
    scale: DeviceArray[f32],
    bias: DeviceArray[f32],
    Y: DeviceArray[f32],
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    epsilon: f32
):
    """
    ONNX InstanceNormalization: Normalize per instance, per channel

    Parameters:
    - X: Input tensor (N×C×H×W)
    - scale: Per-channel scale factors (C,)
    - bias: Per-channel bias terms (C,)
    - Y: Output tensor (N×C×H×W)
    - N: Batch size
    - C: Number of channels
    - H: Height
    - W: Width
    - epsilon: Small constant for numerical stability
    """
    # Each work item processes one (N, C) instance
    instance_idx = get_global_id()

    total_instances = N * C

    if instance_idx < total_instances:
        n = instance_idx // C
        c = instance_idx % C

        spatial_size = H * W
        base_offset = n * C * spatial_size + c * spatial_size

        # Compute mean over spatial dimensions (H×W)
        sum_val = 0.0
        for hw in range(spatial_size):
            sum_val += X[base_offset + hw]
        mean = sum_val / spatial_size

        # Compute variance
        sum_sq = 0.0
        for hw in range(spatial_size):
            diff = X[base_offset + hw] - mean
            sum_sq += diff * diff
        variance = sum_sq / spatial_size

        # Normalize
        std_inv = 1.0 / sqrt(variance + epsilon)
        for hw in range(spatial_size):
            normalized = (X[base_offset + hw] - mean) * std_inv
            Y[base_offset + hw] = scale[c] * normalized + bias[c]
