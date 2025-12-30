"""
ONNX GlobalAveragePool Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html

Applies average pooling over all spatial dimensions.
Equivalent to AveragePool with kernel_size = (H, W).

Shapes:
  - X: (N, C, H, W) - Input
  - Y: (N, C, 1, 1) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def global_average_pool(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    N: u32,
    C: u32,
    H: u32,
    W: u32
):
    """
    ONNX GlobalAveragePool: Average over all spatial dimensions

    Parameters:
    - X: Input tensor (N×C×H×W)
    - Y: Output tensor (N×C×1×1)
    - N: Batch size
    - C: Number of channels
    - H: Input height
    - W: Input width
    """
    # Each work item processes one (N, C) channel
    channel_idx = get_global_id()

    total_channels = N * C

    if channel_idx < total_channels:
        n = channel_idx // C
        c = channel_idx % C

        spatial_size = H * W
        base_offset = (n * C + c) * spatial_size

        # Compute average over H×W
        sum_val = 0.0
        for hw in range(spatial_size):
            sum_val += X[base_offset + hw]

        Y[channel_idx] = sum_val / spatial_size
