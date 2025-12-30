"""
ONNX AveragePool Operation - Average Pooling

ONNX Spec: https://onnx.ai/onnx/operators/onnx__AveragePool.html

Applies average pooling over spatial dimensions.

Simplified Implementation:
  - 2D pooling (H×W dimensions)
  - Square kernel (kernel_size × kernel_size)
  - Stride = kernel_size (non-overlapping)
  - No padding

Shapes:
  - X: (N, C, H, W) - Input
  - Y: (N, C, H_out, W_out) - Output
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def averagepool(
    X: DeviceArray[f32],
    Y: DeviceArray[f32],
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    kernel_size: u32
):
    """
    ONNX AveragePool: Apply average pooling

    Parameters:
    - X: Input tensor (N×C×H×W)
    - Y: Output tensor (N×C×H_out×W_out)
    - N: Batch size
    - C: Number of channels
    - H: Input height
    - W: Input width
    - kernel_size: Pooling kernel size (square)
    """
    idx = get_global_id()

    H_out = H // kernel_size
    W_out = W // kernel_size
    total = N * C * H_out * W_out

    if idx < total:
        # Decode output position
        output_spatial = H_out * W_out
        channel_output = C * output_spatial

        n = idx // channel_output
        remainder = idx % channel_output
        c = remainder // output_spatial
        spatial_idx = remainder % output_spatial

        h_out = spatial_idx // W_out
        w_out = spatial_idx % W_out

        # Input region start
        h_start = h_out * kernel_size
        w_start = w_out * kernel_size

        # Compute average over kernel_size × kernel_size window
        sum_val = 0.0
        kernel_area = kernel_size * kernel_size

        input_channel_offset = (n * C + c) * H * W

        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_idx = h_start + kh
                w_idx = w_start + kw
                input_idx = input_channel_offset + h_idx * W + w_idx
                sum_val += X[input_idx]

        Y[idx] = sum_val / kernel_area
