"""
ONNX Conv Operation - Convolution

ONNX Spec: https://onnx.ai/onnx/operators/onnx__Conv.html

Performs convolution operation on input tensors.

Simplified Implementation:
  - 2D convolution
  - No padding, no dilation, stride=1
  - Single input/output channel

Shapes:
  - X: (H_in, W_in) - Input
  - W: (K_h, K_w) - Kernel
  - Y: (H_out, W_out) - Output
    where H_out = H_in - K_h + 1
          W_out = W_in - K_w + 1
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def conv2d(
    X: DeviceArray[f32],
    W: DeviceArray[f32],
    Y: DeviceArray[f32],
    H_in: u32,
    W_in: u32,
    K_h: u32,
    K_w: u32
):
    """
    ONNX Conv: 2D Convolution

    Parameters:
    - X: Input tensor (H_in × W_in)
    - W: Weight/kernel tensor (K_h × K_w)
    - Y: Output tensor ((H_in - K_h + 1) × (W_in - K_w + 1))
    - H_in: Input height
    - W_in: Input width
    - K_h: Kernel height
    - K_w: Kernel width
    """
    idx = get_global_id()

    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1
    total = H_out * W_out

    if idx < total:
        out_i = idx // W_out
        out_j = idx % W_out

        # Compute convolution at position (out_i, out_j)
        sum = 0.0
        for ki in range(K_h):
            for kj in range(K_w):
                in_i = out_i + ki
                in_j = out_j + kj
                x_val = X[in_i * W_in + in_j]
                w_val = W[ki * K_w + kj]
                sum += x_val * w_val

        Y[idx] = sum
