"""Layer Normalization operation

Normalizes across the last dimension (hidden_dim) for each position.

Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

Where:
- mean = sum(x) / hidden_dim
- var = sum((x - mean)^2) / hidden_dim

Input shape: [batch_size * seq_len, hidden_dim]
(flattened from [batch_size, seq_len, hidden_dim])
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id, sqrt, rsqrt

def layer_norm(
    input: DeviceArray[f32],      # Input tensor [num_positions, hidden_dim]
    gamma: DeviceArray[f32],       # Scale parameter [hidden_dim]
    beta: DeviceArray[f32],        # Bias parameter [hidden_dim]
    output: DeviceArray[f32],      # Output tensor [num_positions, hidden_dim]
    num_positions: u32,            # batch_size * seq_len
    hidden_dim: u32,               # Hidden dimension
    eps: f32,                      # Epsilon for numerical stability
):
    """Layer normalization: normalize across hidden_dim for each position

    For each position (batch, seq):
        mean = sum(x) / hidden_dim
        var = sum((x - mean)^2) / hidden_dim
        output = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    # Each thread processes one position
    pos = get_global_id()

    if pos < num_positions:
        offset = pos * hidden_dim

        # Step 1: Compute mean
        sum_val = 0.0
        for i in range(hidden_dim):
            sum_val += input[offset + i]
        mean = sum_val / float(hidden_dim)

        # Step 2: Compute variance
        var_sum = 0.0
        for i in range(hidden_dim):
            diff = input[offset + i] - mean
            var_sum += diff * diff
        variance = var_sum / float(hidden_dim)

        # Step 3: Normalize and apply scale/bias
        inv_std = rsqrt(variance + eps)
        for i in range(hidden_dim):
            normalized = (input[offset + i] - mean) * inv_std
            output[offset + i] = gamma[i] * normalized + beta[i]
