"""
ONNX Attention Operation (Simplified)

Multi-head attention mechanism used in transformers.

Simplified Implementation:
  - Single-head attention
  - Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V

Shapes:
  - Q: (seq_len, d_model) - Query
  - K: (seq_len, d_model) - Key
  - V: (seq_len, d_model) - Value
  - Output: (seq_len, d_model)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def attention_scores(
    Q: DeviceArray[f32],
    K: DeviceArray[f32],
    Scores: DeviceArray[f32],
    seq_len: u32,
    d_model: u32
):
    """
    ONNX Attention: Compute attention scores Q·K^T / √d_k

    Parameters:
    - Q: Query matrix (seq_len × d_model)
    - K: Key matrix (seq_len × d_model)
    - Scores: Output scores (seq_len × seq_len)
    - seq_len: Sequence length
    - d_model: Model dimension
    """
    idx = get_global_id()

    total = seq_len * seq_len

    if idx < total:
        i = idx // seq_len
        j = idx % seq_len

        # Compute dot product Q[i] · K[j]
        dot = 0.0
        for k in range(d_model):
            dot += Q[i * d_model + k] * K[j * d_model + k]

        # Scale by √d_k
        scale = 1.0 / sqrt(f32(d_model))
        Scores[idx] = dot * scale
