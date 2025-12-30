// Type cast shader: Float32 → Int64
// Implements element-wise type conversion
//
// Operation: output[i] = i64(input[i])
//
// Note: WebGPU doesn't natively support i64, so we use two i32 values
// We only use the lower 32 bits for now (sufficient for shape tensors)

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<i32>;  // Stored as 2x i32 per i64

@group(0) @binding(2)
var<uniform> num_elements: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;

    if (idx >= num_elements) {
        return;
    }

    // Type cast: f32 → i32 (lower 32 bits of i64)
    let value_i32 = i32(input[idx]);

    // Store as i64 (two i32 values: low, high)
    output[idx * 2u] = value_i32;        // Low 32 bits
    output[idx * 2u + 1u] = select(0, -1, value_i32 < 0);  // High 32 bits (sign extension)
}
