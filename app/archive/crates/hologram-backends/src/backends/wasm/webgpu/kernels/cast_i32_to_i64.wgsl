// Type cast shader: Int32 → Int64
// Implements element-wise type conversion
//
// Operation: output[i] = i64(input[i])
//
// Note: WebGPU doesn't natively support i64, so we use two i32 values

@group(0) @binding(0)
var<storage, read> input: array<i32>;

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

    let value = input[idx];

    // Store as i64 (two i32 values: low, high)
    output[idx * 2u] = value;        // Low 32 bits
    output[idx * 2u + 1u] = select(0, -1, value < 0);  // High 32 bits (sign extension)
}
