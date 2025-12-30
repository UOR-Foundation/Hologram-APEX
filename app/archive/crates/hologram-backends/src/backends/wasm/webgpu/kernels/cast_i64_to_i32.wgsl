// Type cast shader: Int64 → Int32
// Implements element-wise type conversion (truncation)
//
// Operation: output[i] = i32(input[i])
//
// Note: WebGPU doesn't natively support i64, so we read two i32 values
// We only use the lower 32 bits (truncation)

@group(0) @binding(0)
var<storage, read> input: array<i32>;  // Stored as 2x i32 per i64

@group(0) @binding(1)
var<storage, read_write> output: array<i32>;

@group(0) @binding(2)
var<uniform> num_elements: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;

    if (idx >= num_elements) {
        return;
    }

    // Read i64 as two i32 values and extract low 32 bits
    let value_low = input[idx * 2u];

    // Truncate to i32
    output[idx] = value_low;
}
