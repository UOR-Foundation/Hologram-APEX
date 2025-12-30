// Type cast shader: Int64 → Float32
// Implements element-wise type conversion
//
// Operation: output[i] = f32(input[i])
//
// Note: WebGPU doesn't natively support i64, so we read two i32 values
// We only use the lower 32 bits for now (sufficient for shape tensors)

@group(0) @binding(0)
var<storage, read> input: array<i32>;  // Stored as 2x i32 per i64

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> num_elements: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;

    if (idx >= num_elements) {
        return;
    }

    // Read i64 as two i32 values (low, high)
    let value_low = input[idx * 2u];

    // Type cast: i32 (low bits of i64) → f32
    output[idx] = f32(value_low);
}
