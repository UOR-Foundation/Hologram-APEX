// Type cast shader: Float32 → Int32
// Implements element-wise type conversion
//
// Operation: output[i] = i32(input[i])

@group(0) @binding(0)
var<storage, read> input: array<f32>;

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

    // Type cast: f32 → i32
    output[idx] = i32(input[idx]);
}
