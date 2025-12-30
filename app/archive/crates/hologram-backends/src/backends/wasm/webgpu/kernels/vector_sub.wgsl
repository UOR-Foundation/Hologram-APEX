// Vector subtraction compute shader
// Implements SplitRange(Sub) generator for element-wise subtraction
//
// Corresponds to: split@c[0..N](Sub)
// Operation: output[i] = input_a[i] - input_b[i]

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Compute linear index from 2D dispatch grid
    // Supports dispatch sizes exceeding 65,535 by using Y dimension
    let idx = global_id.y * 65535u + global_id.x;

    // Bounds check
    if (idx >= arrayLength(&input_a)) {
        return;
    }

    // Element-wise subtraction
    output[idx] = input_a[idx] - input_b[idx];
}
