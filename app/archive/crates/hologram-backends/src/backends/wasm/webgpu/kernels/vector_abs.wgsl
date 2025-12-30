// Vector absolute value compute shader
// Implements MergeRange(Abs) generator for element-wise absolute value
//
// Corresponds to: merge@c[0..N](Abs)
// Operation: output[i] = abs(input[i])
// Note: Unary operation - only uses input_a

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Compute linear index from 2D dispatch grid
    // Supports dispatch sizes exceeding 65,535 by using Y dimension
    let idx = global_id.y * 65535u + global_id.x;

    // Bounds check
    if (idx >= arrayLength(&input)) {
        return;
    }

    // Element-wise absolute value
    output[idx] = abs(input[idx]);
}
