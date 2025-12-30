// Vector sigmoid activation compute shader
// Implements MergeRange(Sigmoid) generator for element-wise sigmoid
//
// Corresponds to: merge@c[0..N](Sigmoid)
// Operation: output[i] = 1.0 / (1.0 + exp(-input[i]))
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

    // Sigmoid: 1 / (1 + exp(-x))
    let x = input[idx];
    output[idx] = 1.0 / (1.0 + exp(-x));
}
