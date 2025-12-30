// Vector square root compute shader
// Implements MergeRange(Sqrt) generator for element-wise square root
//
// Corresponds to: merge@c[0..N](Sqrt)
// Operation: output[i] = sqrt(input[i])
// Note: Unary operation - only uses input_a

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if (idx >= arrayLength(&input)) {
        return;
    }

    // Element-wise square root
    // Note: Returns NaN for negative inputs
    output[idx] = sqrt(input[idx]);
}
