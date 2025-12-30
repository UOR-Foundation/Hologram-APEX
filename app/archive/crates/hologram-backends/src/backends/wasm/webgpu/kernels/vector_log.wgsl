// Vector natural logarithm compute shader
// Implements MergeRange(Log) generator for element-wise natural log
//
// Corresponds to: merge@c[0..N](Log)
// Operation: output[i] = log(input[i])
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

    // Element-wise natural logarithm
    // Note: Returns NaN for negative inputs, -inf for zero
    output[idx] = log(input[idx]);
}
