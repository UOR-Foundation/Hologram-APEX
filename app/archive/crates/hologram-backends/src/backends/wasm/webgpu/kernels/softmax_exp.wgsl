// Softmax exponential compute shader (Pass 2 of softmax)
// Computes exp(input[i] - max) for numerically stable softmax
//
// This is an intermediate step in the softmax computation:
// 1. Find max (ReduceMax)
// 2. Compute exp(x - max) (this shader)
// 3. Sum exponentials (ReduceSum)
// 4. Divide by sum (softmax.wgsl)

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> max_value: array<f32>; // Single element: the max

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input)) {
        return;
    }

    let max_val = max_value[0];

    // Compute exp(x - max) for numerical stability
    output[idx] = exp(input[idx] - max_val);
}
