// Softmax activation compute shader (two-pass algorithm)
// Implements numerically stable softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
//
// This shader implements Pass 2 and 3: exponential and normalization
// Pass 1 (max reduction) uses reduce_max.wgsl
//
// Note: For full softmax implementation, you need:
// 1. ReduceMax to find max value
// 2. This shader (pass 2) to compute exp(x - max)
// 3. ReduceSum to compute sum of exponentials
// 4. This shader (pass 3) to normalize by dividing by sum

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> max_value: array<f32>; // Single element: the max

@group(0) @binding(2)
var<storage, read> sum_value: array<f32>; // Single element: the sum

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input)) {
        return;
    }

    let max_val = max_value[0];
    let sum_val = sum_value[0];

    // Numerically stable softmax:
    // softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    let exp_val = exp(input[idx] - max_val);
    output[idx] = exp_val / sum_val;
}
