// Swap generator compute shader
// Swaps data between two buffers
//
// Corresponds to: swap@c[a<->b]
// Operation: temp = a[i]; a[i] = b[i]; b[i] = temp
//
// Note: This implementation writes to separate output buffers
// since in-place swapping requires careful synchronization

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_a: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output_b: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input_a)) {
        return;
    }

    // Swap: write a's values to b's output, and b's values to a's output
    output_a[idx] = input_b[idx];
    output_b[idx] = input_a[idx];
}
