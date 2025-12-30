// Parallel reduction sum compute shader
// Implements ReduceSum generator with workgroup-level reduction
//
// Corresponds to: reduce_sum@c[0..N]
// Operation: output[0] = sum(input[0..N])
//
// This is Pass 1: Reduce each workgroup to a single value
// Pass 2 (if needed): Final reduction of workgroup results

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Workgroup shared memory for reduction tree
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = arrayLength(&input);

    // Load data into shared memory
    if (gid < n) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    // Reduce 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    var stride = 128u;
    while (stride >= 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes workgroup result to global memory
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
