// Group Normalization compute shader
// Normalizes features within groups of channels
//
// Memory Layout: NCHW (batch, channels, height, width)
// - Input:  [N, C, H, W]
// - Gamma:  [C] (affine scale parameters)
// - Beta:   [C] (affine shift parameters)
// - Output: [N, C, H, W]
//
// Algorithm:
// For each group g:
//   1. Compute mean and variance across spatial dims and channels in group
//   2. Normalize: y = (x - mean) / sqrt(variance + eps)
//   3. Affine transform: y = gamma * y + beta

// Configuration passed as uniforms
struct GroupNormConfig {
    batch_size: u32,
    num_channels: u32,
    height: u32,
    width: u32,
    num_groups: u32,
    channels_per_group: u32,
    spatial_size: u32,       // height * width
    group_size: u32,          // channels_per_group * height * width
    eps: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0)
var<uniform> config: GroupNormConfig;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> gamma: array<f32>;

@group(0) @binding(3)
var<storage, read> beta: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output: array<f32>;

// Shared memory for reduction (one per thread in workgroup)
var<workgroup> shared_data: array<f32, 256>;

// Helper: Calculate linear index for tensor [N, C, H, W]
fn tensor_index(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * (config.num_channels * config.spatial_size) +
           c * config.spatial_size +
           h * config.width +
           w;
}

// Main compute shader
// Each workgroup processes one group in one batch
// Threads cooperate to compute mean, variance, then normalize
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch = workgroup_id.x / config.num_groups;
    let group = workgroup_id.x % config.num_groups;

    // Bounds check
    if (batch >= config.batch_size) {
        return;
    }

    let thread_id = local_id.x;
    let group_start_channel = group * config.channels_per_group;

    // Phase 1: Compute sum for mean (parallel reduction)
    var local_sum: f32 = 0.0;

    // Each thread sums multiple elements
    for (var i = thread_id; i < config.group_size; i = i + 256u) {
        let spatial_idx = i % config.spatial_size;
        let channel_in_group = i / config.spatial_size;
        let channel = group_start_channel + channel_in_group;

        let h = spatial_idx / config.width;
        let w = spatial_idx % config.width;

        let idx = tensor_index(batch, channel, h, w);
        local_sum = local_sum + input[idx];
    }

    // Store in shared memory
    shared_data[thread_id] = local_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride && thread_id + stride < 256u) {
            shared_data[thread_id] = shared_data[thread_id] + shared_data[thread_id + stride];
        }
        workgroupBarrier();
    }

    let mean = shared_data[0] / f32(config.group_size);
    workgroupBarrier();

    // Phase 2: Compute sum of squared differences for variance
    var local_var_sum: f32 = 0.0;

    for (var i = thread_id; i < config.group_size; i = i + 256u) {
        let spatial_idx = i % config.spatial_size;
        let channel_in_group = i / config.spatial_size;
        let channel = group_start_channel + channel_in_group;

        let h = spatial_idx / config.width;
        let w = spatial_idx % config.width;

        let idx = tensor_index(batch, channel, h, w);
        let diff = input[idx] - mean;
        local_var_sum = local_var_sum + diff * diff;
    }

    // Store in shared memory
    shared_data[thread_id] = local_var_sum;
    workgroupBarrier();

    // Parallel reduction for variance
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (thread_id < stride && thread_id + stride < 256u) {
            shared_data[thread_id] = shared_data[thread_id] + shared_data[thread_id + stride];
        }
        workgroupBarrier();
    }

    let variance = shared_data[0] / f32(config.group_size);
    let std_dev = sqrt(variance + config.eps);
    workgroupBarrier();

    // Phase 3: Normalize and apply affine transformation
    for (var i = thread_id; i < config.group_size; i = i + 256u) {
        let spatial_idx = i % config.spatial_size;
        let channel_in_group = i / config.spatial_size;
        let channel = group_start_channel + channel_in_group;

        let h = spatial_idx / config.width;
        let w = spatial_idx % config.width;

        let idx = tensor_index(batch, channel, h, w);

        // Normalize
        let normalized = (input[idx] - mean) / std_dev;

        // Apply affine transformation
        output[idx] = gamma[channel] * normalized + beta[channel];
    }
}
