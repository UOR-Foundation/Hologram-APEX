// 2D Convolution compute shader
// Implements tiled convolution for efficient GPU execution
//
// Memory Layout: NCHW (batch, channels, height, width)
// - Input:   [N, C_in, H, W]
// - Weights: [C_out, C_in, K_h, K_w]
// - Bias:    [C_out]
// - Output:  [N, C_out, H_out, W_out]
//
// Algorithm: Tiled convolution with shared memory for input caching
// - Each workgroup computes a tile of output
// - Input data is cached in shared memory to reduce global memory reads
// - Optimized for 8x8 output tiles per workgroup

// Configuration passed as uniforms (full ONNX Conv spec)
struct ConvConfig {
    batch_size: u32,
    in_channels: u32,
    in_height: u32,
    in_width: u32,
    out_channels: u32,
    out_height: u32,
    out_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_top: u32,
    pad_left: u32,
    pad_bottom: u32,
    pad_right: u32,
    dilation_h: u32,
    dilation_w: u32,
    group: u32,
    has_bias: u32,  // 0 = no bias, 1 = has bias
}

@group(0) @binding(0)
var<uniform> config: ConvConfig;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> weights: array<f32>;

@group(0) @binding(3)
var<storage, read> bias: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output: array<f32>;

// Shared memory for input tile (8x8 output tile + kernel halo)
// For 3x3 kernel: need (8+2)x(8+2) = 100 elements per channel
// Max 16 channels in shared memory: 1600 f32 = 6.4KB (well under 16KB limit)
var<workgroup> shared_input: array<f32, 1600>;

// Helper: Calculate linear index for input tensor [N, C, H, W]
fn input_index(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * (config.in_channels * config.in_height * config.in_width) +
           c * (config.in_height * config.in_width) +
           h * config.in_width +
           w;
}

// Helper: Calculate linear index for weight tensor [C_out, C_in/group, K_h, K_w]
// For grouped convolutions, weight shape is [C_out, C_in/group, K_h, K_w]
fn weight_index(c_out: u32, c_in: u32, kh: u32, kw: u32) -> u32 {
    let channels_per_group = config.in_channels / config.group;
    return c_out * (channels_per_group * config.kernel_h * config.kernel_w) +
           c_in * (config.kernel_h * config.kernel_w) +
           kh * config.kernel_w +
           kw;
}

// Helper: Calculate linear index for output tensor [N, C, H, W]
fn output_index(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * (config.out_channels * config.out_height * config.out_width) +
           c * (config.out_height * config.out_width) +
           h * config.out_width +
           w;
}

// Main compute shader
// Workgroup size: 256 threads in 1D (efficient for memory access)
// Each thread computes ONE output element at (batch, out_c, out_h, out_w)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch for large workgroup counts
    // thread_id = y * 65535 + x
    let thread_id = global_id.y * 65535u + global_id.x;
    let total_outputs = config.batch_size * config.out_channels * config.out_height * config.out_width;

    // Bounds check
    if (thread_id >= total_outputs) {
        return;
    }

    // Decompose linear index into (batch, out_c, out_h, out_w)
    let batch = thread_id / (config.out_channels * config.out_height * config.out_width);
    let remaining1 = thread_id % (config.out_channels * config.out_height * config.out_width);

    let out_c = remaining1 / (config.out_height * config.out_width);
    let remaining2 = remaining1 % (config.out_height * config.out_width);

    let out_h = remaining2 / config.out_width;
    let out_w = remaining2 % config.out_width;

    var sum: f32 = 0.0;

    // For grouped convolutions, determine which group this output channel belongs to
    let channels_per_group = config.in_channels / config.group;
    let out_channels_per_group = config.out_channels / config.group;
    let group_id = out_c / out_channels_per_group;
    let in_c_start = group_id * channels_per_group;
    let in_c_end = in_c_start + channels_per_group;

    // Convolve over input channels (within group) and kernel spatial dimensions
    for (var in_c: u32 = in_c_start; in_c < in_c_end; in_c = in_c + 1u) {
        for (var kh: u32 = 0u; kh < config.kernel_h; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < config.kernel_w; kw = kw + 1u) {
                // Calculate input position with separate stride_h/stride_w, per-side padding, and separate dilation_h/dilation_w
                let in_h_base = out_h * config.stride_h;
                let in_w_base = out_w * config.stride_w;

                let in_h_offset = kh * config.dilation_h;
                let in_w_offset = kw * config.dilation_w;

                // Apply per-side padding
                let in_h_signed = i32(in_h_base) + i32(in_h_offset) - i32(config.pad_top);
                let in_w_signed = i32(in_w_base) + i32(in_w_offset) - i32(config.pad_left);

                // Skip if padding (out of bounds)
                if (in_h_signed < 0 || in_h_signed >= i32(config.in_height) ||
                    in_w_signed < 0 || in_w_signed >= i32(config.in_width)) {
                    continue;
                }

                let in_h = u32(in_h_signed);
                let in_w = u32(in_w_signed);

                // Fetch input and weight (using local in_c for weight indexing within group)
                let input_val = input[input_index(batch, in_c, in_h, in_w)];
                let weight_val = weights[weight_index(out_c, in_c - in_c_start, kh, kw)];

                // Accumulate
                sum = sum + input_val * weight_val;
            }
        }
    }

    // Add bias if present
    if (config.has_bias != 0u) {
        sum = sum + bias[out_c];
    }

    // Write output
    output[output_index(batch, out_c, out_h, out_w)] = sum;
}
