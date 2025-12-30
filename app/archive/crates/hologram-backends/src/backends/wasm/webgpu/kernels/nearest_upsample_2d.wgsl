// Nearest Neighbor 2D Upsampling compute shader
// Increases spatial resolution by repeating pixels
//
// Memory Layout: NCHW (batch, channels, height, width)
// - Input:  [N, C, H, W]
// - Output: [N, C, H*scale, W*scale]
//
// Algorithm: Each output pixel copies from nearest input pixel
// - out_h_in = out_h / scale_factor
// - out_w_in = out_w / scale_factor
// - output[out_h, out_w] = input[out_h_in, out_w_in]

// Configuration passed as uniforms
struct UpsampleConfig {
    batch_size: u32,
    num_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    scale_factor: u32,
}

@group(0) @binding(0)
var<uniform> config: UpsampleConfig;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

// Helper: Calculate linear index for tensor [N, C, H, W]
fn tensor_index(n: u32, c: u32, h: u32, w: u32, height: u32, width: u32, channels: u32) -> u32 {
    return n * (channels * height * width) +
           c * (height * width) +
           h * width +
           w;
}

// Main compute shader
// Workgroup size: 256 threads in 1D (efficient for memory coalescing)
// Each thread processes multiple output pixels
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch for large workgroup counts
    // thread_id = y * 65535 + x
    let thread_id = global_id.y * 65535u + global_id.x;
    let total_outputs = config.batch_size * config.num_channels * config.out_height * config.out_width;

    // Each thread processes one output element
    if (thread_id >= total_outputs) {
        return;
    }

    // Decompose linear index into (batch, channel, out_h, out_w)
    let batch = thread_id / (config.num_channels * config.out_height * config.out_width);
    let remaining1 = thread_id % (config.num_channels * config.out_height * config.out_width);

    let channel = remaining1 / (config.out_height * config.out_width);
    let remaining2 = remaining1 % (config.out_height * config.out_width);

    let out_h = remaining2 / config.out_width;
    let out_w = remaining2 % config.out_width;

    // Calculate corresponding input position (nearest neighbor)
    let in_h = out_h / config.scale_factor;
    let in_w = out_w / config.scale_factor;

    // Bounds check (should always pass for valid upsampling)
    if (in_h >= config.in_height || in_w >= config.in_width) {
        return;
    }

    // Read from input and write to output
    let in_idx = tensor_index(batch, channel, in_h, in_w, config.in_height, config.in_width, config.num_channels);
    let out_idx = tensor_index(batch, channel, out_h, out_w, config.out_height, config.out_width, config.num_channels);

    output[out_idx] = input[in_idx];
}
