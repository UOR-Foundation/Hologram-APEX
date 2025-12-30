// Broadcast subtraction compute shader
// Implements element-wise subtraction with numpy-style broadcasting
//
// Operation: output[i] = input_a[broadcasted_idx_a] - input_b[broadcasted_idx_b]

struct BroadcastParams {
    // Output shape (up to 8 dimensions)
    out_shape_0: u32,
    out_shape_1: u32,
    out_shape_2: u32,
    out_shape_3: u32,
    out_shape_4: u32,
    out_shape_5: u32,
    out_shape_6: u32,
    out_shape_7: u32,

    // Input A shape (up to 8 dimensions)
    a_shape_0: u32,
    a_shape_1: u32,
    a_shape_2: u32,
    a_shape_3: u32,
    a_shape_4: u32,
    a_shape_5: u32,
    a_shape_6: u32,
    a_shape_7: u32,

    // Input B shape (up to 8 dimensions)
    b_shape_0: u32,
    b_shape_1: u32,
    b_shape_2: u32,
    b_shape_3: u32,
    b_shape_4: u32,
    b_shape_5: u32,
    b_shape_6: u32,
    b_shape_7: u32,

    // Number of dimensions for each tensor
    out_ndim: u32,
    a_ndim: u32,
    b_ndim: u32,
    total_elements: u32,
}

@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: BroadcastParams;

// Compute linear index from multidimensional coordinates
fn compute_linear_index(coords: array<u32, 8>, shape: array<u32, 8>, ndim: u32) -> u32 {
    var idx: u32 = 0u;
    var stride: u32 = 1u;

    for (var i: i32 = i32(ndim) - 1; i >= 0; i = i - 1) {
        let dim_idx = u32(i);
        idx += coords[dim_idx] * stride;
        stride *= shape[dim_idx];
    }

    return idx;
}

// Convert linear output index to multidimensional coordinates
fn linear_to_coords(linear_idx: u32, shape: array<u32, 8>, ndim: u32) -> array<u32, 8> {
    var coords: array<u32, 8>;
    var remaining = linear_idx;

    for (var i: i32 = i32(ndim) - 1; i >= 0; i = i - 1) {
        let dim_idx = u32(i);
        coords[dim_idx] = remaining % shape[dim_idx];
        remaining = remaining / shape[dim_idx];
    }

    return coords;
}

// Map output coordinates to input coordinates (handles broadcasting)
fn broadcast_coords(out_coords: array<u32, 8>, out_ndim: u32, in_ndim: u32) -> array<u32, 8> {
    var in_coords: array<u32, 8>;
    let dim_offset = out_ndim - in_ndim;

    for (var i: u32 = 0u; i < in_ndim; i = i + 1u) {
        in_coords[i] = out_coords[dim_offset + i];
    }

    return in_coords;
}

// Clamp coordinates for broadcasting (size-1 dimensions broadcast)
fn clamp_broadcast_coords(coords: array<u32, 8>, shape: array<u32, 8>, ndim: u32) -> array<u32, 8> {
    var result: array<u32, 8>;

    for (var i: u32 = 0u; i < ndim; i = i + 1u) {
        if (shape[i] == 1u) {
            result[i] = 0u;
        } else {
            result[i] = coords[i];
        }
    }

    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;

    if (idx >= params.total_elements) {
        return;
    }

    var out_shape: array<u32, 8>;
    out_shape[0] = params.out_shape_0;
    out_shape[1] = params.out_shape_1;
    out_shape[2] = params.out_shape_2;
    out_shape[3] = params.out_shape_3;
    out_shape[4] = params.out_shape_4;
    out_shape[5] = params.out_shape_5;
    out_shape[6] = params.out_shape_6;
    out_shape[7] = params.out_shape_7;

    var a_shape: array<u32, 8>;
    a_shape[0] = params.a_shape_0;
    a_shape[1] = params.a_shape_1;
    a_shape[2] = params.a_shape_2;
    a_shape[3] = params.a_shape_3;
    a_shape[4] = params.a_shape_4;
    a_shape[5] = params.a_shape_5;
    a_shape[6] = params.a_shape_6;
    a_shape[7] = params.a_shape_7;

    var b_shape: array<u32, 8>;
    b_shape[0] = params.b_shape_0;
    b_shape[1] = params.b_shape_1;
    b_shape[2] = params.b_shape_2;
    b_shape[3] = params.b_shape_3;
    b_shape[4] = params.b_shape_4;
    b_shape[5] = params.b_shape_5;
    b_shape[6] = params.b_shape_6;
    b_shape[7] = params.b_shape_7;

    let out_coords = linear_to_coords(idx, out_shape, params.out_ndim);

    let a_coords_unaligned = broadcast_coords(out_coords, params.out_ndim, params.a_ndim);
    let a_coords = clamp_broadcast_coords(a_coords_unaligned, a_shape, params.a_ndim);
    let a_idx = compute_linear_index(a_coords, a_shape, params.a_ndim);

    let b_coords_unaligned = broadcast_coords(out_coords, params.out_ndim, params.b_ndim);
    let b_coords = clamp_broadcast_coords(b_coords_unaligned, b_shape, params.b_ndim);
    let b_idx = compute_linear_index(b_coords, b_shape, params.b_ndim);

    // Subtraction operation
    output[idx] = input_a[a_idx] - input_b[b_idx];
}
