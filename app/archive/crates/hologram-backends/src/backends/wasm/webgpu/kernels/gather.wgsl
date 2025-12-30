// Gather operation compute shader
// Implements element-wise gathering along a specified axis
//
// Operation: output[i] = input[indices[i]] along axis dimension
//
// Parameters are passed via uniform buffer containing shape/stride information

struct GatherParams {
    // Data tensor properties
    data_shape_0: u32, data_shape_1: u32, data_shape_2: u32, data_shape_3: u32,
    data_shape_4: u32, data_shape_5: u32, data_shape_6: u32, data_shape_7: u32,
    data_strides_0: u32, data_strides_1: u32, data_strides_2: u32, data_strides_3: u32,
    data_strides_4: u32, data_strides_5: u32, data_strides_6: u32, data_strides_7: u32,

    // Output tensor properties
    output_shape_0: u32, output_shape_1: u32, output_shape_2: u32, output_shape_3: u32,
    output_shape_4: u32, output_shape_5: u32, output_shape_6: u32, output_shape_7: u32,
    output_strides_0: u32, output_strides_1: u32, output_strides_2: u32, output_strides_3: u32,
    output_strides_4: u32, output_strides_5: u32, output_strides_6: u32, output_strides_7: u32,

    // Gather parameters
    axis: u32,
    data_ndim: u32,
    output_ndim: u32,
    total_output_elements: u32,

    // Indices properties
    num_indices: u32,
    axis_dim_size: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read> indices: array<i32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: GatherParams;

// Convert linear output index to coordinates
fn linear_to_coords(idx: u32, shape: array<u32, 8>, strides: array<u32, 8>, ndim: u32) -> array<u32, 8> {
    var coords: array<u32, 8>;
    var remaining = idx;

    for (var i = 0u; i < ndim; i = i + 1u) {
        coords[i] = remaining / strides[i];
        remaining = remaining % strides[i];
    }

    return coords;
}

// Convert coordinates to linear input index
fn coords_to_linear(coords: array<u32, 8>, strides: array<u32, 8>, ndim: u32) -> u32 {
    var idx = 0u;
    for (var i = 0u; i < ndim; i = i + 1u) {
        idx = idx + coords[i] * strides[i];
    }
    return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.y * 65535u + global_id.x;

    if (output_idx >= params.total_output_elements) {
        return;
    }

    // Build shape/stride arrays from params
    let output_shape = array<u32, 8>(
        params.output_shape_0, params.output_shape_1, params.output_shape_2, params.output_shape_3,
        params.output_shape_4, params.output_shape_5, params.output_shape_6, params.output_shape_7
    );
    let output_strides = array<u32, 8>(
        params.output_strides_0, params.output_strides_1, params.output_strides_2, params.output_strides_3,
        params.output_strides_4, params.output_strides_5, params.output_strides_6, params.output_strides_7
    );
    let data_strides = array<u32, 8>(
        params.data_strides_0, params.data_strides_1, params.data_strides_2, params.data_strides_3,
        params.data_strides_4, params.data_strides_5, params.data_strides_6, params.data_strides_7
    );

    // Convert output linear index to coordinates
    let out_coords = linear_to_coords(output_idx, output_shape, output_strides, params.output_ndim);

    // Map output coordinates to input coordinates
    // Output shape: data_shape[:axis] + indices_shape + data_shape[axis+1:]
    // So: out_coords[:axis] maps to data_coords[:axis]
    //     out_coords[axis:axis+indices_ndim] indexes into indices array
    //     out_coords[axis+indices_ndim:] maps to data_coords[axis+1:]

    var data_coords: array<u32, 8>;

    // Copy coordinates before axis
    for (var i = 0u; i < params.axis; i = i + 1u) {
        data_coords[i] = out_coords[i];
    }

    // Get index from indices array (output coord at axis position)
    let index_value = indices[out_coords[params.axis]];

    // Handle negative indices
    var actual_index: u32;
    if (index_value < 0) {
        actual_index = u32(i32(params.axis_dim_size) + index_value);
    } else {
        actual_index = u32(index_value);
    }

    // Bounds check
    if (actual_index >= params.axis_dim_size) {
        // Out of bounds - write zero (or could return/error)
        output[output_idx] = 0.0;
        return;
    }

    data_coords[params.axis] = actual_index;

    // Copy coordinates after axis
    for (var i = params.axis + 1u; i < params.data_ndim; i = i + 1u) {
        data_coords[i] = out_coords[i];
    }

    // Convert input coordinates to linear index
    let input_idx = coords_to_linear(data_coords, data_strides, params.data_ndim);

    // Perform gather
    output[output_idx] = input_data[input_idx];
}
