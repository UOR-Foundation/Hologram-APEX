// Mark generator compute shader
// Initializes memory to a specific value (pattern)
//
// Corresponds to: mark@c
// Operation: output[i] = value (fills with constant)

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

// Push constant for the value to fill
// In practice, this will be passed via uniform buffer
@group(0) @binding(1)
var<uniform> value: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&output)) {
        return;
    }

    // Fill with constant value
    output[idx] = value;
}
