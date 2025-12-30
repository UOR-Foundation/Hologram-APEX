// General Matrix Multiplication (GEMM) compute shader
// Implements GPU-native matrix multiplication: C = A * B
//
// Memory Layout:
// - Matrix A: [M, K] (row-major)
// - Matrix B: [K, N] (row-major)
// - Matrix C: [M, N] (row-major)
//
// Algorithm: Tile-based multiplication with shared memory caching
// - Each workgroup computes a TILE_SIZE x TILE_SIZE tile of output matrix C
// - Input tiles from A and B are cached in shared memory to reduce global reads
// - Optimized for 16x16 tiles per workgroup (256 threads)
//
// Performance characteristics:
// - Global memory reads reduced by factor of TILE_SIZE through caching
// - Coalesced memory access patterns for optimal bandwidth
// - Shared memory usage: 2 * TILE_SIZE^2 * 4 bytes = 2KB (well under 16KB limit)

// Configuration passed as uniforms
struct GemmConfig {
    m: u32,  // Rows in A and C
    k: u32,  // Cols in A, rows in B (inner dimension)
    n: u32,  // Cols in B and C
}

@group(0) @binding(0)
var<uniform> config: GemmConfig;

@group(0) @binding(1)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(2)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(3)
var<storage, read_write> matrix_c: array<f32>;

// Shared memory for tile caching
// TILE_SIZE = 16, so each tile is 16x16 = 256 f32 values
const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<f32, 256>;  // 16x16 tile from A
var<workgroup> tile_b: array<f32, 256>;  // 16x16 tile from B

// Helper: Get element from matrix A at (row, col)
fn get_a(row: u32, col: u32) -> f32 {
    if (row >= config.m || col >= config.k) {
        return 0.0;
    }
    return matrix_a[row * config.k + col];
}

// Helper: Get element from matrix B at (row, col)
fn get_b(row: u32, col: u32) -> f32 {
    if (row >= config.k || col >= config.n) {
        return 0.0;
    }
    return matrix_b[row * config.n + col];
}

// Helper: Set element in matrix C at (row, col)
fn set_c(row: u32, col: u32, value: f32) {
    if (row < config.m && col < config.n) {
        matrix_c[row * config.n + col] = value;
    }
}

// Main compute shader
// Workgroup size: 16x16 = 256 threads
// Each workgroup computes a 16x16 tile of output matrix C
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // Global position in output matrix C
    let row = global_id.y;
    let col = global_id.x;

    // Local position within the workgroup tile (0..15, 0..15)
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Accumulator for this thread's output element
    var sum: f32 = 0.0;

    // Number of tiles needed to cover dimension K
    let num_tiles = (config.k + TILE_SIZE - 1u) / TILE_SIZE;

    // Iterate over tiles along the K dimension
    for (var tile_idx: u32 = 0u; tile_idx < num_tiles; tile_idx = tile_idx + 1u) {
        // Load tile from A into shared memory
        // Each thread loads one element: A[row, tile_idx * TILE_SIZE + local_col]
        let a_row = row;
        let a_col = tile_idx * TILE_SIZE + local_col;
        let tile_a_idx = local_row * TILE_SIZE + local_col;
        tile_a[tile_a_idx] = get_a(a_row, a_col);

        // Load tile from B into shared memory
        // Each thread loads one element: B[tile_idx * TILE_SIZE + local_row, col]
        let b_row = tile_idx * TILE_SIZE + local_row;
        let b_col = col;
        let tile_b_idx = local_row * TILE_SIZE + local_col;
        tile_b[tile_b_idx] = get_b(b_row, b_col);

        // Synchronize to ensure all threads have loaded their data
        workgroupBarrier();

        // Compute partial dot product using cached tile data
        // This eliminates TILE_SIZE global memory reads per element
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = tile_a[local_row * TILE_SIZE + k];
            let b_val = tile_b[k * TILE_SIZE + local_col];
            sum = sum + a_val * b_val;
        }

        // Synchronize before loading next tile (prevent race conditions)
        workgroupBarrier();
    }

    // Write final result to output matrix C
    set_c(row, col, sum);
}
