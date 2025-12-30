//! NOTE: All operations in this file are temporarily stubbed during Phase 0 migration.
//! They will be implemented with ISA Programs in Phase 1.

//! Circuit-Based Mathematical Operations (Zero-Overhead)
//!
//! This implementation uses direct Circuit generator construction for all math operations.
//! Operations bypass parsing/canonicalization for maximum performance.
//!
//! ## Key Features
//!
//! 1. **Zero-Overhead Execution**: Direct GeneratorCall construction (no parsing)
//! 2. **Class-based execution**: Operates on 96-class system
//! 3. **Built-in instrumentation**: Operation timing metrics
//! 4. **Performance**: ~100-500ns latency vs ~5-6µs with string parsing
//!
//! ## Architecture
//!
//! ```text
//! Buffer Operation → Direct GeneratorCall Construction
//!   → execute_generators() → ClassMemory (bypasses parsing/canonicalization)
//! ```
//!
//! ## SIMD Fast Path (Phase 4.1)
//!
//! For CPU backend + n ≤ 262,144 elements + f32 type:
//! - Uses inline SIMD kernels (AVX-512/AVX2/SSE4.1)
//! - 881-4,367x faster than ISA execution
//! - Bypasses memory manager locks entirely
//! - Falls back to ISA for large workloads or non-CPU backends

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;
use crate::instrumentation::{ExecutionMetrics, Instant};
use crate::sync::write_lock;

// WebGPU-specific imports
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
use crate::sync::read_lock;

// WebGPU backend for GPU acceleration in WASM
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
use hologram_backends::WebGpuBackend;

// ============================================================================
// Program Caches (Thread-Safe, Lock-Free After First Access)
// ============================================================================

use hologram_backends::program_cache::{ProgramCache, ProgramKey};

static BROADCAST_ADD_CACHE: ProgramCache = ProgramCache::new();

// ============================================================================
// Helper Functions
// ============================================================================

/// SIMD threshold: use inline kernels for n ≤ 262,144 elements (1MB for f32)
const SIMD_THRESHOLD: usize = 262_144;

/// Validate that buffers have correct size
fn validate_buffers<T: bytemuck::Pod>(buffers: &[&Buffer<T>], n: usize, op_name: &str) -> Result<()> {
    for (i, buf) in buffers.iter().enumerate() {
        if buf.len() < n {
            return Err(Error::InvalidOperation(format!(
                "{} buffer {} too small: len={}, need={}",
                op_name,
                i,
                buf.len(),
                n
            )));
        }
    }
    Ok(())
}

/// Try to execute binary operation using inline SIMD kernel
///
/// Returns true if SIMD was used, false if fallback to ISA is needed.
///
/// Conditions for SIMD fast path:
/// - Backend is CpuBackend
/// - n ≤ SIMD_THRESHOLD (262,144 elements)
/// - Type is f32
#[inline(always)]
fn try_inline_simd_binary<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
    kernel: unsafe fn(*const f32, *const f32, *mut f32, usize),
) -> Result<bool> {
    // Check if type is f32
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return Ok(false);
    }

    // Check threshold
    if n > SIMD_THRESHOLD {
        return Ok(false);
    }

    // Zero-overhead fast path: use cached pointers (no locks, no lookups!)
    let ptr_a = a.cached_ptr();
    let ptr_b = b.cached_ptr();
    let ptr_c = c.cached_ptr();

    // Check if pointers are valid (non-null means CPU backend with cached pointers)
    if ptr_a.is_null() || ptr_b.is_null() || ptr_c.is_null() {
        return Ok(false); // Not CPU backend or boundary buffers, use ISA
    }

    // Execute inline SIMD kernel (zero overhead - just pointer arithmetic + SIMD!)
    unsafe { kernel(ptr_a as *const f32, ptr_b as *const f32, ptr_c as *mut f32, n) };

    Ok(true) // SIMD was used
}

/// Try to execute unary operation using inline SIMD kernel
///
/// Returns true if SIMD was used, false if fallback to ISA is needed.
pub(crate) fn try_inline_simd_unary<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
    kernel: unsafe fn(*const f32, *mut f32, usize),
) -> Result<bool> {
    // Check if type is f32
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return Ok(false);
    }

    // Check threshold
    if n > SIMD_THRESHOLD {
        return Ok(false);
    }

    // Zero-overhead fast path: use cached pointers (no locks, no lookups!)
    let ptr_a = a.cached_ptr();
    let ptr_c = c.cached_ptr();

    // Check if pointers are valid (non-null means CPU backend with cached pointers)
    if ptr_a.is_null() || ptr_c.is_null() {
        return Ok(false); // Not CPU backend or boundary buffers, use ISA
    }

    // Execute inline SIMD kernel (zero overhead - just pointer arithmetic + SIMD!)
    unsafe { kernel(ptr_a as *const f32, ptr_c as *mut f32, n) };

    Ok(true) // SIMD was used
}

/// Try to execute binary operation using WebGPU compute shader
///
/// Returns true if WebGPU was used, false if fallback to other paths is needed.
///
/// Conditions for WebGPU fast path:
/// - Backend is WebGpuBackend
/// - Type is f32 (only supported type currently)
/// - Operation has a corresponding WGSL shader
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn try_webgpu_fast_path_binary<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
    shader_name: &str,
) -> Result<bool> {
    // Check if type is f32 (only supported type currently)
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return Ok(false);
    }

    // Check if backend is WebGpuBackend
    let backend_guard = read_lock(&exec.backend);
    let backend_any = backend_guard.as_any();
    let webgpu_backend: &WebGpuBackend = match backend_any.downcast_ref::<WebGpuBackend>() {
        Some(backend) => backend,
        None => return Ok(false), // Not WebGPU backend
    };

    // Get buffer handles (handles both class-based and class-free)
    let handle_a = exec.handle_from_buffer(a)?;
    let handle_b = exec.handle_from_buffer(b)?;
    let handle_c = exec.handle_from_buffer(c)?;

    let gpu_buffer_a = webgpu_backend.get_gpu_buffer(handle_a)?;
    let gpu_buffer_b = webgpu_backend.get_gpu_buffer(handle_b)?;
    let gpu_buffer_c = webgpu_backend.get_gpu_buffer(handle_c)?;

    // Get shader source
    let shader_source = match shader_name {
        "vector_add" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_add.wgsl"),
        "vector_mul" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_mul.wgsl"),
        "vector_sub" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_sub.wgsl"),
        "vector_div" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_div.wgsl"),
        "vector_min" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_min.wgsl"),
        "vector_max" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_max.wgsl"),
        _ => return Ok(false),
    };

    // Get or create pipeline
    let pipeline = webgpu_backend
        .pipeline_cache()
        .get_or_create(shader_name, shader_source, "main")
        .map_err(|e| Error::InvalidOperation(format!("WebGPU pipeline creation failed: {}", e)))?;

    // Create bind group
    let bind_group = webgpu_backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{} Bind Group", shader_name)),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_buffer_c.as_entire_binding(),
            },
        ],
    });

    // Create command encoder
    let mut encoder = webgpu_backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", shader_name)),
        });

    // Dispatch compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{} Pass", shader_name)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Workgroup size matches WGSL @workgroup_size(256)
        let workgroup_size = 256;
        let num_workgroups = (n as u32 + workgroup_size - 1) / workgroup_size;

        // Split dispatch across dimensions to respect 65,535 workgroup limit per dimension
        let (dispatch_x, dispatch_y, dispatch_z) = if num_workgroups <= 65535 {
            (num_workgroups, 1, 1)
        } else {
            // Split across Y dimension
            let dispatch_y = (num_workgroups + 65535 - 1) / 65535;
            let dispatch_x = 65535;
            (dispatch_x, dispatch_y, 1)
        };

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    // Submit commands
    webgpu_backend.queue().submit([encoder.finish()]);

    // Release backend lock
    drop(backend_guard);

    Ok(true) // WebGPU was used
}

/// Try to execute unary operation using WebGPU compute shader
///
/// Returns true if WebGPU was used, false if fallback is needed.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn try_webgpu_fast_path_unary<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
    shader_name: &str,
) -> Result<bool> {
    // Check if type is f32
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return Ok(false);
    }

    // Check if backend is WebGpuBackend
    let backend_guard = read_lock(&exec.backend);
    let backend_any = backend_guard.as_any();
    let webgpu_backend: &WebGpuBackend = match backend_any.downcast_ref::<WebGpuBackend>() {
        Some(backend) => backend,
        None => return Ok(false),
    };

    // Get buffer handles (handles both class-based and class-free)
    let handle_a = exec.handle_from_buffer(a)?;
    let handle_c = exec.handle_from_buffer(c)?;

    let gpu_buffer_a = webgpu_backend.get_gpu_buffer(handle_a)?;
    let gpu_buffer_c = webgpu_backend.get_gpu_buffer(handle_c)?;

    // Get shader source
    let shader_source = match shader_name {
        "vector_abs" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_abs.wgsl"),
        "vector_exp" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_exp.wgsl"),
        "vector_log" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_log.wgsl"),
        "vector_sqrt" => include_str!("../../../hologram-backends/src/backends/wasm/webgpu/kernels/vector_sqrt.wgsl"),
        _ => return Ok(false),
    };

    // Get or create pipeline
    let pipeline = webgpu_backend
        .pipeline_cache()
        .get_or_create(shader_name, shader_source, "main")
        .map_err(|e| Error::InvalidOperation(format!("WebGPU pipeline creation failed: {}", e)))?;

    // Create bind group (2 buffers for unary ops)
    let bind_group = webgpu_backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{} Bind Group", shader_name)),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_buffer_c.as_entire_binding(),
            },
        ],
    });

    // Create encoder and dispatch
    let mut encoder = webgpu_backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", shader_name)),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{} Pass", shader_name)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 256;
        let num_workgroups = (n as u32 + workgroup_size - 1) / workgroup_size;

        // Split dispatch across dimensions to respect 65,535 workgroup limit per dimension
        let (dispatch_x, dispatch_y, dispatch_z) = if num_workgroups <= 65535 {
            (num_workgroups, 1, 1)
        } else {
            // Split across Y dimension
            let dispatch_y = (num_workgroups + 65535 - 1) / 65535;
            let dispatch_x = 65535;
            (dispatch_x, dispatch_y, 1)
        };

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    webgpu_backend.queue().submit([encoder.finish()]);
    drop(backend_guard);

    Ok(true)
}

// ============================================================================
// Binary Vector Operations
// ============================================================================

/// Vector addition: c[i] = a[i] + b[i]
///
/// ## SIMD Fast Path (Phase 4.1)
///
/// For CPU backend, f32 type, n ≤ 262,144 elements:
/// - Uses inline AVX-512/AVX2/SSE4.1 kernels
/// - 881-4,367x faster than ISA execution
/// - Bypasses memory manager locks entirely
///
/// ## ISA Fallback
///
/// For large workloads or non-CPU backends:
/// - Uses precompiled ISA Program
/// - Executes with Rayon parallelization
///
/// # Example
///
/// ```text
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(3072)?;
/// let b = exec.allocate::<f32>(3072)?;
/// let mut c = exec.allocate::<f32>(3072)?;
///
/// ops::math::vector_add(&mut exec, &a, &b, &mut c, 3072)?;
/// ```
#[tracing::instrument(skip(exec, a, b, c), fields(
    n = n,
    elem_size = std::mem::size_of::<T>(),
    total_bytes = n * std::mem::size_of::<T>(),
    type_name = std::any::type_name::<T>()
))]
pub fn vector_add<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    // This is the critical path for small workloads (n ≤ 262K elements)
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_add)? {
        return Ok(()); // Zero overhead: no timing, no validation, no metrics
    }

    // Slow path: WebGPU or ISA execution with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "vector_add")?;

    // Try WebGPU fast path (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_add")? {
        let metrics = ExecutionMetrics::new("vector_add", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "vector_add_webgpu_complete"
        );
        return Ok(());
    }

    // Fall back to ISA execution for large workloads or non-CPU backends
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled ADD program (build-time generated)
    let program = &crate::precompiled_programs::ADD;

    // Calculate launch config: 1 lane per element
    // For n=1024: total_elements=1024, block_size=256 → 4 blocks × 256 lanes = 1024 lanes
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // Set up execution parameters with buffer handles in registers
    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    // Execute precompiled program with parameters
    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    // Instrumentation
    let metrics = ExecutionMetrics::new("vector_add", n, start);
    metrics.log();

    tracing::debug!(
        duration_us = metrics.total_duration_us,
        ops_per_second = metrics.ops_per_second(),
        memory_bandwidth_gbps = metrics.memory_bandwidth_gbps(),
        simd = false,
        "vector_add_isa_complete"
    );

    Ok(())
}

/// Vector subtraction: c[i] = a[i] - b[i]
///
/// ## SIMD Fast Path (Phase 4.1)
///
/// For CPU backend, f32 type, n ≤ 262,144: inline AVX-512/AVX2/SSE4.1 kernels
#[tracing::instrument(skip(exec, a, b, c), fields(
    n = n,
    elem_size = std::mem::size_of::<T>(),
    total_bytes = n * std::mem::size_of::<T>(),
    type_name = std::any::type_name::<T>()
))]
pub fn vector_sub<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_sub)? {
        return Ok(());
    }

    // Slow path with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "vector_sub")?;

    // Try WebGPU fast path (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_sub")? {
        let metrics = ExecutionMetrics::new("vector_sub", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "vector_sub_webgpu_complete"
        );
        return Ok(());
    }

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled SUB program
    let program = &crate::precompiled_programs::SUB;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // Set up execution parameters with buffer handles in registers
    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    // Execute precompiled program
    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("vector_sub", n, start);
    metrics.log();

    Ok(())
}

/// Vector multiplication: c[i] = a[i] * b[i]
///
/// ## SIMD Fast Path (Phase 4.1)
///
/// For CPU backend, f32 type, n ≤ 262,144: inline AVX-512/AVX2/SSE4.1 kernels
#[tracing::instrument(skip(exec, a, b, c), fields(
    n = n,
    elem_size = std::mem::size_of::<T>(),
    total_bytes = n * std::mem::size_of::<T>(),
    type_name = std::any::type_name::<T>()
))]
pub fn vector_mul<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_mul)? {
        return Ok(());
    }

    // Slow path with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "vector_mul")?;

    // Try WebGPU fast path (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_mul")? {
        let metrics = ExecutionMetrics::new("vector_mul", n, start);
        tracing::info!(
            "vector_mul: WebGPU fast path used, n={}, duration_us={}",
            n,
            metrics.total_duration_us
        );
        return Ok(());
    }

    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    tracing::info!("vector_mul: WebGPU fast path NOT used, falling back to ISA, n={}", n);

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled MUL program
    let program = &crate::precompiled_programs::MUL;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("vector_mul", n, start);
    metrics.log();

    Ok(())
}

/// Vector division: c[i] = a[i] / b[i]
///
/// ## SIMD Fast Path (Phase 4.1)
///
/// For CPU backend, f32 type, n ≤ 262,144: inline AVX-512/AVX2/SSE4.1 kernels
#[tracing::instrument(skip(exec, a, b, c), fields(
    n = n,
    elem_size = std::mem::size_of::<T>(),
    total_bytes = n * std::mem::size_of::<T>(),
    type_name = std::any::type_name::<T>()
))]
pub fn vector_div<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_div)? {
        return Ok(());
    }

    // Slow path with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "vector_div")?;

    // Try WebGPU fast path (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_div")? {
        let metrics = ExecutionMetrics::new("vector_div", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "vector_div_webgpu_complete"
        );
        return Ok(());
    }

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled DIV program
    let program = &crate::precompiled_programs::DIV;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("vector_div", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Unary Vector Operations
// ============================================================================

/// Element-wise minimum: c[i] = min(a[i], b[i])
///
/// Uses precompiled ISA Program for optimal performance.
#[tracing::instrument(skip(exec, a, b, c), fields(n = n))]
pub fn min<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_min)? {
        return Ok(()); // Zero overhead: no timing, no validation, no metrics
    }

    // Slow path: WebGPU or ISA execution with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "min")?;

    // Try WebGPU fast path first (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_min")? {
        let metrics = ExecutionMetrics::new("min", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "min_webgpu_complete"
        );
        return Ok(());
    }

    // Get buffer handles (handles both class-based and class-free)
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled MIN program
    let program = &crate::precompiled_programs::VECTOR_MIN;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("min", n, start);
    metrics.log();

    Ok(())
}

/// Element-wise maximum: c[i] = max(a[i], b[i])
///
/// Uses precompiled ISA Program for optimal performance.
#[tracing::instrument(skip(exec, a, b, c), fields(n = n))]
pub fn max<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_binary(exec, a, b, c, n, crate::kernel::inline::vector_max)? {
        return Ok(()); // Zero overhead: no timing, no validation, no metrics
    }

    // Slow path: WebGPU or ISA execution with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "max")?;

    // Try WebGPU fast path first (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_binary(exec, a, b, c, n, "vector_max")? {
        let metrics = ExecutionMetrics::new("max", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "max_webgpu_complete"
        );
        return Ok(());
    }

    // Get buffer handles (handles both class-based and class-free)
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled MAX program
    let program = &crate::precompiled_programs::VECTOR_MAX;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("max", n, start);
    metrics.log();

    Ok(())
}

/// Element-wise absolute value: c[i] = |a[i]|
///
/// Uses SIMD inline kernel for optimal performance (Phase 4.1).
/// Falls back to precompiled ISA Program if SIMD is unavailable.
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn abs<T: bytemuck::Pod + 'static>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if try_inline_simd_unary(exec, a, c, n, crate::kernel::inline::abs)? {
        return Ok(());
    }

    // Slow path with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, c], n, "abs")?;

    // Try WebGPU fast path (Phase 5.2)
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_fast_path_unary(exec, a, c, n, "vector_abs")? {
        let metrics = ExecutionMetrics::new("abs", n, start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "abs_webgpu_complete"
        );
        return Ok(());
    }

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled ABS program
    let program = &crate::precompiled_programs::VECTOR_ABS;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("abs", n, start);
    metrics.log();

    Ok(())
}

/// Element-wise negation: c[i] = -a[i]
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn neg<T: bytemuck::Pod>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "neg")?;

    // Get buffer handles (handles both class-based and class-free)
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Build ISA program: c[i] = -a[i]
    let ty = crate::isa_builder::type_from_rust_type::<T>();
    let program = crate::isa_builder::build_elementwise_unary_op(handle_a, handle_c, n, ty, |dst, src| {
        hologram_backends::Instruction::NEG { ty, dst, src }
    })?;

    // Execute ISA program via backend
    let config = hologram_backends::LaunchConfig::default();
    write_lock(&exec.backend)
        .execute_program(&program, &config)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("neg", n, start);
    metrics.log();

    Ok(())
}

/// Clip values to range: c[i] = clamp(a[i], min_val, max_val)
///
/// Clamps input values to the range [min_val, max_val]:
/// - Values below min_val are set to min_val
/// - Values above max_val are set to max_val
/// - Values within range are unchanged
///
/// # Example
///
/// ```text
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // Clip values to [-1.0, 1.0]
/// clip(&mut exec, &a, &mut c, -1.0f32, 1.0f32, 1024)?;
/// ```
pub fn clip<T: bytemuck::Pod + Copy + std::fmt::Debug>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    min_val: T,
    max_val: T,
    n: usize,
) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "clip")?;

    // Allocate temporary buffers for min/max constants and intermediate result
    let mut min_buf = exec.allocate::<T>(n)?;
    let mut max_buf = exec.allocate::<T>(n)?;
    let mut temp = exec.allocate::<T>(n)?;

    // Fill with constant values
    crate::ops::memory::fill(exec, &mut min_buf, min_val)?;
    crate::ops::memory::fill(exec, &mut max_buf, max_val)?;

    // Step 1: temp = max(a, min_val)
    max(exec, a, &min_buf, &mut temp, n)?;

    // Step 2: c = min(temp, max_val)
    min(exec, &temp, &max_buf, c, n)?;

    let metrics = ExecutionMetrics::new("clip", n, start);
    metrics.log();

    Ok(())
}

/// Scalar addition: c[i] = a[i] + scalar
///
/// Adds a constant scalar value to all elements of the input buffer.
///
/// # Example
///
/// ```text
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // Add 2.5 to all elements
/// scalar_add(&mut exec, &a, &mut c, 2.5f32, 1024)?;
/// ```
pub fn scalar_add<T: bytemuck::Pod + Copy + std::fmt::Debug>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    scalar: T,
    n: usize,
) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "scalar_add")?;

    // Create buffer filled with scalar value
    let mut scalar_buf = exec.allocate::<T>(n)?;
    crate::ops::memory::fill(exec, &mut scalar_buf, scalar)?;

    // Perform vector addition: c = a + scalar_buf
    vector_add(exec, a, &scalar_buf, c, n)?;

    let metrics = ExecutionMetrics::new("scalar_add", n, start);
    metrics.log();

    Ok(())
}

/// Scalar multiplication: c[i] = a[i] * scalar
///
/// Multiplies all elements of the input buffer by a constant scalar value.
///
/// # Example
///
/// ```text
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // Multiply all elements by 2.0
/// scalar_mul(&mut exec, &a, &mut c, 2.0f32, 1024)?;
/// ```
pub fn scalar_mul<T: bytemuck::Pod + Copy + std::fmt::Debug>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    scalar: T,
    n: usize,
) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "scalar_mul")?;

    // Create buffer filled with scalar value
    let mut scalar_buf = exec.allocate::<T>(n)?;
    crate::ops::memory::fill(exec, &mut scalar_buf, scalar)?;

    // Perform vector multiplication: c = a * scalar_buf
    vector_mul(exec, a, &scalar_buf, c, n)?;

    let metrics = ExecutionMetrics::new("scalar_mul", n, start);
    metrics.log();

    Ok(())
}

/// Broadcast add for NCHW tensor format: output[n,c,h,w] = input[n,c,h,w] + bias[c]
///
/// Adds per-channel bias to a 4D tensor in NCHW format (batch, channels, height, width).
/// Each channel gets its own bias value that is broadcasted across all spatial positions.
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `input` - Input tensor buffer [N, C, H, W]
/// * `bias` - Bias buffer [C] (one value per channel)
/// * `output` - Output tensor buffer [N, C, H, W]
/// * `n` - Batch size
/// * `c` - Number of channels
/// * `h` - Height
/// * `w` - Width
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
///
/// // Create tensors: batch=2, channels=3, height=4, width=4
/// let n = 2; let c = 3; let h = 4; let w = 4;
/// let total_size = n * c * h * w;  // 2*3*4*4 = 96 elements
///
/// let input = exec.allocate::<f32>(total_size)?;
/// let bias = exec.allocate::<f32>(c)?;  // 3 bias values
/// let mut output = exec.allocate::<f32>(total_size)?;
///
/// // Add per-channel bias
/// ops::math::broadcast_add_nchw(&mut exec, &input, &bias, &mut output, n, c, h, w)?;
/// ```
#[tracing::instrument(skip(exec, input, bias, output), fields(
    n = n,
    c = c,
    h = h,
    w = w,
    total_elements = n * c * h * w
))]
#[allow(clippy::too_many_arguments)]
pub fn broadcast_add_nchw<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    bias: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Result<()> {
    let start = Instant::now();
    let total_elements = n * c * h * w;

    // Validate buffer sizes
    if input.len() < total_elements {
        return Err(Error::InvalidOperation(format!(
            "Input buffer too small: len={}, need={}",
            input.len(),
            total_elements
        )));
    }
    if bias.len() < c {
        return Err(Error::InvalidOperation(format!(
            "Bias buffer too small: len={}, need={}",
            bias.len(),
            c
        )));
    }
    if output.len() < total_elements {
        return Err(Error::InvalidOperation(format!(
            "Output buffer too small: len={}, need={}",
            output.len(),
            total_elements
        )));
    }

    // Get buffer handles (handles both class-based and class-free)
    let handle_input = exec.handle_from_buffer(input)?.id();
    let handle_bias = exec.handle_from_buffer(bias)?.id();
    let handle_output = exec.handle_from_buffer(output)?.id();

    let ty = crate::isa_builder::type_from_rust_type::<T>();

    // Build cache key based on shape and buffer handles
    let cache_key = ProgramKey::new(
        "broadcast_add_nchw",
        vec![
            handle_input,
            handle_bias,
            handle_output,
            n as u64,
            c as u64,
            h as u64,
            w as u64,
            ty as u64,
        ],
    );

    // Get or create cached program
    let program = BROADCAST_ADD_CACHE.get_or_create(&cache_key, || {
        crate::isa_builder::build_broadcast_add_nchw(handle_input, handle_bias, handle_output, n, c, h, w, ty)
            .expect("Failed to build broadcast_add_nchw program")
    });

    // Execute ISA program via backend
    let config = hologram_backends::LaunchConfig::new(
        hologram_backends::backend::GridDim::linear(1),
        hologram_backends::backend::BlockDim::linear(1),
        hologram_backends::backend::SharedMemoryConfig::none(),
    );

    write_lock(&exec.backend)
        .execute_program(&program, &config)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    // Instrumentation
    let metrics = ExecutionMetrics::new("broadcast_add_nchw", total_elements, start);
    metrics.log();

    tracing::debug!(
        duration_us = metrics.total_duration_us,
        ops_per_second = metrics.ops_per_second(),
        memory_bandwidth_gbps = metrics.memory_bandwidth_gbps(),
        "broadcast_add_nchw_complete"
    );

    Ok(())
}

/// Element-wise square root: c[i] = sqrt(a[i])
///
/// Computes the square root of each element in the input buffer.
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `a` - Input buffer
/// * `c` - Output buffer
/// * `n` - Number of elements to process
///
/// # Example
///
/// ```ignore
/// # use hologram_core::{Executor, Buffer, ops, Result};
/// # fn main() -> Result<()> {
/// let mut exec = Executor::new()?;
/// let mut a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // c = sqrt(a)
/// ops::math::sqrt(&mut exec, &a, &mut c, 1024)?;
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
#[allow(dead_code)]
pub fn sqrt<T: bytemuck::Pod + 'static>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "sqrt")?;

    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled SQRT program
    let program = &crate::precompiled_programs::VECTOR_SQRT;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("sqrt", n, start);
    metrics.log();

    Ok(())
}

/// Element-wise reciprocal square root: c[i] = 1 / sqrt(a[i])
///
/// Computes the reciprocal square root of each element. This is more efficient than
/// computing sqrt and then dividing, commonly used in normalization operations.
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `a` - Input buffer
/// * `c` - Output buffer
/// * `n` - Number of elements to process
///
/// # Example
///
/// ```ignore
/// # use hologram_core::{Executor, Buffer, ops, Result};
/// # fn main() -> Result<()> {
/// let mut exec = Executor::new()?;
/// let mut a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // c = 1 / sqrt(a)
/// ops::math::rsqrt(&mut exec, &a, &mut c, 1024)?;
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
#[allow(dead_code)]
pub fn rsqrt<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, c], n, "rsqrt")?;

    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled RSQRT program
    let program = &crate::precompiled_programs::VECTOR_RSQRT;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("rsqrt", n, start);
    metrics.log();

    Ok(())
}

/// Element-wise power: c[i] = a[i] ^ b[i]
///
/// Raises each element of the first buffer to the power of the corresponding element
/// in the second buffer.
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `a` - Base buffer
/// * `b` - Exponent buffer
/// * `c` - Output buffer
/// * `n` - Number of elements to process
///
/// # Example
///
/// ```
/// # use hologram_core::{Executor, Buffer, ops, Result};
/// # fn main() -> Result<()> {
/// let mut exec = Executor::new()?;
/// let mut a = exec.allocate::<f32>(1024)?;
/// let mut b = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// // c = a ^ b
/// ops::math::pow(&mut exec, &a, &b, &mut c, 1024)?;
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(skip(exec, a, b, c), fields(n = n))]
#[allow(dead_code)]
pub fn pow<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();

    validate_buffers(&[a, b, c], n, "pow")?;

    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_b = exec.handle_from_buffer(b)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled POW program
    let program = &crate::precompiled_programs::VECTOR_POW;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(2), handle_b)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("pow", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Parallel Operation Variants (Phase 3)
// ============================================================================

/// Parallel vector addition: c[i] = a[i] + b[i]
///
/// Uses operation-level chunking for large vectors (n > 10,000).
/// For small vectors, delegates to standard `vector_add` which uses inline kernels.
///
/// # Performance
///
/// - Small (n ≤ 3,072): Inline SIMD kernel (42ns, optimal)
/// - Medium (3K-10K): Standard ISA execution with block+lane parallelism
/// - Large (n > 10K): Chunked execution + block+lane parallelism (2-8x additional speedup)
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(50_000)?;
/// let b = exec.allocate::<f32>(50_000)?;
/// let mut c = exec.allocate::<f32>(50_000)?;
///
/// // Uses chunking for large vectors
/// ops::math::vector_add_par(&mut exec, &a, &b, &mut c, 50_000)?;
/// ```
pub fn vector_add_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // Use parallel chunking helper
    crate::ops::parallel::parallel_binary_op(exec, a, b, c, n, |exec, a, b, c, chunk_n| {
        vector_add(exec, a, b, c, chunk_n)
    })
}

/// Parallel vector subtraction: c[i] = a[i] - b[i]
///
/// Uses operation-level chunking for large vectors.
pub fn vector_sub_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    crate::ops::parallel::parallel_binary_op(exec, a, b, c, n, |exec, a, b, c, chunk_n| {
        vector_sub(exec, a, b, c, chunk_n)
    })
}

/// Parallel vector multiplication: c[i] = a[i] * b[i]
///
/// Uses operation-level chunking for large vectors.
pub fn vector_mul_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    crate::ops::parallel::parallel_binary_op(exec, a, b, c, n, |exec, a, b, c, chunk_n| {
        vector_mul(exec, a, b, c, chunk_n)
    })
}

/// Parallel vector division: c[i] = a[i] / b[i]
///
/// Uses operation-level chunking for large vectors.
pub fn vector_div_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    crate::ops::parallel::parallel_binary_op(exec, a, b, c, n, |exec, a, b, c, chunk_n| {
        vector_div(exec, a, b, c, chunk_n)
    })
}

/// Parallel absolute value: output[i] = |input[i]|
///
/// Uses operation-level chunking for large vectors.
pub fn abs_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    crate::ops::parallel::parallel_unary_op(exec, input, output, n, |exec, input, output, chunk_n| {
        abs(exec, input, output, chunk_n)
    })
}

/// Parallel negation: output[i] = -input[i]
///
/// Uses operation-level chunking for large vectors.
pub fn neg_par<T: bytemuck::Pod + Send + Sync>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    crate::ops::parallel::parallel_unary_op(exec, input, output, n, |exec, input, output, chunk_n| {
        neg(exec, input, output, chunk_n)
    })
}

// ============================================================================
// Broadcast Operations (General Broadcasting)
// ============================================================================

/// Helper function to try WebGPU broadcast operation
///
/// Returns true if WebGPU was used, false if fallback is needed.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn try_webgpu_broadcast_binary<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    a_shape: &[usize],
    b: &Buffer<T>,
    b_shape: &[usize],
    c: &mut Buffer<T>,
    output_shape: &[usize],
    operation: &str,
) -> Result<bool> {
    use crate::sync::write_lock;

    // Check if type is f32 (only supported type currently)
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
        return Ok(false);
    }

    // Get buffer handles
    let handle_a = exec.handle_from_buffer(a)?;
    let handle_b = exec.handle_from_buffer(b)?;
    let handle_c = exec.handle_from_buffer(c)?;

    // Call backend trait method
    let mut backend_guard = write_lock(&exec.backend);
    let result = match operation {
        "broadcast_add" => {
            backend_guard.broadcast_add_f32(handle_a, a_shape, handle_b, b_shape, handle_c, output_shape)
        }
        "broadcast_sub" => {
            backend_guard.broadcast_sub_f32(handle_a, a_shape, handle_b, b_shape, handle_c, output_shape)
        }
        "broadcast_mul" => {
            backend_guard.broadcast_mul_f32(handle_a, a_shape, handle_b, b_shape, handle_c, output_shape)
        }
        "broadcast_div" => {
            backend_guard.broadcast_div_f32(handle_a, a_shape, handle_b, b_shape, handle_c, output_shape)
        }
        _ => return Ok(false),
    };

    match result {
        Ok(_) => Ok(true), // Backend operation succeeded
        Err(hologram_backends::BackendError::UnsupportedOperation(_)) => Ok(false), // Not supported, try fallback
        Err(e) => Err(Error::Backend(e)), // Actual error
    }
}

/// Broadcast addition: c = a + b (with numpy-style broadcasting)
///
/// Performs element-wise addition with broadcasting. Shapes must be broadcast-compatible.
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `a` - Input buffer A
/// * `a_shape` - Shape of input A
/// * `b` - Input buffer B
/// * `b_shape` - Shape of input B
/// * `c` - Output buffer
/// * `output_shape` - Shape of output (must be compatible with broadcast(a_shape, b_shape))
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
///
/// // Broadcast [3, 1] + [1, 4] -> [3, 4]
/// let a = exec.allocate::<f32>(3)?;  // Shape [3, 1]
/// let b = exec.allocate::<f32>(4)?;  // Shape [1, 4]
/// let mut c = exec.allocate::<f32>(12)?;  // Shape [3, 4]
///
/// ops::math::broadcast_add(&mut exec, &a, &[3, 1], &b, &[1, 4], &mut c, &[3, 4])?;
/// ```
#[tracing::instrument(skip(_exec, _a, _b, _c), fields(
    a_shape = ?_a_shape,
    b_shape = ?_b_shape,
    output_shape = ?output_shape
))]
pub fn broadcast_add<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _a_shape: &[usize],
    _b: &Buffer<T>,
    _b_shape: &[usize],
    _c: &mut Buffer<T>,
    output_shape: &[usize],
) -> Result<()> {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _start = Instant::now();
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _total_elements: usize = output_shape.iter().product();

    // Try WebGPU fast path first
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_broadcast_binary(_exec, _a, _a_shape, _b, _b_shape, _c, output_shape, "broadcast_add")? {
        let metrics = ExecutionMetrics::new("broadcast_add", _total_elements, _start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "broadcast_add_webgpu_complete"
        );
        return Ok(());
    }

    // CPU fallback: build ISA program for broadcasting
    // TODO: Implement CPU broadcast support using ISA builder
    Err(Error::InvalidOperation(
        "broadcast_add CPU fallback not yet implemented".to_string(),
    ))
}

/// Broadcast subtraction: c = a - b (with numpy-style broadcasting)
#[tracing::instrument(skip(_exec, _a, _b, _c), fields(
    a_shape = ?_a_shape,
    b_shape = ?_b_shape,
    output_shape = ?output_shape
))]
pub fn broadcast_sub<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _a_shape: &[usize],
    _b: &Buffer<T>,
    _b_shape: &[usize],
    _c: &mut Buffer<T>,
    output_shape: &[usize],
) -> Result<()> {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _start = Instant::now();
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _total_elements: usize = output_shape.iter().product();

    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_broadcast_binary(_exec, _a, _a_shape, _b, _b_shape, _c, output_shape, "broadcast_sub")? {
        let metrics = ExecutionMetrics::new("broadcast_sub", _total_elements, _start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "broadcast_sub_webgpu_complete"
        );
        return Ok(());
    }

    Err(Error::InvalidOperation(
        "broadcast_sub CPU fallback not yet implemented".to_string(),
    ))
}

/// Broadcast multiplication: c = a * b (with numpy-style broadcasting)
#[tracing::instrument(skip(_exec, _a, _b, _c), fields(
    a_shape = ?_a_shape,
    b_shape = ?_b_shape,
    output_shape = ?output_shape
))]
pub fn broadcast_mul<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _a_shape: &[usize],
    _b: &Buffer<T>,
    _b_shape: &[usize],
    _c: &mut Buffer<T>,
    output_shape: &[usize],
) -> Result<()> {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _start = Instant::now();
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _total_elements: usize = output_shape.iter().product();

    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_broadcast_binary(_exec, _a, _a_shape, _b, _b_shape, _c, output_shape, "broadcast_mul")? {
        let metrics = ExecutionMetrics::new("broadcast_mul", _total_elements, _start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "broadcast_mul_webgpu_complete"
        );
        return Ok(());
    }

    Err(Error::InvalidOperation(
        "broadcast_mul CPU fallback not yet implemented".to_string(),
    ))
}

/// Broadcast division: c = a / b (with numpy-style broadcasting)
#[tracing::instrument(skip(_exec, _a, _b, _c), fields(
    a_shape = ?_a_shape,
    b_shape = ?_b_shape,
    output_shape = ?output_shape
))]
pub fn broadcast_div<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _a_shape: &[usize],
    _b: &Buffer<T>,
    _b_shape: &[usize],
    _c: &mut Buffer<T>,
    output_shape: &[usize],
) -> Result<()> {
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _start = Instant::now();
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    let _total_elements: usize = output_shape.iter().product();

    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    if try_webgpu_broadcast_binary(_exec, _a, _a_shape, _b, _b_shape, _c, output_shape, "broadcast_div")? {
        let metrics = ExecutionMetrics::new("broadcast_div", _total_elements, _start);
        tracing::debug!(
            duration_us = metrics.total_duration_us,
            webgpu = true,
            "broadcast_div_webgpu_complete"
        );
        return Ok(());
    }

    Err(Error::InvalidOperation(
        "broadcast_div CPU fallback not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() -> Result<()> {
        // Load kernels before testing
        use hologram_codegen::register_all_kernels_from_directory;
        let _ = register_all_kernels_from_directory("../../target/kernel-libs");

        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(3072)?;
        let mut b = exec.allocate::<f32>(3072)?;
        let mut c = exec.allocate::<f32>(3072)?;

        // Initialize data
        let data_a: Vec<f32> = (0..3072).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..3072).map(|i| (i * 2) as f32).collect();

        a.copy_from_slice(&mut exec, &data_a)?;
        b.copy_from_slice(&mut exec, &data_b)?;

        // Execute vector_add
        vector_add(&mut exec, &a, &b, &mut c, 3072)?;

        // Verify results
        let result = c.to_vec(&exec)?;
        assert_eq!(result[0], 0.0); // 0 + 0
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[2], 6.0); // 2 + 4

        Ok(())
    }

    #[test]
    fn test_min_direct() -> Result<()> {
        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(10)?;
        let mut b = exec.allocate::<f32>(10)?;
        let mut c = exec.allocate::<f32>(10)?;

        // Initialize data
        let data_a: Vec<f32> = vec![-5.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 5.0];
        let data_b: Vec<f32> = vec![1.0; 10]; // max clamp value

        a.copy_from_slice(&mut exec, &data_a)?;
        b.copy_from_slice(&mut exec, &data_b)?;

        // Execute min operation
        min(&mut exec, &a, &b, &mut c, 10)?;

        // Verify results - should clamp all values above 1.0 to 1.0
        let result = c.to_vec(&exec)?;

        assert_eq!(result[0], -5.0); // min(-5.0, 1.0) = -5.0
        assert_eq!(result[1], -1.5); // min(-1.5, 1.0) = -1.5
        assert_eq!(result[2], -1.0); // min(-1.0, 1.0) = -1.0
        assert_eq!(result[3], -0.5); // min(-0.5, 1.0) = -0.5
        assert_eq!(result[4], 0.0); // min(0.0, 1.0) = 0.0
        assert_eq!(result[5], 0.5); // min(0.5, 1.0) = 0.5

        Ok(())
    }

    #[test]
    fn test_max_direct() -> Result<()> {
        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(10)?;
        let mut b = exec.allocate::<f32>(10)?;
        let mut c = exec.allocate::<f32>(10)?;

        // Initialize data
        let data_a: Vec<f32> = vec![-5.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 5.0];
        let data_b: Vec<f32> = vec![-1.0; 10]; // min clamp value

        a.copy_from_slice(&mut exec, &data_a)?;
        b.copy_from_slice(&mut exec, &data_b)?;

        // Execute max operation
        max(&mut exec, &a, &b, &mut c, 10)?;

        // Verify results - should clamp all values below -1.0 to -1.0
        let result = c.to_vec(&exec)?;

        assert_eq!(result[0], -1.0); // max(-5.0, -1.0) = -1.0
        assert_eq!(result[1], -1.0); // max(-1.5, -1.0) = -1.0
        assert_eq!(result[2], -1.0); // max(-1.0, -1.0) = -1.0
        assert_eq!(result[3], -0.5); // max(-0.5, -1.0) = -0.5
        assert_eq!(result[4], 0.0); // max(0.0, -1.0) = 0.0
        assert_eq!(result[5], 0.5); // max(0.5, -1.0) = 0.5

        Ok(())
    }

    #[test]
    fn test_clip() -> Result<()> {
        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(10)?;
        let mut c = exec.allocate::<f32>(10)?;

        // Initialize data with values outside and inside the clip range
        let data: Vec<f32> = vec![-5.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 5.0];
        a.copy_from_slice(&mut exec, &data)?;

        // Clip to [-1.0, 1.0]
        clip(&mut exec, &a, &mut c, -1.0f32, 1.0f32, 10)?;

        // Verify results
        let result = c.to_vec(&exec)?;
        assert_eq!(result[0], -1.0); // -5.0 clamped to -1.0
        assert_eq!(result[1], -1.0); // -1.5 clamped to -1.0
        assert_eq!(result[2], -1.0); // -1.0 unchanged
        assert_eq!(result[3], -0.5); // -0.5 unchanged
        assert_eq!(result[4], 0.0); // 0.0 unchanged
        assert_eq!(result[5], 0.5); // 0.5 unchanged
        assert_eq!(result[6], 1.0); // 1.0 unchanged
        assert_eq!(result[7], 1.0); // 1.5 clamped to 1.0
        assert_eq!(result[8], 1.0); // 2.0 clamped to 1.0
        assert_eq!(result[9], 1.0); // 5.0 clamped to 1.0

        Ok(())
    }

    #[test]
    fn test_scalar_add() -> Result<()> {
        // Load kernels before testing
        use hologram_codegen::register_all_kernels_from_directory;
        let _ = register_all_kernels_from_directory("../../target/kernel-libs");

        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(8)?;
        let mut c = exec.allocate::<f32>(8)?;

        // Initialize data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        a.copy_from_slice(&mut exec, &data)?;

        // Add 2.5 to all elements
        scalar_add(&mut exec, &a, &mut c, 2.5f32, 8)?;

        // Verify results
        let result = c.to_vec(&exec)?;
        assert_eq!(result[0], 3.5); // 1.0 + 2.5
        assert_eq!(result[1], 4.5); // 2.0 + 2.5
        assert_eq!(result[2], 5.5); // 3.0 + 2.5
        assert_eq!(result[3], 6.5); // 4.0 + 2.5
        assert_eq!(result[4], 7.5); // 5.0 + 2.5
        assert_eq!(result[5], 8.5); // 6.0 + 2.5
        assert_eq!(result[6], 9.5); // 7.0 + 2.5
        assert_eq!(result[7], 10.5); // 8.0 + 2.5

        Ok(())
    }

    #[test]
    fn test_scalar_mul() -> Result<()> {
        let mut exec = Executor::new()?;

        // Allocate buffers
        let mut a = exec.allocate::<f32>(8)?;
        let mut c = exec.allocate::<f32>(8)?;

        // Initialize data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        a.copy_from_slice(&mut exec, &data)?;

        // Multiply all elements by 2.0
        scalar_mul(&mut exec, &a, &mut c, 2.0f32, 8)?;

        // Verify results
        let result = c.to_vec(&exec)?;
        assert_eq!(result[0], 2.0); // 1.0 * 2.0
        assert_eq!(result[1], 4.0); // 2.0 * 2.0
        assert_eq!(result[2], 6.0); // 3.0 * 2.0
        assert_eq!(result[3], 8.0); // 4.0 * 2.0
        assert_eq!(result[4], 10.0); // 5.0 * 2.0
        assert_eq!(result[5], 12.0); // 6.0 * 2.0
        assert_eq!(result[6], 14.0); // 7.0 * 2.0
        assert_eq!(result[7], 16.0); // 8.0 * 2.0

        Ok(())
    }

    #[test]
    fn test_broadcast_add_nchw() -> Result<()> {
        let mut exec = Executor::new()?;

        // Test shape: N=2, C=3, H=2, W=2
        let n = 2;
        let c = 3;
        let h = 2;
        let w = 2;
        let total_size = n * c * h * w; // 2*3*2*2 = 24 elements

        // Allocate buffers
        let mut input = exec.allocate::<f32>(total_size)?;
        let mut bias = exec.allocate::<f32>(c)?;
        let mut output = exec.allocate::<f32>(total_size)?;

        // Initialize input: all 1.0
        let input_data = vec![1.0f32; total_size];
        input.copy_from_slice(&mut exec, &input_data)?;

        // Initialize bias: [10.0, 20.0, 30.0]
        let bias_data = vec![10.0f32, 20.0f32, 30.0f32];
        bias.copy_from_slice(&mut exec, &bias_data)?;

        // Apply broadcast add
        broadcast_add_nchw(&mut exec, &input, &bias, &mut output, n, c, h, w)?;

        // Verify results
        let result = output.to_vec(&exec)?;

        // Expected results: each spatial position in channel c gets bias[c] added
        // Batch 0:
        //   Channel 0 (4 elements): 1.0 + 10.0 = 11.0
        //   Channel 1 (4 elements): 1.0 + 20.0 = 21.0
        //   Channel 2 (4 elements): 1.0 + 30.0 = 31.0
        // Batch 1: same pattern

        // Check batch 0, channel 0 (indices 0-3)
        for (i, &val) in result.iter().take(4).enumerate() {
            assert_eq!(val, 11.0, "Batch 0, Channel 0, position {}", i);
        }

        // Check batch 0, channel 1 (indices 4-7)
        for (i, &val) in result.iter().skip(4).take(4).enumerate() {
            assert_eq!(val, 21.0, "Batch 0, Channel 1, position {}", i);
        }

        // Check batch 0, channel 2 (indices 8-11)
        for (i, &val) in result.iter().skip(8).take(4).enumerate() {
            assert_eq!(val, 31.0, "Batch 0, Channel 2, position {}", i);
        }

        // Check batch 1, channel 0 (indices 12-15)
        for (i, &val) in result.iter().skip(12).take(4).enumerate() {
            assert_eq!(val, 11.0, "Batch 1, Channel 0, position {}", i);
        }

        // Check batch 1, channel 1 (indices 16-19)
        for (i, &val) in result.iter().skip(16).take(4).enumerate() {
            assert_eq!(val, 21.0, "Batch 1, Channel 1, position {}", i);
        }

        // Check batch 1, channel 2 (indices 20-23)
        for (i, &val) in result.iter().skip(20).take(4).enumerate() {
            assert_eq!(val, 31.0, "Batch 1, Channel 2, position {}", i);
        }

        Ok(())
    }
}
