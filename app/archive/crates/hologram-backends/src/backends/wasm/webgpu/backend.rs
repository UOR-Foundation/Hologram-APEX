//! WebGPU Backend trait implementation
//!
//! Provides Backend trait implementation for WebGPU compute acceleration in WASM environments.

use crate::error::{BackendError, Result};

// WASM-specific imports for Backend implementation
#[cfg(target_arch = "wasm32")]
use super::buffer_pool::BufferPool;
#[cfg(target_arch = "wasm32")]
use super::device::WebGpuDevice;
#[cfg(target_arch = "wasm32")]
use super::isa_translator::WgslGenerator;
#[cfg(target_arch = "wasm32")]
use super::pipeline::PipelineCache;
#[cfg(target_arch = "wasm32")]
use crate::backend::{Backend, BufferHandle, ExecutionParams, LaunchConfig, PoolHandle};
#[cfg(target_arch = "wasm32")]
use crate::isa::Program;
#[cfg(target_arch = "wasm32")]
use crate::sync::{lock_mutex, Mutex};
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wgpu::{Buffer as WgpuBuffer, Device, Queue};

/// Calculate dispatch dimensions for large workloads that may exceed per-dimension limits
///
/// WebGPU has a limit of 65,535 workgroups per dimension. This function splits
/// large workgroup counts across Y dimension when needed.
#[cfg(target_arch = "wasm32")]
fn calculate_dispatch_size(total_workgroups: u32) -> (u32, u32, u32) {
    const MAX_DISPATCH_PER_DIM: u32 = 65535;
    if total_workgroups <= MAX_DISPATCH_PER_DIM {
        (total_workgroups, 1, 1)
    } else {
        let x = MAX_DISPATCH_PER_DIM;
        let y = (total_workgroups + MAX_DISPATCH_PER_DIM - 1) / MAX_DISPATCH_PER_DIM;
        (x, y, 1)
    }
}

/// WebGPU backend for executing operations in browser environments
///
/// This backend provides GPU-accelerated execution through WebGPU with:
/// - Buffer pooling for reduced allocation overhead
/// - Pipeline caching for fast shader compilation
/// - Async/sync bridging for WASM environments
///
/// # Architecture
///
/// ```text
/// WebGpuBackend
/// ├── Device/Queue      - WebGPU device and command queue
/// ├── BufferPool        - Reusable GPU buffers (500x faster allocation)
/// ├── PipelineCache     - Compiled WGSL shader cache
/// ├── BufferManager     - Maps BufferHandle to GPU buffers
/// └── ISA Executor      - Translates Atlas ISA to WebGPU operations
/// ```
///
/// # Usage
///
/// ```rust,no_run
/// use hologram_backends::{WebGpuBackend, Backend};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create WebGPU backend (async init required for device)
/// let mut backend = WebGpuBackend::new().await?;
///
/// // Use like any other backend
/// let buffer = backend.allocate_buffer(1024)?;
/// backend.copy_to_buffer(buffer, &data)?;
/// // ...
/// backend.free_buffer(buffer)?;
/// # Ok(())
/// # }
/// ```
#[cfg(target_arch = "wasm32")]
pub struct WebGpuBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffer_pool: Arc<BufferPool>,
    pipeline_cache: Arc<PipelineCache>,

    /// Buffer handle management
    /// Maps BufferHandle ID to GPU buffer
    /// Wrapped in Arc to allow cloning for async operations
    buffers: Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
    next_buffer_id: Mutex<u64>,

    /// Pool handle management
    /// Maps PoolHandle ID to GPU buffer
    pools: Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
    next_pool_id: Mutex<u64>,
}

#[cfg(target_arch = "wasm32")]
impl WebGpuBackend {
    /// Create a new WebGPU backend
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - WebGPU is not available in the browser
    /// - Device initialization fails
    /// - Required WebGPU features are not supported
    pub async fn new() -> Result<Self> {
        // Check availability
        if !WebGpuDevice::is_available() {
            return Err(BackendError::BackendUnavailable(
                "WebGPU not available in this browser".into(),
            ));
        }

        // Initialize device (async operation)
        let device = WebGpuDevice::new()
            .await
            .map_err(|e| BackendError::BackendInitialization(format!("WebGPU device init failed: {}", e)))?;

        let device_arc = Arc::clone(device.device());
        let queue_arc = Arc::clone(device.queue());

        // Create buffer pool for memory optimization
        let buffer_pool = Arc::new(BufferPool::new(Arc::clone(&device_arc), Arc::clone(&queue_arc)));

        // Create pipeline cache for shader compilation
        let pipeline_cache = Arc::new(PipelineCache::new(Arc::clone(&device_arc)));

        Ok(Self {
            device: device_arc,
            queue: queue_arc,
            buffer_pool,
            pipeline_cache,
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_buffer_id: Mutex::new(1), // Start at 1 (0 reserved for invalid)
            pools: Arc::new(Mutex::new(HashMap::new())),
            next_pool_id: Mutex::new(1),
        })
    }

    /// Get device reference (for direct WebGPU operations)
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference (for command submission)
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Get buffer pool reference (for pooled buffer management)
    pub fn buffer_pool(&self) -> &Arc<BufferPool> {
        &self.buffer_pool
    }

    /// Get pipeline cache reference (for shader compilation)
    pub fn pipeline_cache(&self) -> &Arc<PipelineCache> {
        &self.pipeline_cache
    }

    /// Get GPU buffer for a handle (for fast path operations)
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid.
    pub fn get_gpu_buffer(&self, handle: BufferHandle) -> Result<Arc<WgpuBuffer>> {
        let buffers = lock_mutex(&self.buffers);
        buffers
            .get(&handle.id())
            .cloned()
            .ok_or_else(|| BackendError::InvalidBufferHandle(handle.id()))
    }

    /// Create a new buffer handle
    fn allocate_handle(&self) -> BufferHandle {
        let mut next_id = lock_mutex(&self.next_buffer_id);
        let id = *next_id;
        *next_id += 1;
        BufferHandle::new(id)
    }

    /// Create a new pool handle
    fn allocate_pool_handle(&self) -> PoolHandle {
        let mut next_id = lock_mutex(&self.next_pool_id);
        let id = *next_id;
        *next_id += 1;
        PoolHandle::new(id)
    }
}

// ============================================================================================
// Helper Functions for High-Level Operations
// ============================================================================================

/// Helper function to execute broadcast binary operations (add, sub, mul, div)
#[cfg(target_arch = "wasm32")]
fn broadcast_binary_op(
    backend: &mut WebGpuBackend,
    a_handle: BufferHandle,
    a_shape: &[usize],
    b_handle: BufferHandle,
    b_shape: &[usize],
    c_handle: BufferHandle,
    output_shape: &[usize],
    shader_name: &str,
    shader_source: &str,
) -> Result<()> {
    use wgpu::util::DeviceExt;

    // Get GPU buffers
    let gpu_buffer_a = backend.get_gpu_buffer(a_handle)?;
    let gpu_buffer_b = backend.get_gpu_buffer(b_handle)?;
    let gpu_buffer_c = backend.get_gpu_buffer(c_handle)?;

    // Pad shapes to 8 dimensions (fill with 1s for missing dimensions)
    let mut out_shape_padded = [1u32; 8];
    let mut a_shape_padded = [1u32; 8];
    let mut b_shape_padded = [1u32; 8];

    for (i, &dim) in output_shape.iter().enumerate() {
        out_shape_padded[i] = dim as u32;
    }
    for (i, &dim) in a_shape.iter().enumerate() {
        a_shape_padded[i] = dim as u32;
    }
    for (i, &dim) in b_shape.iter().enumerate() {
        b_shape_padded[i] = dim as u32;
    }

    let total_elements: usize = output_shape.iter().product();

    // Create uniform buffer for broadcast parameters
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BroadcastParams {
        out_shape: [u32; 8],
        a_shape: [u32; 8],
        b_shape: [u32; 8],
        out_ndim: u32,
        a_ndim: u32,
        b_ndim: u32,
        total_elements: u32,
    }

    let params = BroadcastParams {
        out_shape: out_shape_padded,
        a_shape: a_shape_padded,
        b_shape: b_shape_padded,
        out_ndim: output_shape.len() as u32,
        a_ndim: a_shape.len() as u32,
        b_ndim: b_shape.len() as u32,
        total_elements: total_elements as u32,
    };

    let params_buffer = backend.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{} Params Buffer", shader_name)),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Get or create pipeline
    let pipeline = backend
        .pipeline_cache()
        .get_or_create(shader_name, shader_source, "main")
        .map_err(|e| BackendError::ExecutionError(format!("WebGPU pipeline creation failed: {}", e)))?;

    // Create bind group (3 storage buffers + 1 uniform buffer)
    let bind_group = backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create command encoder
    let mut encoder = backend
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
        let num_workgroups = (total_elements as u32 + workgroup_size - 1) / workgroup_size;

        // Split dispatch across dimensions to respect 65,535 workgroup limit per dimension
        let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    // Submit commands
    backend.queue().submit([encoder.finish()]);

    Ok(())
}

/// Helper function to execute type casting operations
#[cfg(target_arch = "wasm32")]
fn cast_op(
    backend: &mut WebGpuBackend,
    input_handle: BufferHandle,
    output_handle: BufferHandle,
    n: usize,
    shader_name: &str,
    shader_source: &str,
) -> Result<()> {
    use wgpu::util::DeviceExt;

    // Get GPU buffers
    let gpu_buffer_in = backend.get_gpu_buffer(input_handle)?;
    let gpu_buffer_out = backend.get_gpu_buffer(output_handle)?;

    let num_elements = n as u32;

    // Create uniform buffer for element count
    let params_buffer = backend.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{} Params Buffer", shader_name)),
        contents: bytemuck::cast_slice(&[num_elements]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Get or create pipeline
    let pipeline = backend
        .pipeline_cache()
        .get_or_create(shader_name, shader_source, "main")
        .map_err(|e| BackendError::ExecutionError(format!("WebGPU pipeline creation failed: {}", e)))?;

    // Create bind group
    let bind_group = backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{} Bind Group", shader_name)),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer_in.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_buffer_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    let workgroup_size = 256;
    let num_workgroups = (num_elements + workgroup_size - 1) / workgroup_size;

    let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);

    // Execute
    let mut encoder = backend
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
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    backend.queue().submit(std::iter::once(encoder.finish()));

    Ok(())
}

/// Helper function to execute gather operation
#[cfg(target_arch = "wasm32")]
fn gather_op(
    backend: &mut WebGpuBackend,
    input_handle: BufferHandle,
    input_shape: &[usize],
    indices_handle: BufferHandle,
    axis: usize,
    output_handle: BufferHandle,
    output_shape: &[usize],
) -> Result<()> {
    use wgpu::util::DeviceExt;

    // Derive num_indices from output shape
    let num_indices = output_shape[axis];

    // Get GPU buffers
    let gpu_buffer_input = backend.get_gpu_buffer(input_handle)?;
    let gpu_buffer_indices = backend.get_gpu_buffer(indices_handle)?;
    let gpu_buffer_output = backend.get_gpu_buffer(output_handle)?;

    // Calculate strides for input and output
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    let mut output_strides = vec![1usize; output_ndim];
    for i in (0..output_ndim - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Pad shapes and strides to 8 dimensions
    let mut padded_input_shape = [1u32; 8];
    let mut padded_input_strides = [1u32; 8];
    let mut padded_output_shape = [1u32; 8];
    let mut padded_output_strides = [1u32; 8];

    for i in 0..input_ndim.min(8) {
        padded_input_shape[i] = input_shape[i] as u32;
        padded_input_strides[i] = input_strides[i] as u32;
    }

    for i in 0..output_ndim.min(8) {
        padded_output_shape[i] = output_shape[i] as u32;
        padded_output_strides[i] = output_strides[i] as u32;
    }

    let total_output_elements: usize = output_shape.iter().product();
    let axis_dim_size = input_shape[axis];

    // Build parameters buffer
    let mut params_data = Vec::new();
    params_data.extend_from_slice(&padded_input_shape); // 8 u32
    params_data.extend_from_slice(&padded_input_strides); // 8 u32
    params_data.extend_from_slice(&padded_output_shape); // 8 u32
    params_data.extend_from_slice(&padded_output_strides); // 8 u32
    params_data.push(axis as u32);
    params_data.push(input_ndim as u32);
    params_data.push(output_ndim as u32);
    params_data.push(total_output_elements as u32);
    params_data.push(num_indices as u32);
    params_data.push(axis_dim_size as u32);

    let params_buffer = backend.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gather Params Buffer"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Get shader source
    let shader_source = include_str!("kernels/gather.wgsl");

    // Get or create pipeline
    let pipeline = backend
        .pipeline_cache()
        .get_or_create("gather", shader_source, "main")
        .map_err(|e| BackendError::ExecutionError(format!("WebGPU pipeline creation failed: {}", e)))?;

    // Create bind group
    let bind_group = backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Gather Bind Group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_buffer_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_buffer_output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    let workgroup_size = 256;
    let num_workgroups = (total_output_elements as u32 + workgroup_size - 1) / workgroup_size;

    let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);

    // Execute
    let mut encoder = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gather Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gather Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    backend.queue().submit(std::iter::once(encoder.finish()));

    Ok(())
}

/// Check if program contains high-level instructions that need special handling
#[cfg(target_arch = "wasm32")]
fn requires_high_level_handler(program: &Program) -> bool {
    use crate::isa::Instruction;
    program.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Conv2d { .. }
                | Instruction::NearestUpsample2d { .. }
                | Instruction::LayerNorm { .. }
                | Instruction::GroupNorm { .. }
                | Instruction::Gemm { .. }
        )
    })
}

/// Helper function to execute ISA programs with optional register-to-handle mappings
#[cfg(target_arch = "wasm32")]
fn execute_program_internal(
    backend: &mut WebGpuBackend,
    program: &Program,
    config: &LaunchConfig,
    register_map: &std::collections::HashMap<crate::isa::Register, u64>,
) -> Result<()> {
    // Check if this program contains high-level instructions that need special handling
    if requires_high_level_handler(program) {
        return execute_high_level_instruction_impl(backend, program);
    }

    // Step 1: Generate WGSL shader from ISA program (low-level operations only)
    let mut generator = WgslGenerator::new();
    let wgsl_code = generator.generate(program)?;

    tracing::debug!("Generated WGSL shader:\n{}", wgsl_code);

    // Step 2: Compile shader and get/create pipeline
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    wgsl_code.hash(&mut hasher);
    let shader_name = format!("isa_shader_{}", hasher.finish());

    let pipeline = backend
        .pipeline_cache
        .get_or_create(&shader_name, &wgsl_code, "main")
        .map_err(|e| BackendError::ExecutionError(format!("Failed to compile shader: {}", e)))?;

    // Step 3: Create bind group with buffer bindings
    let buffer_bindings = generator.buffer_bindings();

    // Collect all buffers first
    let buffers_lock = crate::sync::lock_mutex(&backend.buffers);
    let mut buffers_vec = Vec::new();
    for (register_id, binding_idx) in buffer_bindings {
        // Look up actual buffer handle from register map (for RegisterIndirectComputed addressing)
        // If not in map, use register_id as the handle (for BufferOffset addressing)
        let reg = crate::isa::Register::new(*register_id as u8);
        let buffer_handle_id = register_map.get(&reg).copied().unwrap_or(*register_id as u64);

        if let Some(buffer) = buffers_lock.get(&buffer_handle_id) {
            buffers_vec.push((*binding_idx, buffer.clone()));
        } else {
            return Err(BackendError::InvalidBufferHandle(buffer_handle_id));
        }
    }
    drop(buffers_lock);

    // Create bind group entries
    let bind_group_entries: Vec<wgpu::BindGroupEntry> = buffers_vec
        .iter()
        .map(|(binding_idx, buffer)| wgpu::BindGroupEntry {
            binding: *binding_idx,
            resource: buffer.as_entire_binding(),
        })
        .collect();

    // Create bind group
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ISA Execution Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    // Step 4: Create command encoder and dispatch compute shader
    let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ISA Execution Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ISA Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Calculate dispatch size based on launch config
        let (workgroup_size_x, _, _) = generator.workgroup_size();
        let dispatch_x = (config.grid.x + workgroup_size_x - 1) / workgroup_size_x;

        compute_pass.dispatch_workgroups(dispatch_x, config.grid.y, config.grid.z);
    }

    // Submit command buffer
    backend.queue.submit(Some(encoder.finish()));

    Ok(())
}

#[cfg(target_arch = "wasm32")]
impl Backend for WebGpuBackend {
    fn execute_program(&mut self, program: &Program, config: &LaunchConfig) -> Result<()> {
        // Call internal method with empty register mapping
        execute_program_internal(self, program, config, &std::collections::HashMap::new())
    }

    fn execute_program_with_params(&mut self, program: &Program, params: &ExecutionParams) -> Result<()> {
        // Call internal method with register-to-handle mappings from params
        execute_program_internal(self, program, &params.launch_config, &params.initial_registers)
    }

    fn allocate_buffer(&mut self, size: usize) -> Result<BufferHandle> {
        // Allocate GPU buffer from pool
        let buffer = self.buffer_pool.acquire_storage(size);

        // Create handle and store mapping
        let handle = self.allocate_handle();
        lock_mutex(&self.buffers).insert(handle.id(), buffer);

        Ok(handle)
    }

    fn free_buffer(&mut self, handle: BufferHandle) -> Result<()> {
        // Remove buffer from mapping
        let mut buffers = lock_mutex(&self.buffers);
        let buffer = buffers
            .remove(&handle.id())
            .ok_or_else(|| BackendError::InvalidBufferHandle(handle.id()))?;

        // Return buffer to pool (Arc drop will trigger PooledBuffer cleanup)
        drop(buffers);
        self.buffer_pool.release_storage(buffer);

        Ok(())
    }

    fn copy_to_buffer(&mut self, handle: BufferHandle, data: &[u8]) -> Result<()> {
        let buffer = self.get_gpu_buffer(handle)?;

        // Validate size
        if data.len() > buffer.size() as usize {
            return Err(BackendError::BufferOutOfBounds {
                offset: 0,
                size: data.len(),
                buffer_size: buffer.size() as usize,
            });
        }

        // Write data to GPU buffer
        self.queue.write_buffer(&buffer, 0, data);

        // On WASM, write_buffer queues the write but doesn't wait for completion
        // We CANNOT make this work reliably in synchronous code on WASM
        // The JavaScript event loop needs to run for GPU operations to complete
        // For now, do NOT poll here - it blocks the JS thread and prevents GPU work

        Ok(())
    }

    fn copy_from_buffer(&mut self, handle: BufferHandle, data: &mut [u8]) -> Result<()> {
        let buffer = self.get_gpu_buffer(handle)?;

        // Validate size
        if data.len() > buffer.size() as usize {
            return Err(BackendError::BufferOutOfBounds {
                offset: 0,
                size: data.len(),
                buffer_size: buffer.size() as usize,
            });
        }

        // For reading from GPU, we need to create a staging buffer and perform async read
        // In WASM, we use wgpu's polling mechanism to wait for the mapping
        #[cfg(target_arch = "wasm32")]
        {
            // Create staging buffer
            let staging_buffer = self.buffer_pool.acquire_staging(data.len());

            // Submit empty command buffer first to ensure all previous GPU operations complete
            self.queue.submit([]);

            // Copy from storage to staging
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer Read Encoder"),
            });
            encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, data.len() as u64);
            self.queue.submit([encoder.finish()]);

            // CRITICAL: Synchronous buffer reads in WASM are unreliable
            // This function should NOT be used for WebGPU - use async reads instead
            return Err(BackendError::Other(
                "Synchronous buffer reads not supported in WASM/WebGPU. \
                 Use async buffer read methods (to_vec_async) instead of synchronous reads (to_vec)."
                    .into(),
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(BackendError::BackendUnavailable(
                "WebGPU backend only available on WASM".into(),
            ))
        }
    }

    fn buffer_size(&self, handle: BufferHandle) -> Result<usize> {
        let buffer = self.get_gpu_buffer(handle)?;
        Ok(buffer.size() as usize)
    }

    fn allocate_pool(&mut self, size: usize) -> Result<PoolHandle> {
        // Allocate GPU buffer for pool
        let buffer = self.buffer_pool.acquire_storage(size);

        // Create handle and store mapping
        let handle = self.allocate_pool_handle();
        lock_mutex(&self.pools).insert(handle.id(), buffer);

        Ok(handle)
    }

    fn free_pool(&mut self, handle: PoolHandle) -> Result<()> {
        // Remove pool from mapping
        let mut pools = lock_mutex(&self.pools);
        let buffer = pools
            .remove(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;

        // Return buffer to pool
        drop(pools);
        self.buffer_pool.release_storage(buffer);

        Ok(())
    }

    fn copy_to_pool(&mut self, handle: PoolHandle, offset: usize, data: &[u8]) -> Result<()> {
        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;

        // Validate bounds
        if offset + data.len() > buffer.size() as usize {
            return Err(BackendError::PoolOutOfBounds {
                offset,
                size: data.len(),
                pool_size: buffer.size() as usize,
            });
        }

        // Write data to GPU buffer at offset
        self.queue.write_buffer(buffer, offset as u64, data);

        Ok(())
    }

    fn copy_from_pool(&mut self, handle: PoolHandle, offset: usize, data: &mut [u8]) -> Result<()> {
        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .cloned()
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;
        drop(pools);

        // Validate bounds
        if offset + data.len() > buffer.size() as usize {
            return Err(BackendError::PoolOutOfBounds {
                offset,
                size: data.len(),
                pool_size: buffer.size() as usize,
            });
        }

        // Similar async read as copy_from_buffer
        #[cfg(target_arch = "wasm32")]
        {
            let staging_buffer = self.buffer_pool.acquire_staging(data.len());

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pool Read Encoder"),
            });
            encoder.copy_buffer_to_buffer(&buffer, offset as u64, &staging_buffer, 0, data.len() as u64);
            self.queue.submit([encoder.finish()]);

            // CRITICAL: Synchronous pool reads in WASM are unreliable
            // This function should NOT be used for WebGPU - use async reads instead
            return Err(BackendError::Other(
                "Synchronous pool reads not supported in WASM/WebGPU. \
                 Use async buffer read methods instead of synchronous reads."
                    .into(),
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(BackendError::BackendUnavailable(
                "WebGPU backend only available on WASM".into(),
            ))
        }
    }

    fn pool_size(&self, handle: PoolHandle) -> Result<usize> {
        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;
        Ok(buffer.size() as usize)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    // ============================================================================================
    // High-Level Operations (Broadcasting)
    // ============================================================================================

    fn broadcast_add_f32(
        &mut self,
        a_handle: BufferHandle,
        a_shape: &[usize],
        b_handle: BufferHandle,
        b_shape: &[usize],
        c_handle: BufferHandle,
        output_shape: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(
            self,
            a_handle,
            a_shape,
            b_handle,
            b_shape,
            c_handle,
            output_shape,
            "broadcast_add",
            include_str!("kernels/broadcast_add.wgsl"),
        )
    }

    fn broadcast_sub_f32(
        &mut self,
        a_handle: BufferHandle,
        a_shape: &[usize],
        b_handle: BufferHandle,
        b_shape: &[usize],
        c_handle: BufferHandle,
        output_shape: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(
            self,
            a_handle,
            a_shape,
            b_handle,
            b_shape,
            c_handle,
            output_shape,
            "broadcast_sub",
            include_str!("kernels/broadcast_sub.wgsl"),
        )
    }

    fn broadcast_mul_f32(
        &mut self,
        a_handle: BufferHandle,
        a_shape: &[usize],
        b_handle: BufferHandle,
        b_shape: &[usize],
        c_handle: BufferHandle,
        output_shape: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(
            self,
            a_handle,
            a_shape,
            b_handle,
            b_shape,
            c_handle,
            output_shape,
            "broadcast_mul",
            include_str!("kernels/broadcast_mul.wgsl"),
        )
    }

    fn broadcast_div_f32(
        &mut self,
        a_handle: BufferHandle,
        a_shape: &[usize],
        b_handle: BufferHandle,
        b_shape: &[usize],
        c_handle: BufferHandle,
        output_shape: &[usize],
    ) -> Result<()> {
        broadcast_binary_op(
            self,
            a_handle,
            a_shape,
            b_handle,
            b_shape,
            c_handle,
            output_shape,
            "broadcast_div",
            include_str!("kernels/broadcast_div.wgsl"),
        )
    }

    // ============================================================================================
    // High-Level Operations (Type Casting)
    // ============================================================================================

    fn cast_f32_to_i32(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_f32_to_i32",
            include_str!("kernels/cast_f32_to_i32.wgsl"),
        )
    }

    fn cast_f32_to_i64(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_f32_to_i64",
            include_str!("kernels/cast_f32_to_i64.wgsl"),
        )
    }

    fn cast_i32_to_f32(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_i32_to_f32",
            include_str!("kernels/cast_i32_to_f32.wgsl"),
        )
    }

    fn cast_i64_to_f32(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_i64_to_f32",
            include_str!("kernels/cast_i64_to_f32.wgsl"),
        )
    }

    fn cast_i32_to_i64(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_i32_to_i64",
            include_str!("kernels/cast_i32_to_i64.wgsl"),
        )
    }

    fn cast_i64_to_i32(&mut self, input_handle: BufferHandle, output_handle: BufferHandle, n: usize) -> Result<()> {
        cast_op(
            self,
            input_handle,
            output_handle,
            n,
            "cast_i64_to_i32",
            include_str!("kernels/cast_i64_to_i32.wgsl"),
        )
    }

    // ============================================================================================
    // High-Level Operations (Indexing)
    // ============================================================================================

    fn gather_f32(
        &mut self,
        input_handle: BufferHandle,
        input_shape: &[usize],
        indices_handle: BufferHandle,
        axis: usize,
        output_handle: BufferHandle,
        output_shape: &[usize],
    ) -> Result<()> {
        gather_op(
            self,
            input_handle,
            input_shape,
            indices_handle,
            axis,
            output_handle,
            output_shape,
        )
    }
}

// WebGPU-specific async methods (not part of Backend trait)
#[cfg(target_arch = "wasm32")]
impl WebGpuBackend {
    /// Get cloneable Arc references for async operations without holding locks
    pub fn get_async_resources(
        &self,
    ) -> (
        Arc<Device>,
        Arc<Queue>,
        Arc<BufferPool>,
        Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
        Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
    ) {
        (
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            Arc::clone(&self.buffer_pool),
            Arc::clone(&self.buffers),
            Arc::clone(&self.pools),
        )
    }

    /// Standalone async buffer copy that doesn't require holding self
    /// This can be called with cloned Arc references, allowing locks to be released
    pub async fn copy_buffer_async_standalone(
        device: Arc<Device>,
        queue: Arc<Queue>,
        buffer_pool: Arc<BufferPool>,
        buffers: Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
        handle: BufferHandle,
        data: &mut [u8],
    ) -> Result<()> {
        // Handle empty buffers - no operation needed
        if data.is_empty() {
            return Ok(());
        }

        // Get the buffer (brief lock)
        let buffer = {
            let bufs = lock_mutex(&buffers);
            bufs.get(&handle.id())
                .ok_or_else(|| BackendError::InvalidBufferHandle(handle.id()))?
                .clone() // Clone the Arc
        }; // Lock released

        if data.len() > buffer.size() as usize {
            return Err(BackendError::Other(format!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                buffer.size()
            )));
        }

        // CRITICAL: Synchronize GPU queue before reading to ensure all prior writes have completed
        // WebGPU processes commands asynchronously - without this sync, reads can happen before
        // prior write operations finish, resulting in garbage data (uninitialized memory)
        let (sync_sender, sync_receiver) = futures_channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            let _ = sync_sender.send(());
        });
        sync_receiver
            .await
            .map_err(|_| BackendError::Other("Queue sync before read failed".into()))?;

        // WebGPU requires buffer sizes and copy sizes to be multiples of 4
        let padded_size = ((data.len() + 3) / 4) * 4;

        // Get staging buffer with padded size
        let staging_buffer = buffer_pool.acquire_staging(padded_size);

        // Copy GPU buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Buffer Copy (Async Standalone)"),
        });

        // WebGPU requires copy size to be a multiple of 4
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, padded_size as u64);
        queue.submit([encoder.finish()]);

        // CRITICAL: Wait for the copy command to complete before mapping
        // Without this, map_async can succeed while the GPU copy is still in flight
        let (copy_sync_sender, copy_sync_receiver) = futures_channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            let _ = copy_sync_sender.send(());
        });
        copy_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Queue sync after copy failed".into()))?;

        // Map the full padded buffer (WebGPU requires mapping size to be multiple of 4)
        let buffer_slice = staging_buffer.slice(..padded_size as u64);
        let (sender, receiver) = futures_channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await the mapping - yields to browser event loop properly
        receiver
            .await
            .map_err(|_| BackendError::Other("Mapping callback channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Buffer mapping failed: {:?}", e)))?;

        // Copy only actual data length from padded buffer
        {
            let mapped_data = buffer_slice.get_mapped_range();
            data.copy_from_slice(&mapped_data[..data.len()]);
        }

        staging_buffer.unmap();
        buffer_pool.release_staging(staging_buffer);

        Ok(())
    }

    /// Standalone async pool read that doesn't require `&self` - avoids holding locks
    ///
    /// This static method takes Arc references to avoid lock contention.
    /// The executor clones Arc references, drops locks, then calls this method.
    pub async fn copy_pool_async_standalone(
        device: Arc<Device>,
        queue: Arc<Queue>,
        buffer_pool: Arc<BufferPool>,
        pools: Arc<Mutex<HashMap<u64, Arc<WgpuBuffer>>>>,
        handle: PoolHandle,
        offset: usize,
        data: &mut [u8],
    ) -> Result<()> {
        // Get pool buffer with brief lock
        let buffer = {
            let pools_guard = lock_mutex(&pools);
            pools_guard
                .get(&handle.id())
                .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?
                .clone()
        }; // Lock released

        let pool_size = buffer.size() as usize;
        if offset + data.len() > pool_size {
            return Err(BackendError::Other(format!(
                "Pool access out of bounds: offset={}, size={}, pool_size={}",
                offset,
                data.len(),
                pool_size
            )));
        }

        // CRITICAL: Synchronize GPU queue before reading to ensure all prior writes have completed
        let (sync_sender, sync_receiver) = futures_channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            let _ = sync_sender.send(());
        });
        sync_receiver
            .await
            .map_err(|_| BackendError::Other("Pool queue sync before read failed".into()))?;

        // WebGPU requires buffer sizes and copy sizes to be multiples of 4
        let padded_size = ((data.len() + 3) / 4) * 4;

        // Get staging buffer with padded size
        let staging_buffer = buffer_pool.acquire_staging(padded_size);

        // Copy pool buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pool Copy (Async Standalone)"),
        });

        // WebGPU requires copy size to be a multiple of 4
        encoder.copy_buffer_to_buffer(&buffer, offset as u64, &staging_buffer, 0, padded_size as u64);
        queue.submit([encoder.finish()]);

        // CRITICAL: Wait for the copy command to complete before mapping
        let (copy_sync_sender, copy_sync_receiver) = futures_channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            let _ = copy_sync_sender.send(());
        });
        copy_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Queue sync after pool copy failed".into()))?;

        // Map the full padded buffer (WebGPU requires mapping size to be multiple of 4)
        let buffer_slice = staging_buffer.slice(..padded_size as u64);
        let (sender, receiver) = futures_channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await mapping callback
        receiver
            .await
            .map_err(|_| BackendError::Other("Pool mapping callback channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Pool mapping failed: {:?}", e)))?;

        // Copy only actual data length from padded buffer
        {
            let mapped_data = buffer_slice.get_mapped_range();
            data.copy_from_slice(&mapped_data[..data.len()]);
        }

        staging_buffer.unmap();
        buffer_pool.release_staging(staging_buffer);

        Ok(())
    }

    /// Async version of copy_to_buffer that properly yields to browser event loop
    ///
    /// This method uses wasm-bindgen-futures to avoid blocking the JavaScript main thread.
    /// Use this instead of copy_to_buffer() when calling from async WASM code to ensure
    /// writes complete before subsequent operations read the data.
    pub async fn copy_to_buffer_async_impl(&mut self, handle: BufferHandle, data: &[u8]) -> Result<()> {
        // Handle empty buffers - no operation needed
        if data.is_empty() {
            return Ok(());
        }

        let buffer = self.get_gpu_buffer(handle)?;

        // Validate size
        if data.len() > buffer.size() as usize {
            return Err(BackendError::BufferOutOfBounds {
                offset: 0,
                size: data.len(),
                buffer_size: buffer.size() as usize,
            });
        }

        // Use staging buffer approach for reliable writes
        // Create staging buffer with MAP_WRITE (for CPU writes) and COPY_SRC (for GPU copy)
        // WebGPU requires buffer sizes to be multiples of 4 when mapped_at_creation is true
        let padded_size = ((data.len() + 3) / 4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Write Staging (Temp)"),
            size: padded_size as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Write data to the mapped staging buffer
        // Only slice the actual data length, not the padded buffer size
        staging
            .slice(..data.len() as u64)
            .get_mapped_range_mut()
            .copy_from_slice(data);

        // Unmap to make the data available to GPU
        staging.unmap();

        // Copy from staging to target buffer
        // WebGPU requires copy size to be a multiple of 4
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Buffer Write (Async)"),
        });
        encoder.copy_buffer_to_buffer(&staging, 0, &buffer, 0, padded_size as u64);

        self.queue.submit([encoder.finish()]);

        // CRITICAL: Wait for write command to be processed before verification
        let (write_sync_sender, write_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = write_sync_sender.send(());
        });
        write_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Write sync channel closed".into()))?;

        // CRITICAL: To ensure GPU write completes, we perform a small verification read
        // The map_async() callback will only fire after GPU processes all prior commands
        // This is the ONLY reliable way to synchronize writes in WebGPU WASM

        // Create small verification staging buffer (just 4 bytes)
        let verify_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Write Verification Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from target buffer to verification staging (must be multiple of 4)
        // This copy command depends on the write completing
        let verify_size = padded_size.min(4) as u64;
        let mut verify_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Write Verification"),
        });
        verify_encoder.copy_buffer_to_buffer(&buffer, 0, &verify_staging, 0, verify_size);

        self.queue.submit([verify_encoder.finish()]);

        // CRITICAL: Wait for verification copy to complete before mapping
        let (verify_sync_sender, verify_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = verify_sync_sender.send(());
        });
        verify_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Verification sync channel closed".into()))?;

        // Map the verification buffer - this callback fires ONLY after GPU processes both commands
        let verify_slice = verify_staging.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();

        verify_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await mapping - this properly yields to browser event loop and waits for GPU
        receiver
            .await
            .map_err(|_| BackendError::Other("Write verification channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Write verification failed: {:?}", e)))?;

        // Unmap verification buffer (we don't need the data, just the synchronization)
        verify_staging.unmap();

        Ok(())
    }

    /// Async version of copy_to_pool with proper GPU synchronization
    ///
    /// This method uses the same dual synchronization pattern as copy_to_buffer_async_impl
    /// to ensure pool writes complete before subsequent reads.
    pub async fn copy_to_pool_async_impl(&mut self, handle: PoolHandle, offset: usize, data: &[u8]) -> Result<()> {
        // Handle empty writes
        if data.is_empty() {
            return Ok(());
        }

        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;

        // Validate bounds
        if offset + data.len() > buffer.size() as usize {
            return Err(BackendError::PoolOutOfBounds {
                offset,
                size: data.len(),
                pool_size: buffer.size() as usize,
            });
        }

        // Create staging buffer for write
        // WebGPU requires buffer sizes to be multiples of 4 when mapped_at_creation is true
        let padded_size = ((data.len() + 3) / 4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Write Staging (Temp)"),
            size: padded_size as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Write data to staging buffer
        // Only slice the actual data length, not the padded buffer size
        staging
            .slice(..data.len() as u64)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        staging.unmap();

        // Copy from staging to pool at offset
        // WebGPU requires copy size to be a multiple of 4
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pool Write (Async)"),
        });
        encoder.copy_buffer_to_buffer(&staging, 0, &buffer, offset as u64, padded_size as u64);

        self.queue.submit([encoder.finish()]);

        // CRITICAL: Wait for write command to be processed
        let (write_sync_sender, write_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = write_sync_sender.send(());
        });
        write_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Pool write sync channel closed".into()))?;

        // CRITICAL: Perform verification read to ensure write completed
        let verify_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pool Write Verification Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy 4 bytes from pool buffer to verification staging
        let mut verify_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pool Write Verification"),
        });
        verify_encoder.copy_buffer_to_buffer(&buffer, offset as u64, &verify_staging, 0, 4.min(data.len() as u64));

        self.queue.submit([verify_encoder.finish()]);

        // CRITICAL: Wait for verification copy to complete before mapping
        let (verify_sync_sender, verify_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = verify_sync_sender.send(());
        });
        verify_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Pool verification sync channel closed".into()))?;

        // Map verification buffer
        let verify_slice = verify_staging.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();

        verify_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await mapping - ensures GPU has processed all commands
        receiver
            .await
            .map_err(|_| BackendError::Other("Pool write verification channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Pool write verification failed: {:?}", e)))?;

        // Unmap verification buffer
        verify_staging.unmap();

        Ok(())
    }

    /// Async version of copy_from_buffer that properly yields to browser event loop
    ///
    /// This method uses wasm-bindgen-futures to avoid blocking the JavaScript main thread.
    /// Use this instead of copy_from_buffer() when calling from async WASM code.
    pub async fn copy_from_buffer_async_impl(&self, handle: BufferHandle, data: &mut [u8]) -> Result<()> {
        // Handle empty buffers - no operation needed
        if data.is_empty() {
            return Ok(());
        }

        let buffers = lock_mutex(&self.buffers);
        let buffer = buffers
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidBufferHandle(handle.id()))?;

        if data.len() > buffer.size() as usize {
            return Err(BackendError::Other(format!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                buffer.size()
            )));
        }

        drop(buffers); // Release lock before async operation

        // CRITICAL: Synchronize GPU queue before reading
        let (sync_sender, sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sync_sender.send(());
        });
        sync_receiver
            .await
            .map_err(|_| BackendError::Other("Buffer queue sync before read failed".into()))?;

        // WebGPU requires buffer sizes and copy sizes to be multiples of 4
        let padded_size = ((data.len() + 3) / 4) * 4;

        // Get staging buffer (returns Arc<Buffer>, not Result)
        let staging_buffer = self.buffer_pool.acquire_staging(padded_size);

        // Re-acquire lock briefly to get buffer reference
        let buffers = lock_mutex(&self.buffers);
        let buffer = buffers
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidBufferHandle(handle.id()))?;

        // Copy GPU buffer to staging buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Buffer Copy (Async)"),
        });

        // WebGPU requires copy size to be a multiple of 4
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, padded_size as u64);

        drop(buffers); // Release lock before submit

        self.queue.submit([encoder.finish()]);

        // CRITICAL: Wait for the copy command to complete before mapping
        let (copy_sync_sender, copy_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = copy_sync_sender.send(());
        });
        copy_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Queue sync after buffer copy failed".into()))?;

        // Map the full padded buffer (WebGPU requires mapping size to be multiple of 4)
        let buffer_slice = staging_buffer.slice(..padded_size as u64);

        // Create a channel for the mapping operation
        let (sender, receiver) = futures_channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await the mapping callback - properly yields to browser event loop
        // wgpu will invoke the callback asynchronously through the JavaScript event loop
        receiver
            .await
            .map_err(|_| BackendError::Other("Mapping callback channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Buffer mapping failed: {:?}", e)))?;

        // Copy only actual data length from padded buffer
        {
            let mapped_data = buffer_slice.get_mapped_range();
            data.copy_from_slice(&mapped_data[..data.len()]);
        }

        // Unmap and return to pool (staging_buffer is already Arc<Buffer>)
        staging_buffer.unmap();
        self.buffer_pool.release_staging(staging_buffer);

        Ok(())
    }

    /// Async version of copy_from_pool that properly yields to browser event loop
    pub async fn copy_from_pool_async(&self, handle: PoolHandle, offset: usize, data: &mut [u8]) -> Result<()> {
        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;

        let pool_size = buffer.size() as usize;
        if offset + data.len() > pool_size {
            return Err(BackendError::Other(format!(
                "Pool access out of bounds: offset={}, size={}, pool_size={}",
                offset,
                data.len(),
                pool_size
            )));
        }

        drop(pools); // Release lock before async operation

        // CRITICAL: Synchronize GPU queue before reading
        let (sync_sender, sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sync_sender.send(());
        });
        sync_receiver
            .await
            .map_err(|_| BackendError::Other("Pool queue sync before read failed".into()))?;

        // WebGPU requires buffer sizes and copy sizes to be multiples of 4
        let padded_size = ((data.len() + 3) / 4) * 4;

        // Get staging buffer with padded size
        let staging_buffer = self.buffer_pool.acquire_staging(padded_size);

        // Re-acquire lock briefly to get buffer reference
        let pools = lock_mutex(&self.pools);
        let buffer = pools
            .get(&handle.id())
            .ok_or_else(|| BackendError::InvalidPoolHandle(handle.id()))?;

        // Copy pool buffer to staging
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pool Copy (Async)"),
        });

        // WebGPU requires copy size to be a multiple of 4
        encoder.copy_buffer_to_buffer(buffer, offset as u64, &staging_buffer, 0, padded_size as u64);

        drop(pools); // Release lock before submit

        self.queue.submit([encoder.finish()]);

        // CRITICAL: Wait for the copy command to complete before mapping
        let (copy_sync_sender, copy_sync_receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = copy_sync_sender.send(());
        });
        copy_sync_receiver
            .await
            .map_err(|_| BackendError::Other("Queue sync after pool copy failed".into()))?;

        // Map the full padded buffer (WebGPU requires mapping size to be multiple of 4)
        let buffer_slice = staging_buffer.slice(..padded_size as u64);
        let (sender, receiver) = futures_channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Await mapping callback - properly yields to browser event loop
        // wgpu will invoke the callback asynchronously through the JavaScript event loop
        receiver
            .await
            .map_err(|_| BackendError::Other("Pool mapping callback channel closed".into()))?
            .map_err(|e| BackendError::Other(format!("Pool mapping failed: {:?}", e)))?;

        // Copy only actual data length from padded buffer
        {
            let mapped_data = buffer_slice.get_mapped_range();
            data.copy_from_slice(&mapped_data[..data.len()]);
        }

        staging_buffer.unmap();
        self.buffer_pool.release_staging(staging_buffer);

        Ok(())
    }
}

// Stub implementation for non-WASM targets
#[cfg(not(target_arch = "wasm32"))]
pub struct WebGpuBackend;

#[cfg(not(target_arch = "wasm32"))]
impl WebGpuBackend {
    pub async fn new() -> Result<Self> {
        Err(BackendError::BackendUnavailable(
            "WebGPU backend only available on WASM target".into(),
        ))
    }
}

/// Execute high-level instruction inline (Conv2d, NearestUpsample2d)
#[cfg(target_arch = "wasm32")]
fn execute_high_level_instruction_impl(backend: &mut WebGpuBackend, program: &Program) -> Result<()> {
    use crate::isa::Instruction;

    // Extract buffer handles from MOV_IMM instructions
    let mut buffer_handles: std::collections::HashMap<u8, u64> = std::collections::HashMap::new();
    for instruction in &program.instructions {
        if let Instruction::MOV_IMM { dst, value, .. } = instruction {
            buffer_handles.insert(dst.index(), *value);
        }
    }

    // Find and execute the high-level instruction
    for instruction in &program.instructions {
        match instruction {
            Instruction::Conv2d {
                input,
                weights,
                bias,
                output,
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                dilation_h,
                dilation_w,
                group,
                has_bias,
                ..
            } => {
                // Get buffer handles
                let input_handle = buffer_handles
                    .get(&input.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(input.index() as u64))?;
                let weights_handle = buffer_handles
                    .get(&weights.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(weights.index() as u64))?;
                let bias_handle = buffer_handles
                    .get(&bias.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(bias.index() as u64))?;
                let output_handle = buffer_handles
                    .get(&output.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(output.index() as u64))?;

                // Get GPU buffers
                let input_buffer = backend.get_gpu_buffer(BufferHandle::new(input_handle))?;
                let weights_buffer = backend.get_gpu_buffer(BufferHandle::new(weights_handle))?;
                let bias_buffer = backend.get_gpu_buffer(BufferHandle::new(bias_handle))?;
                let output_buffer = backend.get_gpu_buffer(BufferHandle::new(output_handle))?;

                // Calculate output dimensions with full ONNX spec
                let out_height = (in_height + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
                let out_width = (in_width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

                // Create config uniform with full ONNX spec parameters
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
                    has_bias: u32,
                    _pad: u32,
                }

                let config = ConvConfig {
                    batch_size: *batch_size,
                    in_channels: *in_channels,
                    in_height: *in_height,
                    in_width: *in_width,
                    out_channels: *out_channels,
                    out_height,
                    out_width,
                    kernel_h: *kernel_h,
                    kernel_w: *kernel_w,
                    stride_h: *stride_h,
                    stride_w: *stride_w,
                    pad_top: *pad_top,
                    pad_left: *pad_left,
                    pad_bottom: *pad_bottom,
                    pad_right: *pad_right,
                    dilation_h: *dilation_h,
                    dilation_w: *dilation_w,
                    group: *group,
                    has_bias: *has_bias,
                    _pad: 0,
                };

                let config_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Conv2d Config"),
                    size: std::mem::size_of::<ConvConfig>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                backend
                    .queue
                    .write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

                let pipeline = backend
                    .pipeline_cache
                    .get_or_create("conv2d", include_str!("kernels/conv2d.wgsl"), "main")
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to create conv2d pipeline: {}", e)))?;

                let bind_group_layout = pipeline.get_bind_group_layout(0);
                let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Conv2d Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: weights_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: bias_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Conv2d Encoder"),
                });

                // Updated dispatch for 1D workgroup layout
                // Each thread computes ONE output element
                let total_outputs = batch_size * out_channels * out_height * out_width;
                let workgroup_size = 256;
                let num_workgroups = (total_outputs + workgroup_size - 1) / workgroup_size;

                // WebGPU limit: 65,535 workgroups per dimension
                // If we exceed this, split across Y dimension
                let (dispatch_x, dispatch_y) = if num_workgroups <= 65535 {
                    (num_workgroups, 1)
                } else {
                    // Split across Y dimension
                    let dispatch_y = (num_workgroups + 65535 - 1) / 65535;
                    let dispatch_x = 65535;
                    (dispatch_x, dispatch_y)
                };

                tracing::info!(
                    "Conv2d: {}x{}x{}x{} -> {}x{}x{}x{}, total_outputs={}, workgroups={}, dispatch=({}, {}, 1)",
                    batch_size,
                    in_channels,
                    in_height,
                    in_width,
                    batch_size,
                    out_channels,
                    out_height,
                    out_width,
                    total_outputs,
                    num_workgroups,
                    dispatch_x,
                    dispatch_y
                );

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Conv2d Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                backend.queue.submit([encoder.finish()]);

                tracing::debug!(
                    "Conv2d completed: {}x{}x{}x{} -> {}x{}x{}x{}, total_outputs={}, workgroups={}",
                    batch_size,
                    in_channels,
                    in_height,
                    in_width,
                    batch_size,
                    out_channels,
                    out_height,
                    out_width,
                    total_outputs,
                    num_workgroups
                );

                return Ok(());
            }
            Instruction::NearestUpsample2d {
                input,
                output,
                batch_size,
                num_channels,
                in_height,
                in_width,
                scale_factor,
                ..
            } => {
                let input_handle = buffer_handles
                    .get(&input.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(input.index() as u64))?;
                let output_handle = buffer_handles
                    .get(&output.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(output.index() as u64))?;

                let input_buffer = backend.get_gpu_buffer(BufferHandle::new(input_handle))?;
                let output_buffer = backend.get_gpu_buffer(BufferHandle::new(output_handle))?;

                let out_height = in_height * scale_factor;
                let out_width = in_width * scale_factor;

                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct UpsampleConfig {
                    batch_size: u32,
                    num_channels: u32,
                    in_height: u32,
                    in_width: u32,
                    out_height: u32,
                    out_width: u32,
                    scale_factor: u32,
                    _pad: u32,
                }

                let config = UpsampleConfig {
                    batch_size: *batch_size,
                    num_channels: *num_channels,
                    in_height: *in_height,
                    in_width: *in_width,
                    out_height,
                    out_width,
                    scale_factor: *scale_factor,
                    _pad: 0,
                };

                let config_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Upsample Config"),
                    size: std::mem::size_of::<UpsampleConfig>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                backend
                    .queue
                    .write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

                let pipeline = backend
                    .pipeline_cache
                    .get_or_create(
                        "nearest_upsample_2d",
                        include_str!("kernels/nearest_upsample_2d.wgsl"),
                        "main",
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to create upsample pipeline: {}", e)))?;

                let bind_group_layout = pipeline.get_bind_group_layout(0);
                let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Upsample Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Upsample Encoder"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Upsample Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let total_outputs = batch_size * num_channels * out_height * out_width;
                    let num_workgroups = (total_outputs + 255) / 256;
                    let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
                }

                backend.queue.submit([encoder.finish()]);

                tracing::debug!(
                    "NearestUpsample2d executed on WebGPU: {}x{}x{}x{} -> {}x{}x{}x{}",
                    batch_size,
                    num_channels,
                    in_height,
                    in_width,
                    batch_size,
                    num_channels,
                    out_height,
                    out_width
                );

                return Ok(());
            }
            Instruction::GroupNorm {
                input,
                gamma,
                beta,
                output,
                batch_size,
                num_channels,
                height,
                width,
                num_groups,
                eps,
                ..
            } => {
                let input_handle = buffer_handles
                    .get(&input.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(input.index() as u64))?;
                let gamma_handle = buffer_handles
                    .get(&gamma.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(gamma.index() as u64))?;
                let beta_handle = buffer_handles
                    .get(&beta.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(beta.index() as u64))?;
                let output_handle = buffer_handles
                    .get(&output.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(output.index() as u64))?;

                let input_buffer = backend.get_gpu_buffer(BufferHandle::new(input_handle))?;
                let gamma_buffer = backend.get_gpu_buffer(BufferHandle::new(gamma_handle))?;
                let beta_buffer = backend.get_gpu_buffer(BufferHandle::new(beta_handle))?;
                let output_buffer = backend.get_gpu_buffer(BufferHandle::new(output_handle))?;

                let spatial_size = height * width;
                let channels_per_group = num_channels / num_groups;
                let group_size = channels_per_group * spatial_size;

                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct GroupNormConfig {
                    batch_size: u32,
                    num_channels: u32,
                    height: u32,
                    width: u32,
                    num_groups: u32,
                    channels_per_group: u32,
                    spatial_size: u32,
                    group_size: u32,
                    eps: f32,
                    _pad1: u32,
                    _pad2: u32,
                    _pad3: u32,
                }

                let config = GroupNormConfig {
                    batch_size: *batch_size,
                    num_channels: *num_channels,
                    height: *height,
                    width: *width,
                    num_groups: *num_groups,
                    channels_per_group,
                    spatial_size,
                    group_size,
                    eps: *eps,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                };

                let config_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GroupNorm Config"),
                    size: std::mem::size_of::<GroupNormConfig>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                backend
                    .queue
                    .write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

                let pipeline = backend
                    .pipeline_cache
                    .get_or_create("group_norm", include_str!("kernels/group_norm.wgsl"), "main")
                    .map_err(|e| {
                        BackendError::ExecutionError(format!("Failed to create group_norm pipeline: {}", e))
                    })?;

                let bind_group_layout = pipeline.get_bind_group_layout(0);
                let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GroupNorm Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: gamma_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: beta_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GroupNorm Encoder"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("GroupNorm Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    let num_workgroups = batch_size * num_groups;

                    tracing::info!(
                        "GroupNorm: batch={}, channels={}, H={}x{}, groups={}, workgroups={}",
                        batch_size,
                        num_channels,
                        height,
                        width,
                        num_groups,
                        num_workgroups
                    );

                    compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                }

                backend.queue.submit([encoder.finish()]);

                tracing::debug!(
                    "GroupNorm executed on WebGPU: batch={}, channels={}, spatial={}x{}, groups={}",
                    batch_size,
                    num_channels,
                    height,
                    width,
                    num_groups
                );

                return Ok(());
            }
            Instruction::Gemm {
                matrix_a,
                matrix_b,
                matrix_c,
                m,
                k,
                n,
                ..
            } => {
                // Get buffer handles from register mappings
                let matrix_a_handle = buffer_handles
                    .get(&matrix_a.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(matrix_a.index() as u64))?;
                let matrix_b_handle = buffer_handles
                    .get(&matrix_b.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(matrix_b.index() as u64))?;
                let matrix_c_handle = buffer_handles
                    .get(&matrix_c.index())
                    .copied()
                    .ok_or_else(|| BackendError::InvalidBufferHandle(matrix_c.index() as u64))?;

                // Get GPU buffers
                let matrix_a_buffer = backend.get_gpu_buffer(BufferHandle::new(matrix_a_handle))?;
                let matrix_b_buffer = backend.get_gpu_buffer(BufferHandle::new(matrix_b_handle))?;
                let matrix_c_buffer = backend.get_gpu_buffer(BufferHandle::new(matrix_c_handle))?;

                // Create config uniform for GEMM
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct GemmConfig {
                    m: u32,
                    k: u32,
                    n: u32,
                    _pad: u32, // Padding for 16-byte alignment
                }

                let config = GemmConfig {
                    m: *m,
                    k: *k,
                    n: *n,
                    _pad: 0,
                };

                // Create uniform buffer for config
                let config_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GEMM Config"),
                    size: std::mem::size_of::<GemmConfig>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                backend
                    .queue
                    .write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

                // Get or compile GEMM shader pipeline
                let pipeline = backend
                    .pipeline_cache
                    .get_or_create("gemm", include_str!("kernels/gemm.wgsl"), "main")
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to create GEMM pipeline: {}", e)))?;

                // Create bind group with config and matrix buffers
                let bind_group_layout = pipeline.get_bind_group_layout(0);
                let bind_group = backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GEMM Bind Group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: matrix_a_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: matrix_b_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: matrix_c_buffer.as_entire_binding(),
                        },
                    ],
                });

                // Create command encoder and compute pass
                let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GEMM Encoder"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("GEMM Pass"),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Calculate dispatch dimensions
                    // Workgroup size is 16x16, so we need (M/16) x (N/16) workgroups
                    const TILE_SIZE: u32 = 16;
                    let workgroups_x = (*m + TILE_SIZE - 1) / TILE_SIZE;
                    let workgroups_y = (*n + TILE_SIZE - 1) / TILE_SIZE;

                    tracing::info!(
                        "GEMM: M={}, K={}, N={}, workgroups={}x{}",
                        m,
                        k,
                        n,
                        workgroups_x,
                        workgroups_y
                    );

                    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
                }

                // Submit command buffer
                backend.queue.submit([encoder.finish()]);

                tracing::debug!(
                    "GEMM executed on WebGPU: [{}, {}] × [{}, {}] → [{}, {}]",
                    m,
                    k,
                    k,
                    n,
                    m,
                    n
                );

                return Ok(());
            }
            Instruction::LayerNorm { .. } => {
                return Err(BackendError::UnsupportedOperation(
                    "LayerNorm not yet implemented for WebGPU backend".to_string(),
                ));
            }
            Instruction::MOV_IMM { .. } => continue,
            _ => {
                return Err(BackendError::execution_error(format!(
                    "Cannot mix high-level operations with low-level ISA instructions: {:?}",
                    instruction
                )));
            }
        }
    }

    Err(BackendError::execution_error(
        "Program contains only MOV_IMM instructions with no actual operation",
    ))
}
