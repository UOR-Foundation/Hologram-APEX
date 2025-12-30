//! WebGPU executor for compute operations
//!
//! Provides high-level execution interface for hologram operations
//! using WebGPU compute shaders.

use super::buffer::HybridBuffer;
use super::device::WebGpuDevice;
use super::pipeline::PipelineCache;
use std::sync::Arc;

/// WebGPU executor for compute operations
///
/// Manages compute pipeline execution, buffer management, and
/// synchronization for GPU-accelerated operations.
///
/// # Example
///
/// ```rust,no_run
/// use hologram_backends::backends::wasm::webgpu::{WebGpuDevice, WebGpuExecutor};
///
/// # async fn example() -> Result<(), String> {
/// // Initialize device
/// let device = WebGpuDevice::new().await?;
///
/// // Create executor
/// let mut executor = WebGpuExecutor::new(device)?;
///
/// // Execute vector addition
/// let a = vec![1.0f32, 2.0, 3.0, 4.0];
/// let b = vec![5.0f32, 6.0, 7.0, 8.0];
/// let result = executor.vector_add(&a, &b).await?;
///
/// assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
/// # Ok(())
/// # }
/// ```
pub struct WebGpuExecutor {
    device: Arc<WebGpuDevice>,
    pipeline_cache: Arc<PipelineCache>,
}

/// Calculate dispatch dimensions for large workloads that may exceed per-dimension limits
///
/// WebGPU has a limit of 65,535 workgroups per dimension. This function splits
/// large workgroup counts across Y dimension when needed.
///
/// # Arguments
///
/// * `total_workgroups` - Total number of workgroups needed
///
/// # Returns
///
/// (x, y, z) tuple for dispatch_workgroups call
fn calculate_dispatch_size(total_workgroups: u32) -> (u32, u32, u32) {
    const MAX_DISPATCH_PER_DIM: u32 = 65535;

    if total_workgroups <= MAX_DISPATCH_PER_DIM {
        (total_workgroups, 1, 1)
    } else {
        // Split across Y dimension
        let x = MAX_DISPATCH_PER_DIM;
        let y = (total_workgroups + MAX_DISPATCH_PER_DIM - 1) / MAX_DISPATCH_PER_DIM;
        (x, y, 1)
    }
}

impl WebGpuExecutor {
    /// Create a new WebGPU executor
    ///
    /// # Arguments
    ///
    /// * `device` - Initialized WebGPU device
    pub fn new(device: WebGpuDevice) -> Result<Self, String> {
        let device_arc = Arc::new(device);
        let pipeline_cache = Arc::new(PipelineCache::new(Arc::clone(device_arc.device())));

        Ok(Self {
            device: device_arc,
            pipeline_cache,
        })
    }

    /// Execute vector addition (MergeRange with Add variant)
    ///
    /// Corresponds to: `merge@c[0..N](Add)`
    ///
    /// # Arguments
    ///
    /// * `a` - First input vector
    /// * `b` - Second input vector
    ///
    /// # Returns
    ///
    /// Vector containing element-wise sum: `output[i] = a[i] + b[i]`
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Input vectors have different lengths
    /// - GPU operation fails
    pub async fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Input vectors must have same length (got {} and {})",
                a.len(),
                b.len()
            ));
        }

        let n = a.len();
        let byte_size = std::mem::size_of_val(a);

        // Create hybrid buffers
        let mut buffer_a = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_b = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_out = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;

        // Write input data to CPU buffers
        buffer_a.write_cpu(a)?;
        buffer_b.write_cpu(b)?;

        // Sync to GPU
        buffer_a.sync_to_gpu()?;
        buffer_b.sync_to_gpu()?;
        buffer_out.sync_to_gpu()?;

        // Get or compile pipeline
        let pipeline =
            self.pipeline_cache
                .get_or_create("vector_add", include_str!("kernels/vector_add.wgsl"), "main")?;

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vector Add Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_out.gpu_buffer().as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute
        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vector Add Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vector Add Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (n as u32).div_ceil(workgroup_size);
            let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        // Submit commands
        self.device.queue().submit([encoder.finish()]);

        // Mark output buffer as GPU dirty
        buffer_out.mark_gpu_dirty();

        // Sync result back to CPU
        buffer_out.sync_to_cpu().await?;

        // Read result
        let result = buffer_out.read_cpu::<f32>()?.to_vec();

        Ok(result)
    }

    /// Execute vector multiplication (MergeRange with Mul variant)
    ///
    /// Corresponds to: `merge@c[0..N](Mul)`
    ///
    /// # Arguments
    ///
    /// * `a` - First input vector
    /// * `b` - Second input vector
    ///
    /// # Returns
    ///
    /// Vector containing element-wise product: `output[i] = a[i] * b[i]`
    pub async fn vector_mul(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Input vectors must have same length (got {} and {})",
                a.len(),
                b.len()
            ));
        }

        let n = a.len();
        let byte_size = std::mem::size_of_val(a);

        // Create hybrid buffers
        let mut buffer_a = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_b = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_out = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;

        // Write input data
        buffer_a.write_cpu(a)?;
        buffer_b.write_cpu(b)?;

        // Sync to GPU
        buffer_a.sync_to_gpu()?;
        buffer_b.sync_to_gpu()?;
        buffer_out.sync_to_gpu()?;

        // Get or compile pipeline
        let pipeline =
            self.pipeline_cache
                .get_or_create("vector_mul", include_str!("kernels/vector_mul.wgsl"), "main")?;

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vector Mul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_out.gpu_buffer().as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vector Mul Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vector Mul Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (n as u32).div_ceil(workgroup_size);
            let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        self.device.queue().submit([encoder.finish()]);

        // Sync result back
        buffer_out.mark_gpu_dirty();
        buffer_out.sync_to_cpu().await?;

        Ok(buffer_out.read_cpu::<f32>()?.to_vec())
    }

    /// Execute vector subtraction (SplitRange with Sub variant)
    ///
    /// Corresponds to: `split@c[0..N](Sub)`
    pub async fn vector_sub(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_binary_op("vector_sub", include_str!("kernels/vector_sub.wgsl"), a, b)
            .await
    }

    /// Execute vector division (SplitRange with Div variant)
    ///
    /// Corresponds to: `split@c[0..N](Div)`
    pub async fn vector_div(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_binary_op("vector_div", include_str!("kernels/vector_div.wgsl"), a, b)
            .await
    }

    /// Execute vector minimum (MergeRange with Min variant)
    ///
    /// Corresponds to: `merge@c[0..N](Min)`
    pub async fn vector_min(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_binary_op("vector_min", include_str!("kernels/vector_min.wgsl"), a, b)
            .await
    }

    /// Execute vector maximum (MergeRange with Max variant)
    ///
    /// Corresponds to: `merge@c[0..N](Max)`
    pub async fn vector_max(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_binary_op("vector_max", include_str!("kernels/vector_max.wgsl"), a, b)
            .await
    }

    /// Execute vector absolute value (MergeRange with Abs variant)
    ///
    /// Corresponds to: `merge@c[0..N](Abs)`
    pub async fn vector_abs(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_abs", include_str!("kernels/vector_abs.wgsl"), input)
            .await
    }

    /// Execute vector exponential (MergeRange with Exp variant)
    ///
    /// Corresponds to: `merge@c[0..N](Exp)`
    pub async fn vector_exp(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_exp", include_str!("kernels/vector_exp.wgsl"), input)
            .await
    }

    /// Execute vector natural logarithm (MergeRange with Log variant)
    ///
    /// Corresponds to: `merge@c[0..N](Log)`
    pub async fn vector_log(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_log", include_str!("kernels/vector_log.wgsl"), input)
            .await
    }

    /// Execute vector square root (MergeRange with Sqrt variant)
    ///
    /// Corresponds to: `merge@c[0..N](Sqrt)`
    pub async fn vector_sqrt(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_sqrt", include_str!("kernels/vector_sqrt.wgsl"), input)
            .await
    }

    /// Execute vector sigmoid activation (MergeRange with Sigmoid variant)
    ///
    /// Corresponds to: `merge@c[0..N](Sigmoid)`
    pub async fn vector_sigmoid(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_sigmoid", include_str!("kernels/vector_sigmoid.wgsl"), input)
            .await
    }

    /// Execute vector hyperbolic tangent (MergeRange with Tanh variant)
    ///
    /// Corresponds to: `merge@c[0..N](Tanh)`
    pub async fn vector_tanh(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        self.execute_unary_op("vector_tanh", include_str!("kernels/vector_tanh.wgsl"), input)
            .await
    }

    /// Helper: Execute binary operation (a op b)
    async fn execute_binary_op(
        &mut self,
        shader_name: &str,
        shader_source: &str,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Input vectors must have same length (got {} and {})",
                a.len(),
                b.len()
            ));
        }

        let n = a.len();
        let byte_size = std::mem::size_of_val(a);

        // Create hybrid buffers
        let mut buffer_a = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_b = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_out = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;

        // Write and sync
        buffer_a.write_cpu(a)?;
        buffer_b.write_cpu(b)?;
        buffer_a.sync_to_gpu()?;
        buffer_b.sync_to_gpu()?;
        buffer_out.sync_to_gpu()?;

        // Get pipeline
        let pipeline = self.pipeline_cache.get_or_create(shader_name, shader_source, "main")?;

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", shader_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_out.gpu_buffer().as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
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
            let num_workgroups = (n as u32).div_ceil(workgroup_size);
            let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        self.device.queue().submit([encoder.finish()]);

        // Sync result back
        buffer_out.mark_gpu_dirty();
        buffer_out.sync_to_cpu().await?;

        Ok(buffer_out.read_cpu::<f32>()?.to_vec())
    }

    /// Helper: Execute unary operation (op input)
    async fn execute_unary_op(
        &mut self,
        shader_name: &str,
        shader_source: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let n = input.len();
        let byte_size = std::mem::size_of_val(input);

        // Create hybrid buffers
        let mut buffer_in = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;
        let mut buffer_out = HybridBuffer::new(
            Arc::clone(self.device.device()),
            Arc::clone(self.device.queue()),
            byte_size,
        )?;

        // Write and sync
        buffer_in.write_cpu(input)?;
        buffer_in.sync_to_gpu()?;
        buffer_out.sync_to_gpu()?;

        // Get pipeline
        let pipeline = self.pipeline_cache.get_or_create(shader_name, shader_source, "main")?;

        // Create bind group (unary operations have 2 bindings, not 3)
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", shader_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_in.gpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_out.gpu_buffer().as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let mut encoder = self
            .device
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
            let num_workgroups = (n as u32).div_ceil(workgroup_size);
            let (dispatch_x, dispatch_y, dispatch_z) = calculate_dispatch_size(num_workgroups);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        self.device.queue().submit([encoder.finish()]);

        // Sync result back
        buffer_out.mark_gpu_dirty();
        buffer_out.sync_to_cpu().await?;

        Ok(buffer_out.read_cpu::<f32>()?.to_vec())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> super::pipeline::CacheStats {
        self.pipeline_cache.stats()
    }

    /// Clear pipeline cache
    pub fn clear_cache(&self) {
        self.pipeline_cache.clear();
    }
}

impl std::fmt::Debug for WebGpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuExecutor")
            .field("device", &self.device)
            .field("cache_stats", &self.cache_stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a WASM environment with WebGPU support
    // They are primarily integration tests that run in wasm-bindgen-test

    #[test]
    fn test_executor_creation_requires_device() {
        // This test just verifies the type signature
        // Actual device creation requires async WASM environment
    }
}
