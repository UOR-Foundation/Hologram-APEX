//! Hybrid buffer management for CPU-GPU synchronization
//!
//! Maintains buffers in both WASM linear memory (CPU) and WebGPU buffers (GPU),
//! tracking synchronization state and providing efficient data transfer.

use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device, Queue};

/// Buffer synchronization state
///
/// Tracks which side (CPU or GPU) has the most recent data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// CPU buffer modified, GPU buffer needs update
    CpuDirty,
    /// GPU buffer modified, CPU buffer needs update
    GpuDirty,
    /// Both buffers in sync
    Synced,
    /// Buffer not yet initialized
    Uninitialized,
}

/// Hybrid buffer maintaining data in both CPU and GPU memory
///
/// Automatically tracks synchronization state and provides methods
/// for efficient data transfer between CPU and GPU.
///
/// # Example
///
/// ```rust,no_run
/// use hologram_backends::backends::wasm::webgpu::buffer::HybridBuffer;
/// use std::sync::Arc;
///
/// # async fn example(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<(), String> {
/// // Create hybrid buffer
/// let mut buffer = HybridBuffer::new(device, queue, 1024)?;
///
/// // Write data on CPU side
/// buffer.write_cpu(&[1.0f32, 2.0, 3.0, 4.0])?;
///
/// // Sync to GPU before compute
/// buffer.sync_to_gpu()?;
///
/// // After GPU compute, sync back to CPU
/// buffer.sync_to_cpu().await?;
///
/// // Read data from CPU side
/// let data = buffer.read_cpu::<f32>()?;
/// # Ok(())
/// # }
/// ```
pub struct HybridBuffer {
    /// CPU-side buffer (WASM linear memory)
    cpu_buffer: Vec<u8>,

    /// GPU-side buffer (wgpu Buffer)
    gpu_buffer: Buffer,

    /// Synchronization state
    state: BufferState,

    /// WebGPU device for buffer operations
    device: Arc<Device>,

    /// WebGPU queue for data transfers
    queue: Arc<Queue>,
}

impl HybridBuffer {
    /// Create a new hybrid buffer
    ///
    /// Allocates buffers on both CPU and GPU with the specified size.
    ///
    /// # Arguments
    ///
    /// * `device` - WebGPU device for GPU buffer creation
    /// * `queue` - WebGPU queue for data transfers
    /// * `size` - Buffer size in bytes
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU buffer allocation fails
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, size: usize) -> Result<Self, String> {
        // Allocate CPU buffer
        let cpu_buffer = vec![0u8; size];

        // Allocate GPU buffer with appropriate usage flags
        let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hybrid Buffer"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            cpu_buffer,
            gpu_buffer,
            state: BufferState::Uninitialized,
            device,
            queue,
        })
    }

    /// Write data to CPU buffer
    ///
    /// Marks the buffer as `CpuDirty` after write.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data to write (type must implement `bytemuck::Pod`)
    ///
    /// # Errors
    ///
    /// Returns `Err` if data size exceeds buffer capacity
    pub fn write_cpu<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<(), String> {
        let bytes = bytemuck::cast_slice(data);

        if bytes.len() > self.cpu_buffer.len() {
            return Err(format!(
                "Data size ({} bytes) exceeds buffer capacity ({} bytes)",
                bytes.len(),
                self.cpu_buffer.len()
            ));
        }

        self.cpu_buffer[..bytes.len()].copy_from_slice(bytes);
        self.state = BufferState::CpuDirty;

        Ok(())
    }

    /// Read data from CPU buffer
    ///
    /// If buffer is `GpuDirty`, automatically syncs from GPU first.
    ///
    /// # Returns
    ///
    /// Returns a slice view of the CPU buffer as type `T`
    ///
    /// # Errors
    ///
    /// Returns `Err` if buffer size is not a multiple of `sizeof(T)`
    pub fn read_cpu<T: bytemuck::Pod>(&self) -> Result<&[T], String> {
        if !self.cpu_buffer.len().is_multiple_of(std::mem::size_of::<T>()) {
            return Err(format!(
                "Buffer size ({}) is not a multiple of type size ({})",
                self.cpu_buffer.len(),
                std::mem::size_of::<T>()
            ));
        }

        Ok(bytemuck::cast_slice(&self.cpu_buffer))
    }

    /// Synchronize data from CPU to GPU
    ///
    /// If buffer is `CpuDirty`, uploads data to GPU and marks as `Synced`.
    /// If already synced or GPU dirty, does nothing.
    pub fn sync_to_gpu(&mut self) -> Result<(), String> {
        if self.state == BufferState::CpuDirty || self.state == BufferState::Uninitialized {
            // Upload data to GPU
            self.queue.write_buffer(&self.gpu_buffer, 0, &self.cpu_buffer);
            self.state = BufferState::Synced;

            tracing::trace!("Synced buffer to GPU ({} bytes)", self.cpu_buffer.len());
        }

        Ok(())
    }

    /// Synchronize data from GPU to CPU
    ///
    /// If buffer is `GpuDirty`, downloads data from GPU and marks as `Synced`.
    /// This is an async operation requiring command submission.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU readback fails
    pub async fn sync_to_cpu(&mut self) -> Result<(), String> {
        if self.state == BufferState::GpuDirty {
            // Create staging buffer for readback
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: self.cpu_buffer.len() as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy GPU buffer to staging buffer
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer Copy Encoder"),
            });

            encoder.copy_buffer_to_buffer(&self.gpu_buffer, 0, &staging_buffer, 0, self.cpu_buffer.len() as u64);

            self.queue.submit([encoder.finish()]);

            // CRITICAL: Wait for the copy command to complete before mapping
            let (copy_sync_sender, copy_sync_receiver) = futures_channel::oneshot::channel();
            self.queue.on_submitted_work_done(move || {
                let _ = copy_sync_sender.send(());
            });
            copy_sync_receiver
                .await
                .map_err(|_| "Queue sync after buffer copy failed".to_string())?;

            // Map staging buffer and read data
            let buffer_slice = staging_buffer.slice(..);
            let (tx, rx) = futures_channel::oneshot::channel();

            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });

            // Poll device until mapping completes
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // Wait for mapping
            rx.await
                .map_err(|_| "Channel closed".to_string())?
                .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

            // Copy data to CPU buffer
            {
                let data = buffer_slice.get_mapped_range();
                self.cpu_buffer.copy_from_slice(&data);
            }

            // Unmap buffer
            staging_buffer.unmap();

            self.state = BufferState::Synced;

            tracing::trace!("Synced buffer from GPU ({} bytes)", self.cpu_buffer.len());
        }

        Ok(())
    }

    /// Mark buffer as GPU dirty
    ///
    /// Call this after GPU compute operations that modify the buffer
    pub fn mark_gpu_dirty(&mut self) {
        self.state = BufferState::GpuDirty;
    }

    /// Get reference to GPU buffer
    pub fn gpu_buffer(&self) -> &Buffer {
        &self.gpu_buffer
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.cpu_buffer.len()
    }

    /// Get current synchronization state
    pub fn state(&self) -> BufferState {
        self.state
    }
}

impl std::fmt::Debug for HybridBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridBuffer")
            .field("size", &self.cpu_buffer.len())
            .field("state", &self.state)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_state_equality() {
        assert_eq!(BufferState::Synced, BufferState::Synced);
        assert_ne!(BufferState::CpuDirty, BufferState::GpuDirty);
    }

    #[test]
    fn test_buffer_state_debug() {
        let state = BufferState::CpuDirty;
        let debug_str = format!("{:?}", state);
        assert_eq!(debug_str, "CpuDirty");
    }
}
