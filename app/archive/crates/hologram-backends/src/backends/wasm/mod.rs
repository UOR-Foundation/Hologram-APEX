//! WASM backend implementation
//!
//! Implementation of the Backend trait for WebAssembly execution.
//! Provides SIMD-accelerated execution using WASM v128 instructions.
//!
//! # Architecture
//!
//! ```text
//! WasmBackend
//! ├── RegisterFile   - 256 registers + 16 predicates
//! ├── MemoryManager  - Linear memory buffers + pools
//! ├── Executor       - Instruction dispatch with SIMD
//! └── SIMD Support   - v128 instructions for vectorization
//! ```
//!
//! # Usage
//!
//! ```rust
//! use hologram_backends::{WasmBackend, Backend, Program};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut backend = WasmBackend::new();
//!
//! // Allocate buffer
//! let buffer = backend.allocate_buffer(1024)?;
//!
//! // Execute program
//! let program = Program::new();
//! let config = Default::default();
//! backend.execute_program(&program, &config)?;
//!
//! backend.free_buffer(buffer)?;
//! # Ok(())
//! # }
//! ```

mod executor_impl;
pub(crate) mod memory;
pub mod simd;
pub mod webgpu;

use crate::backend::{Backend, BufferHandle, LaunchConfig, PoolHandle};
pub use crate::backends::common::RegisterFile;
use crate::error::Result;
use crate::isa::Program;
use executor_impl::WasmExecutor;
use memory::MemoryManager;
use std::sync::Arc;

// Re-export WebGPU backend (WASM GPU acceleration)
#[cfg(feature = "webgpu")]
pub use webgpu::WebGpuBackend;

/// WASM backend for executing Atlas ISA programs
///
/// This implementation executes programs in WebAssembly using:
/// - WASM v128 SIMD instructions for vectorization
/// - Linear memory for buffers and pools
/// - Sequential instruction execution per lane
#[derive(Clone)]
pub struct WasmBackend {
    /// Memory manager (buffers, pools, shared memory)
    /// Uses Arc for shared ownership in wasm-bindgen context
    memory: Arc<MemoryManager>,
}

impl WasmBackend {
    /// Create a new WASM backend
    ///
    /// # Example
    ///
    /// ```rust
    /// use hologram_backends::WasmBackend;
    ///
    /// let backend = WasmBackend::new();
    /// ```
    pub fn new() -> Self {
        Self {
            memory: Arc::new(MemoryManager::new()),
        }
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for WasmBackend {
    fn execute_program(&mut self, program: &Program, config: &LaunchConfig) -> Result<()> {
        // Create executor and delegate execution
        let executor = WasmExecutor::new(Arc::clone(&self.memory));
        executor.execute(program, config)
    }

    fn execute_program_with_params(
        &mut self,
        program: &Program,
        params: &crate::backend::ExecutionParams,
    ) -> Result<()> {
        // Create executor and delegate to execute_with_params
        // This initializes custom registers (R1, R2, R3, etc.) before execution
        let executor = WasmExecutor::new(Arc::clone(&self.memory));
        executor.execute_with_params(program, params)
    }

    fn allocate_buffer(&mut self, size: usize) -> Result<BufferHandle> {
        self.memory.allocate_buffer(size)
    }

    fn free_buffer(&mut self, handle: BufferHandle) -> Result<()> {
        self.memory.free_buffer(handle)
    }

    fn copy_to_buffer(&mut self, handle: BufferHandle, data: &[u8]) -> Result<()> {
        self.memory.copy_to_buffer(handle, data)
    }

    fn copy_from_buffer(&mut self, handle: BufferHandle, data: &mut [u8]) -> Result<()> {
        self.memory.copy_from_buffer(handle, data)
    }

    fn buffer_size(&self, handle: BufferHandle) -> Result<usize> {
        self.memory.buffer_size(handle)
    }

    fn allocate_pool(&mut self, size: usize) -> Result<PoolHandle> {
        self.memory.allocate_pool(size)
    }

    fn free_pool(&mut self, handle: PoolHandle) -> Result<()> {
        self.memory.free_pool(handle)
    }

    fn copy_to_pool(&mut self, handle: PoolHandle, offset: usize, data: &[u8]) -> Result<()> {
        self.memory.copy_to_pool(handle, offset, data)
    }

    fn copy_from_pool(&mut self, handle: PoolHandle, offset: usize, data: &mut [u8]) -> Result<()> {
        self.memory.copy_from_pool(handle, offset, data)
    }

    fn pool_size(&self, handle: PoolHandle) -> Result<usize> {
        self.memory.pool_size(handle)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_backend_creation() {
        let backend = WasmBackend::new();
        assert!(Arc::strong_count(&backend.memory) == 1);
    }

    #[test]
    fn test_wasm_backend_default() {
        let backend: WasmBackend = Default::default();
        assert!(Arc::strong_count(&backend.memory) == 1);
    }

    #[test]
    fn test_wasm_backend_buffer_allocation() {
        let mut backend = WasmBackend::new();

        let buffer = backend.allocate_buffer(1024).unwrap();
        assert_eq!(backend.buffer_size(buffer).unwrap(), 1024);

        backend.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_wasm_backend_pool_allocation() {
        let mut backend = WasmBackend::new();

        let pool = backend.allocate_pool(4096).unwrap();
        assert_eq!(backend.pool_size(pool).unwrap(), 4096);

        backend.free_pool(pool).unwrap();
    }

    #[test]
    fn test_wasm_backend_buffer_copy() {
        let mut backend = WasmBackend::new();

        let buffer = backend.allocate_buffer(16).unwrap();

        let data = b"Hello, WASM!";
        backend.copy_to_buffer(buffer, data).unwrap();

        let mut result = vec![0u8; data.len()];
        backend.copy_from_buffer(buffer, &mut result).unwrap();

        assert_eq!(result, data);

        backend.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_wasm_backend_pool_copy() {
        let mut backend = WasmBackend::new();

        let pool = backend.allocate_pool(1024).unwrap();

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = bytemuck::cast_slice(&data);
        backend.copy_to_pool(pool, 0, bytes).unwrap();

        let mut result = [0.0f32; 4];
        let result_bytes = bytemuck::cast_slice_mut(&mut result);
        backend.copy_from_pool(pool, 0, result_bytes).unwrap();

        assert_eq!(result, data);

        backend.free_pool(pool).unwrap();
    }
}
