//! CPU backend implementation
//!
//! Reference implementation of the Backend trait for CPU execution.
//! Provides sequential and parallel execution of Atlas ISA programs.
//!
//! # Architecture
//!
//! ```text
//! CpuBackend
//! ├── RegisterFile   - 256 registers + 16 predicates
//! ├── MemoryManager  - Buffers + pools + shared memory
//! ├── Executor       - Instruction dispatch and execution
//! └── Parallel       - Rayon-based grid/block execution
//! ```
//!
//! # Usage
//!
//! ```rust
//! use hologram_backends::{CpuBackend, Backend, Program};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut backend = CpuBackend::new();
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

mod boundary_pool;
mod executor_impl;
pub(crate) mod memory;

use crate::backend::{Backend, BufferHandle, LaunchConfig, PoolHandle};
use crate::backends::common::simd;
pub use crate::backends::common::RegisterFile;
use crate::error::Result;
use crate::isa::Program;
use executor_impl::CpuExecutor;
use memory::MemoryManager;
use std::sync::Arc;

/// CPU backend for executing Atlas ISA programs
///
/// This is the reference implementation of the Backend trait.
/// It executes programs on the CPU using:
/// - Sequential instruction execution per lane
/// - Rayon-based parallel execution across blocks
/// - Lock-free per-buffer memory management via DashMap
#[derive(Clone)]
pub struct CpuBackend {
    /// Memory manager (buffers, pools, shared memory)
    /// Per-buffer locking via DashMap - no outer lock for maximum parallelism
    memory: Arc<MemoryManager>,
}

impl CpuBackend {
    /// Create a new CPU backend
    ///
    /// # Example
    ///
    /// ```rust
    /// use hologram_backends::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self {
            memory: Arc::new(MemoryManager::new()),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Recognized operation types
enum Operation {
    BinaryOp {
        handle_b: BufferHandle,
        kernel: unsafe fn(*const f32, *const f32, *mut f32, usize),
    },
    UnaryOp {
        kernel: unsafe fn(*const f32, *mut f32, usize),
    },
}

/// Binary operation types
enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operation types
enum UnaryOpType {
    Abs,
    Neg,
    Relu,
}

impl Backend for CpuBackend {
    fn execute_program(&mut self, program: &Program, config: &LaunchConfig) -> Result<()> {
        // Create executor and delegate execution
        let executor = CpuExecutor::new(Arc::clone(&self.memory));
        executor.execute(program, config)
    }

    fn execute_program_with_params(
        &mut self,
        program: &Program,
        params: &crate::backend::ExecutionParams,
    ) -> Result<()> {
        // Try to recognize and execute as optimized SIMD kernel
        if let Some(result) = self.try_execute_simd_kernel(program, params) {
            return result;
        }

        // Fall back to ISA interpretation
        let executor = CpuExecutor::new(Arc::clone(&self.memory));
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

impl CpuBackend {
    /// Try to recognize and execute program as optimized SIMD kernel
    ///
    /// Recognizes precompiled operations (ADD, SUB, MUL, etc.) and dispatches
    /// directly to SIMD kernels, bypassing ISA interpretation.
    ///
    /// Returns Some(Result) if recognized and executed, None if not recognized.
    fn try_execute_simd_kernel(
        &mut self,
        program: &Program,
        params: &crate::backend::ExecutionParams,
    ) -> Option<Result<()>> {
        // Get element count from R4 (standard convention for precompiled ops)
        let n = *params.initial_registers.get(&crate::isa::Register::new(4))? as usize;

        // Check if this is a small workload (≤ 262K elements, f32)
        if n > 262_144 {
            return None; // Too large for inline SIMD, use ISA fallback
        }

        // Recognize operation by ISA pattern
        // Precompiled operations have specific instruction sequences
        let operation = Self::recognize_operation_with_params(program, params)?;

        // Get buffer handles from registers (R1=input_a, R2=input_b, R3=output)
        let handle_a = BufferHandle(*params.initial_registers.get(&crate::isa::Register::new(1))?);
        let handle_c = BufferHandle(*params.initial_registers.get(&crate::isa::Register::new(3))?);

        match operation {
            Operation::BinaryOp { handle_b, kernel } => {
                // Get pointers from memory manager (zero-copy)
                let ptr_a = self.memory.buffer_as_ptr(handle_a).ok()? as *const f32;
                let ptr_b = self.memory.buffer_as_ptr(handle_b).ok()? as *const f32;
                let ptr_c = self.memory.buffer_as_mut_ptr(handle_c).ok()? as *mut f32;

                // Execute SIMD kernel directly
                unsafe { kernel(ptr_a, ptr_b, ptr_c, n) };
                Some(Ok(()))
            }
            Operation::UnaryOp { kernel } => {
                // Get pointers from memory manager (zero-copy)
                let ptr_a = self.memory.buffer_as_ptr(handle_a).ok()? as *const f32;
                let ptr_c = self.memory.buffer_as_mut_ptr(handle_c).ok()? as *mut f32;

                // Execute SIMD kernel directly
                unsafe { kernel(ptr_a, ptr_c, n) };
                Some(Ok(()))
            }
        }
    }

    /// Recognize precompiled operation from ISA pattern
    fn recognize_operation_with_params(
        program: &Program,
        params: &crate::backend::ExecutionParams,
    ) -> Option<Operation> {
        // Check instruction count (precompiled ops have specific sizes)
        let inst_count = program.instructions.len();

        // Binary operations (ADD, SUB, MUL, DIV) have ~12-15 instructions
        if (10..=20).contains(&inst_count) {
            // Check for binary operation pattern: LDG, LDG, OP, STG
            if let Some(op_type) = Self::detect_binary_op_type(program) {
                // Extract handle_b from R2
                let handle_b = BufferHandle(*params.initial_registers.get(&crate::isa::Register::new(2))?);

                return Some(match op_type {
                    BinaryOpType::Add => Operation::BinaryOp {
                        handle_b,
                        kernel: simd::vector_add_f32,
                    },
                    BinaryOpType::Sub => Operation::BinaryOp {
                        handle_b,
                        kernel: simd::vector_sub_f32,
                    },
                    BinaryOpType::Mul => Operation::BinaryOp {
                        handle_b,
                        kernel: simd::vector_mul_f32,
                    },
                    BinaryOpType::Div => Operation::BinaryOp {
                        handle_b,
                        kernel: simd::vector_div_f32,
                    },
                });
            }
        }

        // Unary operations (ABS, NEG, RELU) have ~8-12 instructions
        if (6..=15).contains(&inst_count) {
            if let Some(op_type) = Self::detect_unary_op_type(program) {
                return Some(match op_type {
                    UnaryOpType::Abs => Operation::UnaryOp {
                        kernel: simd::vector_abs_f32,
                    },
                    UnaryOpType::Neg => Operation::UnaryOp {
                        kernel: simd::vector_neg_f32,
                    },
                    UnaryOpType::Relu => Operation::UnaryOp {
                        kernel: simd::vector_relu_f32,
                    },
                });
            }
        }

        None // Not recognized
    }

    fn detect_binary_op_type(program: &Program) -> Option<BinaryOpType> {
        use crate::isa::Instruction;

        // Look for the arithmetic instruction in the program
        for inst in &program.instructions {
            match inst {
                Instruction::ADD { .. } => return Some(BinaryOpType::Add),
                Instruction::SUB { .. } => return Some(BinaryOpType::Sub),
                Instruction::MUL { .. } => return Some(BinaryOpType::Mul),
                Instruction::DIV { .. } => return Some(BinaryOpType::Div),
                _ => {}
            }
        }
        None
    }

    fn detect_unary_op_type(program: &Program) -> Option<UnaryOpType> {
        use crate::isa::Instruction;

        // Look for the operation type
        for inst in &program.instructions {
            match inst {
                Instruction::ABS { .. } => return Some(UnaryOpType::Abs),
                Instruction::NEG { .. } => return Some(UnaryOpType::Neg),
                // RELU is max(x, 0) - look for MAX instruction
                Instruction::MAX { .. } => return Some(UnaryOpType::Relu),
                _ => {}
            }
        }
        None
    }

    /// Get raw const pointer to buffer memory (for inline SIMD kernels)
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - The buffer handle is valid
    /// - No mutable operations occur on the buffer
    /// - The backend/memory manager is not dropped
    pub fn get_buffer_ptr(&self, handle: BufferHandle) -> Result<*const u8> {
        self.memory.buffer_as_ptr(handle)
    }

    /// Get raw mutable pointer to buffer memory (for inline SIMD kernels)
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - The buffer handle is valid
    /// - No concurrent access occurs
    /// - The backend/memory manager is not dropped
    pub fn get_buffer_mut_ptr(&mut self, handle: BufferHandle) -> Result<*mut u8> {
        self.memory.buffer_as_mut_ptr(handle)
    }

    /// Get raw const pointer to boundary pool class data (for inline SIMD kernels)
    ///
    /// # Arguments
    ///
    /// * `class` - Class index (0-95)
    ///
    /// # Returns
    ///
    /// Const pointer to the start of the class data (12,288 bytes)
    pub fn get_boundary_class_ptr(&self, class: u8) -> Result<*const u8> {
        self.memory.boundary_class_ptr(class)
    }

    /// Get raw mutable pointer to boundary pool class data (for inline SIMD kernels)
    ///
    /// # Arguments
    ///
    /// * `class` - Class index (0-95)
    ///
    /// # Returns
    ///
    /// Mutable pointer to the start of the class data (12,288 bytes)
    pub fn get_boundary_class_mut_ptr(&mut self, class: u8) -> Result<*mut u8> {
        self.memory.boundary_class_ptr_mut(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert!(Arc::strong_count(&backend.memory) == 1);
    }

    #[test]
    fn test_cpu_backend_default() {
        let backend: CpuBackend = Default::default();
        assert!(Arc::strong_count(&backend.memory) == 1);
    }

    #[test]
    fn test_cpu_backend_buffer_allocation() {
        let mut backend = CpuBackend::new();

        let buffer = backend.allocate_buffer(1024).unwrap();
        assert_eq!(backend.buffer_size(buffer).unwrap(), 1024);

        backend.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_cpu_backend_pool_allocation() {
        let mut backend = CpuBackend::new();

        let pool = backend.allocate_pool(4096).unwrap();
        assert_eq!(backend.pool_size(pool).unwrap(), 4096);

        backend.free_pool(pool).unwrap();
    }

    #[test]
    fn test_cpu_backend_buffer_copy() {
        let mut backend = CpuBackend::new();

        let buffer = backend.allocate_buffer(16).unwrap();

        let data = b"Hello, World!";
        backend.copy_to_buffer(buffer, data).unwrap();

        let mut result = vec![0u8; data.len()];
        backend.copy_from_buffer(buffer, &mut result).unwrap();

        assert_eq!(result, data);

        backend.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_cpu_backend_pool_copy() {
        let mut backend = CpuBackend::new();

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
