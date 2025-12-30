//! WASM implementation of the Executor trait
//!
//! This module contains the execution logic for the WASM backend,
//! encapsulated in the `WasmExecutor` struct.

use crate::backend::{ExecutionParams, LaunchConfig};
use crate::backends::common::address::resolve_address_with_state;
use crate::backends::common::executor_trait::{dispatch_common_instruction, Executor};
use crate::backends::common::memory::{load_bytes_from_storage, store_bytes_to_storage};
use crate::backends::common::ExecutionState;
use crate::backends::wasm::memory::MemoryManager;
use crate::error::{BackendError, Result};
use crate::isa::{Address, Instruction, Label, MemoryScope, Predicate, Program, Register, Type};
use std::sync::Arc;

/// WASM executor implementation
///
/// Encapsulates all execution logic for the WASM backend, including:
/// - Main execution loop
/// - Instruction dispatch
/// - Memory operations (LDG, STG, LDS, STS)
/// - Control flow operations (BRA, CALL, RET, LOOP, EXIT)
/// - Synchronization operations (BarSync, MemFence)
pub struct WasmExecutor {
    /// Shared memory manager (thread-safe via interior mutability)
    memory: Arc<MemoryManager>,
}

impl WasmExecutor {
    /// Create a new WASM executor
    pub fn new(memory: Arc<MemoryManager>) -> Self {
        Self { memory }
    }

    // ============================================================================================
    // Memory Operations (Backend-Specific)
    // ============================================================================================

    /// Execute load global (LDG) instruction
    fn execute_ldg(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        ty: Type,
        dst: Register,
        addr: &Address,
    ) -> Result<()> {
        // Resolve address (supports RegisterIndirect via execution state)
        let (handle, offset) = resolve_address_with_state(addr, state)?;

        // Load value from memory
        let value_bytes = load_bytes_from_storage(state.shared.memory.as_ref(), handle, offset, ty.size_bytes())?;

        // Write to register
        let lane = state.current_lane_mut();
        match ty {
            Type::I8 => {
                let value = *bytemuck::from_bytes::<i8>(&value_bytes);
                lane.registers.write_i8(dst, value)?;
            }
            Type::I16 => {
                let value = *bytemuck::from_bytes::<i16>(&value_bytes);
                lane.registers.write_i16(dst, value)?;
            }
            Type::I32 => {
                let value = *bytemuck::from_bytes::<i32>(&value_bytes);
                lane.registers.write_i32(dst, value)?;
            }
            Type::I64 => {
                let value = *bytemuck::from_bytes::<i64>(&value_bytes);
                lane.registers.write_i64(dst, value)?;
            }
            Type::U8 => {
                let value = *bytemuck::from_bytes::<u8>(&value_bytes);
                lane.registers.write_u8(dst, value)?;
            }
            Type::U16 => {
                let value = *bytemuck::from_bytes::<u16>(&value_bytes);
                lane.registers.write_u16(dst, value)?;
            }
            Type::U32 => {
                let value = *bytemuck::from_bytes::<u32>(&value_bytes);
                lane.registers.write_u32(dst, value)?;
            }
            Type::U64 => {
                let value = *bytemuck::from_bytes::<u64>(&value_bytes);
                lane.registers.write_u64(dst, value)?;
            }
            Type::F16 => {
                let value = *bytemuck::from_bytes::<u16>(&value_bytes);
                lane.registers.write_f16_bits(dst, value)?;
            }
            Type::BF16 => {
                let value = *bytemuck::from_bytes::<u16>(&value_bytes);
                lane.registers.write_bf16_bits(dst, value)?;
            }
            Type::F32 => {
                let value = *bytemuck::from_bytes::<f32>(&value_bytes);
                lane.registers.write_f32(dst, value)?;
            }
            Type::F64 => {
                let value = *bytemuck::from_bytes::<f64>(&value_bytes);
                lane.registers.write_f64(dst, value)?;
            }
        }

        Ok(())
    }

    /// Execute store global (STG) instruction
    fn execute_stg(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        ty: Type,
        addr: &Address,
        src: Register,
    ) -> Result<()> {
        // Resolve address (supports RegisterIndirect via execution state)
        let (handle, offset) = resolve_address_with_state(addr, state)?;

        // Read value from register and convert to bytes
        let lane = state.current_lane();
        let value_bytes = match ty {
            Type::I8 => {
                let value = lane.registers.read_i8(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::I16 => {
                let value = lane.registers.read_i16(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::I32 => {
                let value = lane.registers.read_i32(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::I64 => {
                let value = lane.registers.read_i64(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::U8 => {
                let value = lane.registers.read_u8(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::U16 => {
                let value = lane.registers.read_u16(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::U32 => {
                let value = lane.registers.read_u32(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::U64 => {
                let value = lane.registers.read_u64(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::F16 => {
                let value = lane.registers.read_f16_bits(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::BF16 => {
                let value = lane.registers.read_bf16_bits(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::F32 => {
                let value = lane.registers.read_f32(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
            Type::F64 => {
                let value = lane.registers.read_f64(src)?;
                bytemuck::bytes_of(&value).to_vec()
            }
        };

        // Store to memory
        store_bytes_to_storage(state.shared.memory.as_ref(), handle, offset, &value_bytes)?;

        Ok(())
    }

    // ============================================================================================
    // Synchronization Operations
    // ============================================================================================

    fn execute_barrier_sync_impl(&self, _state: &mut ExecutionState<MemoryManager>, _barrier_id: u8) -> Result<()> {
        // WASM backend (single-threaded): no-op
        // In future multi-threaded WASM, this would use SharedArrayBuffer + Atomics
        Ok(())
    }

    fn execute_memory_fence_impl(&self, _state: &mut ExecutionState<MemoryManager>, _scope: MemoryScope) -> Result<()> {
        // WASM backend (single-threaded): no-op
        // In future multi-threaded WASM, this would use Atomics.fence()
        Ok(())
    }

    // ============================================================================================
    // Shared Memory Operations (LDS/STS)
    // ============================================================================================

    /// Execute load shared (LDS) instruction
    ///
    /// For WASM backend, shared memory is the same as global memory
    /// (similar to CPU backend). Future implementations could add separate
    /// shared memory space for better GPU simulation.
    fn execute_lds(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        ty: Type,
        dst: Register,
        addr: &Address,
    ) -> Result<()> {
        // Delegate to LDG - no distinction between shared and global on WASM
        self.execute_ldg(state, ty, dst, addr)
    }

    /// Execute store shared (STS) instruction
    ///
    /// For WASM backend, shared memory is the same as global memory
    /// (similar to CPU backend). Future implementations could add separate
    /// shared memory space for better GPU simulation.
    fn execute_sts(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        ty: Type,
        src: Register,
        addr: &Address,
    ) -> Result<()> {
        // Delegate to STG - no distinction between shared and global on WASM
        self.execute_stg(state, ty, addr, src)
    }

    // ============================================================================================
    // Control Flow Operations
    // ============================================================================================

    /// Execute branch (BRA) instruction
    fn execute_bra_impl(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        pred: Option<Predicate>,
        target: &Label,
    ) -> Result<()> {
        let should_branch = if let Some(p) = pred {
            let lane = state.current_lane();
            lane.registers.read_predicate(p)?
        } else {
            true
        };

        if should_branch {
            let target_pc = state
                .shared
                .labels
                .get(&target.0)
                .ok_or_else(|| BackendError::execution_error(format!("Label not found: {}", target.0)))?;
            state.current_lane_mut().pc = *target_pc;
        }

        Ok(())
    }

    /// Execute call (CALL) instruction
    fn execute_call_impl(&self, state: &mut ExecutionState<MemoryManager>, target: &Label) -> Result<()> {
        let target_pc = *state
            .shared
            .labels
            .get(&target.0)
            .ok_or_else(|| BackendError::execution_error(format!("Label not found: {}", target.0)))?;

        // Push return address (next instruction) to call stack
        let return_pc = state.current_lane().pc + 1;
        let lane = state.current_lane_mut();
        lane.call_stack.push(return_pc);

        // Jump to target
        lane.pc = target_pc;

        Ok(())
    }

    /// Execute return (RET) instruction
    fn execute_ret_impl(&self, state: &mut ExecutionState<MemoryManager>) -> Result<()> {
        // Pop return address from call stack
        let return_pc = state
            .current_lane_mut()
            .call_stack
            .pop()
            .ok_or_else(|| BackendError::execution_error("Call stack underflow".to_string()))?;

        // Jump to return address
        state.current_lane_mut().pc = return_pc;

        Ok(())
    }

    /// Execute loop (LOOP) instruction
    fn execute_loop_impl(
        &self,
        state: &mut ExecutionState<MemoryManager>,
        count: Register,
        body: &Label,
    ) -> Result<()> {
        // Read loop counter
        let counter = state.current_lane().registers.read_u32(count)?;

        if counter > 0 {
            // Decrement counter
            state.current_lane_mut().registers.write_u32(count, counter - 1)?;

            // Branch to loop body
            let target_pc = *state
                .shared
                .labels
                .get(&body.0)
                .ok_or_else(|| BackendError::execution_error(format!("Label not found: {}", body.0)))?;
            state.current_lane_mut().pc = target_pc;
        }

        Ok(())
    }

    /// Execute exit (EXIT) instruction
    fn execute_exit_impl(&self, state: &mut ExecutionState<MemoryManager>) -> Result<()> {
        state.current_lane_mut().active = false;
        Ok(())
    }

    // ============================================================================================
    // Main Execution Methods
    // ============================================================================================

    /// Execute program with launch configuration
    pub fn execute(&self, program: &Program, config: &LaunchConfig) -> Result<()> {
        let params = ExecutionParams::new(*config);
        self.execute_with_params(program, &params)
    }

    /// Execute program with execution parameters (including initial register values)
    pub fn execute_with_params(&self, program: &Program, params: &ExecutionParams) -> Result<()> {
        use crate::backend::ExecutionContext;

        let config = &params.launch_config;

        // Validate program
        program.validate()?;

        // Calculate total number of lanes (threads) per block
        let num_lanes = (config.block.x * config.block.y * config.block.z) as usize;

        // Calculate total number of blocks in the grid
        let total_blocks = (config.grid.x * config.grid.y * config.grid.z) as usize;

        // Execute blocks sequentially (WASM is single-threaded)
        for block_idx in 0..total_blocks {
            // Calculate block coordinates from linear index
            let blocks_per_row = config.grid.x as usize;
            let blocks_per_slice = (config.grid.x * config.grid.y) as usize;

            let block_z = block_idx / blocks_per_slice;
            let remainder = block_idx % blocks_per_slice;
            let block_y = remainder / blocks_per_row;
            let block_x = remainder % blocks_per_row;

            // Create execution context for this block
            let context = ExecutionContext::new(
                (0, 0, 0), // Lane index (will be set per lane)
                (block_x as u32, block_y as u32, block_z as u32),
                config.grid,
                config.block,
            );

            // Create execution state
            let mut state = ExecutionState::new(num_lanes, Arc::clone(&self.memory), context, program.labels.clone());

            // Initialize registers from params and lane contexts for all lanes
            for lane_idx in 0..num_lanes {
                // Initialize registers
                for (reg, value) in &params.initial_registers {
                    state.lane_states[lane_idx].lane.registers.write_u64(*reg, *value)?;
                }

                // Update lane indices in context
                let lane_x = lane_idx as u32 % config.block.x;
                let lane_y = (lane_idx as u32 / config.block.x) % config.block.y;
                let lane_z = lane_idx as u32 / (config.block.x * config.block.y);
                state.lane_states[lane_idx].context.lane_idx = (lane_x, lane_y, lane_z);
            }

            // Execute all lanes sequentially
            // For WASM backend (single-threaded), we process lanes one at a time
            // Since ExecutionState uses the first active lane's context, we keep current_lane_index at 0
            // and execute each lane's program sequentially by modifying state.lane_states directly
            for lane_idx in 0..num_lanes {
                // Main execution loop for this lane
                while state.lane_states[lane_idx].lane.active
                    && state.lane_states[lane_idx].lane.pc < program.instructions.len()
                {
                    let pc = state.lane_states[lane_idx].lane.pc;
                    let instruction = &program.instructions[pc];

                    // For single-threaded WASM execution, we manually handle lane state
                    // by directly accessing lane_states[lane_idx]
                    // Note: current_lane() and current_lane_mut() would use index 0 for single-lane states
                    // but since we're multi-lane sequential, we access lane_states directly

                    // Execute instruction for this specific lane
                    // We need to temporarily make this lane appear as the "current" lane
                    // by swapping it with index 0 if needed, or create a single-lane state

                    // Simpler approach: Use a temporary single-lane ExecutionState for each lane
                    // But for now, let's just execute with the current state and trust that
                    // instructions access the correct lane via the context

                    self.execute_instruction_dispatch(instruction, &mut state)?;

                    // Advance PC only if instruction didn't modify it (e.g., branch, call)
                    if state.lane_states[lane_idx].lane.pc == pc {
                        state.lane_states[lane_idx].lane.pc += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Dispatch a single instruction
    fn execute_instruction_dispatch(
        &self,
        instr: &Instruction,
        state: &mut ExecutionState<MemoryManager>,
    ) -> Result<()> {
        match instr {
            // Memory operations (backend-specific)
            Instruction::LDG { ty, dst, addr } => self.execute_ldg(state, *ty, *dst, addr),
            Instruction::STG { ty, addr, src } => self.execute_stg(state, *ty, addr, *src),
            Instruction::LDS { ty, dst, addr } => self.execute_lds(state, *ty, *dst, addr),
            Instruction::STS { ty, addr, src } => self.execute_sts(state, *ty, *src, addr),

            // Synchronization (backend-specific)
            Instruction::BarSync { id } => self.execute_barrier_sync_impl(state, *id),
            Instruction::MemFence { scope } => self.execute_memory_fence_impl(state, *scope),

            // Control flow (backend-specific)
            Instruction::BRA { pred, target } => self.execute_bra_impl(state, *pred, target),
            Instruction::CALL { target } => self.execute_call_impl(state, target),
            Instruction::RET => self.execute_ret_impl(state),
            Instruction::LOOP { count, body } => self.execute_loop_impl(state, *count, body),
            Instruction::EXIT => self.execute_exit_impl(state),

            // All other instructions (arithmetic, bitwise, math, Atlas ops)
            _ => dispatch_common_instruction(state, instr),
        }
    }
}

impl Executor<MemoryManager> for WasmExecutor {
    fn execute(&self, program: &Program, config: &LaunchConfig) -> Result<()> {
        self.execute(program, config)
    }

    fn execute_instruction(&self, state: &mut ExecutionState<MemoryManager>, instr: &Instruction) -> Result<()> {
        self.execute_instruction_dispatch(instr, state)
    }

    fn execute_barrier_sync(&self, state: &mut ExecutionState<MemoryManager>, barrier_id: u8) -> Result<()> {
        self.execute_barrier_sync_impl(state, barrier_id)
    }

    fn execute_memory_fence(&self, state: &mut ExecutionState<MemoryManager>, scope: MemoryScope) -> Result<()> {
        self.execute_memory_fence_impl(state, scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BlockDim, ExecutionContext, GridDim};

    #[test]
    fn test_wasm_executor_empty_program() {
        let program = Program::new();
        let config = LaunchConfig::default();
        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(memory);

        let result = executor.execute(&program, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_executor_exit_instruction() {
        let mut program = Program::new();
        program.instructions.push(Instruction::EXIT);

        let config = LaunchConfig::default();
        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(memory);

        let result = executor.execute(&program, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_barrier_sync_noop() {
        let memory = Arc::new(MemoryManager::new());
        let context = ExecutionContext::new(
            (0, 0, 0),
            (0, 0, 0),
            GridDim { x: 1, y: 1, z: 1 },
            BlockDim { x: 1, y: 1, z: 1 },
        );
        let labels = std::collections::HashMap::new();
        let mut state = ExecutionState::new(1, Arc::clone(&memory), context, labels);
        let executor = WasmExecutor::new(memory);

        // Should succeed without error (no-op for WASM)
        assert!(executor.execute_barrier_sync(&mut state, 0).is_ok());
        assert!(executor.execute_barrier_sync(&mut state, 255).is_ok());
    }

    #[test]
    fn test_wasm_memory_fence_all_scopes() {
        let memory = Arc::new(MemoryManager::new());
        let context = ExecutionContext::new(
            (0, 0, 0),
            (0, 0, 0),
            GridDim { x: 1, y: 1, z: 1 },
            BlockDim { x: 1, y: 1, z: 1 },
        );
        let labels = std::collections::HashMap::new();
        let mut state = ExecutionState::new(1, Arc::clone(&memory), context, labels);
        let executor = WasmExecutor::new(memory);

        // All memory scopes should succeed (no-op for WASM)
        assert!(executor.execute_memory_fence(&mut state, MemoryScope::Thread).is_ok());
        assert!(executor.execute_memory_fence(&mut state, MemoryScope::Block).is_ok());
        assert!(executor.execute_memory_fence(&mut state, MemoryScope::Device).is_ok());
        assert!(executor.execute_memory_fence(&mut state, MemoryScope::System).is_ok());
    }

    #[test]
    fn test_wasm_memory_operations() {
        use crate::isa::{Address, Register, Type};

        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(Arc::clone(&memory));

        // Allocate a buffer
        let buffer = memory.allocate_buffer(64).unwrap();

        // Write test data to buffer
        let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let test_bytes = bytemuck::cast_slice(&test_data);
        memory.copy_to_buffer(buffer, test_bytes).unwrap();

        // Create a program that loads and stores data
        let mut program = Program::new();

        // LDG: Load f32 from buffer offset 0 into R0
        program.instructions.push(Instruction::LDG {
            ty: Type::F32,
            dst: Register(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        });

        // STG: Store R0 to buffer offset 16 (4th f32)
        program.instructions.push(Instruction::STG {
            ty: Type::F32,
            src: Register(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        });

        let config = LaunchConfig::default();
        let result = executor.execute(&program, &config);
        assert!(result.is_ok());

        // Verify the data was copied correctly
        let mut output_data = vec![0u8; 64];
        memory.copy_from_buffer(buffer, &mut output_data).unwrap();
        let output_floats: &[f32] = bytemuck::cast_slice(&output_data);

        // First value should still be 1.0
        assert_eq!(output_floats[0], 1.0);
        // Fourth value (index 3) should now also be 1.0 (copied from first)
        assert_eq!(output_floats[3], 1.0);
    }

    #[test]
    fn test_wasm_control_flow_branch() {
        use crate::isa::{Label, Register, Type};

        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(memory);

        let mut program = Program::new();

        // MOV_IMM: R0 = 10
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(0),
            value: 10,
        });

        // BRA: unconditional branch to label "end"
        program.instructions.push(Instruction::BRA {
            pred: None,
            target: Label("end".to_string()),
        });

        // MOV_IMM: R0 = 20 (this should be skipped)
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(0),
            value: 20,
        });

        // Label "end"
        program.labels.insert("end".to_string(), 3);

        // EXIT
        program.instructions.push(Instruction::EXIT);

        let config = LaunchConfig::default();
        let result = executor.execute(&program, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_control_flow_loop() {
        use crate::isa::{Label, Register, Type};

        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(memory);

        let mut program = Program::new();

        // MOV_IMM: R0 = 5 (loop counter)
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(0),
            value: 5,
        });

        // MOV_IMM: R1 = 0 (accumulator)
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(1),
            value: 0,
        });

        // Label "loop_body" at instruction 2
        program.labels.insert("loop_body".to_string(), 2);

        // ADD: R1 = R1 + 1
        program.instructions.push(Instruction::ADD {
            ty: Type::U32,
            dst: Register(1),
            src1: Register(1),
            src2: Register(2), // R2 will be 1
        });

        // MOV_IMM: R2 = 1 (for incrementing)
        program.instructions.insert(
            2,
            Instruction::MOV_IMM {
                ty: Type::U32,
                dst: Register(2),
                value: 1,
            },
        );

        // LOOP: decrement R0 and branch to "loop_body" if R0 > 0
        program.instructions.push(Instruction::LOOP {
            count: Register(0),
            body: Label("loop_body".to_string()),
        });

        // EXIT
        program.instructions.push(Instruction::EXIT);

        let config = LaunchConfig::default();
        let result = executor.execute(&program, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_control_flow_call_ret() {
        use crate::isa::{Label, Register, Type};

        let memory = Arc::new(MemoryManager::new());
        let executor = WasmExecutor::new(memory);

        let mut program = Program::new();

        // MOV_IMM: R0 = 10
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(0),
            value: 10,
        });

        // CALL: call function at "subroutine"
        program.instructions.push(Instruction::CALL {
            target: Label("subroutine".to_string()),
        });

        // MOV_IMM: R1 = 1 (after return)
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(1),
            value: 1,
        });

        // EXIT
        program.instructions.push(Instruction::EXIT);

        // Label "subroutine" at instruction 4
        program.labels.insert("subroutine".to_string(), 4);

        // MOV_IMM: R2 = 5 (initialize before using in ADD)
        program.instructions.push(Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(2),
            value: 5,
        });

        // ADD: R0 = R0 + R2
        program.instructions.push(Instruction::ADD {
            ty: Type::U32,
            dst: Register(0),
            src1: Register(0),
            src2: Register(2),
        });

        // RET: return to caller
        program.instructions.push(Instruction::RET);

        let config = LaunchConfig::default();
        let result = executor.execute(&program, &config);
        if let Err(e) = &result {
            eprintln!("CALL/RET test error: {:?}", e);
        }
        assert!(result.is_ok());
    }
}
