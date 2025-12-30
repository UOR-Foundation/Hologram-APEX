//! ISA to WGSL Translator
//!
//! Translates Atlas ISA programs into WebGPU WGSL compute shaders.
//!
//! # Architecture
//!
//! ```text
//! ISA Program → WGSL Generator → Compiled Shader → WebGPU Execution
//!     ↓              ↓                  ↓                ↓
//!  Instructions   Templates         Pipeline         Dispatch
//!  Registers    → Variables      → Bindings      → Workgroups
//!  Memory ops   → Buffer refs    → Storage       → GPU Memory
//! ```

use crate::error::{BackendError, Result};
use crate::isa::{Instruction, Program, Register};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// WGSL shader generator for ISA programs
pub struct WgslGenerator {
    /// Generated WGSL code
    code: String,

    /// Register allocation map (register ID → WGSL variable name)
    registers: HashMap<u32, String>,

    /// Buffer binding points (register ID → binding index)
    buffer_bindings: HashMap<u32, u32>,

    /// Buffer types (register ID → ISA type)
    buffer_types: HashMap<u32, crate::isa::Type>,

    /// Next available binding index
    next_binding: u32,

    /// Workgroup size for the shader
    workgroup_size: (u32, u32, u32),
}

impl WgslGenerator {
    /// Create a new WGSL generator
    pub fn new() -> Self {
        Self {
            code: String::new(),
            registers: HashMap::new(),
            buffer_bindings: HashMap::new(),
            buffer_types: HashMap::new(),
            next_binding: 0,
            workgroup_size: (256, 1, 1), // Default 1D workgroup
        }
    }

    /// Generate WGSL shader from ISA program
    pub fn generate(&mut self, program: &Program) -> Result<String> {
        self.code.clear();
        self.registers.clear();
        self.buffer_bindings.clear();
        self.buffer_types.clear();
        self.next_binding = 0;

        // Write shader header
        self.write_header()?;

        // Analyze program to determine buffer bindings
        self.analyze_buffers(program)?;

        // Write buffer bindings
        self.write_bindings()?;

        // Write main compute function
        self.write_compute_main(program)?;

        Ok(self.code.clone())
    }

    /// Get buffer binding map (register ID → binding index)
    pub fn buffer_bindings(&self) -> &HashMap<u32, u32> {
        &self.buffer_bindings
    }

    /// Get workgroup size
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.workgroup_size
    }

    // ============================================================================================
    // Header Generation
    // ============================================================================================

    fn write_header(&mut self) -> Result<()> {
        writeln!(&mut self.code, "// Generated WGSL shader from Atlas ISA")
            .map_err(|e| BackendError::ExecutionError(format!("Failed to write header: {}", e)))?;
        writeln!(&mut self.code).map_err(|e| BackendError::ExecutionError(format!("Failed to write header: {}", e)))?;
        Ok(())
    }

    // ============================================================================================
    // Buffer Analysis
    // ============================================================================================

    fn analyze_buffers(&mut self, program: &Program) -> Result<()> {
        // Scan all instructions to find buffer references
        for instruction in &program.instructions {
            self.analyze_instruction_buffers(instruction)?;
        }
        Ok(())
    }

    fn analyze_instruction_buffers(&mut self, instruction: &Instruction) -> Result<()> {
        use crate::isa::Address;

        match instruction {
            // Memory operations reference buffers - extract handle from Address and track type
            Instruction::LDG { ty, addr, .. } | Instruction::STG { ty, addr, .. } => {
                match addr {
                    Address::BufferOffset { handle, .. } => {
                        self.ensure_buffer_binding_with_type(*handle as u32, *ty);
                    }
                    Address::RegisterIndirectComputed { handle_reg, .. } => {
                        // handle_reg contains the buffer handle (set via ExecutionParams)
                        // Use the register index as the buffer identifier
                        self.ensure_buffer_binding_with_type(handle_reg.index() as u32, *ty);
                    }
                    // PhiCoordinate addresses use boundary pool, handled separately
                    _ => {}
                }
            }
            Instruction::LDS { addr, .. } | Instruction::STS { addr, .. } => {
                if let Address::BufferOffset { handle, .. } = addr {
                    // LDS/STS default to F32 for now (shared memory typically uses f32)
                    self.ensure_buffer_binding_with_type(*handle as u32, crate::isa::Type::F32);
                }
            }

            // Arithmetic operations reference buffers via registers
            // In WebGPU backend, registers ARE buffers, so we need to bind them
            Instruction::ADD { dst, src1, src2, .. }
            | Instruction::SUB { dst, src1, src2, .. }
            | Instruction::MUL { dst, src1, src2, .. }
            | Instruction::DIV { dst, src1, src2, .. }
            | Instruction::MIN { dst, src1, src2, .. }
            | Instruction::MAX { dst, src1, src2, .. } => {
                self.ensure_buffer_binding(dst.index() as u32);
                self.ensure_buffer_binding(src1.index() as u32);
                self.ensure_buffer_binding(src2.index() as u32);
            }

            // Unary arithmetic operations
            Instruction::ABS { dst, src, .. }
            | Instruction::NEG { dst, src, .. }
            | Instruction::SQRT { dst, src, .. }
            | Instruction::RSQRT { dst, src, .. }
            | Instruction::EXP { dst, src, .. }
            | Instruction::LOG { dst, src, .. }
            | Instruction::LOG2 { dst, src, .. }
            | Instruction::SIN { dst, src, .. }
            | Instruction::COS { dst, src, .. }
            | Instruction::TAN { dst, src, .. }
            | Instruction::TANH { dst, src, .. }
            | Instruction::SIGMOID { dst, src, .. } => {
                self.ensure_buffer_binding(dst.index() as u32);
                self.ensure_buffer_binding(src.index() as u32);
            }

            // FMA operations
            Instruction::FMA { dst, a, b, c, .. } | Instruction::MAD { dst, a, b, c, .. } => {
                self.ensure_buffer_binding(dst.index() as u32);
                self.ensure_buffer_binding(a.index() as u32);
                self.ensure_buffer_binding(b.index() as u32);
                self.ensure_buffer_binding(c.index() as u32);
            }

            // Other instructions
            _ => {}
        }
        Ok(())
    }

    fn ensure_buffer_binding(&mut self, register_id: u32) {
        // For arithmetic operations, default to F32
        self.ensure_buffer_binding_with_type(register_id, crate::isa::Type::F32);
    }

    fn ensure_buffer_binding_with_type(&mut self, register_id: u32, ty: crate::isa::Type) {
        if !self.buffer_bindings.contains_key(&register_id) {
            self.buffer_bindings.insert(register_id, self.next_binding);
            self.buffer_types.insert(register_id, ty);
            self.next_binding += 1;
        } else if !self.buffer_types.contains_key(&register_id) {
            // If binding exists but type doesn't, set it
            self.buffer_types.insert(register_id, ty);
        }
    }

    // ============================================================================================
    // Binding Generation
    // ============================================================================================

    fn write_bindings(&mut self) -> Result<()> {
        // Sort bindings for consistent output
        let mut bindings: Vec<_> = self.buffer_bindings.iter().collect();
        bindings.sort_by_key(|(_, binding)| **binding);

        for (register_id, binding_idx) in bindings {
            let ty = self
                .buffer_types
                .get(register_id)
                .copied()
                .unwrap_or(crate::isa::Type::F32);
            let wgsl_type = self.isa_type_to_wgsl_array_type(ty);

            writeln!(
                &mut self.code,
                "@group(0) @binding({}) var<storage, read_write> buffer_{}: {};",
                binding_idx, register_id, wgsl_type
            )
            .map_err(|e| BackendError::ExecutionError(format!("Failed to write binding: {}", e)))?;
        }

        writeln!(&mut self.code)
            .map_err(|e| BackendError::ExecutionError(format!("Failed to write bindings: {}", e)))?;
        Ok(())
    }

    /// Convert ISA type to WGSL array type declaration
    fn isa_type_to_wgsl_array_type(&self, ty: crate::isa::Type) -> String {
        use crate::isa::Type;
        match ty {
            Type::F32 => "array<f32>".to_string(),
            Type::F64 => "array<f32>".to_string(), // F64 not supported in WebGPU, downcast to f32
            Type::I32 => "array<i32>".to_string(),
            Type::U32 => "array<u32>".to_string(),
            // I64/U64 not natively supported in WGSL, represent as vec2<u32>
            Type::I64 | Type::U64 => "array<vec2<u32>>".to_string(),
            // Smaller types - expand to supported types
            Type::I8 | Type::I16 => "array<i32>".to_string(),
            Type::U8 | Type::U16 => "array<u32>".to_string(),
            Type::F16 | Type::BF16 => "array<f32>".to_string(), // F16 not universally supported
        }
    }

    /// Get the WGSL element type (without array<>)
    fn isa_type_to_wgsl_type(&self, ty: crate::isa::Type) -> &'static str {
        use crate::isa::Type;
        match ty {
            Type::F32 | Type::F64 | Type::F16 | Type::BF16 => "f32",
            Type::I32 | Type::I8 | Type::I16 => "i32",
            Type::U32 | Type::U8 | Type::U16 => "u32",
            Type::I64 | Type::U64 => "vec2<u32>", // 64-bit as vec2
        }
    }

    // ============================================================================================
    // Compute Shader Generation
    // ============================================================================================

    fn write_compute_main(&mut self, program: &Program) -> Result<()> {
        // Write function header
        writeln!(
            &mut self.code,
            "@compute @workgroup_size({}, {}, {})",
            self.workgroup_size.0, self.workgroup_size.1, self.workgroup_size.2
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write compute header: {}", e)))?;

        writeln!(
            &mut self.code,
            "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{"
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write main function: {}", e)))?;

        // Calculate linear thread index from 2D/3D dispatch
        // For large workloads that exceed 65535 workgroups, we dispatch across Y dimension
        // Linear index = Y * 65535 + X (matching calculate_dispatch_size() logic)
        writeln!(&mut self.code, "    let idx = global_id.y * 65535u + global_id.x;")
            .map_err(|e| BackendError::ExecutionError(format!("Failed to write idx: {}", e)))?;

        // Pre-scan instructions to collect register usage
        for instruction in &program.instructions {
            self.collect_instruction_registers(instruction);
        }

        // Declare all registers used in the shader
        if !self.registers.is_empty() {
            writeln!(&mut self.code)
                .map_err(|e| BackendError::ExecutionError(format!("Failed to write newline: {}", e)))?;
            let mut reg_ids: Vec<_> = self.registers.keys().copied().collect();
            reg_ids.sort();
            for reg_id in reg_ids {
                let reg_name = &self.registers[&reg_id];
                writeln!(&mut self.code, "    var {}: f32;", reg_name).map_err(|e| {
                    BackendError::ExecutionError(format!("Failed to write register declaration: {}", e))
                })?;
            }
            writeln!(&mut self.code)
                .map_err(|e| BackendError::ExecutionError(format!("Failed to write newline: {}", e)))?;
        }

        // Generate code for each instruction
        for instruction in &program.instructions {
            self.write_instruction(instruction)?;
        }

        writeln!(&mut self.code, "}}")
            .map_err(|e| BackendError::ExecutionError(format!("Failed to write main close: {}", e)))?;

        Ok(())
    }

    fn write_instruction(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            // Arithmetic operations
            Instruction::ADD { dst, src1, src2, .. } => {
                self.write_binary_op(dst, src1, src2, "+")?;
            }
            Instruction::SUB { dst, src1, src2, .. } => {
                self.write_binary_op(dst, src1, src2, "-")?;
            }
            Instruction::MUL { dst, src1, src2, .. } => {
                self.write_binary_op(dst, src1, src2, "*")?;
            }
            Instruction::DIV { dst, src1, src2, .. } => {
                self.write_binary_op(dst, src1, src2, "/")?;
            }
            Instruction::MIN { dst, src1, src2, .. } => {
                self.write_builtin_binary(dst, src1, src2, "min")?;
            }
            Instruction::MAX { dst, src1, src2, .. } => {
                self.write_builtin_binary(dst, src1, src2, "max")?;
            }

            // Unary operations
            Instruction::ABS { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "abs")?;
            }
            Instruction::NEG { dst, src, .. } => {
                self.write_unary_op(dst, src, "-")?;
            }
            Instruction::SQRT { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "sqrt")?;
            }
            Instruction::RSQRT { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "inverseSqrt")?;
            }
            Instruction::EXP { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "exp")?;
            }
            Instruction::LOG { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "log")?;
            }
            Instruction::LOG2 { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "log2")?;
            }
            Instruction::SIN { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "sin")?;
            }
            Instruction::COS { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "cos")?;
            }
            Instruction::TAN { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "tan")?;
            }
            Instruction::TANH { dst, src, .. } => {
                self.write_builtin_unary(dst, src, "tanh")?;
            }

            // SIGMOID: 1 / (1 + e^(-x))
            Instruction::SIGMOID { dst, src, .. } => {
                writeln!(
                    &mut self.code,
                    "    buffer_{}[idx] = 1.0 / (1.0 + exp(-buffer_{}[idx]));",
                    dst.index() as u32,
                    src.index() as u32
                )
                .map_err(|e| BackendError::ExecutionError(format!("Failed to write sigmoid: {}", e)))?;
            }

            // FMA operations
            Instruction::FMA { dst, a, b, c, .. } => {
                writeln!(
                    &mut self.code,
                    "    buffer_{}[idx] = fma(buffer_{}[idx], buffer_{}[idx], buffer_{}[idx]);",
                    dst.index() as u32,
                    a.index() as u32,
                    b.index() as u32,
                    c.index() as u32
                )
                .map_err(|e| BackendError::ExecutionError(format!("Failed to write FMA: {}", e)))?;
            }
            Instruction::MAD { dst, a, b, c, .. } => {
                writeln!(
                    &mut self.code,
                    "    buffer_{}[idx] = buffer_{}[idx] * buffer_{}[idx] + buffer_{}[idx];",
                    dst.index() as u32,
                    a.index() as u32,
                    b.index() as u32,
                    c.index() as u32
                )
                .map_err(|e| BackendError::ExecutionError(format!("Failed to write MAD: {}", e)))?;
            }

            // Data movement instructions
            Instruction::MOV_IMM { ty, dst, value } => {
                let reg_name = self.get_or_create_register(dst.index() as u32);
                // All registers are f32, so convert value appropriately
                let assignment = match ty {
                    crate::isa::Type::F32 => {
                        let f = f32::from_bits(*value as u32);
                        format!("{}f", f)
                    }
                    crate::isa::Type::U32 => {
                        // Bitcast u32 to f32 (preserves bit pattern)
                        format!("bitcast<f32>({}u)", *value as u32)
                    }
                    crate::isa::Type::I32 => {
                        // Bitcast i32 to f32 (preserves bit pattern)
                        format!("bitcast<f32>({}i)", *value as i32)
                    }
                    crate::isa::Type::U64 | crate::isa::Type::I64 => {
                        // For 64-bit values, take lower 32 bits and bitcast
                        format!("bitcast<f32>({}u)", *value as u32)
                    }
                    _ => {
                        return Err(BackendError::UnsupportedOperation(format!(
                            "Unsupported MOV_IMM type for f32 register: {:?}",
                            ty
                        )));
                    }
                };
                writeln!(&mut self.code, "    {} = {};", reg_name, assignment)
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write MOV_IMM: {}", e)))?;
            }
            Instruction::LDG { ty, dst, addr } => {
                self.write_load(ty, dst, addr)?;
            }
            Instruction::STG { ty, src, addr } => {
                self.write_store(ty, src, addr)?;
            }
            Instruction::MOV { dst, src, .. } => {
                let dst_reg = self.get_or_create_register(dst.index() as u32);
                let src_reg = self.get_or_create_register(src.index() as u32);
                writeln!(&mut self.code, "    {} = {};", dst_reg, src_reg)
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write MOV: {}", e)))?;
            }

            // Other instructions not yet implemented
            _ => {
                writeln!(&mut self.code, "    // TODO: Implement {:?}", instruction)
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write TODO: {}", e)))?;
            }
        }
        Ok(())
    }

    fn write_binary_op(&mut self, dst: &Register, src1: &Register, src2: &Register, op: &str) -> Result<()> {
        writeln!(
            &mut self.code,
            "    buffer_{}[idx] = buffer_{}[idx] {} buffer_{}[idx];",
            dst.index() as u32,
            src1.index() as u32,
            op,
            src2.index() as u32
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write binary op: {}", e)))?;
        Ok(())
    }

    fn write_builtin_binary(&mut self, dst: &Register, src1: &Register, src2: &Register, builtin: &str) -> Result<()> {
        writeln!(
            &mut self.code,
            "    buffer_{}[idx] = {}(buffer_{}[idx], buffer_{}[idx]);",
            dst.index() as u32,
            builtin,
            src1.index() as u32,
            src2.index() as u32
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write builtin binary: {}", e)))?;
        Ok(())
    }

    fn write_unary_op(&mut self, dst: &Register, src: &Register, op: &str) -> Result<()> {
        writeln!(
            &mut self.code,
            "    buffer_{}[idx] = {}buffer_{}[idx];",
            dst.index() as u32,
            op,
            src.index() as u32
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write unary op: {}", e)))?;
        Ok(())
    }

    fn write_builtin_unary(&mut self, dst: &Register, src: &Register, builtin: &str) -> Result<()> {
        writeln!(
            &mut self.code,
            "    buffer_{}[idx] = {}(buffer_{}[idx]);",
            dst.index() as u32,
            builtin,
            src.index() as u32
        )
        .map_err(|e| BackendError::ExecutionError(format!("Failed to write builtin unary: {}", e)))?;
        Ok(())
    }

    fn collect_instruction_registers(&mut self, instruction: &Instruction) {
        // Pre-populate registers used by this instruction
        match instruction {
            Instruction::MOV_IMM { dst, .. } => {
                self.get_or_create_register(dst.index() as u32);
            }
            Instruction::MOV { dst, src, .. } => {
                self.get_or_create_register(dst.index() as u32);
                self.get_or_create_register(src.index() as u32);
            }
            Instruction::LDG { dst, .. } => {
                self.get_or_create_register(dst.index() as u32);
            }
            Instruction::STG { src, .. } => {
                self.get_or_create_register(src.index() as u32);
            }
            _ => {
                // Other instructions use buffer[idx] directly, no register vars needed
            }
        }
    }

    fn get_or_create_register(&mut self, reg_id: u32) -> String {
        use std::collections::hash_map::Entry;
        match self.registers.entry(reg_id) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                let name = format!("r{}", reg_id);
                e.insert(name.clone());
                name
            }
        }
    }

    fn write_load(&mut self, ty: &crate::isa::Type, dst: &Register, addr: &crate::isa::Address) -> Result<()> {
        use crate::isa::Address;
        let dst_reg = self.get_or_create_register(dst.index() as u32);

        match addr {
            Address::BufferOffset { handle, offset } => {
                // Find which binding this buffer handle corresponds to
                let _binding_idx = self
                    .buffer_bindings
                    .get(&(*handle as u32))
                    .ok_or_else(|| BackendError::InvalidBufferHandle(*handle))?;

                // Calculate element index from byte offset
                let elem_size = ty.size_bytes();
                let elem_idx = offset / elem_size;

                // For i64/u64, load vec2<u32> and reinterpret lower 32 bits
                if matches!(ty, crate::isa::Type::I64 | crate::isa::Type::U64) {
                    writeln!(
                        &mut self.code,
                        "    {} = bitcast<f32>(buffer_{}[{}u].x);",
                        dst_reg, handle, elem_idx
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write LDG: {}", e)))?;
                } else {
                    // For other types, bitcast to f32 register
                    writeln!(
                        &mut self.code,
                        "    {} = bitcast<f32>(buffer_{}[{}u]);",
                        dst_reg, handle, elem_idx
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write LDG: {}", e)))?;
                }
            }
            Address::RegisterIndirectComputed {
                handle_reg,
                offset_reg: _,
            } => {
                // handle_reg contains buffer handle (set via ExecutionParams)
                // offset_reg contains computed byte offset (typically global_id.x * element_size)
                // For element-wise operations, we can use global_id.x directly as the index
                let handle = handle_reg.index() as u32;

                let _binding_idx = self
                    .buffer_bindings
                    .get(&handle)
                    .ok_or_else(|| BackendError::InvalidBufferHandle(handle as u64))?;

                // For i64/u64, load vec2<u32> and reinterpret lower 32 bits
                if matches!(ty, crate::isa::Type::I64 | crate::isa::Type::U64) {
                    writeln!(
                        &mut self.code,
                        "    {} = bitcast<f32>(buffer_{}[idx].x);",
                        dst_reg, handle
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write LDG: {}", e)))?;
                } else {
                    // For other types, bitcast to f32 register
                    writeln!(
                        &mut self.code,
                        "    {} = bitcast<f32>(buffer_{}[idx]);",
                        dst_reg, handle
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write LDG: {}", e)))?;
                }
            }
            _ => {
                return Err(BackendError::UnsupportedOperation(format!(
                    "WebGPU backend only supports BufferOffset and RegisterIndirectComputed addressing, got: {:?}",
                    addr
                )));
            }
        }
        Ok(())
    }

    fn write_store(&mut self, ty: &crate::isa::Type, src: &Register, addr: &crate::isa::Address) -> Result<()> {
        use crate::isa::Address;
        let src_reg = self.get_or_create_register(src.index() as u32);

        match addr {
            Address::BufferOffset { handle, offset } => {
                // Find which binding this buffer handle corresponds to
                let _binding_idx = self
                    .buffer_bindings
                    .get(&(*handle as u32))
                    .ok_or_else(|| BackendError::InvalidBufferHandle(*handle))?;

                // Calculate element index from byte offset
                let elem_size = ty.size_bytes();
                let elem_idx = offset / elem_size;

                // For i64/u64, store as vec2<u32> (value in .x, zero in .y)
                if matches!(ty, crate::isa::Type::I64 | crate::isa::Type::U64) {
                    writeln!(
                        &mut self.code,
                        "    buffer_{}[{}u] = vec2<u32>(bitcast<u32>({}), 0u);",
                        handle, elem_idx, src_reg
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write STG: {}", e)))?;
                } else {
                    // For other types, bitcast from f32 register
                    let wgsl_type = self.isa_type_to_wgsl_type(*ty);
                    writeln!(
                        &mut self.code,
                        "    buffer_{}[{}u] = bitcast<{}>({});",
                        handle, elem_idx, wgsl_type, src_reg
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write STG: {}", e)))?;
                }
            }
            Address::RegisterIndirectComputed {
                handle_reg,
                offset_reg: _,
            } => {
                // handle_reg contains buffer handle (set via ExecutionParams)
                // offset_reg contains computed byte offset (typically global_id.x * element_size)
                // For element-wise operations, we can use global_id.x directly as the index
                let handle = handle_reg.index() as u32;

                let _binding_idx = self
                    .buffer_bindings
                    .get(&handle)
                    .ok_or_else(|| BackendError::InvalidBufferHandle(handle as u64))?;

                // For i64/u64, store as vec2<u32> (value in .x, zero in .y)
                if matches!(ty, crate::isa::Type::I64 | crate::isa::Type::U64) {
                    writeln!(
                        &mut self.code,
                        "    buffer_{}[idx] = vec2<u32>(bitcast<u32>({}), 0u);",
                        handle, src_reg
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write STG: {}", e)))?;
                } else {
                    // For other types, bitcast from f32 register
                    let wgsl_type = self.isa_type_to_wgsl_type(*ty);
                    writeln!(
                        &mut self.code,
                        "    buffer_{}[idx] = bitcast<{}>({});",
                        handle, wgsl_type, src_reg
                    )
                    .map_err(|e| BackendError::ExecutionError(format!("Failed to write STG: {}", e)))?;
                }
            }
            _ => {
                return Err(BackendError::UnsupportedOperation(format!(
                    "WebGPU backend only supports BufferOffset and RegisterIndirectComputed addressing, got: {:?}",
                    addr
                )));
            }
        }
        Ok(())
    }
}

impl Default for WgslGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{Instruction, Program, Register, Type};

    #[test]
    fn test_simple_add() {
        let mut generator = WgslGenerator::new();

        let program = Program {
            instructions: vec![Instruction::ADD {
                ty: Type::F32,
                dst: Register::new(0),
                src1: Register::new(1),
                src2: Register::new(2),
            }],
        };

        let wgsl = generator.generate(&program).unwrap();
        assert!(wgsl.contains("buffer_0"));
        assert!(wgsl.contains("buffer_1"));
        assert!(wgsl.contains("buffer_2"));
        assert!(wgsl.contains("buffer_0[idx] = buffer_1[idx] + buffer_2[idx]"));
    }
}
