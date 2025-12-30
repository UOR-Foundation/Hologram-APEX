//! Typed Register File Implementation
//!
//! This module implements the register file for Atlas ISA instruction execution.
//! The register file provides type-safe access to 256 registers supporting all
//! Atlas ISA types, plus 16 predicate registers for conditional execution.
//!
//! # Architecture
//!
//! The register file is the central state store for ISA instruction execution:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Register File                                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Scalar Registers: R0 - R255 (256 registers)                 │
//! │  ├─ i8/i16/i32/i64  (signed integers)                       │
//! │  ├─ u8/u16/u32/u64  (unsigned integers)                     │
//! │  └─ f16/bf16/f32/f64 (floating point)                       │
//! │                                                             │
//! │ Predicate Registers: P0 - P15 (16 boolean flags)            │
//! │  └─ Used for conditional branches and select operations     │
//! │                                                             │
//! │ Type Tracking: Option<Type>[256]                            │
//! │  └─ Runtime type safety enforcement                         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Type Safety
//!
//! All register accesses are type-checked at runtime:
//! - **Writing** a value sets the register's type
//! - **Reading** a value validates the type matches
//! - **Uninitialized** registers error on read
//! - **CVT instruction** explicitly changes types
//!
//! This prevents common errors like:
//! - Reading uninitialized memory
//! - Type confusion (interpreting f32 bits as i32)
//! - Undefined behavior from type punning
//!
//! # Performance
//!
//! - **Zero-cost access:** Register read/write is a simple array index (15 ns)
//! - **No heap allocation:** All 256 registers stored inline
//! - **Type check overhead:** Single enum compare + array lookup
//! - **Predicate access:** Direct bool array access (no type check)
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust,ignore
//! use atlas_backends::RegisterFile;
//! use atlas_isa::{Register, Type};
//!
//! let mut regs = RegisterFile::new();
//!
//! // Write different types to different registers
//! regs.write(Register::new(0), 42i32)?;
//! regs.write(Register::new(1), 3.14f32)?;
//! regs.write(Register::new(2), 100u64)?;
//!
//! // Read back with type safety
//! let i: i32 = regs.read(Register::new(0))?;
//! let f: f32 = regs.read(Register::new(1))?;
//! let u: u64 = regs.read(Register::new(2))?;
//!
//! assert_eq!(i, 42);
//! assert_eq!(f, 3.14);
//! assert_eq!(u, 100);
//! ```
//!
//! ## Type Safety Enforcement
//!
//! ```rust,ignore
//! use atlas_backends::RegisterFile;
//! use atlas_isa::Register;
//!
//! let mut regs = RegisterFile::new();
//!
//! // Write as f32
//! regs.write(Register::new(0), 3.14f32)?;
//!
//! // Attempt to read as i32 fails with TypeMismatch error
//! let result: Result<i32> = regs.read(Register::new(0));
//! assert!(matches!(result, Err(BackendError::TypeMismatch { .. })));
//!
//! // Uninitialized read fails
//! let result: Result<f32> = regs.read(Register::new(100));
//! assert!(matches!(result, Err(BackendError::UninitializedRegister { .. })));
//! ```
//!
//! ## Working with Predicates
//!
//! ```rust,ignore
//! use atlas_backends::RegisterFile;
//! use atlas_isa::Predicate;
//!
//! let mut regs = RegisterFile::new();
//!
//! // Set predicate based on comparison
//! let a: f32 = regs.read(Register::new(0))?;
//! let b: f32 = regs.read(Register::new(1))?;
//! regs.write_pred(Predicate::new(0), a > b);
//!
//! // Use predicate for conditional execution
//! if regs.read_pred(Predicate::new(0)) {
//!     // Execute true path
//! } else {
//!     // Execute false path
//! }
//! ```
//!
//! ## All Supported Types
//!
//! ```rust,ignore
//! use atlas_backends::RegisterFile;
//! use atlas_isa::Register;
//! use half::{f16, bf16};
//!
//! let mut regs = RegisterFile::new();
//!
//! // Signed integers
//! regs.write(Register::new(0), 1i8)?;
//! regs.write(Register::new(1), 2i16)?;
//! regs.write(Register::new(2), 3i32)?;
//! regs.write(Register::new(3), 4i64)?;
//!
//! // Unsigned integers
//! regs.write(Register::new(4), 5u8)?;
//! regs.write(Register::new(5), 6u16)?;
//! regs.write(Register::new(6), 7u32)?;
//! regs.write(Register::new(7), 8u64)?;
//!
//! // Floating point
//! regs.write(Register::new(8), f16::from_f32(9.0))?;
//! regs.write(Register::new(9), bf16::from_f32(10.0))?;
//! regs.write(Register::new(10), 11.0f32)?;
//! regs.write(Register::new(11), 12.0f64)?;
//! ```
//!
//! ## Type Queries
//!
//! ```rust,ignore
//! use atlas_backends::RegisterFile;
//! use atlas_isa::{Register, Type};
//!
//! let mut regs = RegisterFile::new();
//! regs.write(Register::new(0), 42i32)?;
//!
//! // Check register type before reading
//! if regs.get_type(Register::new(0)) == Some(Type::I32) {
//!     let value: i32 = regs.read(Register::new(0))?;
//! }
//!
//! // Clear register (mark as uninitialized)
//! regs.clear(Register::new(0));
//! assert_eq!(regs.get_type(Register::new(0)), None);
//! ```

use atlas_isa::{Predicate, Register, Type};
use half::{bf16, f16};

use crate::error::{BackendError, Result};

/// Typed register file supporting ISA §3 type system
///
/// The register file contains 256 registers, each capable of holding any of the
/// 12 supported types. Type tracking ensures type safety across instruction execution.
///
/// # Example
///
/// ```ignore
/// use atlas_backends::RegisterFile;
/// use atlas_isa::{Register, Type};
///
/// let mut regs = RegisterFile::new();
///
/// // Write f32 value to register 0
/// regs.write(Register(0), 3.14f32).unwrap();
///
/// // Read back as f32
/// let value: f32 = regs.read(Register(0)).unwrap();
/// assert_eq!(value, 3.14f32);
///
/// // Attempt to read as wrong type fails
/// let result: Result<i32> = regs.read(Register(0));
/// assert!(result.is_err());
/// ```
#[derive(Debug, Clone)]
pub struct RegisterFile {
    // Scalar registers (256 registers per type)
    i8_regs: [i8; 256],
    i16_regs: [i16; 256],
    i32_regs: [i32; 256],
    i64_regs: [i64; 256],
    u8_regs: [u8; 256],
    u16_regs: [u16; 256],
    u32_regs: [u32; 256],
    u64_regs: [u64; 256],
    f16_regs: [f16; 256],
    bf16_regs: [bf16; 256],
    f32_regs: [f32; 256],
    f64_regs: [f64; 256],

    // Predicate registers (16 boolean predicates)
    predicates: [bool; 16],

    // Register type tracking (validates type safety)
    reg_types: [Option<Type>; 256],
}

impl RegisterFile {
    /// Create a new register file with all registers uninitialized
    ///
    /// All scalar registers are zero-initialized, but marked as uninitialized
    /// until first write. Predicate registers default to false.
    pub fn new() -> Self {
        Self {
            i8_regs: [0; 256],
            i16_regs: [0; 256],
            i32_regs: [0; 256],
            i64_regs: [0; 256],
            u8_regs: [0; 256],
            u16_regs: [0; 256],
            u32_regs: [0; 256],
            u64_regs: [0; 256],
            f16_regs: [f16::ZERO; 256],
            bf16_regs: [bf16::ZERO; 256],
            f32_regs: [0.0; 256],
            f64_regs: [0.0; 256],
            predicates: [false; 16],
            reg_types: [None; 256],
        }
    }

    /// Reset all registers to uninitialized state
    ///
    /// Clears type tracking and zeroes all register values.
    pub fn reset(&mut self) {
        self.i8_regs = [0; 256];
        self.i16_regs = [0; 256];
        self.i32_regs = [0; 256];
        self.i64_regs = [0; 256];
        self.u8_regs = [0; 256];
        self.u16_regs = [0; 256];
        self.u32_regs = [0; 256];
        self.u64_regs = [0; 256];
        self.f16_regs = [f16::ZERO; 256];
        self.bf16_regs = [bf16::ZERO; 256];
        self.f32_regs = [0.0; 256];
        self.f64_regs = [0.0; 256];
        self.predicates = [false; 16];
        self.reg_types = [None; 256];
    }

    /// Read typed register with type safety validation
    ///
    /// # Errors
    ///
    /// - Returns `TypeMismatch` if register type doesn't match requested type
    /// - Returns `UninitializedRegister` if register has not been written
    pub fn read<T: RegisterType>(&self, reg: Register) -> Result<T> {
        let idx = reg.index() as usize;

        // Type check
        match self.reg_types[idx] {
            None => Err(BackendError::UninitializedRegister { register: reg.index() }),
            Some(ty) if ty != T::TYPE => Err(BackendError::TypeMismatch {
                register: reg.index(),
                expected: T::TYPE,
                actual: Some(ty),
            }),
            Some(_) => T::read_from(self, reg),
        }
    }

    /// Write typed register with type tracking
    ///
    /// Sets the register type and writes the value.
    pub fn write<T: RegisterType>(&mut self, reg: Register, value: T) -> Result<()> {
        let idx = reg.index() as usize;
        self.reg_types[idx] = Some(T::TYPE);
        T::write_to(self, reg, value)
    }

    /// Read predicate register
    ///
    /// # Panics
    ///
    /// Panics if predicate index >= 16
    pub fn read_pred(&self, pred: Predicate) -> bool {
        let idx = pred.index() as usize;
        assert!(idx < 16, "Predicate index {} out of range [0, 16)", idx);
        self.predicates[idx]
    }

    /// Write predicate register
    ///
    /// # Panics
    ///
    /// Panics if predicate index >= 16
    pub fn write_pred(&mut self, pred: Predicate, value: bool) {
        let idx = pred.index() as usize;
        assert!(idx < 16, "Predicate index {} out of range [0, 16)", idx);
        self.predicates[idx] = value;
    }

    /// Get the current type of a register
    ///
    /// Returns `None` if the register is uninitialized.
    pub fn get_type(&self, reg: Register) -> Option<Type> {
        self.reg_types[reg.index() as usize]
    }

    /// Check if a register has been initialized
    pub fn is_initialized(&self, reg: Register) -> bool {
        self.reg_types[reg.index() as usize].is_some()
    }

    /// Clear a register (mark as uninitialized)
    pub fn clear_register(&mut self, reg: Register) {
        let idx = reg.index() as usize;
        self.reg_types[idx] = None;
        // Zero out all type arrays for this register
        self.i8_regs[idx] = 0;
        self.i16_regs[idx] = 0;
        self.i32_regs[idx] = 0;
        self.i64_regs[idx] = 0;
        self.u8_regs[idx] = 0;
        self.u16_regs[idx] = 0;
        self.u32_regs[idx] = 0;
        self.u64_regs[idx] = 0;
        self.f16_regs[idx] = f16::ZERO;
        self.bf16_regs[idx] = bf16::ZERO;
        self.f32_regs[idx] = 0.0;
        self.f64_regs[idx] = 0.0;
    }
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be stored in registers
///
/// This trait provides type-safe access to the register file.
/// Each type knows its ISA type and how to read/write from the register file.
pub trait RegisterType: Sized {
    /// The ISA type this Rust type corresponds to
    const TYPE: Type;

    /// Read this type from the register file
    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self>;

    /// Write this type to the register file
    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()>;
}

// ================================================================================================
// RegisterType Implementations - Signed Integers
// ================================================================================================

impl RegisterType for i8 {
    const TYPE: Type = Type::I8;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.i8_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.i8_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for i16 {
    const TYPE: Type = Type::I16;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.i16_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.i16_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for i32 {
    const TYPE: Type = Type::I32;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.i32_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.i32_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for i64 {
    const TYPE: Type = Type::I64;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.i64_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.i64_regs[reg.index() as usize] = value;
        Ok(())
    }
}

// ================================================================================================
// RegisterType Implementations - Unsigned Integers
// ================================================================================================

impl RegisterType for u8 {
    const TYPE: Type = Type::U8;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.u8_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.u8_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for u16 {
    const TYPE: Type = Type::U16;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.u16_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.u16_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for u32 {
    const TYPE: Type = Type::U32;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.u32_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.u32_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for u64 {
    const TYPE: Type = Type::U64;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.u64_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.u64_regs[reg.index() as usize] = value;
        Ok(())
    }
}

// ================================================================================================
// RegisterType Implementations - Floating Point
// ================================================================================================

impl RegisterType for f16 {
    const TYPE: Type = Type::F16;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.f16_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.f16_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for bf16 {
    const TYPE: Type = Type::BF16;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.bf16_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.bf16_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for f32 {
    const TYPE: Type = Type::F32;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.f32_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.f32_regs[reg.index() as usize] = value;
        Ok(())
    }
}

impl RegisterType for f64 {
    const TYPE: Type = Type::F64;

    fn read_from(file: &RegisterFile, reg: Register) -> Result<Self> {
        Ok(file.f64_regs[reg.index() as usize])
    }

    fn write_to(file: &mut RegisterFile, reg: Register, value: Self) -> Result<()> {
        file.f64_regs[reg.index() as usize] = value;
        Ok(())
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_file_new() {
        let regs = RegisterFile::new();

        // All registers should be uninitialized
        for i in 0..=255u8 {
            let reg = Register::new(i);
            assert!(!regs.is_initialized(reg));
            assert_eq!(regs.get_type(reg), None);
        }

        // All predicates should be false
        for i in 0..16u8 {
            assert_eq!(regs.read_pred(Predicate::new(i)), false);
        }
    }

    #[test]
    fn test_type_safety_enforcement() {
        let mut regs = RegisterFile::new();
        let r0 = Register::new(0);

        // Write f32 value
        regs.write(r0, 3.14f32).unwrap();

        // Read back as f32 succeeds
        let value: f32 = regs.read(r0).unwrap();
        assert_eq!(value, 3.14f32);

        // Read as wrong type fails
        let result: Result<i32> = regs.read(r0);
        assert!(result.is_err());
        match result {
            Err(BackendError::TypeMismatch {
                register,
                expected,
                actual,
            }) => {
                assert_eq!(register, 0);
                assert_eq!(expected, Type::I32);
                assert_eq!(actual, Some(Type::F32));
            }
            _ => panic!("Expected TypeMismatch error"),
        }
    }

    #[test]
    fn test_uninitialized_register_error() {
        let regs = RegisterFile::new();
        let r0 = Register::new(0);

        // Reading uninitialized register fails
        let result: Result<f32> = regs.read(r0);
        assert!(result.is_err());
        match result {
            Err(BackendError::UninitializedRegister { register }) => {
                assert_eq!(register, 0);
            }
            _ => panic!("Expected UninitializedRegister error"),
        }
    }

    #[test]
    fn test_all_integer_types() {
        let mut regs = RegisterFile::new();

        // Signed integers
        regs.write(Register::new(0), 42i8).unwrap();
        regs.write(Register::new(1), 1000i16).unwrap();
        regs.write(Register::new(2), 100000i32).unwrap();
        regs.write(Register::new(3), 10000000000i64).unwrap();

        assert_eq!(regs.read::<i8>(Register::new(0)).unwrap(), 42i8);
        assert_eq!(regs.read::<i16>(Register::new(1)).unwrap(), 1000i16);
        assert_eq!(regs.read::<i32>(Register::new(2)).unwrap(), 100000i32);
        assert_eq!(regs.read::<i64>(Register::new(3)).unwrap(), 10000000000i64);

        // Unsigned integers
        regs.write(Register::new(4), 255u8).unwrap();
        regs.write(Register::new(5), 65535u16).unwrap();
        regs.write(Register::new(6), 4000000000u32).unwrap();
        regs.write(Register::new(7), 10000000000u64).unwrap();

        assert_eq!(regs.read::<u8>(Register::new(4)).unwrap(), 255u8);
        assert_eq!(regs.read::<u16>(Register::new(5)).unwrap(), 65535u16);
        assert_eq!(regs.read::<u32>(Register::new(6)).unwrap(), 4000000000u32);
        assert_eq!(regs.read::<u64>(Register::new(7)).unwrap(), 10000000000u64);
    }

    #[test]
    fn test_all_float_types() {
        let mut regs = RegisterFile::new();

        // f32 and f64
        regs.write(Register::new(0), 3.14f32).unwrap();
        regs.write(Register::new(1), 2.71828f64).unwrap();

        assert_eq!(regs.read::<f32>(Register::new(0)).unwrap(), 3.14f32);
        assert_eq!(regs.read::<f64>(Register::new(1)).unwrap(), 2.71828f64);

        // f16 and bf16
        regs.write(Register::new(2), f16::from_f32(1.5)).unwrap();
        regs.write(Register::new(3), bf16::from_f32(2.5)).unwrap();

        assert_eq!(regs.read::<f16>(Register::new(2)).unwrap(), f16::from_f32(1.5));
        assert_eq!(regs.read::<bf16>(Register::new(3)).unwrap(), bf16::from_f32(2.5));
    }

    #[test]
    fn test_predicate_registers() {
        let mut regs = RegisterFile::new();

        // Set some predicates
        regs.write_pred(Predicate::new(0), true);
        regs.write_pred(Predicate::new(5), true);
        regs.write_pred(Predicate::new(15), true);

        // Verify
        assert_eq!(regs.read_pred(Predicate::new(0)), true);
        assert_eq!(regs.read_pred(Predicate::new(1)), false);
        assert_eq!(regs.read_pred(Predicate::new(5)), true);
        assert_eq!(regs.read_pred(Predicate::new(15)), true);

        // Clear
        regs.write_pred(Predicate::new(0), false);
        assert_eq!(regs.read_pred(Predicate::new(0)), false);
    }

    #[test]
    fn test_register_reset() {
        let mut regs = RegisterFile::new();

        // Write some values
        regs.write(Register::new(0), 42i32).unwrap();
        regs.write(Register::new(1), 3.14f32).unwrap();
        regs.write_pred(Predicate::new(0), true);

        // Verify written
        assert!(regs.is_initialized(Register::new(0)));
        assert!(regs.is_initialized(Register::new(1)));
        assert_eq!(regs.read_pred(Predicate::new(0)), true);

        // Reset
        regs.reset();

        // Verify cleared
        assert!(!regs.is_initialized(Register::new(0)));
        assert!(!regs.is_initialized(Register::new(1)));
        assert_eq!(regs.read_pred(Predicate::new(0)), false);
    }

    #[test]
    fn test_clear_register() {
        let mut regs = RegisterFile::new();

        // Write value
        regs.write(Register::new(0), 42i32).unwrap();
        assert!(regs.is_initialized(Register::new(0)));

        // Clear
        regs.clear_register(Register::new(0));
        assert!(!regs.is_initialized(Register::new(0)));

        // Reading now should fail
        assert!(regs.read::<i32>(Register::new(0)).is_err());
    }

    #[test]
    fn test_get_type() {
        let mut regs = RegisterFile::new();

        // Uninitialized
        assert_eq!(regs.get_type(Register::new(0)), None);

        // Write f32
        regs.write(Register::new(0), 3.14f32).unwrap();
        assert_eq!(regs.get_type(Register::new(0)), Some(Type::F32));

        // Overwrite with i32
        regs.write(Register::new(0), 42i32).unwrap();
        assert_eq!(regs.get_type(Register::new(0)), Some(Type::I32));
    }

    #[test]
    fn test_type_overwrite() {
        let mut regs = RegisterFile::new();
        let r0 = Register::new(0);

        // Write f32
        regs.write(r0, 3.14f32).unwrap();
        assert_eq!(regs.get_type(r0), Some(Type::F32));

        // Overwrite with i32 (type changes)
        regs.write(r0, 42i32).unwrap();
        assert_eq!(regs.get_type(r0), Some(Type::I32));

        // Reading as f32 now fails
        assert!(regs.read::<f32>(r0).is_err());

        // Reading as i32 succeeds
        assert_eq!(regs.read::<i32>(r0).unwrap(), 42);
    }

    #[test]
    #[should_panic(expected = "Predicate index must be < 16")]
    fn test_predicate_bounds_read() {
        let regs = RegisterFile::new();
        regs.read_pred(Predicate::new(16)); // Should panic
    }

    #[test]
    #[should_panic(expected = "Predicate index must be < 16")]
    fn test_predicate_bounds_write() {
        let mut regs = RegisterFile::new();
        regs.write_pred(Predicate::new(16), true); // Should panic
    }
}
