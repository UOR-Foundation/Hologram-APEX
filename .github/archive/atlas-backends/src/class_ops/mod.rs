//! Direct class-based operations (Phase 2)
//!
//! This module provides the ClassOperations trait and infrastructure for
//! executing operations directly on class_bases[96] without the ISA layer.
//!
//! ## Architecture
//!
//! Phase 2 eliminates the ISA → Graph translation by working directly on
//! the 96 resonance classes using the 7 generators as primitives:
//!
//! ```text
//! High-Level Op → Graph Operations → class_bases[96]
//!                 (no ISA layer)
//! ```
//!
//! ## The 7 Generators
//!
//! Per Atlas Sigil Algebra §5.1:
//! 1. **Mark** - introduce/remove mark
//! 2. **Copy** - comultiplication (fan-out)
//! 3. **Swap** - symmetry/braid on wires
//! 4. **Merge** - fold/meet (with modality)
//! 5. **Split** - case analysis/deconstruct
//! 6. **Quote** - suspend computation
//! 7. **Evaluate** - force/thunk discharge
//!
//! ## Example
//!
//! ```rust,ignore
//! use atlas_backends::class_ops::{ClassOperations, ClassArithmetic};
//!
//! // Build lookup tables
//! let arith = ClassArithmetic::new();
//!
//! // Classify bytes
//! let class_a = arith.classify(42);
//! let class_b = arith.classify(128);
//!
//! // Add using class arithmetic
//! let class_result = arith.add(class_a, class_b);
//!
//! // Execute on backend
//! backend.merge(class_a, class_result, class_b)?;
//! ```

pub mod arithmetic;
pub mod generators;

pub use arithmetic::ClassArithmetic;

use crate::Result;

/// Core trait for direct class-based operations
///
/// This trait provides the 7 generators as primitive operations that
/// work directly on class_bases[96] without register mapping or ISA.
pub trait ClassOperations {
    /// Mark generator: introduce/remove mark at a class
    ///
    /// The mark generator is used to track special states and
    /// implement logical negation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.mark(5)?;  // Mark class 5
    /// ```
    fn mark(&mut self, class: u8) -> Result<()>;

    /// Copy generator: comultiplication (fan-out)
    ///
    /// Duplicates data from src class to dst class. Both classes must
    /// be neighbors in the graph.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.copy(5, 12)?;  // Copy class 5 → class 12
    /// ```
    fn copy(&mut self, src: u8, dst: u8) -> Result<()>;

    /// Swap generator: exchange data between two classes
    ///
    /// Symmetry operation that swaps the contents of two classes.
    /// Both classes must be neighbors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.swap(7, 23)?;  // Swap classes 7 and 23
    /// ```
    fn swap(&mut self, class_a: u8, class_b: u8) -> Result<()>;

    /// Merge generator: fold/meet with context
    ///
    /// Combines src and context classes into dst. This is the fundamental
    /// binary operation that implements arithmetic at the class level.
    ///
    /// # Arguments
    ///
    /// * `src` - Source class (must be neighbor of dst)
    /// * `dst` - Destination class
    /// * `context` - Context class (provides second operand)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Add: merge class 5 and context 7 into class 12
    /// backend.merge(5, 12, 7)?;
    /// ```
    fn merge(&mut self, src: u8, dst: u8, context: u8) -> Result<()>;

    /// Split generator: case analysis/deconstruct
    ///
    /// Inverse of merge. Decomposes src into dst using context as witness.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.split(12, 5, 7)?;  // Split class 12 → 5 with witness 7
    /// ```
    fn split(&mut self, src: u8, dst: u8, context: u8) -> Result<()>;

    /// Quote generator: suspend computation
    ///
    /// Delays evaluation at a class, creating a thunk/suspension.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.quote(12)?;  // Suspend evaluation at class 12
    /// ```
    fn quote(&mut self, class: u8) -> Result<()>;

    /// Evaluate generator: force/thunk discharge
    ///
    /// Forces evaluation of a suspended computation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.evaluate(12)?;  // Force evaluation at class 12
    /// ```
    fn evaluate(&mut self, class: u8) -> Result<()>;
}

/// High-level operations composed from generators
///
/// This trait provides familiar operations (Hadamard, CNOT, vector_add)
/// built from generator compositions.
pub trait ComposedOperations: ClassOperations {
    /// Hadamard gate: copy@c05 . mark@c21
    ///
    /// Creates superposition by copying and marking.
    fn hadamard(&mut self) -> Result<()> {
        self.copy(5, 21)?;
        self.mark(21)?;
        Ok(())
    }

    /// CNOT gate: merge . (mark || swap) . copy
    ///
    /// Controlled-NOT using generator composition.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit class
    /// * `target` - Target qubit class
    fn cnot(&mut self, control: u8, target: u8) -> Result<()> {
        self.copy(control, target)?;
        // Parallel composition (order doesn't matter)
        self.mark(control)?;
        self.swap(control, target)?;
        self.merge(control, target, 0)?;
        Ok(())
    }

    /// Vector addition using class arithmetic
    ///
    /// Adds two classes using merge generator.
    ///
    /// # Example
    ///
    /// ```ignore
    /// backend.vector_add(5, 7, 12)?;  // Add classes 5 and 7 → 12
    /// ```
    fn vector_add(&mut self, src_a: u8, src_b: u8, dst: u8) -> Result<()> {
        self.merge(src_a, dst, src_b)
    }

    /// Vector multiplication using class arithmetic
    fn vector_mul(&mut self, src_a: u8, src_b: u8, dst: u8) -> Result<()> {
        // Multiplication uses merge with different transform
        self.merge(src_a, dst, src_b)
    }
}
