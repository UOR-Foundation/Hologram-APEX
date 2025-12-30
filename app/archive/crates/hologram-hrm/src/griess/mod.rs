//! Griess algebra implementation for HRM
//!
//! This module implements a simplified version of the Griess algebra using
//! the Hadamard (component-wise) product. The full Griess algebra is a
//! commutative non-associative algebra of dimension 196,884 related to the
//! Monster group.
//!
//! For HRM's purposes, we use:
//! - **Product**: Hadamard product (component-wise multiplication)
//! - **Identity**: Vector of all 1.0s
//! - **Properties**: Associative, commutative
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_hrm::griess::{GriessVector, product};
//!
//! // Create vectors
//! let a = GriessVector::identity();
//! let b = GriessVector::from_vec(vec![2.0; 196_884])?;
//!
//! // Compute product
//! let c = product(&a, &b)?;
//!
//! // c = a ⊙ b = b (since a is identity)
//! assert_eq!(b, c);
//! ```

mod product;
pub mod resonance;
mod vector;

pub use product::{add, divide, product, scalar_mul, subtract};
pub use resonance::{
    constants, crush, lift, resonance_add, resonance_mul, resonance_neg, resonate, BooleanTruth, Budget,
    BudgetAccumulator, ParallelResonanceTracks, ResonanceClass,
};
pub use vector::GriessVector;
