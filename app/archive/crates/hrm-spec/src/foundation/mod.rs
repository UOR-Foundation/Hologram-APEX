//! Foundation Layer - Mathematical Primitives
//!
//! This module defines the fundamental mathematical structures from first principles:
//! - Group axioms and abstract groups
//! - Ring axioms and field structures
//! - Lattice theory and order structures
//! - Homomorphisms and structure-preserving maps
//! - Exact arbitrary-precision arithmetic

pub mod group;
pub mod ring;
pub mod lattice;
pub mod homomorphism;
pub mod exactmath;

pub use group::Group;
pub use ring::Ring;
pub use lattice::Lattice;
pub use homomorphism::Homomorphism;
pub use exactmath::Exact;
