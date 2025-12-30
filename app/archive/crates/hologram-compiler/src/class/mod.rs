//! Class System and Type Definitions
//!
//! This module defines the 96-class system and related types.
//! Classes are organized as: class(h₂, d, ℓ) = 24h₂ + 8d + ℓ

pub mod class_system;
pub mod multi_class;
pub mod types;

pub use class_system::*;
pub use multi_class::*;
pub use types::*;
