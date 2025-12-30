//! Automorphism Group Operations
//!
//! This module implements automorphism group operations for the 96-class system.
//! The automorphism group Aut(Atlas₁₂₂₈₈) has 2,048 elements: D₈ × T₈ × S₁₆.

pub mod automorphism_group;
pub mod automorphism_search;

pub use automorphism_group::*;
pub use automorphism_search::*;
