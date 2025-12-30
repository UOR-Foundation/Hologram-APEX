//! Canonicalization and Pattern Rewriting
//!
//! This module implements canonicalization through pattern-based rewriting.
//! Applies rules like H² = I, X² = I, HXH = Z to reduce operation count.

pub mod canonical_repr;
pub mod canonicalization;
pub mod pattern;
pub mod rewrite;
pub mod rules;

pub use canonical_repr::*;
pub use canonicalization::*;
pub use pattern::*;
pub use rewrite::*;
pub use rules::*;
