//! Intermediate Representation (IR) for Compile-Time Canonicalization
//!
//! This module provides an IR used ONLY during compilation for:
//! - Circuit → IR conversion
//! - IR-level canonicalization and rewrites
//! - IR → ISA Program lowering
//!
//! **CRITICAL**: IR is compile-time only. Runtime uses ISA Programs, not IR.
//!
//! ## Architecture
//!
//! ```text
//! Circuit (String)
//!     ↓ parse
//! Circuit AST
//!     ↓ build
//! IR (IRNode)
//!     ↓ normalize
//! Canonical IR
//!     ↓ lower
//! ISA Program
//! ```
//!
//! ## IR Structure
//!
//! - **Atom**: Single generator call (mark, copy, swap, merge, split, quote, evaluate)
//! - **Seq**: Sequential composition (A; B)
//! - **Par**: Parallel composition (A | B)
//! - **Transform**: Automorphism transform (R, T, M, S)

pub mod builder;
pub mod lowering;
pub mod nodes;
pub mod normalization;

pub use builder::*;
pub use lowering::*;
pub use nodes::*;
pub use normalization::*;
