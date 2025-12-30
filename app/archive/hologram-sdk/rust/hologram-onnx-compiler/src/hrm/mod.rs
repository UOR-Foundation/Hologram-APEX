//! # MoonshineHRM Components
//!
//! This module provides the core components used by the simplified ONNX compiler:
//!
//! - **Graph IR**: HologramGraph (petgraph-based computation graph)
//! - **Operators**: Macro-generated ONNX operators with flexible arity
//! - **Types**: Shared types for compilation pipeline
//! - **Numeric**: Numeric trait for type-generic operations
//! - **Shape Inference**: ONNX shape propagation engine

pub mod graph;
pub mod numeric;
pub mod ops;
pub mod shape_inference;
pub mod types;

// Re-exports
pub use graph::{EmbeddedInput, GraphOptimizer, HologramGraph, NodeId, OptimizationPass};
pub use shape_inference::ShapeInference;
pub use types::{
    AddressSpace, CollectionManifest, CompilationCheckpoint, DiscretizationStrategy, EmbeddingCache, FactorizedResults,
    OperationMetadata, OperationStats, PerfectHashTable,
};
