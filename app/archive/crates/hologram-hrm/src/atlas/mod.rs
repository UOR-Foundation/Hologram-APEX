//! Atlas partition: 96 canonical Griess vectors
//!
//! The Atlas provides the canonical vectors that form the basis of the HRM
//! address space. Each of the 96 base-96 digits corresponds to a unique
//! 196,884-dimensional vector generated deterministically using SplitMix64 PRNG.
//!
//! # Components
//!
//! - **PRNG**: SplitMix64 pseudo-random number generator
//! - **Generator**: Deterministic Atlas vector generation
//! - **Service**: High-level Atlas management and caching
//!
//! # Example
//!
//! ```ignore
//! use hologram_hrm::atlas::Atlas;
//!
//! // Create Atlas service
//! let atlas = Atlas::with_cache()?;
//!
//! // Get canonical vector for class 42
//! let vector = atlas.get_vector(42)?;
//! assert_eq!(vector.len(), 196_884);
//!
//! // Vector is normalized (unit length)
//! let norm = vector.norm();
//! assert!((norm - 1.0).abs() < 1e-6);
//! ```

pub mod generator;
pub mod prng;
pub mod scaled;
pub mod service;

// Re-exports
pub use generator::{are_orthogonal, generate_all_atlas_vectors, generate_atlas_vector, Modality, SigilComponents};
pub use prng::SplitMix64;
pub use scaled::ScaledAtlas;
pub use service::Atlas;
