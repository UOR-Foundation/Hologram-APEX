//! Factor extraction and address factorization
//!
//! This module provides factorization methods for mapping inputs to addresses:
//!
//! - **Hash-based factorization**: Fast, deterministic mapping (production-ready)
//! - **Algebraic factorization**: Mathematical decomposition (future)
//!
//! # Hash-Based Factorization
//!
//! The hash-based approach provides O(1) deterministic mapping from input patterns
//! to addresses without requiring mathematical factorization of Griess vectors.
//! This is the recommended approach for production use.
//!
//! # Future: Algebraic Factorization
//!
//! The algebraic factor extraction operator P: GriessVector → (Vector, Vector)
//! will decompose a Griess vector into two factor vectors whose Hadamard product
//! equals the original. This will be implemented in a future version for
//! factorization analysis and research purposes.

pub mod hash_factorization;

pub use hash_factorization::{compute_data_hash, factorize_data, factorize_hash, ExtendedAddress};
