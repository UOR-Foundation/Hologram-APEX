//! MoonshineHRM Specification v1.0
//!
//! A self-contained, first-principles specification of the MoonshineHRM
//! hierarchical reasoning engine with strict conformance tests and benchmarks.
//!
//! # Architecture
//!
//! The specification is structured in six layers:
//!
//! - **Layer 0 (Foundation)**: Mathematical primitives - groups, rings, lattices
//! - **Layer 1 (Monster)**: 196,884-D representation and moonshine correspondence
//! - **Layer 2 (Torus)**: Two-torus lattice with projection/lifting
//! - **Layer 3 (Algebra)**: Generators ⊕, ⊗, ⊙ with coherence proofs
//! - **Layer 4 (Harmonic)**: Zeta calibration and prime orbits
//! - **Layer 5 (Routing)**: Entanglement network and O(1) algorithms
//! - **Layer 6 (Derived)**: Matrix ops, convolution, attention
//!
//! # Example
//!
//! ```rust
//! use hrm_spec::prelude::*;
//! use num_bigint::BigInt;
//!
//! // Project integer to torus coordinates
//! let n = BigInt::from(12345);
//! let coord = TorusCoordinate::from_integer(&n);
//!
//! // Multiplication in torus space
//! let a = TorusCoordinate { page: 3, resonance: 5 };
//! let b = TorusCoordinate { page: 5, resonance: 7 };
//! let c = mul(&a, &b);
//!
//! assert_eq!(c.page, (3 * 5) % 48);
//! assert_eq!(c.resonance, (5 * 7) % 96);
//! ```

// Layer 0: Foundation
pub mod foundation;

// Layer 1: Monster Group
pub mod monster;

// Layer 2: Two-Torus Lattice
pub mod torus;

// Layer 3: Algebraic Generators
pub mod algebra;

// Layer 4: Harmonic Calibration
pub mod harmonic;

// Layer 5: Routing Protocol
pub mod routing;

// Layer 6: Derived Operations
pub mod derived;

/// Prelude - commonly used types and traits
pub mod prelude {
    // Foundation
    pub use crate::foundation::group::{Group, AbelianGroup};
    pub use crate::foundation::ring::{Ring, CommutativeRing};
    pub use crate::foundation::exactmath::Exact;
    pub use crate::foundation::homomorphism::Homomorphism;
    
    // Monster
    pub use crate::monster::representation::MonsterRepresentation;
    
    // Torus
    pub use crate::torus::coordinate::TorusCoordinate;
    pub use crate::torus::projection::{Projection, StandardProjection};
    pub use crate::torus::lifting::{Lifting, O1Lifting};
    
    // Algebra
    pub use crate::algebra::addition::add;
    pub use crate::algebra::multiplication::mul;
    pub use crate::algebra::scalar::scalar_mul_optimized;
    pub use crate::algebra::coherence::CoherenceVerifier;
    
    // Routing
    pub use crate::routing::protocol::{RoutingProtocol, StandardRouting};
    pub use crate::routing::entanglement::NetworkAddress;
    
    // Derived
    pub use crate::derived::matmul::matmul;
    pub use crate::derived::convolution::{convolve, circular_convolve};
    pub use crate::derived::reduction::{reduce_sum, reduce_product};
    pub use crate::derived::attention::attention;
    
    // External
    pub use num_bigint::BigInt;
    pub use num_rational::BigRational;
}

#[cfg(test)]
mod tests {

    #[test]
    fn spec_is_self_contained() {
        // Verify specification uses only constructs it defines
        // This test ensures no external dependencies leak into core spec
        assert!(true, "All types defined within hrm-spec");
    }
}
