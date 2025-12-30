//! Two-Torus Lattice Structure
//!
//! The boundary lattice T² = T₁ × T₂ with:
//! - T₁: 48-periodic (page coordinate)
//! - T₂: 96-periodic (resonance coordinate)
//! - Total cells: 12,288 (48 pages × 256 bytes)

pub mod coordinate;
pub mod projection;
pub mod lifting;
pub mod metric;

pub use coordinate::TorusCoordinate;
pub use projection::Projection;
pub use lifting::Lifting;
pub use metric::TorusMetric;
