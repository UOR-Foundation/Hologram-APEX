//! Monster Metric on Torus

use super::coordinate::TorusCoordinate;

/// Distance metric on torus
pub trait TorusMetric {
    /// Compute distance between two coordinates
    fn distance(&self, a: &TorusCoordinate, b: &TorusCoordinate) -> f64;
}

// Implementation deferred to Phase 2
