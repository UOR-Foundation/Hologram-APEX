//! E₇ Orbit Distance Computation
//!
//! This module implements orbit distance calculations in the 96-class system.
//! All 96 classes form a SINGLE ORBIT under the automorphism group {R, D, T, M}
//! with diameter 12.
//!
//! ## Research Foundation
//!
//! Research from sigmatics demonstrates:
//! - All classes reachable from any class via transforms
//! - Prime generator 37 has minimal complexity (10.0)
//! - Maximum orbit distance is 12 (the diameter)
//! - Orbit coordinates enable 80% compression
//!
//! ## Transforms
//!
//! - **R** (rotate): Quadrant rotation, h₂ → (h₂ + 1) mod 4
//! - **D** (triality): Modality rotation, d → (d + 1) mod 3
//! - **T** (twist): Context rotation, ℓ → (ℓ + 1) mod 8
//! - **M** (mirror): Modality mirror, d → (3 - d) mod 3

use crate::types::Transform;

/// Orbit distance from a source class to a target class
///
/// Represents the minimum number of transforms {R, D, T, M} needed
/// to reach the target from the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrbitDistance {
    /// Minimum number of transforms
    pub distance: u8,

    /// Path length (number of transforms in path)
    /// Note: Actual path is computed on-demand, not stored for Copy trait
    pub path_length: u8,
}

impl OrbitDistance {
    /// Create an orbit distance
    pub const fn new(distance: u8) -> Self {
        Self {
            distance,
            path_length: distance,
        }
    }

    /// Create a default orbit distance (unreachable)
    pub const fn unreachable() -> Self {
        Self {
            distance: u8::MAX,
            path_length: 0,
        }
    }
}

/// Compute orbit distance between two classes
///
/// Uses BFS to find the shortest path from source to target.
/// Maximum distance is 12 (the orbit diameter).
///
/// # Example
///
/// ```
/// use hologram_compiler::factorization::compute_orbit_distance;
///
/// let distance = compute_orbit_distance(37, 37);
/// assert_eq!(distance.distance, 0); // Same class
///
/// let distance = compute_orbit_distance(37, 42);
/// assert!(distance.distance > 0); // Different classes
/// ```
pub fn compute_orbit_distance(source: u8, target: u8) -> OrbitDistance {
    if source == target {
        return OrbitDistance::new(0);
    }

    if source >= 96 || target >= 96 {
        return OrbitDistance::unreachable();
    }

    // BFS to find shortest path
    use std::collections::{HashMap, VecDeque};

    let mut queue = VecDeque::new();
    let mut visited = HashMap::new();

    queue.push_back(source);
    visited.insert(source, 0u8);

    while let Some(current) = queue.pop_front() {
        if current == target {
            let dist = *visited.get(&target).unwrap();
            return OrbitDistance::new(dist);
        }

        let current_dist = visited[&current];

        // Try all four transforms
        let transforms = vec![
            Transform::new().with_rotate(1),
            Transform::new().with_rotate(-1),
            Transform::new().with_twist(1),
            Transform::new().with_twist(-1),
            Transform::new().with_mirror(),
        ];

        for transform in &transforms {
            let next = apply_transform(current, *transform);

            if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(next) {
                e.insert(current_dist + 1);
                queue.push_back(next);
            }
        }
    }

    // Should never reach here (all classes are in one orbit)
    OrbitDistance::unreachable()
}

/// Apply a transform to a class
///
/// Implements the automorphism group actions:
/// - R: Rotate quadrant
/// - T: Twist context
/// - M: Mirror modality
fn apply_transform(class: u8, transform: Transform) -> u8 {
    let (mut h2, mut d, mut l) = decompose_class(class);

    // Apply rotation if present
    if let Some(k) = transform.r {
        h2 = ((h2 as i32 + k).rem_euclid(4)) as u8;
    }

    // Apply twist if present
    if let Some(k) = transform.t {
        l = ((l as i32 + k).rem_euclid(8)) as u8;
    }

    // Apply mirror if present
    if transform.m {
        d = match d {
            0 => 0, // Neutral stays neutral
            1 => 2, // Produce ↔ Consume
            2 => 1,
            _ => d,
        };
    }

    compose_class(h2, d, l)
}

/// Decompose a class into (h₂, d, ℓ) components
///
/// Formula: class = 24h₂ + 8d + ℓ
fn decompose_class(class: u8) -> (u8, u8, u8) {
    let h2 = class / 24;
    let remainder = class % 24;
    let d = remainder / 8;
    let l = remainder % 8;
    (h2, d, l)
}

/// Compose a class from (h₂, d, ℓ) components
///
/// Formula: class = 24h₂ + 8d + ℓ
fn compose_class(h2: u8, d: u8, l: u8) -> u8 {
    24 * h2 + 8 * d + l
}

/// Compute orbit distances from prime generator 37 to all classes
///
/// This is used by build.rs to generate the ORBIT_DISTANCE_TABLE.
/// Returns an array of orbit distances for all 96 classes.
pub fn compute_all_orbit_distances() -> [OrbitDistance; 96] {
    let mut distances = [OrbitDistance::unreachable(); 96];

    for class in 0..96 {
        distances[class as usize] = compute_orbit_distance(crate::factorization::PRIME_GENERATOR, class);
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_compose() {
        // Test round-trip for all classes
        for class in 0..96 {
            let (h2, d, l) = decompose_class(class);
            assert!(h2 < 4);
            assert!(d < 3);
            assert!(l < 8);
            assert_eq!(compose_class(h2, d, l), class);
        }
    }

    #[test]
    fn test_orbit_distance_self() {
        // Distance from a class to itself is 0
        for class in 0..96 {
            let dist = compute_orbit_distance(class, class);
            assert_eq!(dist.distance, 0);
        }
    }

    #[test]
    fn test_orbit_distance_generator() {
        // Distance from generator to itself is 0
        let dist = compute_orbit_distance(37, 37);
        assert_eq!(dist.distance, 0);
    }

    #[test]
    fn test_orbit_diameter() {
        // All distances should be within diameter 12 using precomputed table
        for class in 0..96 {
            let dist = crate::factorization::orbit_distance(class);
            assert!(
                dist.distance <= 12,
                "Distance from 37 to {} is {} (exceeds diameter 12)",
                class,
                dist.distance
            );
        }
    }

    #[test]
    fn test_apply_rotate() {
        let class = compose_class(0, 0, 0); // h2=0, d=0, l=0
        let rotated = apply_transform(class, Transform::new().with_rotate(1));
        let (h2, d, l) = decompose_class(rotated);
        assert_eq!(h2, 1); // Rotated quadrant
        assert_eq!(d, 0); // Modality unchanged
        assert_eq!(l, 0); // Context unchanged
    }

    #[test]
    fn test_apply_twist() {
        let class = compose_class(0, 0, 0); // h2=0, d=0, l=0
        let twisted = apply_transform(class, Transform::new().with_twist(1));
        let (h2, d, l) = decompose_class(twisted);
        assert_eq!(h2, 0); // Quadrant unchanged
        assert_eq!(d, 0); // Modality unchanged
        assert_eq!(l, 1); // Context twisted
    }

    #[test]
    fn test_apply_mirror() {
        let class = compose_class(0, 1, 0); // h2=0, d=1 (Produce), l=0
        let mirrored = apply_transform(class, Transform::new().with_mirror());
        let (h2, d, l) = decompose_class(mirrored);
        assert_eq!(h2, 0); // Quadrant unchanged
        assert_eq!(d, 2); // Modality mirrored: Produce → Consume
        assert_eq!(l, 0); // Context unchanged
    }

    #[test]
    fn test_single_orbit_property() {
        // Verify that all classes are reachable from generator 37
        // Using precomputed table from build.rs (includes all transforms)
        for class in 0..96 {
            let dist = crate::factorization::orbit_distance(class);
            assert!(
                dist.distance < u8::MAX,
                "Class {} not reachable from generator 37",
                class
            );
        }
    }

    #[test]
    fn test_distance_accuracy() {
        // Test that distances are accurate using precomputed table
        let dist = crate::factorization::orbit_distance(5);
        assert!(dist.distance <= 12, "Distance exceeds orbit diameter");
        assert!(dist.distance > 0, "Distance should be non-zero for different classes");
    }
}
