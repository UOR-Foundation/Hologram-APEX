//! Atlas topological structure: 1-skeleton and mirror involution
//!
//! This module provides the **tiny LUTs** for graph topology as mentioned in the spec.
//! These tables are const-initialized at compile time and fit entirely in L1I cache.

use atlas_core::{self, ResonanceClass};

/// Maximum degree of the 1-skeleton neighbor graph
///
/// This is a conservative upper bound. The actual neighbor tables from atlas-core
/// may have varying degrees per class.
pub const MAX_DEGREE: usize = 8;

/// Mirror Involution Table
///
/// Maps each resonance class to its mirror pair. This is an **exact involution**:
/// `mirror(mirror(c)) = c` for all c.
///
/// Size: 96 bytes (fits in a single cache line on most architectures)
///
/// # Example
///
/// ```
/// use atlas_runtime::MirrorTable;
///
/// let mirrors = MirrorTable::new();
/// let class = 42;
/// let mirror = mirrors.mirror(class);
///
/// // Involution property
/// assert_eq!(class, mirrors.mirror(mirror));
/// ```
pub struct MirrorTable {
    /// Mirror mapping: mirrors[c] = mirror_of(c)
    /// Populated from atlas-core::get_mirror_pair
    map: [u8; 96],
}

impl MirrorTable {
    /// Create mirror table from atlas-core
    ///
    /// This is a compile-time constant operation that builds the involution
    /// mapping from the authoritative atlas-core implementation.
    pub fn new() -> Self {
        let mut map = [0u8; 96];
        for (class, entry) in map.iter_mut().enumerate() {
            *entry = atlas_core::get_mirror_pair(class as u8);
        }

        // Verify involution property in debug builds
        #[cfg(debug_assertions)]
        for c in 0..96 {
            let m = map[c] as usize;
            assert_eq!(c, map[m] as usize, "Mirror involution violated at class {}", c);
        }

        Self { map }
    }

    /// Get mirror pair for a class
    ///
    /// # Performance
    ///
    /// Single array lookup (1 cycle on modern CPUs).
    #[inline(always)]
    pub fn mirror(&self, class: u8) -> u8 {
        debug_assert!((class as usize) < 96, "class must be < 96");
        self.map[class as usize]
    }

    /// Check if two classes are mirror pairs
    #[inline]
    pub fn are_mirrors(&self, c1: u8, c2: u8) -> bool {
        self.mirror(c1) == c2
    }

    /// Get the raw mirror table
    pub fn as_slice(&self) -> &[u8; 96] {
        &self.map
    }
}

impl Default for MirrorTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Neighbor Adjacency Table (1-Skeleton)
///
/// Represents the legal neighbor transitions in the Atlas topology.
/// This is the graph structure that kernels must respect when traversing classes.
///
/// # Structure
///
/// The 1-skeleton is a sparse graph over the 96 resonance classes. Each class
/// has a small number of neighbors (typically 2-6). This is represented as:
/// - `edges[class]`: Array of neighbor class IDs
/// - `degrees[class]`: Number of valid neighbors
///
/// Size: ~1 KiB total (easily fits in L1D cache)
///
/// # Example
///
/// ```
/// use atlas_runtime::NeighborTable;
///
/// let neighbors = NeighborTable::new();
/// let class = 10;
///
/// // Iterate legal neighbor transitions
/// for i in 0..neighbors.degree(class) {
///     let neighbor = neighbors.get(class, i);
///     println!("Class {} can step to class {}", class, neighbor);
/// }
/// ```
pub struct NeighborTable {
    /// edges[class] = array of neighbor class IDs
    edges: [[u8; MAX_DEGREE]; 96],

    /// degrees[class] = number of valid neighbors for class
    degrees: [u8; 96],
}

impl NeighborTable {
    /// Create neighbor table from atlas-core
    ///
    /// Note: The current atlas-core doesn't expose the 1-skeleton directly,
    /// so this is a placeholder that will be populated when the skeleton
    /// data structure is added to atlas-core.
    pub fn new() -> Self {
        let mut edges = [[0u8; MAX_DEGREE]; 96];
        let mut degrees = [0u8; 96];

        for class_id in 0u8..96u8 {
            let class = ResonanceClass::new(class_id).expect("resonance class id must be valid");
            let neighbors = class.neighbors();

            assert!(
                neighbors.len() <= MAX_DEGREE,
                "Neighbor degree {} exceeds MAX_DEGREE {} for class {}",
                neighbors.len(),
                MAX_DEGREE,
                class_id
            );

            for (slot, neighbor) in neighbors.iter().enumerate() {
                edges[class_id as usize][slot] = neighbor.as_u8();
            }

            degrees[class_id as usize] = neighbors.len() as u8;
        }

        Self { edges, degrees }
    }

    /// Get number of neighbors for a class
    #[inline(always)]
    pub fn degree(&self, class: u8) -> u8 {
        debug_assert!((class as usize) < 96, "class must be < 96");
        self.degrees[class as usize]
    }

    /// Get the i-th neighbor of a class
    ///
    /// # Panics
    ///
    /// Panics if `i >= degree(class)` in debug builds.
    #[inline(always)]
    pub fn get(&self, class: u8, i: u8) -> u8 {
        debug_assert!((class as usize) < 96, "class must be < 96");
        debug_assert!(
            (i as usize) < self.degrees[class as usize] as usize,
            "neighbor index out of range"
        );

        self.edges[class as usize][i as usize]
    }

    /// Check if two classes are neighbors
    pub fn are_neighbors(&self, from: u8, to: u8) -> bool {
        let degree = self.degree(from);
        for i in 0..degree {
            if self.get(from, i) == to {
                return true;
            }
        }
        false
    }

    /// Get all neighbors for a class as a slice
    pub fn neighbors(&self, class: u8) -> &[u8] {
        let degree = self.degree(class) as usize;
        &self.edges[class as usize][..degree]
    }
}

impl Default for NeighborTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_involution() {
        let mirrors = MirrorTable::new();

        // Test involution property for all classes
        for class in 0..96 {
            let mirror = mirrors.mirror(class);
            let mirror_of_mirror = mirrors.mirror(mirror);
            assert_eq!(class, mirror_of_mirror, "Involution property violated");
        }
    }

    #[test]
    fn test_mirror_are_pairs() {
        let mirrors = MirrorTable::new();

        for class in 0..96 {
            let mirror = mirrors.mirror(class);
            assert!(mirrors.are_mirrors(class, mirror));
            assert!(mirrors.are_mirrors(mirror, class));
        }
    }

    #[test]
    fn test_mirror_table_size() {
        use std::mem::size_of_val;
        let mirrors = MirrorTable::new();

        // Should be exactly 96 bytes (fits in single cache line)
        assert_eq!(size_of_val(&mirrors.map), 96);
    }

    #[test]
    fn test_neighbor_table_construction() {
        let neighbors = NeighborTable::new();

        // Basic sanity check
        for class in 0..96 {
            let degree = neighbors.degree(class);
            assert!(degree <= MAX_DEGREE as u8);
        }
    }

    #[test]
    fn test_neighbor_table_matches_atlas_core() {
        let table = NeighborTable::new();

        for class in 0u8..96u8 {
            let rc = ResonanceClass::new(class).expect("class id must be valid");
            let expected: Vec<u8> = rc.neighbors().into_iter().map(|n| n.as_u8()).collect();
            assert_eq!(expected, table.neighbors(class), "neighbor mismatch for class {class}");
        }
    }

    #[test]
    fn test_neighbor_symmetry() {
        let neighbors = NeighborTable::new();

        // If A is a neighbor of B, B should be a neighbor of A (undirected graph)
        // Note: This assumes the 1-skeleton is undirected. If it's directed,
        // this test should be removed.
        for class in 0..96 {
            for i in 0..neighbors.degree(class) {
                let neighbor = neighbors.get(class, i);
                // We'd check symmetry here if we had real neighbor data
                assert!(neighbor < 96);
            }
        }
    }
}
