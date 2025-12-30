//! Topology table builders for Atlas backends
//!
//! Provides helpers to materialize the topology lookup tables required by
//! backends during initialization. The data is sourced from the authoritative
//! `atlas-core` crate to ensure perfect agreement with Atlas invariants.

use atlas_core::{get_mirror_pair, ResonanceClass};

use crate::types::{NEIGHBOR_SLOTS, RESONANCE_CLASS_COUNT};

/// Compute the mirror involution table for all resonance classes.
///
/// The returned array satisfies `mirrors[mirrors[c] as usize] == c` for every
/// class `c` in `[0, 96)`.
pub fn compute_mirrors() -> [u8; RESONANCE_CLASS_COUNT] {
    let mut mirrors = [0u8; RESONANCE_CLASS_COUNT];

    for (class_id, slot) in mirrors.iter_mut().enumerate() {
        let mirror = get_mirror_pair(class_id as u8);
        debug_assert!(mirror < RESONANCE_CLASS_COUNT as u8);
        *slot = mirror;
    }

    debug_assert!(mirrors.iter().enumerate().all(|(idx, &mirror)| {
        let mirror_idx = mirror as usize;
        mirror_idx < RESONANCE_CLASS_COUNT && mirrors[mirror_idx] == idx as u8
    }));

    mirrors
}

/// Compute the 1-skeleton (adjacency) table for the Atlas resonance graph.
///
/// Empty neighbor slots are filled with `u8::MAX` for ergonomic iteration.
pub fn compute_1_skeleton() -> [[u8; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT] {
    let mut neighbors = [[u8::MAX; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT];

    for (class_id, slots) in neighbors.iter_mut().enumerate() {
        let class = ResonanceClass::new(class_id as u8).expect("class id must be valid in compute_1_skeleton");
        let mut local_neighbors = class.neighbors();
        local_neighbors.sort();

        debug_assert!(local_neighbors.len() <= NEIGHBOR_SLOTS);
        for (slot, neighbor) in local_neighbors.into_iter().enumerate() {
            slots[slot] = neighbor.id();
        }
    }

    // Verify bidirectional connectivity in debug builds.
    for (class_id, slots) in neighbors.iter().enumerate() {
        for neighbor in slots.iter().copied().filter(|&n| n != u8::MAX) {
            let neighbor_idx = neighbor as usize;
            debug_assert!(neighbor_idx < RESONANCE_CLASS_COUNT);
            debug_assert!(
                neighbors[neighbor_idx].contains(&(class_id as u8)),
                "neighbor relationship must be bidirectional: {class_id} -> {neighbor_idx}"
            );
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn mirrors_form_involution() {
        let mirrors = compute_mirrors();
        for (idx, &mirror) in mirrors.iter().enumerate() {
            assert!(mirror < RESONANCE_CLASS_COUNT as u8);
            let back = mirrors[mirror as usize];
            assert_eq!(back, idx as u8, "mirror involution broken at {idx}");
        }
    }

    #[test]
    fn neighbors_match_degree_and_are_bidirectional() {
        let table = compute_1_skeleton();

        for class_id in 0..RESONANCE_CLASS_COUNT {
            let class = ResonanceClass::new(class_id as u8).unwrap();
            let expected_degree = class.degree();
            let row = &table[class_id];
            let actual_neighbors: Vec<_> = row.iter().copied().filter(|&n| n != u8::MAX).collect();
            assert_eq!(
                actual_neighbors.len(),
                expected_degree,
                "neighbor count mismatch for class {class_id}"
            );

            for neighbor in actual_neighbors {
                let neighbor_idx = neighbor as usize;
                assert!(table[neighbor_idx].contains(&(class_id as u8)));
            }
        }
    }

    // =====================================================================
    // Property-Based Tests for Topology Invariants
    // =====================================================================

    proptest! {
        #[test]
        fn prop_mirror_involution_for_any_class(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let mirrors = compute_mirrors();
            let mirror = mirrors[class_id as usize];
            let double_mirror = mirrors[mirror as usize];

            // Involution: mirror(mirror(x)) = x
            prop_assert_eq!(
                double_mirror,
                class_id,
                "Mirror involution failed for class {}: mirror({}) = {}, mirror({}) = {}",
                class_id,
                class_id,
                mirror,
                mirror,
                double_mirror
            );

            // Mirror must be in valid range
            prop_assert!(
                mirror < RESONANCE_CLASS_COUNT as u8,
                "Mirror {} for class {} is out of bounds",
                mirror,
                class_id
            );
        }

        #[test]
        fn prop_neighbors_are_bidirectional(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let neighbors = &table[class_id as usize];

            for &neighbor in neighbors.iter().filter(|&&n| n != u8::MAX) {
                let neighbor_idx = neighbor as usize;

                // Bidirectionality: if B is a neighbor of A, then A is a neighbor of B
                prop_assert!(
                    table[neighbor_idx].contains(&class_id),
                    "Neighbor relationship not bidirectional: class {} lists {} as neighbor, but {} doesn't list {} back",
                    class_id,
                    neighbor,
                    neighbor,
                    class_id
                );
            }
        }

        #[test]
        fn prop_neighbor_count_matches_degree(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let class = ResonanceClass::new(class_id).unwrap();
            let expected_degree = class.degree();
            let neighbors = &table[class_id as usize];
            let actual_count = neighbors.iter().filter(|&&n| n != u8::MAX).count();

            prop_assert_eq!(
                actual_count,
                expected_degree,
                "Neighbor count mismatch for class {}: expected {} (degree), got {} (actual neighbors)",
                class_id,
                expected_degree,
                actual_count
            );
        }

        #[test]
        fn prop_neighbors_are_unique(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let neighbors: Vec<u8> = table[class_id as usize]
                .iter()
                .copied()
                .filter(|&n| n != u8::MAX)
                .collect();

            let mut sorted = neighbors.clone();
            sorted.sort();
            sorted.dedup();

            prop_assert_eq!(
                sorted.len(),
                neighbors.len(),
                "Class {} has duplicate neighbors: {:?}",
                class_id,
                neighbors
            );
        }

        #[test]
        fn prop_neighbors_are_in_bounds(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let neighbors = &table[class_id as usize];

            for &neighbor in neighbors.iter().filter(|&&n| n != u8::MAX) {
                prop_assert!(
                    neighbor < RESONANCE_CLASS_COUNT as u8,
                    "Neighbor {} of class {} is out of bounds",
                    neighbor,
                    class_id
                );
            }
        }

        #[test]
        fn prop_no_self_neighbors(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let neighbors = &table[class_id as usize];

            for &neighbor in neighbors.iter().filter(|&&n| n != u8::MAX) {
                prop_assert_ne!(
                    neighbor,
                    class_id,
                    "Class {} lists itself as a neighbor",
                    class_id
                );
            }
        }

        #[test]
        fn prop_neighbor_slots_sorted(class_id in 0u8..RESONANCE_CLASS_COUNT as u8) {
            let table = compute_1_skeleton();
            let neighbors: Vec<u8> = table[class_id as usize]
                .iter()
                .copied()
                .filter(|&n| n != u8::MAX)
                .collect();

            let mut sorted = neighbors.clone();
            sorted.sort();

            prop_assert_eq!(
                &neighbors,
                &sorted,
                "Neighbors of class {} are not sorted: {:?}",
                class_id,
                neighbors
            );
        }
    }
}
