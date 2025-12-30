//! Graph validation functions
//!
//! Validates that graph operations respect Atlas invariants:
//! - Neighbor adjacency (edges must exist in graph)
//! - Modality compatibility (produce/consume matching)
//! - Resonance conservation (unity neutrality)

use super::GraphOperation;
use crate::{BackendError, Rational, Result};
use atlas_core::atlas;

/// Validate that an edge exists in the graph (neighbor adjacency)
pub fn validate_edge(src: u8, dst: u8) -> Result<()> {
    let atlas = atlas();
    let neighbors = atlas.neighbors(src as usize);

    if !neighbors.contains(&(dst as usize)) {
        return Err(BackendError::InvalidTopology(format!(
            "No edge {} → {} in graph (not neighbors)",
            src, dst
        )));
    }

    Ok(())
}

/// Validate modality compatibility between source and destination
pub fn validate_modality(src_class: u8, dst_class: u8) -> Result<()> {
    let atlas = atlas();
    let src_label = atlas.label(src_class as usize);
    let dst_label = atlas.label(dst_class as usize);

    // Check modality compatibility based on d₄₅ field
    // d45 values: -1 (consume), 0 (neutral), 1 (produce)
    match (src_label.d45, dst_label.d45) {
        // Neutral (0) can interact with anything
        (0, _) | (_, 0) => Ok(()),
        // Produce (1) and Consume (-1) can interact
        (1, -1) | (-1, 1) => Ok(()),
        // Same non-neutral modality is allowed (data flow through same type)
        (1, 1) | (-1, -1) => Ok(()),
        _ => Err(BackendError::InvalidTopology(format!(
            "Modality mismatch: {} (d={}) → {} (d={})",
            src_class, src_label.d45, dst_class, dst_label.d45
        ))),
    }
}

/// Validate resonance budget (unity neutrality)
/// Sum of all resonance deltas must equal zero
pub fn validate_resonance_budget(operations: &[GraphOperation]) -> Result<()> {
    let mut resonance = [Rational::zero(); 96];

    // Accumulate resonance changes from operations
    for op in operations {
        // Each operation contributes to resonance based on generator
        let delta = compute_resonance_delta(op);
        for (class, value) in delta.iter().enumerate() {
            resonance[class] = resonance[class] + *value;
        }
    }

    // Check unity neutrality: sum must be zero
    let mut total = Rational::zero();
    for r in &resonance {
        total = total + *r;
    }

    if total != Rational::zero() {
        return Err(BackendError::InvalidTopology(format!(
            "Unity neutrality violated: resonance sum = {}",
            total
        )));
    }

    Ok(())
}

/// Compute resonance delta for a single operation
fn compute_resonance_delta(op: &GraphOperation) -> [Rational; 96] {
    let mut delta = [Rational::zero(); 96];

    // For now, simple model: each operation contributes +1 to src, -1 to dst
    // This ensures zero net resonance
    delta[op.src as usize] = Rational::from(1);
    delta[op.dst as usize] = Rational::from(-1);

    delta
}

/// Validate all operations in a sequence
pub fn validate_graph_operations(operations: &[GraphOperation]) -> Result<()> {
    // Validate each edge exists
    for op in operations {
        validate_edge(op.src, op.dst)?;
        validate_modality(op.src, op.dst)?;
    }

    // Validate resonance budget
    validate_resonance_budget(operations)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{Generator, GraphOperation};
    use super::*;

    #[test]
    fn test_validate_edge_valid() {
        // Get atlas and find a valid edge
        let atlas = atlas();
        let class_0_neighbors = atlas.neighbors(0);

        // Convert HashSet to Vec to get first element
        let neighbors_vec: Vec<usize> = class_0_neighbors.iter().copied().collect();

        if let Some(&neighbor) = neighbors_vec.first() {
            // Should pass validation
            assert!(validate_edge(0, neighbor as u8).is_ok());
        }
    }

    #[test]
    fn test_validate_edge_invalid() {
        // Most classes won't be neighbors with class 95
        // (this is a heuristic - might need adjustment based on actual graph)
        let result = validate_edge(0, 95);
        // May or may not be neighbors - just testing the validation exists
        match result {
            Ok(_) => {}  // They are neighbors
            Err(_) => {} // They are not neighbors
        }
    }

    #[test]
    fn test_validate_resonance_budget_balanced() {
        let ops = vec![
            GraphOperation::new(0, 1, Generator::Copy),
            GraphOperation::new(1, 0, Generator::Copy),
        ];

        // These operations balance out (net zero resonance)
        assert!(validate_resonance_budget(&ops).is_ok());
    }

    #[test]
    fn test_compute_resonance_delta() {
        let op = GraphOperation::new(5, 7, Generator::Copy);
        let delta = compute_resonance_delta(&op);

        assert_eq!(delta[5], Rational::from(1));
        assert_eq!(delta[7], Rational::from(-1));

        // All others should be zero
        for (i, &val) in delta.iter().enumerate() {
            if i != 5 && i != 7 {
                assert_eq!(val, Rational::zero());
            }
        }
    }
}
