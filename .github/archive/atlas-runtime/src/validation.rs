//! Runtime validation for Atlas invariants
//!
//! Implements validation checks as specified in Atlas Runtime Spec §14.

use atlas_isa::KernelMetadata;

use crate::error::{AtlasError, Result};
use crate::space::AtlasSpace;

/// Validation layer for Atlas invariants
///
/// Provides runtime checks for:
/// - Phase window constraints
/// - Boundary footprint validation
/// - Neighbor legality (1-skeleton)
/// - Unity neutrality
/// - Mirror safety
pub struct Validator;

impl Validator {
    /// Validate that current phase is within kernel's declared window
    ///
    /// Per Atlas Runtime Spec §8.4: A kernel with `(phase_begin, phase_span)`
    /// only executes when `P ∈ [begin, begin+span) mod 768`.
    pub fn check_phase_window(space: &AtlasSpace, metadata: &KernelMetadata) -> Result<()> {
        let current_phase = space.phase().get();

        if !space
            .phase()
            .in_window(metadata.phase.begin as u16, metadata.phase.span as u16)
        {
            return Err(AtlasError::phase_window(
                current_phase,
                metadata.phase.begin as u16,
                metadata.phase.span as u16,
            ));
        }

        Ok(())
    }

    /// Validate boundary coordinates are within declared footprint
    ///
    /// Per Atlas Runtime Spec §8.3: Accesses via BOUND.MAP must remain within
    /// `(x_min..x_max, y_min..y_max)` when declared.
    pub fn check_boundary_window(metadata: &KernelMetadata, x: u8, y: u8) -> Result<()> {
        if !metadata.uses_boundary {
            return Ok(()); // No boundary constraints
        }

        let footprint = &metadata.boundary;

        if x < footprint.page_min || x >= footprint.page_max || y < footprint.byte_min || y >= footprint.byte_max {
            return Err(AtlasError::boundary_violation(
                x,
                y,
                (footprint.page_min, footprint.page_max),
                (footprint.byte_min, footprint.byte_max),
            ));
        }

        Ok(())
    }

    /// Validate neighbor traversal is legal (1-skeleton)
    ///
    /// Per Atlas Runtime Spec §8.5: Legal neighbor traversals must use NBR.*;
    /// attempts to use non-neighbors MAY be rejected in checked builds.
    pub fn check_neighbor_legal(space: &AtlasSpace, from_class: u8, to_class: u8) -> Result<()> {
        // Self-loops are always legal
        if from_class == to_class {
            return Ok(());
        }

        // Mirror transitions are always legal
        if space.mirrors().mirror(from_class) == to_class {
            return Ok(());
        }

        // Check 1-skeleton
        if !space.neighbors().are_neighbors(from_class, to_class) {
            return Err(AtlasError::NeighborIllegal {
                from: from_class,
                to: to_class,
            });
        }

        Ok(())
    }

    /// Validate mirror safety for a kernel
    ///
    /// Per Atlas Runtime Spec §8.1: If `mirror_safe=1`, the kernel's effect
    /// is invariant under classwise mapping `c → mirror(c)`.
    pub fn check_mirror_safety(space: &AtlasSpace, metadata: &KernelMetadata) -> Result<()> {
        if !metadata.mirror_safe {
            return Ok(()); // Not required to be mirror-safe
        }

        // For each class in the mask, verify its mirror is also in the mask
        for class in metadata.classes_mask.active_classes() {
            let mirror = space.mirrors().mirror(class.as_u8());
            let mirror_class = atlas_isa::ResonanceClass::new(mirror).map_err(|_| AtlasError::InvalidClass(mirror))?;

            if !metadata.classes_mask.is_set(mirror_class) {
                return Err(AtlasError::MirrorSafetyViolation {
                    class: class.as_u8(),
                    mirror,
                });
            }
        }

        Ok(())
    }

    /// Validate unity neutrality
    ///
    /// Per Atlas Runtime Spec §8.2: If `unity_neutral=1`, the net change to
    /// `R[96]` over the kernel launch must be zero.
    pub fn check_unity_neutrality(space: &AtlasSpace) -> Result<()> {
        space.resonance().check_unity_neutral()
    }

    /// Validate kernel metadata before launch
    ///
    /// Comprehensive pre-launch checks:
    /// - Phase window
    /// - Mirror safety (mask completeness)
    /// - Metadata self-consistency
    pub fn validate_launch(space: &AtlasSpace, metadata: &KernelMetadata) -> Result<()> {
        // Check phase window
        Self::check_phase_window(space, metadata)?;

        // Check mirror safety
        Self::check_mirror_safety(space, metadata)?;

        // Validate metadata itself
        metadata
            .validate()
            .map_err(|e| AtlasError::InvalidMetadata(e.to_string()))?;

        Ok(())
    }

    /// Validate post-launch (after kernel execution)
    ///
    /// Checks invariants that must hold after execution:
    /// - Unity neutrality (if declared)
    pub fn validate_post_launch(space: &AtlasSpace, metadata: &KernelMetadata) -> Result<()> {
        if metadata.unity_neutral {
            Self::check_unity_neutrality(space)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::AtlasRatio;
    use atlas_isa::{BoundaryFootprint, ClassMask, PhaseWindow};

    fn create_test_metadata(name: &str) -> KernelMetadata {
        KernelMetadata {
            name: name.to_string(),
            classes_mask: ClassMask::all(),
            mirror_safe: false,
            unity_neutral: false,
            uses_boundary: false,
            boundary: BoundaryFootprint::default(),
            phase: PhaseWindow::full(),
        }
    }

    #[test]
    fn test_phase_window_valid() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");

        // Set phase and window to overlap
        space.phase().set(10);
        metadata.phase.begin = 5;
        metadata.phase.span = 20;

        assert!(Validator::check_phase_window(&space, &metadata).is_ok());
    }

    #[test]
    fn test_phase_window_invalid() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");

        // Set phase outside window
        space.phase().set(100);
        metadata.phase.begin = 5;
        metadata.phase.span = 20;

        assert!(Validator::check_phase_window(&space, &metadata).is_err());
    }

    #[test]
    fn test_boundary_window_valid() {
        let mut metadata = create_test_metadata("test");
        metadata.uses_boundary = true;
        metadata.boundary.page_min = 10;
        metadata.boundary.page_max = 20;
        metadata.boundary.byte_min = 50;
        metadata.boundary.byte_max = 150;

        assert!(Validator::check_boundary_window(&metadata, 15, 100).is_ok());
    }

    #[test]
    fn test_boundary_window_invalid_x() {
        let mut metadata = create_test_metadata("test");
        metadata.uses_boundary = true;
        metadata.boundary.page_min = 10;
        metadata.boundary.page_max = 20;
        metadata.boundary.byte_min = 50;
        metadata.boundary.byte_max = 150;

        // x out of range
        assert!(Validator::check_boundary_window(&metadata, 5, 100).is_err());
    }

    #[test]
    fn test_boundary_window_invalid_y() {
        let mut metadata = create_test_metadata("test");
        metadata.uses_boundary = true;
        metadata.boundary.page_min = 10;
        metadata.boundary.page_max = 20;
        metadata.boundary.byte_min = 50;
        metadata.boundary.byte_max = 150;

        // y out of range
        assert!(Validator::check_boundary_window(&metadata, 15, 200).is_err());
    }

    #[test]
    fn test_neighbor_legal_self() {
        let space = AtlasSpace::new();

        // Self-loops are always legal
        assert!(Validator::check_neighbor_legal(&space, 10, 10).is_ok());
    }

    #[test]
    fn test_neighbor_legal_mirror() {
        let space = AtlasSpace::new();

        // Mirror transitions are always legal
        let class = 10u8;
        let mirror = space.mirrors().mirror(class);

        assert!(Validator::check_neighbor_legal(&space, class, mirror).is_ok());
    }

    #[test]
    fn test_mirror_safety_complete() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");
        metadata.mirror_safe = true;

        // Create mask with mirror pairs
        let class = atlas_isa::ResonanceClass::new(10).unwrap();
        let mirror_id = space.mirrors().mirror(10);
        let mirror = atlas_isa::ResonanceClass::new(mirror_id).unwrap();

        let mut mask = ClassMask::empty();
        mask.set(class);
        mask.set(mirror);
        metadata.classes_mask = mask;

        assert!(Validator::check_mirror_safety(&space, &metadata).is_ok());
    }

    #[test]
    fn test_mirror_safety_incomplete() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");
        metadata.mirror_safe = true;

        // Create mask without mirror pair
        let class = atlas_isa::ResonanceClass::new(10).unwrap();
        let mut mask = ClassMask::empty();
        mask.set(class);
        metadata.classes_mask = mask;

        assert!(Validator::check_mirror_safety(&space, &metadata).is_err());
    }

    #[test]
    fn test_unity_neutrality_balanced() {
        let space = AtlasSpace::new();

        // Add balanced deltas
        space.resonance().add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        space.resonance().add(1, AtlasRatio::new_raw(-1, 2)).unwrap();

        assert!(Validator::check_unity_neutrality(&space).is_ok());
    }

    #[test]
    fn test_unity_neutrality_imbalanced() {
        let space = AtlasSpace::new();

        // Add unbalanced delta
        space.resonance().add(0, AtlasRatio::new_raw(1, 2)).unwrap();

        assert!(Validator::check_unity_neutrality(&space).is_err());
    }

    #[test]
    fn test_unity_neutrality_overflow_propagates() {
        let space = AtlasSpace::new();
        // Seed accumulator near numeric limits so the neutral check must handle overflow.
        space.resonance().set(0, AtlasRatio::new_raw(i64::MAX, 1)).unwrap();
        space.resonance().set(1, AtlasRatio::new_raw(1, 1)).unwrap();

        let err = Validator::check_unity_neutrality(&space).expect_err("overflow should propagate through validator");
        assert!(matches!(err, AtlasError::ResonanceOverflow { .. }));
    }

    #[test]
    fn test_validate_launch_success() {
        let space = AtlasSpace::new();
        let metadata = create_test_metadata("test");

        assert!(Validator::validate_launch(&space, &metadata).is_ok());
    }

    #[test]
    fn test_validate_post_launch_unity() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");
        metadata.unity_neutral = true;

        // Balanced resonance
        space.resonance().add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        space.resonance().add(1, AtlasRatio::new_raw(-1, 2)).unwrap();

        assert!(Validator::validate_post_launch(&space, &metadata).is_ok());
    }

    #[test]
    fn test_validate_post_launch_unity_fail() {
        let space = AtlasSpace::new();
        let mut metadata = create_test_metadata("test");
        metadata.unity_neutral = true;

        // Unbalanced resonance
        space.resonance().add(0, AtlasRatio::new_raw(1, 2)).unwrap();

        assert!(Validator::validate_post_launch(&space, &metadata).is_err());
    }
}
