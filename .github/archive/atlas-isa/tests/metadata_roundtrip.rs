//! Integration tests for Atlas ISA metadata
//!
//! These tests focus on ensuring kernel metadata remains consistent across
//! serialization boundaries and helper APIs. They intentionally avoid any
//! device-specific assumptions so that Atlas stays target-agnostic.

use atlas_isa::metadata::{BoundaryFootprint, ClassMask, KernelMetadata, PhaseWindow};
use atlas_isa::uor::ResonanceClass;

#[test]
fn metadata_json_roundtrip_preserves_fields() {
    let mut meta = KernelMetadata::new("vector_add");
    meta.mirror_safe = true;
    meta.unity_neutral = true;
    meta.uses_boundary = true;
    meta.boundary.page_min = 4;
    meta.boundary.page_max = 12;
    meta.boundary.byte_min = 8;
    meta.boundary.byte_max = 200;
    meta.phase.begin = 64;
    meta.phase.span = 128;

    let mut mask = ClassMask::empty();
    mask.set(ResonanceClass::new(3).unwrap());
    mask.set(ResonanceClass::new(15).unwrap());
    mask.set(ResonanceClass::new(42).unwrap());
    meta.classes_mask = mask;

    // Ensure the metadata validates before serialization
    let mirror = atlas_core::get_mirror_pair(3);
    meta.classes_mask
        .set(ResonanceClass::new(mirror).expect("mirror class must exist"));

    let encoded = serde_json::to_string(&meta).expect("metadata serializes");
    let decoded: KernelMetadata = serde_json::from_str(&encoded).expect("metadata deserializes");

    assert_eq!(decoded.name, meta.name);
    assert_eq!(decoded.boundary, meta.boundary);
    assert_eq!(decoded.phase, meta.phase);
    assert_eq!(decoded.mirror_safe, meta.mirror_safe);
    assert_eq!(decoded.unity_neutral, meta.unity_neutral);
    assert_eq!(decoded.uses_boundary, meta.uses_boundary);
    assert_eq!(decoded.classes_mask.count(), meta.classes_mask.count());
}

#[test]
fn boundary_footprint_contains_points() {
    let footprint = BoundaryFootprint {
        page_min: 2,
        page_max: 6,
        byte_min: 10,
        byte_max: 120,
    };

    assert!(footprint.contains(2, 10));
    assert!(footprint.contains(4, 64));
    assert!(footprint.contains(6, 120));

    assert!(!footprint.contains(1, 10));
    assert!(!footprint.contains(7, 10));
    assert!(!footprint.contains(4, 9));
    assert!(!footprint.contains(4, 121));
}

#[test]
fn phase_window_wraps_correctly() {
    let window = PhaseWindow { begin: 700, span: 100 };

    // Covers [700, 768) ∪ [0, 32)
    for phase in [700u32, 750, 0, 10, 31] {
        assert!(window.contains(phase), "phase {phase} should be inside window");
    }

    for phase in [699u32, 32, 400, 699] {
        assert!(!window.contains(phase), "phase {phase} should be outside window");
    }
}

#[test]
fn mirror_safe_metadata_requires_pairs() {
    let mut meta = KernelMetadata::new("mirror_sensitive");
    meta.mirror_safe = true;

    let class = ResonanceClass::new(25).unwrap();
    meta.classes_mask = ClassMask::single(class);

    // Without the mirror pair validation must fail
    assert!(meta.validate().is_err());

    let mirror = atlas_core::get_mirror_pair(class.as_u8());
    meta.classes_mask
        .set(ResonanceClass::new(mirror).expect("mirror class must exist"));

    // With the mirror present validation should succeed
    assert!(meta.validate().is_ok());
}
