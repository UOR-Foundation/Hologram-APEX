use atlas_core::{check_unity_neutrality, verify_unity_neutrality, AtlasRatio};
use atlas_isa::{ClassMask, KernelMetadata, ResonanceClass};

#[test]
fn atlas_ratio_layout_matches_c() {
    use std::mem::{align_of, size_of};

    assert_eq!(size_of::<AtlasRatio>(), 16);
    assert_eq!(align_of::<AtlasRatio>(), 8);
}

#[test]
fn abi_unity_neutrality_roundtrip() {
    let deltas = [
        AtlasRatio::new_raw(1, 2),
        AtlasRatio::new_raw(-1, 2),
        AtlasRatio::new_raw(2, 3),
        AtlasRatio::new_raw(-2, 3),
    ];

    let ok = unsafe { check_unity_neutrality(deltas.as_ptr(), deltas.len()) };
    assert!(ok, "expected neutrality to hold");

    let mut broken = deltas;
    broken[0] = AtlasRatio::new_raw(1, 3);
    let ok = unsafe { check_unity_neutrality(broken.as_ptr(), broken.len()) };
    assert!(!ok, "expected neutrality violation to fail");
}

#[test]
fn hologram_style_metadata_validation() {
    let mut meta = KernelMetadata::new("hologram-driver-kernel");
    meta.unity_neutral = true;
    meta.mirror_safe = true;

    let class = ResonanceClass::classify(5);
    let mirror = class.mirror();
    meta.classes_mask = ClassMask::empty();
    meta.classes_mask.set(class);
    meta.classes_mask.set(mirror);

    meta.validate().expect("metadata should pass invariant checks");

    let mut delta = [AtlasRatio::new_raw(0, 1); 96];
    delta[class.as_u8() as usize] = AtlasRatio::new_raw(3, 8);
    delta[mirror.as_u8() as usize] = AtlasRatio::new_raw(-3, 8);

    verify_unity_neutrality(&delta).expect("runtime neutrality check should pass");
}
