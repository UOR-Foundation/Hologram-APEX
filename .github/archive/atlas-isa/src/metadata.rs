//! Kernel metadata for Atlas invariant enforcement

use crate::{constants::*, uor::ResonanceClass, AtlasError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// 96-bit class mask represented as two u64 values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassMask {
    pub low: u64,  // Classes 0-63
    pub high: u32, // Classes 64-95
}

impl ClassMask {
    /// Create empty mask (no classes)
    pub const fn empty() -> Self {
        Self { low: 0, high: 0 }
    }

    /// Create mask with all classes enabled
    pub const fn all() -> Self {
        Self {
            low: u64::MAX,
            high: 0xFFFF_FFFF, // Only lower 32 bits used for classes 64-95
        }
    }

    /// Create mask with single class
    pub fn single(class: ResonanceClass) -> Self {
        let mut mask = Self::empty();
        mask.set(class);
        mask
    }

    /// Check if a class is enabled in this mask
    pub fn is_set(&self, class: ResonanceClass) -> bool {
        let id = class.as_u8() as u32;
        if id < 64 {
            (self.low & (1u64 << id)) != 0
        } else {
            (self.high & (1u32 << (id - 64))) != 0
        }
    }

    /// Enable a class in this mask
    pub fn set(&mut self, class: ResonanceClass) {
        let id = class.as_u8() as u32;
        if id < 64 {
            self.low |= 1u64 << id;
        } else {
            self.high |= 1u32 << (id - 64);
        }
    }

    /// Disable a class in this mask
    pub fn clear(&mut self, class: ResonanceClass) {
        let id = class.as_u8() as u32;
        if id < 64 {
            self.low &= !(1u64 << id);
        } else {
            self.high &= !(1u32 << (id - 64));
        }
    }

    /// Check if two masks have overlapping classes
    pub fn overlaps(&self, other: &Self) -> bool {
        (self.low & other.low) != 0 || (self.high & other.high) != 0
    }

    /// Check if this mask is disjoint from another (no overlap)
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.overlaps(other)
    }

    /// Compute intersection of two masks
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            low: self.low & other.low,
            high: self.high & other.high,
        }
    }

    /// Compute union of two masks
    pub fn union(&self, other: &Self) -> Self {
        Self {
            low: self.low | other.low,
            high: self.high | other.high,
        }
    }

    /// Count number of active classes
    pub fn count(&self) -> u32 {
        self.low.count_ones() + self.high.count_ones()
    }

    /// Get list of active class IDs
    pub fn active_classes(&self) -> Vec<ResonanceClass> {
        let mut classes = Vec::new();
        for i in 0..64 {
            if (self.low & (1u64 << i)) != 0 {
                classes
                    .push(ResonanceClass::new(i as u8).expect("bitmask indices 0..63 must be valid resonance classes"));
            }
        }
        for i in 0..32 {
            let class_id = 64 + i;
            if class_id < 96 && (self.high & (1u32 << i)) != 0 {
                classes.push(
                    ResonanceClass::new(class_id as u8)
                        .expect("bitmask indices 64..95 must be valid resonance classes"),
                );
            }
        }
        classes
    }
}

impl Default for ClassMask {
    fn default() -> Self {
        Self::all()
    }
}

/// Boundary footprint: (page, byte) ranges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryFootprint {
    pub page_min: u8,
    pub page_max: u8,
    pub byte_min: u8,
    pub byte_max: u8,
}

impl BoundaryFootprint {
    /// Create footprint covering entire boundary
    pub fn full() -> Self {
        Self {
            page_min: 0,
            page_max: (PAGES - 1) as u8,
            byte_min: 0,
            byte_max: (BYTES_PER_PAGE - 1) as u8,
        }
    }

    /// Validate that coordinates are in range
    pub fn validate(&self) -> Result<()> {
        if self.page_min >= PAGES as u8 {
            return Err(AtlasError::InvalidPage(self.page_min as u32));
        }
        if self.page_max >= PAGES as u8 {
            return Err(AtlasError::InvalidPage(self.page_max as u32));
        }
        if self.page_min > self.page_max {
            return Err(AtlasError::InvalidMetadata("page_min > page_max".to_string()));
        }
        // Note: byte values are u8, so byte_min/max are always < 256
        if self.byte_min > self.byte_max {
            return Err(AtlasError::InvalidMetadata("byte_min > byte_max".to_string()));
        }
        Ok(())
    }

    /// Check if (page, byte) is within this footprint
    pub fn contains(&self, page: u8, byte: u8) -> bool {
        page >= self.page_min && page <= self.page_max && byte >= self.byte_min && byte <= self.byte_max
    }
}

impl Default for BoundaryFootprint {
    fn default() -> Self {
        Self::full()
    }
}

/// Phase window [begin, begin+span) mod 768
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseWindow {
    pub begin: u32,
    pub span: u32,
}

impl PhaseWindow {
    /// Create window covering all phases
    pub fn full() -> Self {
        Self {
            begin: 0,
            span: PHASE_MODULUS,
        }
    }

    /// Validate phase window
    pub fn validate(&self) -> Result<()> {
        if self.begin >= PHASE_MODULUS {
            return Err(AtlasError::InvalidPhase(self.begin));
        }
        if self.span > PHASE_MODULUS {
            return Err(AtlasError::InvalidMetadata(format!(
                "phase span {} exceeds modulus {}",
                self.span, PHASE_MODULUS
            )));
        }
        Ok(())
    }

    /// Check if a phase is within this window
    pub fn contains(&self, phase: u32) -> bool {
        let phase = phase % PHASE_MODULUS;
        if self.span >= PHASE_MODULUS {
            return true; // Full window
        }

        let end = (self.begin + self.span) % PHASE_MODULUS;
        if self.begin < end {
            // No wrap: [begin, end)
            phase >= self.begin && phase < end
        } else {
            // Wraps around: [begin, 768) ∪ [0, end)
            phase >= self.begin || phase < end
        }
    }
}

impl Default for PhaseWindow {
    fn default() -> Self {
        Self::full()
    }
}

/// Complete kernel metadata for Atlas invariant enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Name of the kernel
    pub name: String,

    /// 96-bit mask of classes this kernel accesses
    pub classes_mask: ClassMask,

    /// Kernel effect invariant under class-wise mirror pairing
    pub mirror_safe: bool,

    /// Net change to R\[96] must be zero; deltas are provided as `AtlasRatio` arrays
    pub unity_neutral: bool,

    /// Kernel uses boundary lens for addressing
    pub uses_boundary: bool,

    /// Boundary footprint (if uses_boundary)
    pub boundary: BoundaryFootprint,

    /// Phase window for execution (modulo 768)
    pub phase: PhaseWindow,
}

impl KernelMetadata {
    /// Create default metadata with name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            classes_mask: ClassMask::all(),
            mirror_safe: false,
            unity_neutral: false,
            uses_boundary: false,
            boundary: BoundaryFootprint::default(),
            phase: PhaseWindow::default(),
        }
    }

    /// Validate all metadata fields
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(AtlasError::InvalidMetadata("kernel name cannot be empty".to_string()));
        }

        if self.classes_mask.count() == 0 {
            return Err(AtlasError::InvalidMetadata(
                "kernel must access at least one class".to_string(),
            ));
        }

        self.boundary.validate()?;
        self.phase.validate()?;

        if self.mirror_safe {
            for class in self.classes_mask.active_classes() {
                let mirror = atlas_core::get_mirror_pair(class.as_u8());
                if mirror >= RESONANCE_CLASSES as u8 {
                    return Err(AtlasError::InvalidMetadata(format!(
                        "mirror pair is undefined for class {}",
                        class.as_u8()
                    )));
                }

                let mirror_class = ResonanceClass::new(mirror).map_err(|_| {
                    AtlasError::InvalidMetadata(format!("mirror class {} fell outside the resonance range", mirror))
                })?;

                if !self.classes_mask.is_set(mirror_class) {
                    return Err(AtlasError::InvalidMetadata(format!(
                        "mirror pair {} missing for class {}",
                        mirror,
                        class.as_u8()
                    )));
                }
            }
        }

        Ok(())
    }
}

impl fmt::Display for KernelMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Kernel '{}': {} classes, mirror_safe={}, unity_neutral={}, uses_boundary={}",
            self.name,
            self.classes_mask.count(),
            self.mirror_safe,
            self.unity_neutral,
            self.uses_boundary
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_mask_empty() {
        let mask = ClassMask::empty();
        assert_eq!(mask.count(), 0);
        assert!(!mask.is_set(ResonanceClass::new(0).unwrap()));
    }

    #[test]
    fn test_class_mask_all() {
        let mask = ClassMask::all();
        assert_eq!(mask.count(), 96);
        for i in 0..96 {
            assert!(mask.is_set(ResonanceClass::new(i).unwrap()));
        }
    }

    #[test]
    fn test_class_mask_single() {
        let mask = ClassMask::single(ResonanceClass::new(42).unwrap());
        assert_eq!(mask.count(), 1);
        assert!(mask.is_set(ResonanceClass::new(42).unwrap()));
        assert!(!mask.is_set(ResonanceClass::new(41).unwrap()));
        assert!(!mask.is_set(ResonanceClass::new(43).unwrap()));
    }

    #[test]
    fn test_class_mask_set_clear() {
        let mut mask = ClassMask::empty();
        mask.set(ResonanceClass::new(10).unwrap());
        mask.set(ResonanceClass::new(70).unwrap());
        mask.set(ResonanceClass::new(95).unwrap());

        assert_eq!(mask.count(), 3);
        assert!(mask.is_set(ResonanceClass::new(10).unwrap()));
        assert!(mask.is_set(ResonanceClass::new(70).unwrap()));
        assert!(mask.is_set(ResonanceClass::new(95).unwrap()));

        mask.clear(ResonanceClass::new(70).unwrap());
        assert_eq!(mask.count(), 2);
        assert!(!mask.is_set(ResonanceClass::new(70).unwrap()));
    }

    #[test]
    fn test_class_mask_overlaps() {
        let mut mask1 = ClassMask::empty();
        mask1.set(ResonanceClass::new(10).unwrap());
        mask1.set(ResonanceClass::new(20).unwrap());

        let mut mask2 = ClassMask::empty();
        mask2.set(ResonanceClass::new(20).unwrap());
        mask2.set(ResonanceClass::new(30).unwrap());

        assert!(mask1.overlaps(&mask2));
        assert!(!mask1.is_disjoint(&mask2));

        let mut mask3 = ClassMask::empty();
        mask3.set(ResonanceClass::new(40).unwrap());
        assert!(!mask1.overlaps(&mask3));
        assert!(mask1.is_disjoint(&mask3));
    }

    #[test]
    fn test_metadata_mirror_safe_requires_pairs() {
        let mut meta = KernelMetadata::new("mirror-test");
        meta.mirror_safe = true;
        meta.classes_mask = ClassMask::single(ResonanceClass::new(10).unwrap());

        // Missing mirror partner must fail validation
        assert!(meta.validate().is_err());

        let mirror = atlas_core::get_mirror_pair(10);
        meta.classes_mask
            .set(ResonanceClass::new(mirror).expect("mirror id valid"));
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_phase_window_invalid_span_rejected() {
        let mut meta = KernelMetadata::new("phase-test");
        meta.phase.span = PHASE_MODULUS + 1;
        assert!(meta.validate().is_err());

        meta.phase.span = PHASE_MODULUS;
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_metadata_requires_non_empty_mask() {
        let mut meta = KernelMetadata::new("mask-test");
        meta.classes_mask = ClassMask::empty();
        assert!(meta.validate().is_err());

        meta.classes_mask.set(ResonanceClass::new(0).expect("class 0 valid"));
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_class_mask_operations() {
        let mut mask1 = ClassMask::empty();
        mask1.set(ResonanceClass::new(10).unwrap());
        mask1.set(ResonanceClass::new(20).unwrap());

        let mut mask2 = ClassMask::empty();
        mask2.set(ResonanceClass::new(20).unwrap());
        mask2.set(ResonanceClass::new(30).unwrap());

        let intersection = mask1.intersection(&mask2);
        assert_eq!(intersection.count(), 1);
        assert!(intersection.is_set(ResonanceClass::new(20).unwrap()));

        let union = mask1.union(&mask2);
        assert_eq!(union.count(), 3);
        assert!(union.is_set(ResonanceClass::new(10).unwrap()));
        assert!(union.is_set(ResonanceClass::new(20).unwrap()));
        assert!(union.is_set(ResonanceClass::new(30).unwrap()));
    }

    #[test]
    fn test_class_mask_active_classes() {
        let mut mask = ClassMask::empty();
        mask.set(ResonanceClass::new(5).unwrap());
        mask.set(ResonanceClass::new(70).unwrap());
        mask.set(ResonanceClass::new(90).unwrap());

        let active = mask.active_classes();
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].as_u8(), 5);
        assert_eq!(active[1].as_u8(), 70);
        assert_eq!(active[2].as_u8(), 90);
    }

    #[test]
    fn test_boundary_footprint_validation() {
        let valid = BoundaryFootprint {
            page_min: 0,
            page_max: 10,
            byte_min: 0,
            byte_max: 100,
        };
        assert!(valid.validate().is_ok());

        let invalid_page = BoundaryFootprint {
            page_min: 50,
            page_max: 50,
            byte_min: 0,
            byte_max: 100,
        };
        assert!(invalid_page.validate().is_err());
    }

    #[test]
    fn test_boundary_footprint_contains() {
        let footprint = BoundaryFootprint {
            page_min: 10,
            page_max: 20,
            byte_min: 50,
            byte_max: 150,
        };

        assert!(footprint.contains(15, 100));
        assert!(!footprint.contains(5, 100));
        assert!(!footprint.contains(15, 200));
    }

    #[test]
    fn test_phase_window_contains() {
        let window = PhaseWindow { begin: 100, span: 50 };
        assert!(window.contains(100));
        assert!(window.contains(125));
        assert!(window.contains(149));
        assert!(!window.contains(150));
        assert!(!window.contains(99));
    }

    #[test]
    fn test_phase_window_wraps() {
        // Window [700, 50) wraps around: [700, 768) ∪ [0, 50)
        let window = PhaseWindow { begin: 700, span: 118 };

        assert!(window.contains(700));
        assert!(window.contains(750));
        assert!(window.contains(0));
        assert!(window.contains(49));
        assert!(!window.contains(50));
        assert!(!window.contains(699));
    }

    #[test]
    fn test_kernel_metadata_validation() {
        let valid = KernelMetadata::new("test_kernel");
        assert!(valid.validate().is_ok());

        let mut invalid = KernelMetadata::new("");
        assert!(invalid.validate().is_err());

        invalid.name = "test".to_string();
        invalid.classes_mask = ClassMask::empty();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_kernel_metadata_mirror_safe_requires_pairs() {
        let class = ResonanceClass::new(1).unwrap();
        let mut metadata = KernelMetadata::new("mirror-check");
        metadata.mirror_safe = true;
        metadata.classes_mask = ClassMask::single(class);

        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_kernel_metadata_mirror_safe_accepts_pairs() {
        let class = ResonanceClass::new(1).unwrap();
        let mirror = atlas_core::get_mirror_pair(class.as_u8());
        let mut metadata = KernelMetadata::new("mirror-ok");
        metadata.mirror_safe = true;

        let mut classes = ClassMask::empty();
        classes.set(class);
        classes.set(ResonanceClass::new(mirror).unwrap());
        metadata.classes_mask = classes;

        assert!(metadata.validate().is_ok());
    }
}
