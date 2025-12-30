//! Helper types and utilities for CPU backend

use std::{alloc::Layout, ptr::NonNull};

use crate::{arch::ArchOps, types::MemoryPool};

#[derive(Debug, Clone)]
pub(super) struct AllocationRecord {
    pub(super) ptr: NonNull<u8>,
    pub(super) size: usize,
    // Phase 8/9 scaffolding: Will be used for memory alignment validation
    #[allow(dead_code)]
    pub(super) alignment: usize,
    // Phase 8/9 scaffolding: Will be used for pool-specific operations
    #[allow(dead_code)]
    pub(super) pool: MemoryPool,
    pub(super) location: AllocationLocation,
}

#[derive(Debug, Clone)]
pub(super) enum AllocationLocation {
    Boundary { _offset: usize },
    Linear { layout: Layout },
}

/// Binary operation types for SIMD fast paths.
///
/// Specifies which binary operation to apply via architecture-specific
/// SIMD instructions (AVX2/AVX-512 on x86_64, NEON on ARM).
#[derive(Debug, Clone, Copy)]
pub(super) enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOpKind {
    /// Apply this binary operation using architecture-specific SIMD instructions.
    ///
    /// # Safety
    ///
    /// Pointers must be valid and aligned, and `len` must not exceed buffer capacities.
    #[inline(always)]
    pub(super) unsafe fn apply(self, arch: &dyn ArchOps, dst: *mut f32, a: *const f32, b: *const f32, len: usize) {
        match self {
            Self::Add => arch.simd_add_f32(dst, a, b, len),
            Self::Sub => arch.simd_sub_f32(dst, a, b, len),
            Self::Mul => arch.simd_mul_f32(dst, a, b, len),
            Self::Div => arch.simd_div_f32(dst, a, b, len),
        }
    }
}

/// Scalar operation types for SIMD fast paths.
///
/// Specifies which scalar operation (operation with a constant) to apply
/// via architecture-specific SIMD instructions.
#[derive(Debug, Clone, Copy)]
pub(super) enum ScalarOpKind {
    Add,
    Mul,
}

impl ScalarOpKind {
    /// Apply this scalar operation using architecture-specific SIMD instructions.
    ///
    /// # Safety
    ///
    /// Pointers must be valid and aligned, and `len` must not exceed buffer capacities.
    #[inline]
    pub(super) unsafe fn apply(self, arch: &dyn ArchOps, dst: *mut f32, input: *const f32, scalar: f32, len: usize) {
        match self {
            Self::Add => arch.scalar_add_f32(dst, input, scalar, len),
            Self::Mul => arch.scalar_mul_f32(dst, input, scalar, len),
        }
    }
}
