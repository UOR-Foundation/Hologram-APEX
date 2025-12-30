//! Architecture-specific SIMD and prefetch helpers
//!
//! Provides an abstraction over ISA-specific capabilities required by the CPU
//! backend. Each supported architecture exposes an implementation of
//! [`ArchOps`], while other targets fall back to scalar operations.

use std::sync::atomic::{fence, Ordering};

use atlas_runtime::addressing::CLASS_STRIDE;

const CACHE_LINE_BYTES: usize = 64;

/// Cache hierarchy information reported by architecture.
///
/// Each architecture implementation reports its cache configuration,
/// allowing the backend to adapt without hardcoding assumptions.
#[derive(Debug, Clone, Copy)]
pub struct CacheHierarchy {
    /// L1 data cache size in KB (per core, private)
    pub l1_data_kb: u32,

    /// L2 cache size in KB (per core, may be private or shared)
    pub l2_kb: u32,

    /// L3 cache size in KB (shared across cores), None if not present
    pub l3_kb: Option<u32>,

    /// Cache line size in bytes (typically 64 on modern architectures)
    pub cache_line_bytes: u32,
}

/// ISA-specific operations used by the CPU backend.
pub trait ArchOps: Send + Sync + core::fmt::Debug {
    /// Prefetch the cache line containing `ptr` into L1.
    fn prefetch_l1(&self, ptr: *const u8);

    /// Prefetch an entire resonance class (12,288 bytes) into L1.
    ///
    /// The implementation MUST issue cache-line prefetches with a 64-byte
    /// stride so that the full class working set is resident before compute
    /// begins, matching the Activate phase defined in the backend spec.
    fn activate_class_l1(&self, class_base: *const u8) {
        debug_assert!(!class_base.is_null(), "class base pointer must be non-null");
        debug_assert_eq!(
            CLASS_STRIDE % CACHE_LINE_BYTES,
            0,
            "class stride must be cache-line aligned"
        );

        for offset in (0..CLASS_STRIDE).step_by(CACHE_LINE_BYTES) {
            let line_ptr = class_base.wrapping_add(offset);
            self.prefetch_l1(line_ptr);
        }
    }

    /// Element-wise addition of two `len`-element f32 buffers: `dst = a + b`.
    fn simd_add_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize);

    /// Element-wise subtraction of two `len`-element f32 buffers: `dst = a - b`.
    fn simd_sub_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize);

    /// Element-wise multiplication of two `len`-element f32 buffers: `dst = a * b`.
    fn simd_mul_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize);

    /// Element-wise division of two `len`-element f32 buffers: `dst = a / b`.
    fn simd_div_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize);

    /// Add scalar to each element: `dst = input + scalar`.
    fn scalar_add_f32(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize);

    /// Multiply scalar with each element: `dst = input * scalar`.
    fn scalar_mul_f32(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize);

    /// Memory fence with acquire semantics.
    fn fence_acquire(&self) {
        fence(Ordering::Acquire);
    }

    /// Memory fence with release semantics.
    fn fence_release(&self) {
        fence(Ordering::Release);
    }

    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;

    /// Report cache hierarchy for this architecture.
    ///
    /// This allows the backend to:
    /// - Verify Atlas structure alignment with cache
    /// - Log cache configuration at startup
    /// - Make cache-aware scheduling decisions (future)
    ///
    /// Each architecture reports realistic values:
    /// - x86_64: Assumes L3 present (safe for modern systems)
    /// - ARM: L3 may or may not exist
    /// - Microcontrollers: Minimal or no cache
    fn cache_hierarchy(&self) -> CacheHierarchy;
}

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::X86Arch;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::NeonArch;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod scalar;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use scalar::ScalarArch;

/// Obtain the architecture implementation for the current target.
pub fn current_arch() -> Box<dyn ArchOps> {
    #[cfg(target_arch = "x86_64")]
    {
        Box::new(x86_64::X86Arch::new())
    }
    #[cfg(target_arch = "aarch64")]
    {
        Box::new(aarch64::NeonArch::new())
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Box::new(scalar::ScalarArch::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_add_matches_scalar() {
        let arch = current_arch();
        let len = 33;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] + b[i]);
        }
    }

    #[test]
    fn class_activation_prefetches_without_panic() {
        let arch = current_arch();
        let buffer = vec![0u8; CLASS_STRIDE];
        arch.activate_class_l1(buffer.as_ptr());
    }

    // =====================================================================
    // Comprehensive SIMD Tests
    // =====================================================================

    #[test]
    fn simd_sub_correctness() {
        let arch = current_arch();
        let len = 100;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| (i * 2) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32).collect();

        arch.simd_sub_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] - b[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn simd_mul_correctness() {
        let arch = current_arch();
        let len = 100;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..len).map(|_i| 2.0).collect();

        arch.simd_mul_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] * b[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn simd_div_correctness() {
        let arch = current_arch();
        let len = 100;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| (i * 2) as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| if i == 0 { 1.0 } else { i as f32 }).collect();

        arch.simd_div_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            let expected = a[i] / b[i];
            assert!(
                (dst[i] - expected).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn simd_scalar_add_correctness() {
        let arch = current_arch();
        let len = 100;
        let mut dst = vec![0.0f32; len];
        let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let scalar = 42.0;

        arch.scalar_add_f32(dst.as_mut_ptr(), input.as_ptr(), scalar, len);

        for i in 0..len {
            assert_eq!(dst[i], input[i] + scalar, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn simd_scalar_mul_correctness() {
        let arch = current_arch();
        let len = 100;
        let mut dst = vec![0.0f32; len];
        let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let scalar = 3.0;

        arch.scalar_mul_f32(dst.as_mut_ptr(), input.as_ptr(), scalar, len);

        for i in 0..len {
            assert_eq!(dst[i], input[i] * scalar, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn simd_handles_unaligned_length() {
        // Test with lengths that don't align to SIMD boundaries
        let arch = current_arch();

        for len in [1, 7, 15, 17, 31, 33, 63, 65] {
            let mut dst = vec![0.0f32; len];
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|_i| 1.0).collect();

            arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

            for i in 0..len {
                assert_eq!(dst[i], a[i] + b[i], "Failed for length {} at index {}", len, i);
            }
        }
    }

    #[test]
    fn simd_empty_array() {
        let arch = current_arch();
        let mut dst = vec![];
        let a = vec![];
        let b = vec![];

        // Should not panic
        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 0);
    }

    #[test]
    fn simd_single_element() {
        let arch = current_arch();
        let mut dst = vec![0.0f32];
        let a = vec![5.0f32];
        let b = vec![3.0f32];

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 1);

        assert_eq!(dst[0], 8.0);
    }

    #[test]
    fn simd_large_array_avx512_boundary() {
        // Test at AVX-512 boundary (16 elements)
        let arch = current_arch();
        let len = 16;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] + b[i]);
        }
    }

    #[test]
    fn simd_large_array_avx2_boundary() {
        // Test at AVX2 boundary (8 elements)
        let arch = current_arch();
        let len = 8;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|_i| 10.0).collect();

        arch.simd_mul_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] * b[i]);
        }
    }

    #[test]
    fn simd_negative_numbers() {
        let arch = current_arch();
        let len = 50;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| -(i as f32)).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32).collect();

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            assert_eq!(dst[i], a[i] + b[i]);
        }
    }

    #[test]
    fn simd_floating_point_edge_cases() {
        let arch = current_arch();
        let len = 10;
        let mut dst = vec![0.0f32; len];
        let a = vec![0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 1000.0, -1000.0, 0.0001, -0.0001];
        let b = vec![0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 1000.0, 1000.0, 0.0001, 0.0001];

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        for i in 0..len {
            let expected = a[i] + b[i];
            assert!(
                (dst[i] - expected).abs() < 1e-6,
                "Mismatch at {}: got {}, expected {}",
                i,
                dst[i],
                expected
            );
        }
    }

    #[test]
    fn simd_memory_barriers() {
        // Test that SIMD operations respect memory ordering
        let arch = current_arch();
        let len = 1000;
        let mut dst = vec![0.0f32; len];
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|_i| 1.0).collect();

        arch.simd_add_f32(dst.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        // All values should be computed correctly
        for i in 0..len {
            assert_eq!(dst[i], a[i] + b[i], "Memory ordering issue at index {}", i);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_avx2_vs_scalar_equivalence() {
        // Verify AVX2 and scalar paths produce identical results
        let len = 100;
        let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32) * 1.3).collect();

        let mut simd_result = vec![0.0f32; len];
        let mut scalar_result = vec![0.0f32; len];

        let arch = current_arch();
        arch.simd_add_f32(simd_result.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);

        // Compute scalar reference
        for i in 0..len {
            scalar_result[i] = a[i] + b[i];
        }

        for i in 0..len {
            assert_eq!(simd_result[i], scalar_result[i], "SIMD/scalar mismatch at index {}", i);
        }
    }

    #[test]
    fn simd_fence_acquire_release() {
        // Test memory fence operations
        let arch = current_arch();

        arch.fence_acquire();
        arch.fence_release();

        // Should not panic and should compile
    }

    #[test]
    fn simd_prefetch_multiple_classes() {
        let arch = current_arch();
        let buffer1 = vec![0u8; CLASS_STRIDE];
        let buffer2 = vec![0u8; CLASS_STRIDE];
        let buffer3 = vec![0u8; CLASS_STRIDE];

        // Should handle multiple prefetches without issues
        arch.activate_class_l1(buffer1.as_ptr());
        arch.activate_class_l1(buffer2.as_ptr());
        arch.activate_class_l1(buffer3.as_ptr());
    }
}
