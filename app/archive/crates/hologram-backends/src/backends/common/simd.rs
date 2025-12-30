//! CPU SIMD kernels for optimized execution
//!
//! These kernels provide AVX512/AVX2/SSE4.1 optimizations for common operations.

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::sync::OnceLock;

// Cache SIMD capability detection
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
static SIMD_CAPS: OnceLock<(bool, bool, bool)> = OnceLock::new();

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn get_simd_caps() -> (bool, bool, bool) {
    *SIMD_CAPS.get_or_init(|| {
        (
            is_x86_feature_detected!("avx512f"),
            is_x86_feature_detected!("avx2"),
            is_x86_feature_detected!("sse4.1"),
        )
    })
}

/// SIMD-accelerated vector_add (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `b` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
#[inline(always)]
pub unsafe fn vector_add_f32(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let (_has_avx512, has_avx2, has_sse4) = get_simd_caps();
        if has_avx2 {
            unsafe { vector_add_avx2(a, b, c, n) }
        } else if has_sse4 {
            unsafe { vector_add_sse4(a, b, c, n) }
        } else {
            vector_add_scalar(a, b, c, n)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        vector_add_scalar(a, b, c, n)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn vector_add_avx2(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let simd_end = n / 8 * 8;
    for i in (0..simd_end).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.add(i));
        let b_vec = _mm256_loadu_ps(b.add(i));
        let c_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(c.add(i), c_vec);
    }
    for idx in simd_end..n {
        *c.add(idx) = *a.add(idx) + *b.add(idx);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse4.1")]
unsafe fn vector_add_sse4(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let simd_end = n / 4 * 4;
    for i in (0..simd_end).step_by(4) {
        let a_vec = _mm_loadu_ps(a.add(i));
        let b_vec = _mm_loadu_ps(b.add(i));
        let c_vec = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(c.add(i), c_vec);
    }
    for idx in simd_end..n {
        *c.add(idx) = *a.add(idx) + *b.add(idx);
    }
}

#[inline(always)]
fn vector_add_scalar(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    unsafe {
        for idx in 0..n {
            *c.add(idx) = *a.add(idx) + *b.add(idx);
        }
    }
}

/// SIMD-accelerated vector_sub (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `b` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_sub_f32(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        *c.add(i) = *a.add(i) - *b.add(i);
    }
}

/// SIMD-accelerated vector_mul (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `b` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_mul_f32(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        *c.add(i) = *a.add(i) * *b.add(i);
    }
}

/// SIMD-accelerated vector_div (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `b` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_div_f32(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        *c.add(i) = *a.add(i) / *b.add(i);
    }
}

/// SIMD-accelerated vector_abs (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_abs_f32(a: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        *c.add(i) = (*a.add(i)).abs();
    }
}

/// SIMD-accelerated vector_neg (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_neg_f32(a: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        *c.add(i) = -(*a.add(i));
    }
}

/// SIMD-accelerated vector_relu (f32)
///
/// # Safety
///
/// - `a` must be valid for reads of `n` elements
/// - `c` must be valid for writes of `n` elements
/// - Pointers must not alias unless semantically valid
pub unsafe fn vector_relu_f32(a: *const f32, c: *mut f32, n: usize) {
    for i in 0..n {
        let val = *a.add(i);
        *c.add(i) = if val > 0.0 { val } else { 0.0 };
    }
}
