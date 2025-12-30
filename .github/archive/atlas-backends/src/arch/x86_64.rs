use std::arch::x86_64::*;

use super::{ArchOps, CacheHierarchy};

#[derive(Debug, Clone, Copy)]
struct X86Capabilities {
    avx512f: bool,
    avx2: bool,
}

impl X86Capabilities {
    fn detect() -> Self {
        Self {
            avx512f: is_x86_feature_detected!("avx512f"),
            avx2: is_x86_feature_detected!("avx2"),
        }
    }
}

#[derive(Debug)]
pub struct X86Arch {
    caps: X86Capabilities,
}

impl X86Arch {
    pub fn new() -> Self {
        Self {
            caps: X86Capabilities::detect(),
        }
    }

    #[inline]
    fn apply_binary(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
        if len == 0 {
            return;
        }

        unsafe {
            if self.caps.avx512f {
                binary_op_avx512(dst, a, b, len, op);
            } else if self.caps.avx2 {
                binary_op_avx2(dst, a, b, len, op);
            } else {
                binary_op_scalar(dst, a, b, len, op);
            }
        }
    }

    #[inline]
    fn apply_scalar(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
        if len == 0 {
            return;
        }
        unsafe {
            if self.caps.avx512f {
                scalar_op_avx512(dst, input, scalar, len, op);
            } else if self.caps.avx2 {
                scalar_op_avx2(dst, input, scalar, len, op);
            } else {
                scalar_op_scalar(dst, input, scalar, len, op);
            }
        }
    }
}

impl Default for X86Arch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
enum ScalarOp {
    Add,
    Mul,
}

impl ArchOps for X86Arch {
    fn prefetch_l1(&self, ptr: *const u8) {
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    fn simd_add_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize) {
        self.apply_binary(dst, a, b, len, BinaryOp::Add);
    }

    fn simd_sub_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize) {
        self.apply_binary(dst, a, b, len, BinaryOp::Sub);
    }

    fn simd_mul_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize) {
        self.apply_binary(dst, a, b, len, BinaryOp::Mul);
    }

    fn simd_div_f32(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize) {
        self.apply_binary(dst, a, b, len, BinaryOp::Div);
    }

    fn scalar_add_f32(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize) {
        self.apply_scalar(dst, input, scalar, len, ScalarOp::Add);
    }

    fn scalar_mul_f32(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize) {
        self.apply_scalar(dst, input, scalar, len, ScalarOp::Mul);
    }

    fn name(&self) -> &'static str {
        if self.caps.avx512f {
            "x86_64-avx512"
        } else if self.caps.avx2 {
            "x86_64-avx2"
        } else {
            "x86_64-scalar"
        }
    }

    fn cache_hierarchy(&self) -> CacheHierarchy {
        // x86_64: Assume L3 present (safe for modern desktop/server/cloud)
        //
        // Rationale:
        // - All modern x86_64 CPUs have L3 (Intel Core, AMD Ryzen, Xeon, EPYC)
        // - Cloud environments (AWS, Azure, GCP) all expose L3
        // - CI/CD environments (GitHub Actions, Codespaces) have L3
        // - Microcontrollers use different architectures (ARM Cortex-M, RISC-V)
        //
        // Atlas boundary pool (1.18 MB) fits comfortably in typical L3 (8-128 MB)
        CacheHierarchy {
            l1_data_kb: 32,       // Universal on x86_64
            l2_kb: 256,           // Conservative baseline (256 KB - 1 MB typical)
            l3_kb: Some(8192),    // 8 MB (safe assumption for modern x86_64)
            cache_line_bytes: 64, // Universal on x86_64
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_op_avx512(dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
    let mut i = 0usize;
    while i + 16 <= len {
        let lhs = _mm512_loadu_ps(a.add(i));
        let rhs = _mm512_loadu_ps(b.add(i));
        let res = match op {
            BinaryOp::Add => _mm512_add_ps(lhs, rhs),
            BinaryOp::Sub => _mm512_sub_ps(lhs, rhs),
            BinaryOp::Mul => _mm512_mul_ps(lhs, rhs),
            BinaryOp::Div => _mm512_div_ps(lhs, rhs),
        };
        _mm512_storeu_ps(dst.add(i), res);
        i += 16;
    }
    binary_op_scalar(dst.add(i), a.add(i), b.add(i), len - i, op);
}

#[target_feature(enable = "avx2")]
unsafe fn binary_op_avx2(dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
    let mut i = 0usize;
    while i + 8 <= len {
        let lhs = _mm256_loadu_ps(a.add(i));
        let rhs = _mm256_loadu_ps(b.add(i));
        let res = match op {
            BinaryOp::Add => _mm256_add_ps(lhs, rhs),
            BinaryOp::Sub => _mm256_sub_ps(lhs, rhs),
            BinaryOp::Mul => _mm256_mul_ps(lhs, rhs),
            BinaryOp::Div => _mm256_div_ps(lhs, rhs),
        };
        _mm256_storeu_ps(dst.add(i), res);
        i += 8;
    }
    binary_op_scalar(dst.add(i), a.add(i), b.add(i), len - i, op);
}

unsafe fn binary_op_scalar(dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
    for idx in 0..len {
        let lhs = *a.add(idx);
        let rhs = *b.add(idx);
        *dst.add(idx) = match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        };
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn scalar_op_avx512(dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
    let scalar_vec = _mm512_set1_ps(scalar);
    let mut i = 0usize;
    while i + 16 <= len {
        let data = _mm512_loadu_ps(input.add(i));
        let res = match op {
            ScalarOp::Add => _mm512_add_ps(data, scalar_vec),
            ScalarOp::Mul => _mm512_mul_ps(data, scalar_vec),
        };
        _mm512_storeu_ps(dst.add(i), res);
        i += 16;
    }
    scalar_op_scalar(dst.add(i), input.add(i), scalar, len - i, op);
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_op_avx2(dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
    let scalar_vec = _mm256_set1_ps(scalar);
    let mut i = 0usize;
    while i + 8 <= len {
        let data = _mm256_loadu_ps(input.add(i));
        let res = match op {
            ScalarOp::Add => _mm256_add_ps(data, scalar_vec),
            ScalarOp::Mul => _mm256_mul_ps(data, scalar_vec),
        };
        _mm256_storeu_ps(dst.add(i), res);
        i += 8;
    }
    scalar_op_scalar(dst.add(i), input.add(i), scalar, len - i, op);
}

unsafe fn scalar_op_scalar(dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
    for idx in 0..len {
        let value = *input.add(idx);
        *dst.add(idx) = match op {
            ScalarOp::Add => value + scalar,
            ScalarOp::Mul => value * scalar,
        };
    }
}
