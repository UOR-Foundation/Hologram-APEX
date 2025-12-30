use core::arch::aarch64::*;

use super::{ArchOps, CacheHierarchy};

#[derive(Debug, Default)]
pub struct NeonArch;

impl NeonArch {
    pub fn new() -> Self {
        Self
    }

    #[inline]
    fn apply_binary(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
        if len == 0 {
            return;
        }
        unsafe {
            binary_op_neon(dst, a, b, len, op);
        }
    }

    #[inline]
    fn apply_scalar(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
        if len == 0 {
            return;
        }
        unsafe {
            scalar_op_neon(dst, input, scalar, len, op);
        }
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

impl ArchOps for NeonArch {
    fn prefetch_l1(&self, ptr: *const u8) {
        unsafe {
            core::arch::asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ptr, options(nostack, preserves_flags));
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
        "aarch64-neon"
    }

    fn cache_hierarchy(&self) -> CacheHierarchy {
        // ARM: L3 may or may not exist (varies by implementation)
        //
        // Rationale:
        // - Apple Silicon (M1/M2/M3): Has large L2 (up to 12 MB), variable L3
        // - ARM Cortex-A (mobile/embedded): Usually has L2, sometimes L3
        // - ARM servers (Graviton, Ampere): Usually have L3
        //
        // Conservative assumptions: Report L3 as optional
        CacheHierarchy {
            l1_data_kb: 32,       // Typical for ARM Cortex-A
            l2_kb: 256,           // Conservative (128 KB - 1 MB typical)
            l3_kb: Some(4096),    // 4 MB if present (less common than x86_64)
            cache_line_bytes: 64, // Common on ARMv8/ARMv9
        }
    }
}

unsafe fn binary_op_neon(dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = vld1q_f32(a.add(i));
        let rhs = vld1q_f32(b.add(i));
        let res = match op {
            BinaryOp::Add => vaddq_f32(lhs, rhs),
            BinaryOp::Sub => vsubq_f32(lhs, rhs),
            BinaryOp::Mul => vmulq_f32(lhs, rhs),
            BinaryOp::Div => vdivq_f32(lhs, rhs),
        };
        vst1q_f32(dst.add(i), res);
        i += 4;
    }
    for idx in i..len {
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

unsafe fn scalar_op_neon(dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
    let scalar_vec = vdupq_n_f32(scalar);
    let mut i = 0usize;
    while i + 4 <= len {
        let data = vld1q_f32(input.add(i));
        let res = match op {
            ScalarOp::Add => vaddq_f32(data, scalar_vec),
            ScalarOp::Mul => vmulq_f32(data, scalar_vec),
        };
        vst1q_f32(dst.add(i), res);
        i += 4;
    }
    for idx in i..len {
        let value = *input.add(idx);
        *dst.add(idx) = match op {
            ScalarOp::Add => value + scalar,
            ScalarOp::Mul => value * scalar,
        };
    }
}
