#[derive(Debug, Default)]
pub struct ScalarArch;

use super::{ArchOps, CacheHierarchy};

impl ScalarArch {
    pub fn new() -> Self {
        Self
    }

    #[inline]
    fn apply_binary(&self, dst: *mut f32, a: *const f32, b: *const f32, len: usize, op: BinaryOp) {
        unsafe {
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
    }

    #[inline]
    fn apply_scalar(&self, dst: *mut f32, input: *const f32, scalar: f32, len: usize, op: ScalarOp) {
        unsafe {
            for idx in 0..len {
                let value = *input.add(idx);
                *dst.add(idx) = match op {
                    ScalarOp::Add => value + scalar,
                    ScalarOp::Mul => value * scalar,
                };
            }
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

impl ArchOps for ScalarArch {
    fn prefetch_l1(&self, _ptr: *const u8) {}

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
        "scalar"
    }

    fn cache_hierarchy(&self) -> CacheHierarchy {
        // Scalar/Microcontrollers: Minimal or no cache
        //
        // Rationale:
        // - This is the fallback for architectures without SIMD
        // - Microcontrollers (STM32, ESP32) have minimal cache (0-64 KB)
        // - Embedded systems may have no cache at all
        // - RISC-V implementations vary widely
        //
        // Report minimal assumptions - no L3, potentially no L1/L2
        CacheHierarchy {
            l1_data_kb: 0,        // May not have L1 cache
            l2_kb: 0,             // Likely no L2 cache
            l3_kb: None,          // Definitely no L3 cache
            cache_line_bytes: 64, // Assume 64 if cache exists at all
        }
    }
}
