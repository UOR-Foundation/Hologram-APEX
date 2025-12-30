//! Matrix operations (MatMul, Gemm)
//!
//! Linear algebra operations with full generic type support.

use super::{OnnxHRMNode, Result};
use crate::error::CompilerError;
use crate::hrm::numeric::Numeric;
use hologram_hrm::Atlas;

/// MatMul operation (matrix multiplication)
///
/// Computes C = A @ B (matrix multiplication).
///
/// For 2D matrices:
/// - A: [M, K]
/// - B: [K, N]
/// - C: [M, N]
///
/// Note: This is a simplified implementation. Full ONNX MatMul supports
/// higher-dimensional tensors with broadcasting.
#[derive(Debug, Clone)]
pub struct MatMulOp {
    /// M dimension (rows of A, rows of C)
    pub m: usize,
    /// K dimension (cols of A, rows of B)
    pub k: usize,
    /// N dimension (cols of B, cols of C)
    pub n: usize,
}

impl MatMulOp {
    /// Create a new MatMul operation with specified dimensions
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in A (and C)
    /// * `k` - Number of columns in A / rows in B
    /// * `n` - Number of columns in B (and C)
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }

    /// Construct from ONNX node shapes
    ///
    /// Extracts m, k, n from input tensor shapes.
    pub fn from_shapes(input_shapes: &[Vec<i64>]) -> Result<Self> {
        crate::validate_input_count!(input_shapes, 2, "MatMul");

        let (m, k) = crate::extract_2d_dims!(input_shapes[0], "MatMul input A");
        let (k2, n) = crate::extract_2d_dims!(input_shapes[1], "MatMul input B");

        if k != k2 {
            return Err(CompilerError::InvalidModel(format!(
                "MatMul dimension mismatch: A has {} cols, B has {} rows",
                k, k2
            )));
        }

        Ok(Self::new(m, k, n))
    }
}

impl<T: Numeric> OnnxHRMNode<T> for MatMulOp {
    fn op_type(&self) -> &'static str {
        "MatMul"
    }

    fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
        self.validate_inputs(inputs)?;

        let a = inputs[0];
        let b = inputs[1];

        // Perform matrix multiplication C = A @ B
        let mut c = vec![T::zero(); self.m * self.n];

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k in 0..self.k {
                    sum = sum.add(a[i * self.k + k].mul(b[k * self.n + j]));
                }
                c[i * self.n + j] = sum;
            }
        }

        Ok(c)
    }

    fn validate_inputs(&self, inputs: &[&[T]]) -> Result<()> {
        crate::validate_input_count!(inputs, 2, "MatMul");
        crate::validate_input_size!(inputs[0], self.m * self.k, "MatMul input A");
        crate::validate_input_size!(inputs[1], self.k * self.n, "MatMul input B");
        Ok(())
    }
}

/// Gemm operation (General Matrix Multiplication)
///
/// Computes Y = alpha * A @ B + beta * C
///
/// This is the BLAS GEMM operation, commonly used in neural networks.
#[derive(Debug, Clone)]
pub struct GemmOp {
    /// M dimension (rows of A, rows of Y)
    pub m: usize,
    /// K dimension (cols of A, rows of B)
    pub k: usize,
    /// N dimension (cols of B, cols of Y)
    pub n: usize,
    /// Alpha scalar multiplier
    pub alpha: f32,
    /// Beta scalar multiplier
    pub beta: f32,
    /// Transpose A before multiplication
    pub trans_a: bool,
    /// Transpose B before multiplication
    pub trans_b: bool,
}

impl GemmOp {
    /// Create a new Gemm operation
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in A (and Y)
    /// * `k` - Number of columns in A / rows in B
    /// * `n` - Number of columns in B (and Y)
    /// * `alpha` - Scalar multiplier for A @ B
    /// * `beta` - Scalar multiplier for C
    pub fn new(m: usize, k: usize, n: usize, alpha: f32, beta: f32) -> Self {
        Self {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a: false,
            trans_b: false,
        }
    }

    /// Set transpose flags
    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.trans_a = trans_a;
        self.trans_b = trans_b;
        self
    }

    /// Construct from ONNX node shapes
    ///
    /// Extracts m, k, n from input tensor shapes.
    /// Uses default alpha=1.0, beta=1.0.
    pub fn from_shapes(input_shapes: &[Vec<i64>]) -> Result<Self> {
        if input_shapes.len() < 2 {
            return Err(CompilerError::InvalidModel(format!(
                "Gemm requires at least 2 input shapes, got {}",
                input_shapes.len()
            )));
        }

        let (m, k) = crate::extract_2d_dims!(input_shapes[0], "Gemm input A");
        let (_, n) = crate::extract_2d_dims!(input_shapes[1], "Gemm input B");

        Ok(Self::new(m, k, n, 1.0, 1.0))
    }
}

impl<T: Numeric> OnnxHRMNode<T> for GemmOp {
    fn op_type(&self) -> &'static str {
        "Gemm"
    }

    fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
        self.validate_inputs(inputs)?;

        let a = inputs[0];
        let b = inputs[1];
        let c = if inputs.len() > 2 { inputs[2] } else { &[] };

        // Compute Y = alpha * A @ B + beta * C
        let mut y = vec![T::zero(); self.m * self.n];

        let alpha_t = T::from_f32(self.alpha);
        let beta_t = T::from_f32(self.beta);

        // Compute A @ B
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for k_idx in 0..self.k {
                    let a_val = if self.trans_a {
                        a[k_idx * self.m + i]
                    } else {
                        a[i * self.k + k_idx]
                    };
                    let b_val = if self.trans_b {
                        b[j * self.k + k_idx]
                    } else {
                        b[k_idx * self.n + j]
                    };
                    sum = sum.add(a_val.mul(b_val));
                }
                y[i * self.n + j] = alpha_t.mul(sum);
            }
        }

        // Add beta * C if C is provided
        if !c.is_empty() {
            for i in 0..self.m * self.n {
                y[i] = y[i].add(beta_t.mul(c[i]));
            }
        }

        Ok(y)
    }

    fn validate_inputs(&self, inputs: &[&[T]]) -> Result<()> {
        crate::validate_input_count_range!(inputs, 2, 3, "Gemm");

        let expected_a_size = if self.trans_a { self.k * self.m } else { self.m * self.k };
        let expected_b_size = if self.trans_b { self.n * self.k } else { self.k * self.n };

        crate::validate_input_size!(inputs[0], expected_a_size, "Gemm input A");
        crate::validate_input_size!(inputs[1], expected_b_size, "Gemm input B");

        if inputs.len() == 3 {
            crate::validate_input_size!(inputs[2], self.m * self.n, "Gemm input C");
        }

        Ok(())
    }
}
