//! Single-Node Tests for Matrix Operators
//!
//! Tests the following ONNX matrix operators in isolation:
//! - MatMul: Matrix multiplication
//! - Gemm: General matrix multiplication (Y = alpha*A@B + beta*C)

use crate::onnx_single_node::*;
use hologram_hrm::Atlas;
use hologram_onnx_compiler::hrm::ops::{GemmOp, MatMulOp, OnnxHRMNode};

#[cfg(test)]
mod test_matmul {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        let atlas = Atlas::with_cache().unwrap();
        let matmul_op = MatMulOp::new(2, 2, 2); // m=2, k=2, n=2

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // 2x2 result

        let result = matmul_op.execute(&atlas, &[&a, &b]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_identity() {
        let atlas = Atlas::with_cache().unwrap();
        let matmul_op = MatMulOp::new(2, 2, 2);

        // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let identity = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let expected = a.clone();

        let result = matmul_op.execute(&atlas, &[&a, &identity]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_different_shapes() {
        let atlas = Atlas::with_cache().unwrap();
        let matmul_op = MatMulOp::new(2, 3, 2); // [2,3] @ [3,2] = [2,2]

        // [2,3] @ [3,2] = [2,2]
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let expected = vec![58.0, 64.0, 139.0, 154.0]; // 2x2

        let result = matmul_op.execute(&atlas, &[&a, &b]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_zero() {
        let atlas = Atlas::with_cache().unwrap();
        let matmul_op = MatMulOp::new(2, 2, 2);

        // [[1,2],[3,4]] @ [[0,0],[0,0]] = [[0,0],[0,0]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let zero = vec![0.0, 0.0, 0.0, 0.0];
        let expected = vec![0.0, 0.0, 0.0, 0.0];

        let result = matmul_op.execute(&atlas, &[&a, &zero]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }
}

#[cfg(test)]
mod test_gemm {
    use super::*;

    #[test]
    fn test_gemm_basic() {
        let atlas = Atlas::with_cache().unwrap();
        let gemm_op = GemmOp::new(2, 2, 2, 1.0, 0.0); // m=2, k=2, n=2, alpha=1.0, beta=0.0

        // With alpha=1, beta=0: Y = A@B
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![0.0, 0.0, 0.0, 0.0]; // Not used when beta=0
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        let result = gemm_op.execute(&atlas, &[&a, &b, &c]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_gemm_with_alpha() {
        let atlas = Atlas::with_cache().unwrap();
        let gemm_op = GemmOp::new(2, 2, 2, 2.0, 0.0); // alpha=2.0

        // Y = 2.0 * A@B
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];
        let expected = vec![38.0, 44.0, 86.0, 100.0]; // 2x the basic result

        let result = gemm_op.execute(&atlas, &[&a, &b, &c]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_gemm_with_beta() {
        let atlas = Atlas::with_cache().unwrap();
        let gemm_op = GemmOp::new(2, 2, 2, 1.0, 1.0); // beta=1.0

        // Y = A@B + C
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let expected = vec![20.0, 23.0, 44.0, 51.0]; // Basic result + 1

        let result = gemm_op.execute(&atlas, &[&a, &b, &c]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        let atlas = Atlas::with_cache().unwrap();
        let gemm_op = GemmOp::new(2, 2, 2, 2.0, 0.5); // alpha=2.0, beta=0.5

        // Y = 2.0 * A@B + 0.5 * C
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![2.0, 4.0, 6.0, 8.0];
        let expected = vec![39.0, 46.0, 89.0, 104.0]; // 2*(A@B) + 0.5*C

        let result = gemm_op.execute(&atlas, &[&a, &b, &c]).unwrap();
        assert_tensors_equal(&result, &expected, 1e-6);
    }
}
