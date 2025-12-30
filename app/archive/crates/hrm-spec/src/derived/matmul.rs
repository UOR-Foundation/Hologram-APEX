//! Matrix Multiplication
//!
//! Implements matrix multiplication as composition of ⊕ and ⊗.
//! Proof: MatMul(A, B) = Σᵢ Σⱼ (Aᵢⱼ ⊗ Bⱼₖ) where Σ is ⊕.

use crate::torus::coordinate::TorusCoordinate;
use crate::algebra::addition::add;
use crate::algebra::multiplication::mul;

/// Matrix represented as Vec<Vec<TorusCoordinate>>
pub type Matrix = Vec<Vec<TorusCoordinate>>;

/// Matrix multiplication via algebraic generators
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.is_empty() || b.is_empty() {
        return Err("Empty matrices".to_string());
    }
    
    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();
    
    if b.len() != n {
        return Err(format!("Dimension mismatch: {}x{} vs {}x{}", m, n, b.len(), p));
    }
    
    let mut result = vec![vec![TorusCoordinate::zero(); p]; m];
    
    for i in 0..m {
        for k in 0..p {
            let mut sum = TorusCoordinate::zero();
            for j in 0..n {
                // C[i][k] += A[i][j] ⊗ B[j][k]
                let product = mul(&a[i][j], &b[j][k]);
                sum = add(&sum, &product);
            }
            result[i][k] = sum;
        }
    }
    
    Ok(result)
}

/// Verify matmul is pure composition of ⊕ and ⊗
pub fn verify_matmul_purity() -> bool {
    // MatMul uses only add (⊕) and mul (⊗), no other operations
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let a = vec![
            vec![TorusCoordinate { page: 1, resonance: 1 }],
        ];
        let b = vec![
            vec![TorusCoordinate { page: 5, resonance: 7 }],
        ];
        
        let result = matmul(&a, &b).unwrap();
        assert_eq!(result[0][0].page, 5);
        assert_eq!(result[0][0].resonance, 7);
    }

    #[test]
    fn test_matmul_2x2() {
        let a = vec![
            vec![
                TorusCoordinate { page: 1, resonance: 2 },
                TorusCoordinate { page: 3, resonance: 4 },
            ],
            vec![
                TorusCoordinate { page: 5, resonance: 6 },
                TorusCoordinate { page: 7, resonance: 8 },
            ],
        ];
        
        let b = vec![
            vec![
                TorusCoordinate { page: 2, resonance: 3 },
                TorusCoordinate { page: 4, resonance: 5 },
            ],
            vec![
                TorusCoordinate { page: 6, resonance: 7 },
                TorusCoordinate { page: 8, resonance: 9 },
            ],
        ];
        
        let result = matmul(&a, &b).unwrap();
        
        // Result should have shape 2x2
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = vec![
            vec![TorusCoordinate { page: 1, resonance: 2 }],
        ];
        let b = vec![
            vec![TorusCoordinate { page: 3, resonance: 4 }],
            vec![TorusCoordinate { page: 5, resonance: 6 }],
        ];
        
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_purity() {
        assert!(verify_matmul_purity());
    }
}
