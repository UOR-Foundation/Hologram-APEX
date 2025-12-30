//! Attention Mechanisms
//!
//! Implements attention as composition of matmul, scaling, and softmax.
//! Attention(Q, K, V) = softmax(Q ⊗ Kᵀ / √d) ⊗ V

use crate::torus::coordinate::TorusCoordinate;
use crate::derived::matmul::{Matrix, matmul};


/// Scaled dot-product attention (simplified)
pub fn attention(query: &Matrix, key: &Matrix, value: &Matrix) -> Result<Matrix, String> {
    if query.is_empty() || key.is_empty() || value.is_empty() {
        return Err("Empty matrices".to_string());
    }
    
    // Q ⊗ Kᵀ
    let key_transpose = transpose(key);
    let scores = matmul(query, &key_transpose)?;
    
    // Scale by √d (simplified: just use d=1 for now)
    // In full implementation: scores / √d_k
    
    // Softmax (simplified: identity for now)
    // In full implementation: exp normalization
    let attention_weights = scores;
    
    // Attention ⊗ V
    matmul(&attention_weights, value)
}

/// Matrix transpose
fn transpose(matrix: &Matrix) -> Matrix {
    if matrix.is_empty() {
        return vec![];
    }
    
    let rows = matrix.len();
    let cols = matrix[0].len();
    
    let mut result = vec![vec![TorusCoordinate::zero(); rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j].clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = vec![
            vec![
                TorusCoordinate { page: 1, resonance: 2 },
                TorusCoordinate { page: 3, resonance: 4 },
            ],
            vec![
                TorusCoordinate { page: 5, resonance: 6 },
                TorusCoordinate { page: 7, resonance: 8 },
            ],
        ];
        
        let transposed = transpose(&matrix);
        assert_eq!(transposed.len(), 2);
        assert_eq!(transposed[0].len(), 2);
        assert_eq!(transposed[0][0].page, 1);
        assert_eq!(transposed[0][1].page, 5);
        assert_eq!(transposed[1][0].page, 3);
        assert_eq!(transposed[1][1].page, 7);
    }

    #[test]
    fn test_attention_simple() {
        let query = vec![
            vec![TorusCoordinate { page: 1, resonance: 2 }],
        ];
        let key = vec![
            vec![TorusCoordinate { page: 1, resonance: 1 }],
        ];
        let value = vec![
            vec![TorusCoordinate { page: 3, resonance: 5 }],
        ];
        
        let result = attention(&query, &key, &value).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
    }

    #[test]
    fn test_attention_dimension_check() {
        let query = vec![
            vec![TorusCoordinate::zero(); 4],
            vec![TorusCoordinate::zero(); 4],
        ];
        let key = vec![
            vec![TorusCoordinate::zero(); 4],
            vec![TorusCoordinate::zero(); 4],
            vec![TorusCoordinate::zero(); 4],
        ];
        let value = vec![
            vec![TorusCoordinate::zero(); 8],
            vec![TorusCoordinate::zero(); 8],
            vec![TorusCoordinate::zero(); 8],
        ];
        
        let result = attention(&query, &key, &value).unwrap();
        // Result should be (2, 8) = Q rows × V cols
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 8);
    }
}
