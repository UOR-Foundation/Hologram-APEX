//! Convolution
//!
//! Implements convolution as sliding window composition of ⊕ and ⊗.
//! Conv(f, g)[n] = Σₖ f[k] ⊗ g[n-k]

use crate::torus::coordinate::TorusCoordinate;
use crate::algebra::addition::add;
use crate::algebra::multiplication::mul;

/// Signal represented as Vec<TorusCoordinate>
pub type Signal = Vec<TorusCoordinate>;

/// Convolution via algebraic generators
pub fn convolve(signal: &Signal, kernel: &Signal) -> Signal {
    if signal.is_empty() || kernel.is_empty() {
        return vec![];
    }
    
    let n = signal.len();
    let m = kernel.len();
    let output_len = n + m - 1;
    
    let mut result = vec![TorusCoordinate::zero(); output_len];
    
    for i in 0..output_len {
        let mut sum = TorusCoordinate::zero();
        for j in 0..m {
            if i >= j && i - j < n {
                // result[i] += signal[i-j] ⊗ kernel[j]
                let product = mul(&signal[i - j], &kernel[j]);
                sum = add(&sum, &product);
            }
        }
        result[i] = sum;
    }
    
    result
}

/// Circular convolution (wraps around)
pub fn circular_convolve(signal: &Signal, kernel: &Signal) -> Signal {
    if signal.is_empty() || kernel.is_empty() {
        return vec![];
    }
    
    let n = signal.len();
    let m = kernel.len();
    let mut result = vec![TorusCoordinate::zero(); n];
    
    for i in 0..n {
        let mut sum = TorusCoordinate::zero();
        for j in 0..m {
            let idx = (i + n - j) % n;
            let product = mul(&signal[idx], &kernel[j]);
            sum = add(&sum, &product);
        }
        result[i] = sum;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_simple() {
        let signal = vec![
            TorusCoordinate { page: 1, resonance: 2 },
            TorusCoordinate { page: 3, resonance: 4 },
        ];
        let kernel = vec![
            TorusCoordinate { page: 2, resonance: 3 },
        ];
        
        let result = convolve(&signal, &kernel);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_convolve_length() {
        let signal = vec![TorusCoordinate::zero(); 10];
        let kernel = vec![TorusCoordinate::zero(); 5];
        
        let result = convolve(&signal, &kernel);
        assert_eq!(result.len(), 14);  // 10 + 5 - 1
    }

    #[test]
    fn test_circular_convolve_length() {
        let signal = vec![TorusCoordinate::zero(); 10];
        let kernel = vec![TorusCoordinate::zero(); 5];
        
        let result = circular_convolve(&signal, &kernel);
        assert_eq!(result.len(), 10);  // Same as signal length
    }

    #[test]
    fn test_convolve_identity_kernel() {
        let signal = vec![
            TorusCoordinate { page: 5, resonance: 7 },
        ];
        let kernel = vec![
            TorusCoordinate::one(),
        ];
        
        let result = convolve(&signal, &kernel);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page, 5);
        assert_eq!(result[0].resonance, 7);
    }
}
