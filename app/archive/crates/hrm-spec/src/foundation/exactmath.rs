//! Exact Arbitrary-Precision Arithmetic
//!
//! All operations in MoonshineHRM use exact arithmetic with no floating-point
//! approximations. This module defines the exact number types.

use num_bigint::BigInt;
use num_rational::BigRational;
use std::fmt;

/// Exact number type (no floating point!)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exact {
    /// Arbitrary-precision integer
    Integer(BigInt),
    
    /// Arbitrary-precision rational (p/q)
    Rational(BigRational),
    
    /// Algebraic number (roots, etc.)
    Algebraic(AlgebraicNumber),
}

/// Algebraic number representation
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlgebraicNumber {
    /// Minimal polynomial coefficients
    pub coefficients: Vec<BigRational>,
    
    /// Isolating interval [lower, upper]
    pub interval: (BigRational, BigRational),
}

impl Exact {
    /// Create from integer
    pub fn from_integer(n: BigInt) -> Self {
        Exact::Integer(n)
    }
    
    /// Create from rational
    pub fn from_rational(r: BigRational) -> Self {
        Exact::Rational(r)
    }
    
    /// Addition (exact)
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Exact::Integer(a), Exact::Integer(b)) => {
                Exact::Integer(a + b)
            }
            (Exact::Rational(a), Exact::Rational(b)) => {
                Exact::Rational(a + b)
            }
            _ => unimplemented!("Mixed exact types"),
        }
    }
    
    /// Multiplication (exact)
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Exact::Integer(a), Exact::Integer(b)) => {
                Exact::Integer(a * b)
            }
            (Exact::Rational(a), Exact::Rational(b)) => {
                Exact::Rational(a * b)
            }
            _ => unimplemented!("Mixed exact types"),
        }
    }
    
    /// Square root (exact algebraic result)
    pub fn sqrt(&self) -> Result<Self, String> {
        match self {
            Exact::Integer(n) => {
                // Check if perfect square
                let sqrt = n.sqrt();
                if &(&sqrt * &sqrt) == n {
                    Ok(Exact::Integer(sqrt))
                } else {
                    // Return algebraic: root of x² - n
                    Ok(Exact::Algebraic(AlgebraicNumber {
                        coefficients: vec![
                            BigRational::from(BigInt::from(-1) * n),
                            BigRational::from(BigInt::from(0)),
                            BigRational::from(BigInt::from(1)),
                        ],
                        interval: (
                            BigRational::from(BigInt::from(0)),
                            BigRational::from(n.clone()),
                        ),
                    }))
                }
            }
            _ => Err("sqrt not implemented for this type".to_string()),
        }
    }
    
    /// Comparison (exact)
    pub fn cmp_exact(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Exact::Integer(a), Exact::Integer(b)) => a.cmp(b),
            (Exact::Rational(a), Exact::Rational(b)) => a.cmp(b),
            _ => unimplemented!("Mixed exact comparisons"),
        }
    }
}

impl fmt::Display for Exact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exact::Integer(n) => write!(f, "{}", n),
            Exact::Rational(r) => write!(f, "{}", r),
            Exact::Algebraic(a) => write!(f, "algebraic({:?})", a.coefficients),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exact_addition() {
        let a = Exact::from_integer(BigInt::from(123));
        let b = Exact::from_integer(BigInt::from(456));
        let c = a.add(&b);
        
        assert_eq!(c, Exact::from_integer(BigInt::from(579)));
    }
    
    #[test]
    fn test_exact_multiplication() {
        let a = Exact::from_integer(BigInt::from(123));
        let b = Exact::from_integer(BigInt::from(456));
        let c = a.mul(&b);
        
        assert_eq!(c, Exact::from_integer(BigInt::from(56088)));
    }
    
    #[test]
    fn test_exact_sqrt() {
        let a = Exact::from_integer(BigInt::from(144));
        let sqrt_a = a.sqrt().unwrap();
        
        assert_eq!(sqrt_a, Exact::from_integer(BigInt::from(12)));
    }
    
    #[test]
    fn test_algebraic_sqrt() {
        let a = Exact::from_integer(BigInt::from(2));
        let sqrt_a = a.sqrt().unwrap();
        
        // √2 is algebraic, not a perfect square
        match sqrt_a {
            Exact::Algebraic(_) => assert!(true),
            _ => panic!("Expected algebraic number"),
        }
    }
}
