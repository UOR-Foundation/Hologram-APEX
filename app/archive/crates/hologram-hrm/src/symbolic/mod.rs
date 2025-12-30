//! Symbolic integer representation with base-96 conversion
//!
//! This module provides the `SymbolicInteger` type, which represents arbitrary-precision
//! integers in base-96 form. This is the fundamental data type for HRM address resolution.
//!
//! # Base-96 Representation
//!
//! Every integer can be uniquely represented as a sequence of base-96 digits:
//! ```text
//! n = d₀ + d₁·96 + d₂·96² + d₃·96³ + ...
//! ```
//! where each digit dᵢ ∈ [0, 95].
//!
//! # Example
//!
//! ```ignore
//! use hologram_hrm::symbolic::SymbolicInteger;
//!
//! // From u64
//! let sym = SymbolicInteger::from(1961u64);
//! let digits = sym.to_base96();
//! println!("1961 in base-96: {:?}", digits);
//!
//! // From arbitrary-precision integer
//! use num_bigint::BigUint;
//! let big = BigUint::from(123456789u64);
//! let sym = SymbolicInteger::from(big);
//! ```

use crate::{Error, Result};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use std::fmt;

/// Symbolic integer in base-96 representation
///
/// This wraps a `BigUint` and provides conversions to/from base-96 digit sequences.
/// The base-96 representation is fundamental to the HRM embedding operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicInteger {
    /// The underlying arbitrary-precision integer
    value: BigUint,
}

impl SymbolicInteger {
    /// The base used for digit conversion (96 classes)
    pub const BASE: u32 = 96;

    /// Create a SymbolicInteger from a BigUint
    pub fn from_biguint(value: BigUint) -> Self {
        Self { value }
    }

    /// Create a SymbolicInteger representing zero
    pub fn zero() -> Self {
        Self { value: BigUint::zero() }
    }

    /// Create a SymbolicInteger representing one
    pub fn one() -> Self {
        Self { value: BigUint::one() }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Get the underlying BigUint value
    pub fn value(&self) -> &BigUint {
        &self.value
    }

    /// Convert to u64 if possible
    ///
    /// Returns None if the value is too large to fit in u64.
    pub fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }

    /// Convert to base-96 digit sequence
    ///
    /// Returns a vector of digits [d₀, d₁, d₂, ...] where:
    /// - d₀ is the least significant digit (units place)
    /// - Each digit is in [0, 95]
    /// - The integer equals: d₀ + d₁·96 + d₂·96² + ...
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sym = SymbolicInteger::from(1961u64);
    /// let digits = sym.to_base96();
    /// // 1961 = 25 + 20·96 = 25 + 1920
    /// assert_eq!(digits, vec![25, 20]);
    /// ```
    pub fn to_base96(&self) -> Vec<u8> {
        if self.is_zero() {
            return vec![0];
        }

        let mut digits = Vec::new();
        let mut n = self.value.clone();
        let base = BigUint::from(Self::BASE);

        while !n.is_zero() {
            let digit = (&n % &base).to_u8().expect("Digit should fit in u8");
            digits.push(digit);
            n /= &base;
        }

        digits
    }

    /// Create a SymbolicInteger from base-96 digits
    ///
    /// # Arguments
    ///
    /// * `digits` - Base-96 digits [d₀, d₁, d₂, ...] where d₀ is least significant
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidBase96Digit` if any digit >= 96.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sym = SymbolicInteger::from_base96(&[25, 20])?;
    /// assert_eq!(sym.to_u64(), Some(1961));
    /// ```
    pub fn from_base96(digits: &[u8]) -> Result<Self> {
        // Validate all digits are in [0, 95]
        for &digit in digits {
            if digit >= Self::BASE as u8 {
                return Err(Error::InvalidBase96Digit(digit));
            }
        }

        let mut value = BigUint::zero();
        let base = BigUint::from(Self::BASE);
        let mut power = BigUint::one();

        for &digit in digits {
            value += BigUint::from(digit) * &power;
            power *= &base;
        }

        Ok(Self { value })
    }

    /// Get the number of base-96 digits
    ///
    /// This is the length of the digit sequence returned by `to_base96()`.
    pub fn digit_count(&self) -> usize {
        if self.is_zero() {
            1
        } else {
            self.to_base96().len()
        }
    }

    /// Hash the integer value for address caching
    ///
    /// This produces a deterministic u64 hash suitable for HashMap keys.
    pub fn hash_value(&self) -> u64 {
        use sha2::{Digest, Sha256};

        let bytes = self.value.to_bytes_le();
        let hash = Sha256::digest(&bytes);

        // Use first 8 bytes as u64
        u64::from_le_bytes([hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7]])
    }
}

// Conversions from primitive types

impl From<u64> for SymbolicInteger {
    fn from(value: u64) -> Self {
        Self {
            value: BigUint::from(value),
        }
    }
}

impl From<u32> for SymbolicInteger {
    fn from(value: u32) -> Self {
        Self {
            value: BigUint::from(value),
        }
    }
}

impl From<u8> for SymbolicInteger {
    fn from(value: u8) -> Self {
        Self {
            value: BigUint::from(value),
        }
    }
}

impl From<BigUint> for SymbolicInteger {
    fn from(value: BigUint) -> Self {
        Self { value }
    }
}

// Display implementation

impl fmt::Display for SymbolicInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let zero = SymbolicInteger::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.to_u64(), Some(0));
        assert_eq!(zero.to_base96(), vec![0]);
    }

    #[test]
    fn test_one() {
        let one = SymbolicInteger::one();
        assert!(!one.is_zero());
        assert_eq!(one.to_u64(), Some(1));
        assert_eq!(one.to_base96(), vec![1]);
    }

    #[test]
    fn test_from_u64() {
        let sym = SymbolicInteger::from(1961u64);
        assert_eq!(sym.to_u64(), Some(1961));
    }

    #[test]
    fn test_to_base96_small() {
        let sym = SymbolicInteger::from(95u64);
        assert_eq!(sym.to_base96(), vec![95]);

        let sym = SymbolicInteger::from(96u64);
        assert_eq!(sym.to_base96(), vec![0, 1]);

        let sym = SymbolicInteger::from(97u64);
        assert_eq!(sym.to_base96(), vec![1, 1]);
    }

    #[test]
    fn test_to_base96_1961() {
        let sym = SymbolicInteger::from(1961u64);
        let digits = sym.to_base96();

        // 1961 = 41 + 20·96
        assert_eq!(digits, vec![41, 20]);

        // Verify reconstruction
        let reconstructed = 41 + 20 * 96;
        assert_eq!(reconstructed, 1961);
    }

    #[test]
    fn test_from_base96() {
        let sym = SymbolicInteger::from_base96(&[41, 20]).unwrap();
        assert_eq!(sym.to_u64(), Some(1961));
    }

    #[test]
    fn test_from_base96_invalid_digit() {
        let result = SymbolicInteger::from_base96(&[96]);
        assert!(matches!(result, Err(Error::InvalidBase96Digit(96))));

        let result = SymbolicInteger::from_base96(&[10, 96, 20]);
        assert!(matches!(result, Err(Error::InvalidBase96Digit(96))));
    }

    #[test]
    fn test_roundtrip_conversion() {
        for value in [0, 1, 95, 96, 97, 1961, 9216, 65536, 1000000] {
            let sym = SymbolicInteger::from(value);
            let digits = sym.to_base96();
            let reconstructed = SymbolicInteger::from_base96(&digits).unwrap();
            assert_eq!(sym, reconstructed, "Roundtrip failed for {}", value);
            assert_eq!(reconstructed.to_u64(), Some(value));
        }
    }

    #[test]
    fn test_digit_count() {
        assert_eq!(SymbolicInteger::from(0u64).digit_count(), 1);
        assert_eq!(SymbolicInteger::from(1u64).digit_count(), 1);
        assert_eq!(SymbolicInteger::from(95u64).digit_count(), 1);
        assert_eq!(SymbolicInteger::from(96u64).digit_count(), 2);
        assert_eq!(SymbolicInteger::from(9215u64).digit_count(), 2); // 96^2 - 1
        assert_eq!(SymbolicInteger::from(9216u64).digit_count(), 3); // 96^2
    }

    #[test]
    fn test_hash_value() {
        let sym1 = SymbolicInteger::from(1961u64);
        let sym2 = SymbolicInteger::from(1961u64);
        let sym3 = SymbolicInteger::from(1962u64);

        // Same value produces same hash
        assert_eq!(sym1.hash_value(), sym2.hash_value());

        // Different values produce different hashes (with very high probability)
        assert_ne!(sym1.hash_value(), sym3.hash_value());
    }

    #[test]
    fn test_large_number() {
        // Test with a number larger than u64::MAX
        let large = BigUint::from(u64::MAX) * BigUint::from(2u32);
        let sym = SymbolicInteger::from(large.clone());

        // Should not fit in u64
        assert_eq!(sym.to_u64(), None);

        // But base-96 conversion should still work
        let digits = sym.to_base96();
        let reconstructed = SymbolicInteger::from_base96(&digits).unwrap();
        assert_eq!(sym, reconstructed);
        assert_eq!(reconstructed.value(), &large);
    }

    #[test]
    fn test_display() {
        let sym = SymbolicInteger::from(1961u64);
        assert_eq!(format!("{}", sym), "1961");
    }
}
