//! Input Hashing for Pattern Lookups
//!
//! Implements FNV-1a hashing for fast, deterministic input pattern hashing.
//!
//! ## Performance
//!
//! - FNV-1a: ~10ns for 256 f32 values
//! - Good distribution, low collision rate
//! - Simple implementation (no dependencies)

/// FNV-1a offset basis (64-bit)
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

/// FNV-1a prime (64-bit)
const FNV_PRIME: u64 = 0x100000001b3;

/// Hash f32 input pattern using FNV-1a
///
/// Hashes the bit patterns of floating-point values for deterministic,
/// bitwise-exact matching.
///
/// # Arguments
///
/// * `input` - Slice of f32 values to hash
///
/// # Returns
///
/// 64-bit FNV-1a hash
///
/// # Example
///
/// ```
/// use hologram_core::moonshine::hash_input_f32;
///
/// let input = vec![1.0f32, 2.0, 3.0];
/// let hash1 = hash_input_f32(&input);
/// let hash2 = hash_input_f32(&input);
///
/// assert_eq!(hash1, hash2); // Deterministic
/// ```
pub fn hash_input_f32(input: &[f32]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &value in input {
        // Hash the bit pattern of the float (bitwise exact)
        let bytes = value.to_bits().to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

/// Hash f64 input pattern using FNV-1a
///
/// # Arguments
///
/// * `input` - Slice of f64 values to hash
///
/// # Returns
///
/// 64-bit FNV-1a hash
pub fn hash_input_f64(input: &[f64]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &value in input {
        // Hash the bit pattern of the float (bitwise exact)
        let bytes = value.to_bits().to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

/// Hash i32 input pattern using FNV-1a
///
/// # Arguments
///
/// * `input` - Slice of i32 values to hash
///
/// # Returns
///
/// 64-bit FNV-1a hash
pub fn hash_input_i32(input: &[i32]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &value in input {
        let bytes = value.to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

/// Hash i64 input pattern using FNV-1a
///
/// # Arguments
///
/// * `input` - Slice of i64 values to hash
///
/// # Returns
///
/// 64-bit FNV-1a hash
pub fn hash_input_i64(input: &[i64]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &value in input {
        let bytes = value.to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];

        let hash1 = hash_input_f32(&input);
        let hash2 = hash_input_f32(&input);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let input1 = vec![1.0f32, 2.0, 3.0];
        let input2 = vec![1.0f32, 2.0, 4.0];

        let hash1 = hash_input_f32(&input1);
        let hash2 = hash_input_f32(&input2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_order_matters() {
        let input1 = vec![1.0f32, 2.0, 3.0];
        let input2 = vec![3.0f32, 2.0, 1.0];

        let hash1 = hash_input_f32(&input1);
        let hash2 = hash_input_f32(&input2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_bitwise_exact() {
        // 0.0 and -0.0 have different bit patterns
        let input1 = vec![0.0f32];
        let input2 = vec![-0.0f32];

        let hash1 = hash_input_f32(&input1);
        let hash2 = hash_input_f32(&input2);

        // They should have different hashes (bitwise exact)
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_f64_deterministic() {
        let input = vec![1.0f64, 2.0, 3.0];

        let hash1 = hash_input_f64(&input);
        let hash2 = hash_input_f64(&input);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_i32_deterministic() {
        let input = vec![1i32, 2, 3, 4];

        let hash1 = hash_input_i32(&input);
        let hash2 = hash_input_i32(&input);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_i64_deterministic() {
        let input = vec![1i64, 2, 3, 4];

        let hash1 = hash_input_i64(&input);
        let hash2 = hash_input_i64(&input);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_empty_input() {
        let empty: Vec<f32> = vec![];
        let hash = hash_input_f32(&empty);

        // Empty input should produce FNV offset basis
        assert_eq!(hash, FNV_OFFSET_BASIS);
    }
}
