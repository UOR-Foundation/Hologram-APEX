//! Griess algebra vectors (196,884-dimensional)
//!
//! This module implements the fundamental vector type for the Griess algebra,
//! using Apache Arrow's Float64Array for zero-copy operations.

use crate::{Error, Result, GRIESS_DIMENSION};
use arrow_array::Float64Array;
use std::sync::Arc;

/// A 196,884-dimensional vector in the Griess algebra
///
/// The Griess algebra is the commutative non-associative algebra underlying
/// the Monster group. For HRM, we use a simplified version where the product
/// is component-wise (Hadamard product).
///
/// # Memory Layout
///
/// Vectors are stored using Arrow's Float64Array, providing:
/// - Zero-copy access to underlying f64 slice
/// - Efficient memory layout (contiguous buffer)
/// - Reference-counted sharing (cheap clones)
///
/// # Example
///
/// ```rust,ignore
/// use hologram_hrm::griess::GriessVector;
///
/// // Create from Vec
/// let data: Vec<f64> = vec![1.0; 196_884];
/// let v = GriessVector::from_vec(data)?;
///
/// // Zero-copy access
/// let slice: &[f64] = v.as_slice();
///
/// // Cheap clone (Arc)
/// let v2 = v.clone();
/// ```
#[derive(Debug)]
pub struct GriessVector {
    /// Arrow Float64Array for zero-copy operations
    data: Arc<Float64Array>,
}

impl GriessVector {
    /// Griess algebra dimension (196,884)
    pub const DIMENSION: usize = GRIESS_DIMENSION;

    /// Create a Griess vector from a Vec<f64>
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidDimension` if the vector length is not 196,884.
    pub fn from_vec(data: Vec<f64>) -> Result<Self> {
        if data.len() != Self::DIMENSION {
            return Err(Error::InvalidDimension(data.len()));
        }
        Ok(Self {
            data: Arc::new(Float64Array::from(data)),
        })
    }

    /// Create a Griess vector from an Arrow Float64Array
    ///
    /// This is a zero-copy operation if the array is already the correct size.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidDimension` if the array length is not 196,884.
    pub fn from_arrow(data: Arc<Float64Array>) -> Result<Self> {
        if data.len() != Self::DIMENSION {
            return Err(Error::InvalidDimension(data.len()));
        }
        Ok(Self { data })
    }

    /// Zero-copy access to the underlying f64 slice
    ///
    /// This returns a direct reference to the Arrow buffer's values,
    /// with no allocation or copying.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data.values()
    }

    /// Get the Arrow Float64Array (for storage/serialization)
    pub fn as_arrow(&self) -> &Arc<Float64Array> {
        &self.data
    }

    /// Get a mutable copy of the data as Vec<f64>
    ///
    /// This allocates a new Vec and copies the data. Use sparingly.
    pub fn to_vec(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }

    /// Create the identity element (all 1.0s)
    ///
    /// For Hadamard product, the identity is a vector of all 1.0s:
    /// ```text
    /// v ⊙ I = I ⊙ v = v
    /// ```
    pub fn identity() -> Self {
        Self::from_vec(vec![1.0; Self::DIMENSION]).expect("Identity vector has correct dimension")
    }

    /// Create the zero element (all 0.0s)
    pub fn zero() -> Self {
        Self::from_vec(vec![0.0; Self::DIMENSION]).expect("Zero vector has correct dimension")
    }

    /// Get the dimension of this vector
    #[inline]
    pub fn len(&self) -> usize {
        Self::DIMENSION
    }

    /// Check if the vector is empty
    ///
    /// Since GriessVector is always 196,884-dimensional, this always returns false.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Check if the vector is the zero vector (within tolerance)
    pub fn is_zero(&self, tolerance: f64) -> bool {
        self.as_slice().iter().all(|&x| x.abs() < tolerance)
    }

    /// Check if the vector is close to the identity (within tolerance)
    pub fn is_near_identity(&self, tolerance: f64) -> bool {
        self.as_slice().iter().all(|&x| (x - 1.0).abs() < tolerance)
    }

    /// Compute the L2 norm (Euclidean length)
    pub fn norm(&self) -> f64 {
        self.as_slice().iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize to unit length (L2 norm = 1)
    ///
    /// Returns a new normalized vector.
    pub fn normalize(&self) -> Result<Self> {
        let norm = self.norm();
        if norm < 1e-10 {
            return Err(Error::DecodingFailed("Cannot normalize zero vector".to_string()));
        }

        let normalized: Vec<f64> = self.as_slice().iter().map(|&x| x / norm).collect();
        Self::from_vec(normalized)
    }

    /// Compute the inner product (dot product) with another vector
    pub fn inner_product(&self, other: &Self) -> f64 {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    /// Compute Euclidean distance to another vector
    pub fn distance(&self, other: &Self) -> f64 {
        self.distance_squared(other).sqrt()
    }

    /// Compute squared Euclidean distance to another vector
    ///
    /// This is faster than `distance()` when you only need to compare distances,
    /// as it avoids the expensive sqrt() operation.
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> f64 {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
    }
}

/// Cheap cloning via Arc (reference counting)
///
/// Cloning a GriessVector is O(1) as it only increments the reference count.
impl Clone for GriessVector {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
        }
    }
}

impl PartialEq for GriessVector {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_vector() {
        let data: Vec<f64> = vec![1.0; GRIESS_DIMENSION];
        let v = GriessVector::from_vec(data).unwrap();
        assert_eq!(v.len(), GRIESS_DIMENSION);
    }

    #[test]
    fn test_invalid_dimension() {
        let data: Vec<f64> = vec![1.0; 100];
        let result = GriessVector::from_vec(data);
        assert!(matches!(result, Err(Error::InvalidDimension(100))));
    }

    #[test]
    fn test_identity() {
        let id = GriessVector::identity();
        assert_eq!(id.len(), GRIESS_DIMENSION);
        assert!(id.as_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_zero() {
        let z = GriessVector::zero();
        assert_eq!(z.len(), GRIESS_DIMENSION);
        assert!(z.as_slice().iter().all(|&x| x == 0.0));
        assert!(z.is_zero(1e-10));
    }

    #[test]
    fn test_norm() {
        let data: Vec<f64> = vec![1.0; GRIESS_DIMENSION];
        let v = GriessVector::from_vec(data).unwrap();
        let norm = v.norm();
        let expected = (GRIESS_DIMENSION as f64).sqrt();
        assert!((norm - expected).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let data: Vec<f64> = vec![2.0; GRIESS_DIMENSION];
        let v = GriessVector::from_vec(data).unwrap();
        let normalized = v.normalize().unwrap();
        let norm = normalized.norm();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let v1 = GriessVector::identity();
        let v2 = GriessVector::identity();
        let dot = v1.inner_product(&v2);
        assert_eq!(dot, GRIESS_DIMENSION as f64);
    }

    #[test]
    fn test_distance() {
        let v1 = GriessVector::identity();
        let v2 = GriessVector::zero();
        let dist = v1.distance(&v2);
        let expected = (GRIESS_DIMENSION as f64).sqrt();
        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_clone_is_cheap() {
        let v1 = GriessVector::identity();
        let v2 = v1.clone();
        assert_eq!(v1, v2);
        // Arc clone doesn't copy data, just increments refcount
        assert!(Arc::ptr_eq(&v1.data, &v2.data));
    }
}
