//! Four normed division algebras: ℝ, ℂ, ℍ, 𝕆
//!
//! The four normed division algebras form the foundation of the HRM geometric structure:
//!
//! - **ℝ** (Real numbers, dim 1): Associative, commutative
//! - **ℂ** (Complex numbers, dim 2): Associative, non-commutative
//! - **ℍ** (Quaternions, dim 4): Non-associative, non-commutative
//! - **𝕆** (Octonions, dim 8): Non-associative, non-commutative, non-alternative
//!
//! These correspond to the (h₂, d, ℓ) structure in Atlas generation:
//! - h₂ ∈ ℤ₄: quaternionic quadrant (ℝ → ℂ → ℍ progression)
//! - d ∈ ℤ₃: octonionic triality (𝕆 structure)
//! - ℓ ∈ ℤ₈: Clifford algebra context

#![allow(missing_docs)]
#![allow(clippy::needless_range_loop)]

use crate::algebra::Ring;
use crate::Result;

/// Real algebra ℝ (dimension 1)
///
/// The simplest normed division algebra - just real numbers.
/// Forms the foundation for all higher division algebras.
#[derive(Debug, Clone, PartialEq)]
pub struct Real {
    pub value: f64,
}

impl Real {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn norm(&self) -> f64 {
        self.value.abs()
    }
}

impl Ring for Real {
    fn zero() -> Self {
        Self { value: 0.0 }
    }

    fn one() -> Self {
        Self { value: 1.0 }
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            value: self.value + other.value,
        })
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            value: self.value * other.value,
        })
    }

    fn neg(&self) -> Result<Self> {
        Ok(Self { value: -self.value })
    }

    fn is_zero(&self) -> bool {
        self.value.abs() < 1e-10
    }
}

/// Complex algebra ℂ (dimension 2)
///
/// Complex numbers: a + bi where i² = -1
/// Associative and commutative.
#[derive(Debug, Clone, PartialEq)]
pub struct Complex {
    pub re: f64, // Real part
    pub im: f64, // Imaginary part
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Ring for Complex {
    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            re: self.re + other.re,
            im: self.im + other.im,
        })
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        Ok(Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        })
    }

    fn neg(&self) -> Result<Self> {
        Ok(Self {
            re: -self.re,
            im: -self.im,
        })
    }

    fn is_zero(&self) -> bool {
        self.norm() < 1e-10
    }
}

/// Quaternion algebra ℍ (dimension 4)
///
/// Quaternions: a + bi + cj + dk where i² = j² = k² = ijk = -1
/// Non-commutative but still associative.
///
/// Multiplication rules:
/// - ij = k, jk = i, ki = j
/// - ji = -k, kj = -i, ik = -j
#[derive(Debug, Clone, PartialEq)]
pub struct Quaternion {
    pub w: f64, // Real part
    pub x: f64, // i coefficient
    pub y: f64, // j coefficient
    pub z: f64, // k coefficient
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Ring for Quaternion {
    fn zero() -> Self {
        Self {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn one() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            w: self.w + other.w,
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        })
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        // Hamilton product (non-commutative)
        Ok(Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        })
    }

    fn neg(&self) -> Result<Self> {
        Ok(Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        })
    }

    fn is_zero(&self) -> bool {
        self.norm() < 1e-10
    }
}

/// Octonion algebra 𝕆 (dimension 8)
///
/// Octonions: 8-dimensional non-associative algebra
/// The largest normed division algebra (by Hurwitz's theorem).
///
/// Basis: {1, e₁, e₂, e₃, e₄, e₅, e₆, e₇}
/// Multiplication is non-associative and non-commutative.
#[derive(Debug, Clone, PartialEq)]
pub struct Octonion {
    pub components: [f64; 8],
}

impl Octonion {
    pub fn new(components: [f64; 8]) -> Self {
        Self { components }
    }

    pub fn from_scalar(s: f64) -> Self {
        let mut components = [0.0; 8];
        components[0] = s;
        Self { components }
    }

    pub fn norm(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        let mut conj = self.components;
        for i in 1..8 {
            conj[i] = -conj[i];
        }
        Self { components: conj }
    }
}

impl Ring for Octonion {
    fn zero() -> Self {
        Self { components: [0.0; 8] }
    }

    fn one() -> Self {
        let mut components = [0.0; 8];
        components[0] = 1.0;
        Self { components }
    }

    fn add(&self, other: &Self) -> Result<Self> {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = self.components[i] + other.components[i];
        }
        Ok(Self { components: result })
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        // Octonion multiplication using the Cayley-Dickson construction
        // Split into two quaternions: (a, b) * (c, d) = (ac - d*b, da + bc*)

        let self_q1 = Quaternion::new(
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
        );
        let self_q2 = Quaternion::new(
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7],
        );

        let other_q1 = Quaternion::new(
            other.components[0],
            other.components[1],
            other.components[2],
            other.components[3],
        );
        let other_q2 = Quaternion::new(
            other.components[4],
            other.components[5],
            other.components[6],
            other.components[7],
        );

        // First quaternion: ac - d*b
        let ac = self_q1.mul(&other_q1)?;
        let d_conj = other_q2.conjugate();
        let d_conj_b = d_conj.mul(&self_q2)?;
        let result_q1 = ac.add(&d_conj_b.neg()?)?;

        // Second quaternion: da + bc*
        let da = other_q2.mul(&self_q1)?;
        let b_conj = self_q2.conjugate();
        let bc_conj = b_conj.mul(&other_q1)?;
        let result_q2 = da.add(&bc_conj)?;

        Ok(Self {
            components: [
                result_q1.w,
                result_q1.x,
                result_q1.y,
                result_q1.z,
                result_q2.w,
                result_q2.x,
                result_q2.y,
                result_q2.z,
            ],
        })
    }

    fn neg(&self) -> Result<Self> {
        let mut result = [0.0; 8];
        for i in 0..8 {
            result[i] = -self.components[i];
        }
        Ok(Self { components: result })
    }

    fn is_zero(&self) -> bool {
        self.norm() < 1e-10
    }
}

/// Division algebra type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionAlgebraType {
    Real,
    Complex,
    Quaternion,
    Octonion,
}

impl DivisionAlgebraType {
    pub fn dimension(&self) -> usize {
        match self {
            Self::Real => 1,
            Self::Complex => 2,
            Self::Quaternion => 4,
            Self::Octonion => 8,
        }
    }

    pub fn is_associative(&self) -> bool {
        matches!(self, Self::Real | Self::Complex | Self::Quaternion)
    }

    pub fn is_commutative(&self) -> bool {
        matches!(self, Self::Real | Self::Complex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_ring() {
        let a = Real::new(3.0);
        let b = Real::new(4.0);

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.value, 7.0);

        let product = a.mul(&b).unwrap();
        assert_eq!(product.value, 12.0);

        assert_eq!(Real::zero().value, 0.0);
        assert_eq!(Real::one().value, 1.0);
    }

    #[test]
    fn test_complex_ring() {
        let a = Complex::new(1.0, 2.0); // 1 + 2i
        let b = Complex::new(3.0, 4.0); // 3 + 4i

        let sum = a.add(&b).unwrap();
        assert_eq!(sum, Complex::new(4.0, 6.0));

        // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let product = a.mul(&b).unwrap();
        assert!((product.re + 5.0).abs() < 1e-10);
        assert!((product.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_ring() {
        let _a = Quaternion::new(1.0, 0.0, 0.0, 0.0); // 1
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0); // i
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0); // j
        let k = Quaternion::new(0.0, 0.0, 0.0, 1.0); // k

        // Test i² = -1
        let i_squared = i.mul(&i).unwrap();
        assert_eq!(i_squared, Quaternion::new(-1.0, 0.0, 0.0, 0.0));

        // Test ij = k
        let ij = i.mul(&j).unwrap();
        assert_eq!(ij, k);

        // Test ji = -k (non-commutative)
        let ji = j.mul(&i).unwrap();
        assert_eq!(ji, k.neg().unwrap());
    }

    #[test]
    fn test_octonion_ring() {
        let one = Octonion::one();
        let e1 = Octonion::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Test 1 * e1 = e1
        let product = one.mul(&e1).unwrap();
        assert_eq!(product, e1);

        // Test e1² = -1
        let e1_squared = e1.mul(&e1).unwrap();
        assert!((e1_squared.components[0] + 1.0).abs() < 1e-10);
        for i in 1..8 {
            assert!(e1_squared.components[i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_division_algebra_types() {
        assert_eq!(DivisionAlgebraType::Real.dimension(), 1);
        assert_eq!(DivisionAlgebraType::Complex.dimension(), 2);
        assert_eq!(DivisionAlgebraType::Quaternion.dimension(), 4);
        assert_eq!(DivisionAlgebraType::Octonion.dimension(), 8);

        assert!(DivisionAlgebraType::Real.is_associative());
        assert!(DivisionAlgebraType::Complex.is_associative());
        assert!(DivisionAlgebraType::Quaternion.is_associative());
        assert!(!DivisionAlgebraType::Octonion.is_associative());

        assert!(DivisionAlgebraType::Real.is_commutative());
        assert!(DivisionAlgebraType::Complex.is_commutative());
        assert!(!DivisionAlgebraType::Quaternion.is_commutative());
        assert!(!DivisionAlgebraType::Octonion.is_commutative());
    }

    #[test]
    fn test_norms() {
        let r = Real::new(3.0);
        assert_eq!(r.norm(), 3.0);

        let c = Complex::new(3.0, 4.0);
        assert_eq!(c.norm(), 5.0); // 3-4-5 triangle

        let q = Quaternion::new(1.0, 2.0, 2.0, 4.0);
        assert_eq!(q.norm(), 5.0); // sqrt(1 + 4 + 4 + 16) = 5

        let o = Octonion::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(o.norm(), 1.0);
    }
}
