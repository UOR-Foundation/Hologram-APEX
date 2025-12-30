//! Type definitions for Atlas backends

use std::{
    fmt,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Number of resonance classes defined by the Atlas ISA.
pub const RESONANCE_CLASS_COUNT: usize = 96;

/// Maximum neighbor slots per resonance class.
pub const NEIGHBOR_SLOTS: usize = 6;

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

/// Exact rational number with canonicalized representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational {
    numerator: i64,
    denominator: u64,
}

impl Rational {
    /// Create a new rational number and reduce it to canonical form.
    pub fn new(numerator: i64, denominator: u64) -> Self {
        assert_ne!(denominator, 0, "denominator cannot be zero");
        Self::from_parts(numerator as i128, denominator as u128)
    }

    /// Canonical zero value.
    pub const fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    /// Canonical one value.
    pub const fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    /// Get the numerator component.
    pub const fn numerator(&self) -> i64 {
        self.numerator
    }

    /// Get the denominator component (always positive).
    pub const fn denominator(&self) -> u64 {
        self.denominator
    }

    /// Returns true when the rational is exactly zero.
    pub const fn is_zero(&self) -> bool {
        self.numerator == 0
    }

    fn from_parts(numerator: i128, denominator: u128) -> Self {
        assert!(denominator != 0, "denominator cannot be zero");

        if numerator == 0 {
            return Self::zero();
        }

        let gcd = gcd_u128(numerator.unsigned_abs(), denominator);
        let reduced_num = numerator / gcd as i128;
        let reduced_den = denominator / gcd;

        let numerator = i64::try_from(reduced_num).expect("reduced numerator does not fit in i64");
        let denominator = u64::try_from(reduced_den).expect("reduced denominator does not fit in u64");

        Self { numerator, denominator }
    }

    /// Sum a sequence of rationals with exact arithmetic.
    pub fn sum<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        iter.into_iter().fold(Self::zero(), |acc, item| acc + item)
    }

    /// Compute the multiplicative inverse of the rational.
    pub fn reciprocal(self) -> Self {
        assert_ne!(self.numerator, 0, "cannot take reciprocal of zero");

        let numerator = if self.numerator < 0 {
            -(self.denominator as i128)
        } else {
            self.denominator as i128
        };
        let denominator = self.numerator.unsigned_abs() as u128;

        Self::from_parts(numerator, denominator)
    }

    /// Compute absolute value of the rational.
    pub fn abs(self) -> Self {
        if self.numerator < 0 {
            Self {
                numerator: -self.numerator,
                denominator: self.denominator,
            }
        } else {
            self
        }
    }

    /// Convert rational to f32 (may lose precision for large rationals).
    ///
    /// This performs floating-point division which may introduce rounding error.
    /// Use only when approximate output is acceptable (e.g., final results to user).
    pub fn to_f32(self) -> f32 {
        if self.numerator == 0 {
            return 0.0;
        }
        (self.numerator as f64 / self.denominator as f64) as f32
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<i64> for Rational {
    fn from(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }
}

impl From<f32> for Rational {
    /// Convert f32 to exact rational using IEEE 754 representation.
    ///
    /// This conversion is mathematically exact for all finite f32 values.
    /// Special values (infinity, NaN) will panic.
    ///
    /// # Panics
    ///
    /// Panics if the f32 value is infinity or NaN.
    fn from(value: f32) -> Self {
        // Special case: zero (both +0.0 and -0.0)
        if value == 0.0 {
            return Self::zero();
        }

        // Reject inf/NaN
        assert!(value.is_finite(), "cannot convert inf/NaN to Rational");

        // Extract IEEE 754 components
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        // Handle denormalized numbers (exponent == 0)
        let (mantissa_value, exp_bias) = if exponent == 0 {
            // Denormalized: value = (-1)^sign × 2^(-126) × (0.mantissa)
            (mantissa as i128, -126 - 23)
        } else {
            // Normalized: value = (-1)^sign × 2^(exp-127) × (1.mantissa)
            // mantissa_value includes implicit leading 1
            ((1i128 << 23) | mantissa as i128, exponent - 127 - 23)
        };

        // Now we have: value = (-1)^sign × mantissa_value × 2^exp_bias
        // Convert to rational: numerator / denominator

        let (numerator, denominator) = if exp_bias >= 0 {
            // Positive exponent: multiply mantissa by 2^exp_bias
            let shift = exp_bias as u32;
            if shift < 64 {
                (mantissa_value << shift, 1u128)
            } else {
                // Overflow risk - use from_parts which will handle it
                (mantissa_value * (1i128 << 63), 1u128 << (63 - shift))
            }
        } else {
            // Negative exponent: denominator is 2^(-exp_bias)
            (mantissa_value, 1u128 << (-exp_bias as u32))
        };

        // Apply sign
        let signed_numerator = if sign == 1 { -numerator } else { numerator };

        Self::from_parts(signed_numerator, denominator)
    }
}

impl From<&f32> for Rational {
    fn from(value: &f32) -> Self {
        Self::from(*value)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.numerator == 0 {
            return rhs;
        }
        if rhs.numerator == 0 {
            return self;
        }

        let gcd = gcd_u64(self.denominator, rhs.denominator);
        let lhs_multiplier = rhs.denominator / gcd;
        let rhs_multiplier = self.denominator / gcd;

        let numerator =
            (self.numerator as i128) * lhs_multiplier as i128 + (rhs.numerator as i128) * rhs_multiplier as i128;
        let denominator = (self.denominator / gcd) as u128 * rhs.denominator as u128;

        Self::from_parts(numerator, denominator)
    }
}

impl AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for Rational {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.numerator == 0 || rhs.numerator == 0 {
            return Self::zero();
        }

        let lhs_sign = if self.numerator < 0 { -1 } else { 1 };
        let rhs_sign = if rhs.numerator < 0 { -1 } else { 1 };

        let mut lhs_abs = (self.numerator as i128).unsigned_abs();
        let mut rhs_abs = (rhs.numerator as i128).unsigned_abs();
        let mut lhs_den = self.denominator as u128;
        let mut rhs_den = rhs.denominator as u128;

        let gcd1 = gcd_u128(lhs_abs, rhs_den);
        if gcd1 > 1 {
            lhs_abs /= gcd1;
            rhs_den /= gcd1;
        }

        let gcd2 = gcd_u128(rhs_abs, lhs_den);
        if gcd2 > 1 {
            rhs_abs /= gcd2;
            lhs_den /= gcd2;
        }

        let numerator_abs = lhs_abs
            .checked_mul(rhs_abs)
            .expect("numerator overflow during multiplication");
        let denominator = lhs_den
            .checked_mul(rhs_den)
            .expect("denominator overflow during multiplication");

        debug_assert!(numerator_abs <= i128::MAX as u128, "numerator exceeds i128 range");

        let sign = lhs_sign * rhs_sign;
        let numerator = if sign < 0 {
            -(numerator_abs as i128)
        } else {
            numerator_abs as i128
        };

        Self::from_parts(numerator, denominator)
    }
}

impl MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert_ne!(rhs.numerator, 0, "cannot divide by zero");

        if self.numerator == 0 {
            return Self::zero();
        }

        let lhs_sign = if self.numerator < 0 { -1 } else { 1 };
        let rhs_sign = if rhs.numerator < 0 { -1 } else { 1 };

        let mut lhs_abs = (self.numerator as i128).unsigned_abs();
        let mut rhs_abs = (rhs.numerator as i128).unsigned_abs();
        let mut lhs_den = self.denominator as u128;
        let mut rhs_den = rhs.denominator as u128;

        let gcd_num = gcd_u128(lhs_abs, rhs_abs);
        if gcd_num > 1 {
            lhs_abs /= gcd_num;
            rhs_abs /= gcd_num;
        }

        let gcd_den = gcd_u128(rhs_den, lhs_den);
        if gcd_den > 1 {
            rhs_den /= gcd_den;
            lhs_den /= gcd_den;
        }

        let numerator_abs = lhs_abs
            .checked_mul(rhs_den)
            .expect("numerator overflow during division");
        let denominator = lhs_den
            .checked_mul(rhs_abs)
            .expect("denominator overflow during division");

        debug_assert!(numerator_abs <= i128::MAX as u128, "numerator exceeds i128 range");

        let sign = lhs_sign * rhs_sign;
        let numerator = if sign < 0 {
            -(numerator_abs as i128)
        } else {
            numerator_abs as i128
        };

        Self::from_parts(numerator, denominator)
    }
}

impl DivAssign for Rational {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Sum for Rational {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + item)
    }
}

impl<'a> Sum<&'a Rational> for Rational {
    fn sum<I: Iterator<Item = &'a Rational>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, item| acc + *item)
    }
}

/// Lookup tables describing Atlas topology information.
#[derive(Debug, Clone)]
pub struct TopologyTables {
    mirrors: [u8; RESONANCE_CLASS_COUNT],
    neighbors: [[u8; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT],
}

impl TopologyTables {
    /// Build topology tables from canonical mirror and neighbor mappings.
    pub fn new(mirrors: [u8; RESONANCE_CLASS_COUNT], neighbors: [[u8; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT]) -> Self {
        Self { mirrors, neighbors }
    }

    /// Mirror mapping accessor.
    pub const fn mirrors(&self) -> &[u8; RESONANCE_CLASS_COUNT] {
        &self.mirrors
    }

    /// Neighbor table accessor.
    pub const fn neighbors(&self) -> &[[u8; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT] {
        &self.neighbors
    }

    /// Mirror partner for a resonance class.
    pub fn mirror_of(&self, class: usize) -> u8 {
        self.mirrors[class]
    }

    /// Neighbor slots for a resonance class.
    pub fn neighbors_of(&self, class: usize) -> &[u8; NEIGHBOR_SLOTS] {
        &self.neighbors[class]
    }
}

/// Backend-specific handle to allocated memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendHandle(pub u64);

/// Topology description for buffer allocation.
///
/// Unlike traditional allocation (size_bytes), Atlas allocation is informed
/// by the data's structure in Atlas space.
#[derive(Debug, Clone)]
pub struct BufferTopology {
    /// Resonance classes occupied by this data
    pub active_classes: Vec<u8>,

    /// Φ-encoded (page, byte) coordinates
    pub phi_coordinates: Vec<(u8, u8)>,

    /// Preferred phase for temporal locality
    pub phase_affinity: Option<u16>,

    /// Memory pool selection
    pub pool: MemoryPool,

    /// Size in bytes (for pool allocation)
    pub size_bytes: usize,

    /// Alignment requirement
    pub alignment: usize,
}

impl Default for BufferTopology {
    fn default() -> Self {
        Self {
            active_classes: Vec::new(),
            phi_coordinates: Vec::new(),
            phase_affinity: None,
            pool: MemoryPool::Linear,
            size_bytes: 0,
            alignment: 8,
        }
    }
}

/// Memory pool selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPool {
    /// Boundary-addressed, cache-resident (1.18 MB fixed)
    ///
    /// CPU: Pinned to L2 cache
    /// GPU: Device L2 cache
    /// Quantum: Primary qubit registers
    Boundary,

    /// Linear-addressed, RAM-resident (unlimited)
    ///
    /// CPU: Regular heap allocation
    /// GPU: Global device memory
    /// Quantum: Auxiliary classical memory
    Linear,
}

impl fmt::Display for MemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryPool::Boundary => write!(f, "Boundary (cache-resident)"),
            MemoryPool::Linear => write!(f, "Linear (RAM-resident)"),
        }
    }
}

/// Execution context for operations.
///
/// Provides all information needed for topology-aware execution.
#[derive(Debug, Clone)]
pub struct ExecutionContext<'a> {
    /// Current phase (0-767)
    pub phase: u16,

    /// Resonance classes active in this operation
    pub active_classes: Vec<u8>,

    /// Total number of elements being operated on
    pub n_elements: usize,

    /// Parallelism hint (optional)
    pub parallelism: Option<ParallelismHint>,

    /// Resonance accumulator snapshot for all 96 classes
    pub resonance: [Rational; RESONANCE_CLASS_COUNT],

    /// Topology lookup tables shared by the backend
    pub topology: &'a TopologyTables,
}

impl<'a> ExecutionContext<'a> {
    /// Create a new execution context with zeroed resonance accumulators.
    pub fn new(topology: &'a TopologyTables) -> Self {
        Self {
            phase: 0,
            active_classes: Vec::new(),
            n_elements: 0,
            parallelism: None,
            resonance: [Rational::zero(); RESONANCE_CLASS_COUNT],
            topology,
        }
    }
}

/// Parallelism strategy hint.
#[derive(Debug, Clone, Copy)]
pub enum ParallelismHint {
    /// Sequential execution (debugging)
    Sequential,

    /// Parallel across classes
    Classes,

    /// Parallel within classes (SIMD)
    SIMD,

    /// Auto-detect based on workload
    Auto,
}

// PHASE 3 NOTE: Tests below use the old Operation-based API which has been removed.
// These tests are temporarily disabled and will be rewritten in Phase 5-6 to use
// the new Program-based ISA instruction execution API.
// Phase 8/9 scaffolding: operation_api feature will be added in future phases
#[allow(unexpected_cfgs)] // operation_api feature will be added in Phase 8/9
#[cfg(all(test, feature = "operation_api"))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn dummy_topology() -> TopologyTables {
        let mut mirrors = [0u8; RESONANCE_CLASS_COUNT];
        for (idx, slot) in mirrors.iter_mut().enumerate() {
            *slot = idx as u8;
        }
        let neighbors = [[u8::MAX; NEIGHBOR_SLOTS]; RESONANCE_CLASS_COUNT];
        TopologyTables::new(mirrors, neighbors)
    }

    #[test]
    fn test_buffer_topology_default() {
        let topo = BufferTopology::default();
        assert_eq!(topo.active_classes.len(), 0);
        assert_eq!(topo.pool, MemoryPool::Linear);
        assert_eq!(topo.alignment, 8);
    }

    #[test]
    fn test_execution_context_new_initializes_resonance() {
        let tables = dummy_topology();
        let ctx = ExecutionContext::new(&tables);
        assert_eq!(ctx.phase, 0);
        assert_eq!(ctx.active_classes.len(), 0);
        assert_eq!(ctx.n_elements, 0);
        assert!(ctx.resonance.iter().all(|r| *r == Rational::zero()));
        assert!(std::ptr::eq(ctx.topology, &tables));
    }

    #[test]
    fn test_operation_display() {
        let op = Operation::VectorAdd {
            a: BackendHandle(0),
            b: BackendHandle(1),
            c: BackendHandle(2),
            n: 1024,
        };
        assert_eq!(format!("{}", op), "VectorAdd(n=1024)");
    }

    #[test]
    fn rational_new_canonicalizes() {
        let r = Rational::new(6, 8);
        assert_eq!(r.numerator(), 3);
        assert_eq!(r.denominator(), 4);

        let negative = Rational::new(-6, 8);
        assert_eq!(negative.numerator(), -3);
        assert_eq!(negative.denominator(), 4);
    }

    #[test]
    fn rational_arithmetic_basic() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        let sum = a + b;
        assert_eq!(sum.numerator(), 5);
        assert_eq!(sum.denominator(), 6);

        let product = a * b;
        assert_eq!(product.numerator(), 1);
        assert_eq!(product.denominator(), 6);

        let difference = a - b;
        assert_eq!(difference.numerator(), 1);
        assert_eq!(difference.denominator(), 6);
    }

    prop_compose! {
        fn rational_strategy()(num in -1000i32..=1000, denom in 1u32..=64) -> Rational {
            Rational::new(num as i64, denom as u64)
        }
    }

    proptest! {
        #[test]
        fn rational_add_commutative(a in rational_strategy(), b in rational_strategy()) {
            prop_assert_eq!(a + b, b + a);
        }

        #[test]
        fn rational_mul_associative(a in rational_strategy(), b in rational_strategy(), c in rational_strategy()) {
            let left = (a * b) * c;
            let right = a * (b * c);
            prop_assert_eq!(left, right);
        }

        #[test]
        fn rational_add_identity(a in rational_strategy()) {
            let zero = Rational::zero();
            prop_assert_eq!(a + zero, a);
            prop_assert_eq!(zero + a, a);
        }

        #[test]
        fn rational_mul_identity(a in rational_strategy()) {
            let one = Rational::from(1i64);
            prop_assert_eq!(a * one, a);
            prop_assert_eq!(one * a, a);
        }

        #[test]
        fn rational_mul_zero(a in rational_strategy()) {
            let zero = Rational::zero();
            prop_assert_eq!(a * zero, zero);
            prop_assert_eq!(zero * a, zero);
        }

        #[test]
        fn rational_sub_self_is_zero(a in rational_strategy()) {
            prop_assert_eq!(a - a, Rational::zero());
        }

        #[test]
        fn rational_add_sub_inverse(a in rational_strategy(), b in rational_strategy()) {
            let sum = a + b;
            let back = sum - b;
            prop_assert_eq!(back, a);
        }

        #[test]
        fn rational_mul_distributive(a in rational_strategy(), b in rational_strategy(), c in rational_strategy()) {
            let left = a * (b + c);
            let right = (a * b) + (a * c);
            prop_assert_eq!(left, right);
        }

        #[test]
        fn rational_div_self_is_one(a in rational_strategy()) {
            // Skip zero division
            if a == Rational::zero() {
                return Ok(());
            }
            let one = Rational::from(1i64);
            prop_assert_eq!(a / a, one);
        }

        #[test]
        fn rational_mul_div_inverse(a in rational_strategy(), b in rational_strategy()) {
            // Skip zero division
            if b == Rational::zero() {
                return Ok(());
            }
            let product = a * b;
            let back = product / b;
            prop_assert_eq!(back, a);
        }

        #[test]
        fn rational_abs_positive_or_zero(a in rational_strategy()) {
            let abs = a.abs();
            prop_assert!(abs.numerator() >= 0);
        }

        #[test]
        fn rational_abs_idempotent(a in rational_strategy()) {
            let abs1 = a.abs();
            let abs2 = abs1.abs();
            prop_assert_eq!(abs1, abs2);
        }

        #[test]
        fn rational_abs_preserves_zero(a in rational_strategy()) {
            if a == Rational::zero() {
                prop_assert_eq!(a.abs(), Rational::zero());
            }
        }

        #[test]
        fn rational_from_f32_finite(val in -1000.0f32..=1000.0f32) {
            if !val.is_finite() {
                return Ok(());
            }
            let r = Rational::from(val);
            let back = r.to_f32();

            // Allow small floating-point error
            let error = (back - val).abs();
            prop_assert!(
                error < 0.001 || (val.abs() > 1000.0 && error / val.abs() < 0.001),
                "Conversion error too large: {} -> {:?} -> {}, error = {}",
                val,
                r,
                back,
                error
            );
        }

        #[test]
        fn rational_canonicalization_gcd(num in -1000i64..=1000, denom in 1u64..=100) {
            let r = Rational::new(num, denom);
            // After canonicalization, gcd(numerator, denominator) should be 1
            let gcd = {
                fn gcd(mut a: u64, mut b: u64) -> u64 {
                    while b != 0 {
                        let temp = b;
                        b = a % b;
                        a = temp;
                    }
                    a
                }
                gcd(r.numerator().abs() as u64, r.denominator())
            };
            prop_assert_eq!(gcd, 1, "GCD of numerator {} and denominator {} should be 1", r.numerator(), r.denominator());
        }
    }

    #[test]
    fn rational_to_f32_zero() {
        let zero = Rational::zero();
        assert_eq!(zero.to_f32(), 0.0);
    }
}
