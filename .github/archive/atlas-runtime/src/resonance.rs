//! Resonance accumulator R\[96\]
//!
//! Implements the per-class resonance state tracking as specified in
//! Atlas Runtime Spec §6.1 - Resonance Arithmetic (Canonical).
//!
//! ## Normative Semantics (§6.1)
//!
//! Resonance values are **exact rationals** in canonical form:
//! - `d > 0` (denominator always positive)
//! - `gcd(|n|, d) = 1` (fully reduced)
//! - Sign on numerator only
//! - Zero is `0/1`
//!
//! The `RES_ACCUM(c, a/b)` operation is defined as:
//! 1. Let `R[c] = n/d`. Compute `n' = n·b + a·d`, `d' = d·b`
//! 2. Compute `g = gcd(|n'|, |d'|)`
//! 3. Set `n'' = n'/g`, `d'' = d'/g`. If `d'' < 0`, flip signs
//! 4. Write back `R[c] = n''/d''`

use atlas_core::AtlasRatio;
use num_integer::Integer; // For gcd
use parking_lot::RwLock;
use tracing::warn;

use crate::error::{AtlasError, Result};

fn validate_ratio(ratio: AtlasRatio) -> Result<()> {
    if ratio.denom == 0 {
        return Err(AtlasError::ResonanceZeroDenominator);
    }

    if ratio.denom < 0 {
        return Err(AtlasError::ResonanceNonCanonical {
            numer: ratio.numer,
            denom: ratio.denom,
        });
    }

    if ratio.numer == 0 {
        if ratio.denom != 1 {
            return Err(AtlasError::ResonanceNonCanonical {
                numer: ratio.numer,
                denom: ratio.denom,
            });
        }
        return Ok(());
    }

    let gcd = ratio.numer.abs().gcd(&ratio.denom);
    if gcd != 1 {
        return Err(AtlasError::ResonanceNonCanonical {
            numer: ratio.numer,
            denom: ratio.denom,
        });
    }

    Ok(())
}

/// Canonical resonance accumulation per §6.1
///
/// Implements exact rational addition with GCD reduction to maintain canonical form:
/// - d > 0 (denominator always positive)
/// - gcd(|n|, d) = 1 (fully reduced)
/// - Sign on numerator only
///
/// # Algorithm (§6.1)
///
/// Given current = n/d and delta = a/b:
/// 1. Compute n' = n·b + a·d, d' = d·b
/// 2. Compute g = gcd(|n'|, |d'|)
/// 3. Set n'' = n'/g, d'' = d'/g
/// 4. If d'' < 0, flip signs: n'' = -n'', d'' = -d''
///
/// # Invariant
///
/// The result is always in canonical form.
fn res_accum(current: AtlasRatio, delta: AtlasRatio) -> Result<AtlasRatio> {
    // Promote to 128-bit to avoid intermediate overflow
    let n_prime = (current.numer as i128) * (delta.denom as i128) + (delta.numer as i128) * (current.denom as i128);
    let d_prime = (current.denom as i128) * (delta.denom as i128);

    // Step 2: Compute GCD for reduction
    let g = n_prime.abs().gcd(&d_prime.abs());

    // Step 3: Reduce by GCD
    let mut n_final = n_prime / g;
    let mut d_final = d_prime / g;

    // Step 4: Ensure denominator is positive (canonical form)
    if d_final < 0 {
        n_final = -n_final;
        d_final = -d_final;
    }

    // Step 5: Ensure result fits in i64 (AtlasRatio representation)
    if n_final < i64::MIN as i128
        || n_final > i64::MAX as i128
        || d_final < i64::MIN as i128
        || d_final > i64::MAX as i128
    {
        warn!(
            target: "atlas::resonance",
            numer = n_final,
            denom = d_final,
            "resonance accumulation overflow"
        );
        return Err(AtlasError::ResonanceOverflow {
            numer: n_final,
            denom: d_final,
        });
    }

    let result = AtlasRatio::new_raw(n_final as i64, d_final as i64);
    debug_assert!(
        validate_ratio(result).is_ok(),
        "canonical accumulation must produce canonical ratio"
    );
    Ok(result)
}

/// Absolute value of a rational
fn ratio_abs(r: AtlasRatio) -> AtlasRatio {
    AtlasRatio::new_raw(r.numer.abs(), r.denom.abs())
}

/// Compare two rationals: returns true if a > b
///
/// Uses cross-multiplication to avoid division:
/// a/b > c/d ⟺ a·d > c·b (when b,d > 0)
fn ratio_gt(a: AtlasRatio, b: AtlasRatio) -> bool {
    // Assume canonical form: denominators are positive
    debug_assert!(
        a.denom > 0 && b.denom > 0,
        "denominators must be positive (canonical form)"
    );
    a.numer * b.denom > b.numer * a.denom
}

/// Resonance Accumulator R\[96\]
///
/// Tracks the accumulated resonance delta for each of the 96 classes.
/// Each entry is an exact rational number (`AtlasRatio` from atlas-core)
/// representing the net change to that class's resonance.
///
/// # Unity Neutrality
///
/// For kernels marked `unity_neutral`, the sum of all deltas must be zero:
/// ```text
/// Σ(R[0] + R[1] + ... + R[95]) = 0
/// ```
///
/// This is verified using exact rational arithmetic to avoid floating-point
/// error accumulation.
///
/// # Example
///
/// ```
/// use atlas_runtime::ResonanceAccumulator;
/// use atlas_core::AtlasRatio;
///
/// let mut accum = ResonanceAccumulator::new();
///
/// // Add delta to class 10
/// let delta = AtlasRatio::new_raw(1, 2); // 1/2
/// accum.add(10, delta).unwrap();
///
/// // Verify it was recorded
/// assert_eq!(accum.get(10), delta);
/// ```
pub struct ResonanceAccumulator {
    /// Per-class resonance values (exact rationals)
    /// Protected by RwLock for concurrent access
    values: RwLock<[AtlasRatio; 96]>,
}

impl ResonanceAccumulator {
    /// Create a new accumulator with all classes at zero resonance
    pub fn new() -> Self {
        Self {
            values: RwLock::new([AtlasRatio::new_raw(0, 1); 96]),
        }
    }

    /// Get current resonance value for a class (lock-free read)
    pub fn get(&self, class: u8) -> AtlasRatio {
        debug_assert!((class as usize) < 96, "class must be < 96");
        self.values.read()[class as usize]
    }

    /// Add delta to a class's resonance using canonical accumulation (§6.1)
    pub fn add(&self, class: u8, delta: AtlasRatio) -> Result<()> {
        debug_assert!((class as usize) < 96, "class must be < 96");
        validate_ratio(delta)?;

        let mut values = self.values.write();
        debug_assert!(
            validate_ratio(values[class as usize]).is_ok(),
            "stored resonance value must remain canonical"
        );
        let current = values[class as usize];
        let updated = match res_accum(current, delta) {
            Ok(updated) => updated,
            Err(err @ AtlasError::ResonanceOverflow { .. }) => {
                warn!(
                    target: "atlas::resonance",
                    class,
                    current_numer = current.numer,
                    current_denom = current.denom,
                    delta_numer = delta.numer,
                    delta_denom = delta.denom,
                    "resonance accumulation overflow"
                );
                return Err(err);
            }
            Err(err) => return Err(err),
        };
        values[class as usize] = updated;
        Ok(())
    }

    /// Set resonance value for a class directly
    pub fn set(&self, class: u8, value: AtlasRatio) -> Result<()> {
        debug_assert!((class as usize) < 96, "class must be < 96");
        validate_ratio(value)?;
        let mut values = self.values.write();
        values[class as usize] = value;
        Ok(())
    }

    /// Add an array of deltas (one per class)
    ///
    /// This is the typical interface for kernel execution:
    /// the kernel produces a delta array and it's added to the accumulator.
    /// Uses canonical accumulation (§6.1) for each class.
    pub fn add_deltas(&self, deltas: &[AtlasRatio; 96]) -> Result<()> {
        for ratio in deltas.iter() {
            validate_ratio(*ratio)?;
        }

        let mut values = self.values.write();
        for i in 0..96 {
            let current = values[i];
            let delta = deltas[i];
            let updated = match res_accum(current, delta) {
                Ok(updated) => updated,
                Err(err @ AtlasError::ResonanceOverflow { .. }) => {
                    warn!(
                        target: "atlas::resonance",
                        class = i,
                        current_numer = current.numer,
                        current_denom = current.denom,
                        delta_numer = delta.numer,
                        delta_denom = delta.denom,
                        "resonance accumulation overflow"
                    );
                    return Err(err);
                }
                Err(err) => return Err(err),
            };
            values[i] = updated;
        }
        Ok(())
    }

    /// Get a snapshot of all resonance values
    pub fn snapshot(&self) -> [AtlasRatio; 96] {
        *self.values.read()
    }

    /// Reset all resonance values to zero
    pub fn reset(&self) {
        let mut values = self.values.write();
        for i in 0..96 {
            values[i] = AtlasRatio::new_raw(0, 1);
        }
    }

    /// Check unity neutrality: sum of all resonances must be zero
    ///
    /// This is the critical invariant check for `unity_neutral` kernels.
    /// Uses exact rational arithmetic to avoid floating-point errors.
    ///
    /// # Errors
    ///
    /// Returns `UnityNeutralityViolation` if the sum is non-zero.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::ResonanceAccumulator;
    /// use atlas_core::AtlasRatio;
    ///
    /// let accum = ResonanceAccumulator::new();
    ///
    /// // Add balanced deltas
    /// accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();   // +1/2
    /// accum.add(1, AtlasRatio::new_raw(-1, 2)).unwrap();  // -1/2
    ///
    /// // Should be neutral
    /// assert!(accum.check_unity_neutral().is_ok());
    /// ```
    pub fn check_unity_neutral(&self) -> Result<()> {
        let sum = self.sum()?;
        if sum.numer == 0 && sum.denom > 0 {
            Ok(())
        } else {
            Err(AtlasError::UnityNeutralityViolation)
        }
    }

    /// Compute sum of all resonance values
    ///
    /// Uses exact rational arithmetic with canonical accumulation (§6.1).
    /// The sum should be zero for unity-neutral execution.
    pub fn sum(&self) -> Result<AtlasRatio> {
        let values = self.values.read();
        let mut sum = AtlasRatio::new_raw(0, 1);
        for i in 0..96 {
            sum = res_accum(sum, values[i])?;
        }
        Ok(sum)
    }

    /// Check if a delta array is unity-neutral (static check)
    ///
    /// This can be used to validate deltas before adding them to the accumulator.
    /// Uses canonical accumulation (§6.1) for exact rational arithmetic.
    pub fn check_neutrality(deltas: &[AtlasRatio; 96]) -> Result<()> {
        for ratio in deltas.iter() {
            validate_ratio(*ratio)?;
        }
        let mut sum = AtlasRatio::new_raw(0, 1);
        for delta in deltas {
            sum = res_accum(sum, *delta)?;
        }

        if sum.numer == 0 && sum.denom > 0 {
            Ok(())
        } else {
            Err(AtlasError::UnityNeutralityViolation)
        }
    }

    /// Get maximum absolute resonance across all classes
    ///
    /// Useful for monitoring resonance drift and detecting anomalies.
    pub fn max_abs(&self) -> AtlasRatio {
        let values = self.values.read();
        let mut max = AtlasRatio::new_raw(0, 1);

        for i in 0..96 {
            let abs_val = ratio_abs(values[i]);
            if ratio_gt(abs_val, max) {
                max = abs_val;
            }
        }

        max
    }

    /// Get statistics about resonance distribution
    pub fn stats(&self) -> Result<ResonanceStats> {
        let values = self.values.read();
        let mut sum = AtlasRatio::new_raw(0, 1);
        for v in values.iter() {
            sum = res_accum(sum, *v)?;
        }

        let max = {
            let mut m = AtlasRatio::new_raw(0, 1);
            for v in values.iter() {
                let abs_v = ratio_abs(*v);
                if ratio_gt(abs_v, m) {
                    m = abs_v;
                }
            }
            m
        };

        let non_zero_count = values.iter().filter(|v| v.numer != 0).count();

        Ok(ResonanceStats {
            sum,
            max_abs: max,
            non_zero_classes: non_zero_count,
        })
    }
}

impl Default for ResonanceAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about resonance accumulator state
#[derive(Debug, Clone, Copy)]
pub struct ResonanceStats {
    /// Sum of all resonances (should be zero for unity-neutral)
    pub sum: AtlasRatio,

    /// Maximum absolute resonance value
    pub max_abs: AtlasRatio,

    /// Number of classes with non-zero resonance
    pub non_zero_classes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_integer::Integer;
    use proptest::prelude::*;

    fn canonical_ratio_strategy() -> impl Strategy<Value = AtlasRatio> {
        (any::<i64>(), 1i64..=65_535i64).prop_map(|(numer, denom)| {
            let mut n = numer;
            let mut d = denom.abs();
            if n == 0 {
                return AtlasRatio::new_raw(0, 1);
            }

            let gcd = n.abs().gcd(&d);
            n /= gcd;
            d /= gcd;
            if d < 0 {
                n = -n;
                d = -d;
            }
            AtlasRatio::new_raw(n, d)
        })
    }

    #[test]
    fn test_resonance_accumulator_new() {
        let accum = ResonanceAccumulator::new();

        for class in 0..96 {
            assert_eq!(accum.get(class), AtlasRatio::new_raw(0, 1));
        }
    }

    #[test]
    fn test_resonance_add() {
        let accum = ResonanceAccumulator::new();

        let delta = AtlasRatio::new_raw(3, 4);
        accum.add(10, delta).unwrap();

        assert_eq!(accum.get(10), delta);
        assert_eq!(accum.get(11), AtlasRatio::new_raw(0, 1));
    }

    #[test]
    fn test_resonance_add_multiple() {
        let accum = ResonanceAccumulator::new();

        accum.add(10, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(10, AtlasRatio::new_raw(1, 4)).unwrap();

        // 1/2 + 1/4 = 3/4
        assert_eq!(accum.get(10), AtlasRatio::new_raw(3, 4));
    }

    #[test]
    fn test_resonance_reset() {
        let accum = ResonanceAccumulator::new();

        accum.add(10, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(20, AtlasRatio::new_raw(2, 3)).unwrap();

        accum.reset();

        for class in 0..96 {
            assert_eq!(accum.get(class), AtlasRatio::new_raw(0, 1));
        }
    }

    #[test]
    fn test_unity_neutrality_balanced() {
        let accum = ResonanceAccumulator::new();

        // Add balanced deltas: +1/2 and -1/2
        accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(1, AtlasRatio::new_raw(-1, 2)).unwrap();

        assert!(accum.check_unity_neutral().is_ok());
    }

    #[test]
    fn test_unity_neutrality_imbalanced() {
        let accum = ResonanceAccumulator::new();

        // Add unbalanced delta
        accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();

        assert!(accum.check_unity_neutral().is_err());
    }

    #[test]
    fn test_check_neutrality_static() {
        let mut deltas = [AtlasRatio::new_raw(0, 1); 96];

        // Balanced
        deltas[0] = AtlasRatio::new_raw(1, 3);
        deltas[1] = AtlasRatio::new_raw(-1, 3);

        assert!(ResonanceAccumulator::check_neutrality(&deltas).is_ok());

        // Imbalanced
        deltas[2] = AtlasRatio::new_raw(1, 5);
        assert!(ResonanceAccumulator::check_neutrality(&deltas).is_err());
    }

    #[test]
    fn test_resonance_sum() {
        let accum = ResonanceAccumulator::new();

        accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(1, AtlasRatio::new_raw(1, 3)).unwrap();
        accum.add(2, AtlasRatio::new_raw(1, 6)).unwrap();

        // 1/2 + 1/3 + 1/6 = 3/6 + 2/6 + 1/6 = 1
        let sum = accum.sum().unwrap();
        assert_eq!(sum, AtlasRatio::new_raw(1, 1));
    }

    #[test]
    fn test_max_abs() {
        let accum = ResonanceAccumulator::new();

        accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(1, AtlasRatio::new_raw(-3, 4)).unwrap(); // Larger absolute value
        accum.add(2, AtlasRatio::new_raw(1, 8)).unwrap();

        let max = accum.max_abs();
        assert_eq!(max, AtlasRatio::new_raw(3, 4));
    }

    #[test]
    fn test_stats() {
        let accum = ResonanceAccumulator::new();

        accum.add(0, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(1, AtlasRatio::new_raw(-1, 2)).unwrap();
        accum.add(5, AtlasRatio::new_raw(2, 3)).unwrap();
        accum.add(6, AtlasRatio::new_raw(-2, 3)).unwrap();

        let stats = accum.stats().unwrap();

        assert_eq!(stats.sum, AtlasRatio::new_raw(0, 1));
        assert_eq!(stats.max_abs, AtlasRatio::new_raw(2, 3));
        assert_eq!(stats.non_zero_classes, 4);
    }

    #[test]
    fn test_snapshot() {
        let accum = ResonanceAccumulator::new();

        accum.add(10, AtlasRatio::new_raw(1, 2)).unwrap();
        accum.add(20, AtlasRatio::new_raw(2, 3)).unwrap();

        let snapshot = accum.snapshot();

        assert_eq!(snapshot[10], AtlasRatio::new_raw(1, 2));
        assert_eq!(snapshot[20], AtlasRatio::new_raw(2, 3));
        assert_eq!(snapshot[0], AtlasRatio::new_raw(0, 1));
    }

    #[test]
    fn test_add_deltas() {
        let accum = ResonanceAccumulator::new();
        let mut deltas = [AtlasRatio::new_raw(0, 1); 96];

        deltas[0] = AtlasRatio::new_raw(1, 4);
        deltas[1] = AtlasRatio::new_raw(-1, 4);

        accum.add_deltas(&deltas).unwrap();

        assert_eq!(accum.get(0), AtlasRatio::new_raw(1, 4));
        assert_eq!(accum.get(1), AtlasRatio::new_raw(-1, 4));
    }

    #[test]
    fn add_rejects_zero_denominator() {
        let accum = ResonanceAccumulator::new();
        let err = accum
            .add(0, AtlasRatio::new_raw(1, 0))
            .expect_err("zero denominator must be rejected");
        assert!(matches!(err, AtlasError::ResonanceZeroDenominator));
    }

    #[test]
    fn add_rejects_non_canonical_ratio() {
        let accum = ResonanceAccumulator::new();
        let err = accum
            .add(0, AtlasRatio::new_raw(2, 4))
            .expect_err("ratio must be reduced");
        assert!(matches!(err, AtlasError::ResonanceNonCanonical { numer: 2, denom: 4 }));
    }

    proptest! {
        #[test]
        fn prop_resonance_add_preserves_canonical(
            class in 0u8..96u8,
            base in canonical_ratio_strategy(),
            delta in canonical_ratio_strategy(),
        ) {
            let accum = ResonanceAccumulator::new();
            accum.set(class, base).unwrap();

            match accum.add(class, delta) {
                Ok(()) => {
                    let result = accum.get(class);
                    prop_assert!(result.denom > 0);
                    if result.numer == 0 {
                        prop_assert_eq!(result.denom, 1);
                    } else {
                        prop_assert_eq!(result.numer.abs().gcd(&result.denom), 1);
                    }
                }
                Err(AtlasError::ResonanceOverflow { .. }) => {
                    // Overflow is allowed; accumulator remains unchanged and canonical.
                    let result = accum.get(class);
                    prop_assert!(result.denom > 0);
                    prop_assert_eq!(result, base);
                }
                Err(err) => panic!("unexpected error: {err}"),
            }
        }
    }
}
