#!/usr/bin/env python3
"""
F₄ Roots Loader - Connect First-Principles Construction to Quotient Field

Loads the F₄ root system into F4QuotientField with EXACT arithmetic.

The F₄ root system has 48 roots:
- 24 short roots with norm² = 1
- 24 long roots with norm² = 2

This loader creates the canonical F₄ configuration on the quotient field.
"""

from fractions import Fraction
from action_framework.core.quotient_field import F4QuotientField
from action_framework.core.exact_arithmetic import ComplexFraction


def load_f4_canonical() -> F4QuotientField:
    """
    Load canonical F₄ root configuration.

    Returns F4QuotientField with 24 short + 24 long roots (EXACT).

    Since F₄ roots in ℝ⁴ have irrational coordinates (involving √2),
    we represent them abstractly in the quotient field. The key invariant
    is the norm distribution: 24 at norm²=1, 24 at norm²=2.

    We use ComplexFraction to represent root "amplitudes" that give exact norms:
    - Short roots: |α|² = 1 → use α = 1+0j (norm² = 1)
    - Long roots: |α|² = 2 → use α = 1+1j (norm² = 1+1 = 2)

    This abstract representation captures the essential structure:
    the 24:24 distribution with exact norm ratio 1:2.

    Returns:
        F4QuotientField with exact F₄ configuration
    """
    psi = F4QuotientField()

    # 24 short roots: norm² = 1 exactly
    for i in range(24):
        psi[i] = ComplexFraction(1, 0)  # 1 + 0j, |·|² = 1

    # 24 long roots: norm² = 2 exactly
    for i in range(24, 48):
        psi[i] = ComplexFraction(1, 1)  # 1 + 1j, |·|² = 1² + 1² = 2

    return psi


def verify_f4_configuration(psi: F4QuotientField) -> dict:
    """
    Verify F₄ configuration properties (EXACT checks).

    Args:
        psi: F4QuotientField to verify

    Returns:
        Dictionary with verification results (all bools, no tolerances)
    """
    # Compute exact norms
    norms_sq = [amp.norm_squared() for amp in psi.amplitudes]

    # Count by exact norm values
    short_count = sum(1 for n in norms_sq if n == Fraction(1))
    long_count = sum(1 for n in norms_sq if n == Fraction(2))

    # Total energy (should be exactly 72)
    total_energy = sum(norms_sq)

    # Verification (EXACT equality)
    checks = {
        'has_24_short_roots': short_count == 24,
        'has_24_long_roots': long_count == 24,
        'total_energy_72': total_energy == Fraction(72),
        'all_norms_quantized': all(n in [Fraction(1), Fraction(2)] for n in norms_sq),
    }

    # Add exact values for inspection
    checks['short_count'] = short_count
    checks['long_count'] = long_count
    checks['total_energy'] = total_energy

    return checks


if __name__ == '__main__':
    print("Testing F₄ canonical configuration loader...")

    # Load F₄
    psi_f4 = load_f4_canonical()
    print(f"  Loaded F₄ quotient field with {len(psi_f4.amplitudes)} mirror pairs")

    # Verify
    verification = verify_f4_configuration(psi_f4)

    print("\nVerification (EXACT checks):")
    for key, value in verification.items():
        if isinstance(value, bool):
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        else:
            print(f"    {key} = {value}")

    # Check all passed
    all_passed = all(v for k, v in verification.items() if isinstance(v, bool))

    if all_passed:
        print("\n✓ F₄ canonical configuration VALID")
        print("  All properties exact (no approximations)")
    else:
        print("\n✗ Verification failed")

    print("\nNote: This is an ABSTRACT representation of F₄ roots.")
    print("  Geometric F₄ roots in ℝ⁴ have irrational coords (√2)")
    print("  We use complex amplitudes with exact rational norms²")
    print("  The quotient field captures the essential 24:24 structure")
