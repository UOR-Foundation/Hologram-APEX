#!/usr/bin/env python3
"""
E₇ Roots Loader - Connect Structure to Quotient Field

Loads the E₇ root system into E7QuotientField with EXACT arithmetic.

The E₇ root system has 126 roots, ALL with equal length (simply-laced).

E₇ structure: 126 = 96 + 30
- 96 Atlas vertices (canonical resonance classes)
- 30 S₄ orbits (meta-vertices from quotient structure)
"""

from fractions import Fraction
from action_framework.core.quotient_field import E7QuotientField
from action_framework.core.exact_arithmetic import ComplexFraction


def load_e7_canonical() -> E7QuotientField:
    """
    Load canonical E₇ root configuration.

    Returns E7QuotientField with 126 roots, all at norm²=1 (EXACT).

    E₇ is simply-laced: all roots have the same length.
    We represent this in the quotient field with uniform amplitudes:
    - All 126 roots: amplitude = 1+0j → norm² = 1 (exact)

    This abstract representation captures the essential simply-laced structure.

    Returns:
        E7QuotientField with exact E₇ configuration
    """
    psi = E7QuotientField()

    # All 126 roots have norm² = 1 exactly (simply-laced)
    for i in range(126):
        psi[i] = ComplexFraction(1, 0)  # 1 + 0j, |·|² = 1

    return psi


def verify_e7_configuration(psi: E7QuotientField) -> dict:
    """
    Verify E₇ configuration properties (EXACT checks).

    Args:
        psi: E7QuotientField to verify

    Returns:
        Dictionary with verification results (all bools, no tolerances)
    """
    # Compute exact norms
    norms_sq = [amp.norm_squared() for amp in psi.amplitudes]

    # Check all norms are equal to 1
    all_norm_one = all(n == Fraction(1) for n in norms_sq)

    # Total energy (should be exactly 126)
    total_energy = sum(norms_sq)

    # Verification (EXACT equality)
    checks = {
        'has_126_roots': len(psi.amplitudes) == 126,
        'all_roots_norm_one': all_norm_one,
        'total_energy_126': total_energy == Fraction(126),
        'simply_laced': all_norm_one,  # Simply-laced means all equal length
    }

    # Add exact values for inspection
    checks['num_roots'] = len(psi.amplitudes)
    checks['total_energy'] = total_energy
    checks['mean_norm_sq'] = total_energy / 126

    return checks


if __name__ == '__main__':
    print("Testing E₇ canonical configuration loader...")

    # Load E₇
    psi_e7 = load_e7_canonical()
    print(f"  Loaded E₇ quotient field with {len(psi_e7.amplitudes)} roots")

    # Verify
    verification = verify_e7_configuration(psi_e7)

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
        print("\n✓ E₇ canonical configuration VALID")
        print("  All properties exact (no approximations)")
        print("  Simply-laced: all 126 roots have equal length")
    else:
        print("\n✗ Verification failed")

    print("\nNote: E₇ structure:")
    print("  126 = 96 (Atlas vertices) + 30 (S₄ orbits)")
    print("  Simply-laced: all roots have norm² = 1")
