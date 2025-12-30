#!/usr/bin/env python3
"""
E₇ Critical Point Verification - EXACT

Verifies that the E₇ root configuration is an EXACT critical point
of the E₇ action: ∂S/∂ψ = 0 (no tolerances, no approximations).

This validates the hypothesis: the action's function signature
RECOGNIZES the E₇ structure.

E₇ is SIMPLY-LACED: all 126 roots have equal length.
E₇ structure: 126 = 96 (Atlas vertices) + 30 (S₄ orbits)
"""

from fractions import Fraction
from action_framework.core.quotient_field import E7QuotientField
from action_framework.core.exact_arithmetic import ComplexFraction
from action_framework.sectors.e7_root_action import E7RootAction, E7ActionWeights
from action_framework.loaders.e7_loader import load_e7_canonical


def verify_e7_is_critical_point(
    psi: E7QuotientField,
    action: E7RootAction,
    verbose: bool = True
) -> bool:
    """
    Verify E₇ configuration is EXACT critical point: ∂S/∂ψ = 0.

    Args:
        psi: E₇ quotient field configuration
        action: E₇ action functional
        verbose: Print detailed output

    Returns:
        True if ∂S/∂ψ = 0 exactly (all 126 components), False otherwise
    """
    if verbose:
        print("=" * 70)
        print("E₇ CRITICAL POINT VERIFICATION (EXACT)")
        print("=" * 70)

    # Compute energy (should be 0 for correct config)
    E = action.energy(psi)
    if verbose:
        print(f"\n1. Energy E = {E}")
        if E == Fraction(0):
            print("   ✓ Energy is EXACTLY zero")
        else:
            print(f"   ✗ Energy is {E}, not zero")

    # Compute gradient (should be exactly zero for all 126 components)
    grad = action.gradient(psi)
    if verbose:
        print("\n2. Gradient ∂S/∂ψ check:")

    # Check each component exactly
    zero_components = 0
    nonzero_components = []

    for i in range(126):
        if grad[i] == ComplexFraction.zero():
            zero_components += 1
        else:
            nonzero_components.append((i, grad[i]))

    if verbose:
        print(f"   Zero components: {zero_components}/126")
        if nonzero_components:
            print(f"   Non-zero components: {len(nonzero_components)}")
            # Show first few non-zero
            for i, val in nonzero_components[:5]:
                print(f"     ∂S/∂ψ[{i}] = {val}")
            if len(nonzero_components) > 5:
                print(f"     ... and {len(nonzero_components) - 5} more")

    # Exact stationarity check
    is_stationary = zero_components == 126

    if verbose:
        print("\n3. Stationarity check:")
        if is_stationary:
            print("   ✓ ∂S/∂ψ = 0 EXACTLY (all 126 components)")
            print("   ✓ E₇ configuration IS a critical point")
        else:
            print(f"   ✗ {len(nonzero_components)} components non-zero")
            print("   ✗ E₇ configuration is NOT a critical point")

    # Compute statistics
    stats = action.compute_root_statistics(psi)
    if verbose:
        print("\n4. E₇ structure verification:")
        print(f"   Number of roots: {stats['num_roots']}/126")
        print(f"   Mean norm²: {stats['mean_norm_sq']} (expect 1)")
        print(f"   All equal norm (simply-laced): {stats['all_equal_norm']}")
        print(f"   Total energy: {stats['total_energy']} (expect 126)")
        print(f"   Norm variance: {stats['norm_variance']} (expect 0)")

        # Verify structure
        structure_ok = (
            stats['num_roots'] == 126 and
            stats['mean_norm_sq'] == Fraction(1) and
            stats['all_equal_norm'] == True and
            stats['total_energy'] == Fraction(126) and
            stats['norm_variance'] == Fraction(0)
        )
        if structure_ok:
            print("   ✓ E₇ structure EXACT (simply-laced)")
        else:
            print("   ✗ E₇ structure deviates from canonical")

    if verbose:
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        if is_stationary and E == Fraction(0):
            print("✓✓✓ E₇ IS AN EXACT CRITICAL POINT ✓✓✓")
            print("\nHypothesis VALIDATED:")
            print("  - Action functional recognizes E₇ structure")
            print("  - ∂S/∂ψ = 0 EXACTLY (no tolerances)")
            print("  - Energy E = 0 EXACTLY")
            print("  - All arithmetic exact (Fraction-based)")
            print("  - Simply-laced: all 126 roots have equal length")
            print("  - Structure: 126 = 96 (Atlas) + 30 (S₄ orbits)")
        else:
            print("✗ Verification failed")
            if not is_stationary:
                print(f"  Gradient not zero: {len(nonzero_components)}/126 components non-zero")
            if E != Fraction(0):
                print(f"  Energy not zero: E = {E}")
        print("=" * 70)

    return is_stationary and E == Fraction(0)


if __name__ == '__main__':
    print("E₇ Critical Point Verification Test\n")

    # Load canonical E₇ configuration
    print("Loading E₇ canonical configuration...")
    psi_e7 = load_e7_canonical()
    print(f"  Loaded {len(psi_e7.amplitudes)} roots (simply-laced)\n")

    # Create E₇ action with default weights
    weights = E7ActionWeights(
        lambda_uniform_norm=Fraction(1),
        lambda_energy_conservation=Fraction(1),
        lambda_simply_laced=Fraction(0)  # Redundant with uniform norm
    )
    action = E7RootAction(weights)
    print(f"E₇ Action initialized with weights:")
    print(f"  λ_uniform_norm = {weights.lambda_uniform_norm}")
    print(f"  λ_energy = {weights.lambda_energy_conservation}")
    print()

    # Verify critical point
    is_critical = verify_e7_is_critical_point(psi_e7, action, verbose=True)

    # Exit code
    import sys
    sys.exit(0 if is_critical else 1)
