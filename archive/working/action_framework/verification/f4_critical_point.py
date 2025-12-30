#!/usr/bin/env python3
"""
F₄ Critical Point Verification - EXACT

Verifies that the F₄ root configuration is an EXACT critical point
of the F₄ action: ∂S/∂ψ = 0 (no tolerances, no approximations).

This validates the user's hypothesis: the action's function signature
RECOGNIZES the F₄ structure.
"""

from fractions import Fraction
from action_framework.core.quotient_field import F4QuotientField
from action_framework.core.exact_arithmetic import ComplexFraction
from action_framework.sectors.f4_root_action import F4RootAction, F4ActionWeights
from action_framework.loaders.f4_loader import load_f4_canonical


def verify_f4_is_critical_point(
    psi: F4QuotientField,
    action: F4RootAction,
    verbose: bool = True
) -> bool:
    """
    Verify F₄ configuration is EXACT critical point: ∂S/∂ψ = 0.

    Args:
        psi: F₄ quotient field configuration
        action: F₄ action functional
        verbose: Print detailed output

    Returns:
        True if ∂S/∂ψ = 0 exactly (all 48 components), False otherwise
    """
    if verbose:
        print("=" * 70)
        print("F₄ CRITICAL POINT VERIFICATION (EXACT)")
        print("=" * 70)

    # Compute energy (should be 0 for correct config)
    E = action.energy(psi)
    if verbose:
        print(f"\n1. Energy E = {E}")
        if E == Fraction(0):
            print("   ✓ Energy is EXACTLY zero")
        else:
            print(f"   ✗ Energy is {E}, not zero")

    # Compute gradient (should be exactly zero for all 48 components)
    grad = action.gradient(psi)
    if verbose:
        print("\n2. Gradient ∂S/∂ψ check:")

    # Check each component exactly
    zero_components = 0
    nonzero_components = []

    for i in range(48):
        if grad[i] == ComplexFraction.zero():
            zero_components += 1
        else:
            nonzero_components.append((i, grad[i]))

    if verbose:
        print(f"   Zero components: {zero_components}/48")
        if nonzero_components:
            print(f"   Non-zero components: {len(nonzero_components)}")
            # Show first few non-zero
            for i, val in nonzero_components[:5]:
                print(f"     ∂S/∂ψ[{i}] = {val}")
            if len(nonzero_components) > 5:
                print(f"     ... and {len(nonzero_components) - 5} more")

    # Exact stationarity check
    is_stationary = zero_components == 48

    if verbose:
        print("\n3. Stationarity check:")
        if is_stationary:
            print("   ✓ ∂S/∂ψ = 0 EXACTLY (all 48 components)")
            print("   ✓ F₄ configuration IS a critical point")
        else:
            print(f"   ✗ {len(nonzero_components)} components non-zero")
            print("   ✗ F₄ configuration is NOT a critical point")

    # Compute statistics
    stats = action.compute_root_statistics(psi)
    if verbose:
        print("\n4. F₄ structure verification:")
        print(f"   Short roots: {stats['num_short']}/24")
        print(f"   Long roots: {stats['num_long']}/24")
        print(f"   Mean short norm²: {stats['mean_short_norm_sq']} (expect 1)")
        print(f"   Mean long norm²: {stats['mean_long_norm_sq']} (expect 2)")
        print(f"   Total energy: {stats['total_energy']} (expect 72)")

        # Verify structure
        structure_ok = (
            stats['num_short'] == 24 and
            stats['num_long'] == 24 and
            stats['mean_short_norm_sq'] == Fraction(1) and
            stats['mean_long_norm_sq'] == Fraction(2) and
            stats['total_energy'] == Fraction(72)
        )
        if structure_ok:
            print("   ✓ F₄ structure EXACT")
        else:
            print("   ✗ F₄ structure deviates from canonical")

    if verbose:
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        if is_stationary and E == Fraction(0):
            print("✓✓✓ F₄ IS AN EXACT CRITICAL POINT ✓✓✓")
            print("\nHypothesis VALIDATED:")
            print("  - Action functional recognizes F₄ structure")
            print("  - ∂S/∂ψ = 0 EXACTLY (no tolerances)")
            print("  - Energy E = 0 EXACTLY")
            print("  - All arithmetic exact (Fraction-based)")
        else:
            print("✗ Verification failed")
            if not is_stationary:
                print(f"  Gradient not zero: {len(nonzero_components)}/48 components non-zero")
            if E != Fraction(0):
                print(f"  Energy not zero: E = {E}")
        print("=" * 70)

    return is_stationary and E == Fraction(0)


if __name__ == '__main__':
    print("F₄ Critical Point Verification Test\n")

    # Load canonical F₄ configuration
    print("Loading F₄ canonical configuration...")
    psi_f4 = load_f4_canonical()
    print(f"  Loaded {len(psi_f4.amplitudes)} mirror pairs\n")

    # Create F₄ action with default weights
    weights = F4ActionWeights(
        lambda_norm_quantization=Fraction(1),
        lambda_energy_conservation=Fraction(1),
        lambda_orthogonality=Fraction(0)  # Not yet implemented
    )
    action = F4RootAction(weights)
    print(f"F₄ Action initialized with weights:")
    print(f"  λ_norm = {weights.lambda_norm_quantization}")
    print(f"  λ_energy = {weights.lambda_energy_conservation}")
    print()

    # Verify critical point
    is_critical = verify_f4_is_critical_point(psi_f4, action, verbose=True)

    # Exit code
    import sys
    sys.exit(0 if is_critical else 1)
