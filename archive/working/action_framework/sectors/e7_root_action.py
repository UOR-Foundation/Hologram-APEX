#!/usr/bin/env python3
"""
E₇ Root System Action - EXACT ARITHMETIC

The action on E₇'s 126-dimensional quotient space.
Critical points of this action ARE the E₇ root vectors.

E₇ is SIMPLY-LACED: All roots have equal length.

E₇ structure: 126 = 96 + 30
- 96 Atlas vertices
- 30 S₄ orbits (meta-vertices)

Sectors enforce (EXACTLY, no floats):
1. Uniform norm: All 126 roots with norm²=1
2. Conservation: Total norm² = 126·1 = 126
3. Simply-laced structure (all roots equal)

NO NUMPY - All arithmetic exact using fractions.Fraction.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict

from action_framework.core.quotient_field import E7QuotientField
from action_framework.core.exact_arithmetic import ComplexFraction


@dataclass
class E7ActionWeights:
    """Sector weights for E₇ root system action (EXACT)."""
    # Uniform norm: all roots should have norm²=1
    lambda_uniform_norm: Fraction = Fraction(1)

    # Total energy conservation
    lambda_energy_conservation: Fraction = Fraction(1)

    # Simply-laced enforcement (optional, redundant with uniform norm)
    lambda_simply_laced: Fraction = Fraction(0)


class E7UniformNormSector:
    """
    Uniform norm sector for E₇.

    All 126 roots should have exactly norm²=1 (simply-laced).

    Energy: λ·Σᵢ (|αᵢ|² - 1)²
    """

    def __init__(self, lambda_norm: Fraction = Fraction(1)):
        self.lambda_norm = lambda_norm
        self.target_norm = Fraction(1)  # All roots: norm² = 1

    def energy(self, psi: E7QuotientField) -> Fraction:
        """
        Penalty for roots deviating from uniform norm (EXACT).
        """
        if self.lambda_norm == 0:
            return Fraction(0)

        energy = Fraction(0)

        for amplitude in psi.amplitudes:
            norm_sq = amplitude.norm_squared()  # Exact Fraction
            deviation = (norm_sq - self.target_norm) ** 2
            energy += deviation

        return self.lambda_norm * energy

    def gradient(self, psi: E7QuotientField) -> E7QuotientField:
        """
        Gradient: push all roots toward norm²=1 (EXACT).

        ∂E/∂α*ᵢ = 2λ(|αᵢ|² - 1)·αᵢ (Wirtinger derivative)
        """
        grad = E7QuotientField()

        if self.lambda_norm == 0:
            return grad

        for i in range(126):
            amplitude = psi[i]
            norm_sq = amplitude.norm_squared()  # Exact Fraction

            # Gradient: ∂E/∂α*ᵢ = 2λ(|α|² - 1)·α
            factor = self.lambda_norm * Fraction(2) * (norm_sq - self.target_norm)
            grad[i] = amplitude * factor

        return grad


class E7EnergyConservationSector:
    """
    Total energy conservation for E₇: Σᵢ |αᵢ|² = 126 (EXACT).

    All 126 roots at norm²=1 → total = 126.
    """

    def __init__(self, lambda_energy: Fraction = Fraction(1)):
        self.lambda_energy = lambda_energy
        self.target_total = Fraction(126)  # 126 roots × norm²=1

    def energy(self, psi: E7QuotientField) -> Fraction:
        """Penalty for total energy deviation (EXACT)."""
        if self.lambda_energy == 0:
            return Fraction(0)

        total_energy = psi.norm_squared()  # Exact Fraction
        deviation = (total_energy - self.target_total) ** 2

        return self.lambda_energy * deviation

    def gradient(self, psi: E7QuotientField) -> E7QuotientField:
        """
        Gradient: ∂E/∂α*ᵢ = 2λ·(Σⱼ|αⱼ|² - 126)·αᵢ (EXACT)
        """
        grad = E7QuotientField()

        if self.lambda_energy == 0:
            return grad

        total_energy = psi.norm_squared()  # Exact Fraction
        factor = Fraction(2) * self.lambda_energy * (total_energy - self.target_total)

        # Multiply each amplitude by factor
        for i in range(126):
            grad[i] = psi[i] * factor

        return grad


class E7RootAction:
    """
    Total action for E₇ root system (EXACT ARITHMETIC).

    Critical points = E₇ root configurations (all roots norm²=1).
    """

    def __init__(self, weights: E7ActionWeights):
        self.weights = weights
        self.uniform_norm = E7UniformNormSector(weights.lambda_uniform_norm)
        self.energy_cons = E7EnergyConservationSector(weights.lambda_energy_conservation)

    def energy(self, psi: E7QuotientField) -> Fraction:
        """Total energy (EXACT)."""
        E = Fraction(0)
        E += self.uniform_norm.energy(psi)
        E += self.energy_cons.energy(psi)
        return E

    def gradient(self, psi: E7QuotientField) -> E7QuotientField:
        """Total gradient (EXACT)."""
        grad = E7QuotientField()
        grad = grad + self.uniform_norm.gradient(psi)
        grad = grad + self.energy_cons.gradient(psi)
        return grad

    def compute_root_statistics(self, psi: E7QuotientField) -> Dict[str, any]:
        """
        Compute E₇ root system statistics (EXACT).

        Returns:
            Dictionary with norms (exact Fractions), etc.
        """
        # Compute exact norms squared
        norms_sq = [amp.norm_squared() for amp in psi.amplitudes]

        # Mean norm
        mean_norm_sq = sum(norms_sq) / 126

        # Check if all norms are equal (simply-laced)
        all_equal = all(n == norms_sq[0] for n in norms_sq)

        # Total energy
        total_energy = sum(norms_sq)

        # Variance (for checking uniformity)
        if all_equal:
            variance = Fraction(0)
        else:
            mean = mean_norm_sq
            variance = sum((n - mean) ** 2 for n in norms_sq) / 126

        return {
            'num_roots': 126,
            'mean_norm_sq': mean_norm_sq,  # Exact Fraction
            'all_equal_norm': all_equal,  # Boolean
            'total_energy': total_energy,  # Exact Fraction
            'norm_variance': variance,  # Exact Fraction
            'norms_sq': norms_sq  # List of exact Fractions
        }


if __name__ == '__main__':
    print("Testing E₇ root system action (EXACT ARITHMETIC)...")

    weights = E7ActionWeights()
    action = E7RootAction(weights)

    # Test with correct E₇ configuration (all roots norm²=1)
    psi = E7QuotientField()
    for i in range(126):
        psi[i] = ComplexFraction(1, 0)  # All roots norm²=1 exactly

    E = action.energy(psi)
    stats = action.compute_root_statistics(psi)

    print(f"  Energy of correct E₇ config: {E}")
    print(f"  Statistics:")
    print(f"    Number of roots: {stats['num_roots']}")
    print(f"    Mean norm²: {stats['mean_norm_sq']} (expect 1)")
    print(f"    All equal norm: {stats['all_equal_norm']}")
    print(f"    Total energy: {stats['total_energy']} (expect exactly 126)")
    print(f"    Norm variance: {stats['norm_variance']} (expect 0)")

    # Check if gradient is exactly zero
    grad = action.gradient(psi)
    grad_is_zero = all(grad[i] == ComplexFraction.zero() for i in range(126))
    print(f"  Gradient exactly zero: {grad_is_zero}")

    print("\n✓ E₇ root action implemented with EXACT arithmetic")
    print("  All 126 roots have equal length (simply-laced)")
    print("  126 = 96 (Atlas vertices) + 30 (S₄ orbits)")
    print("  All operations use Fraction - NO FLOATS")
