#!/usr/bin/env python3
"""
Atlas-Specific Lagrangian Sectors

These sectors are derived from the Generator axioms (A0-A3) and Atlas polytope structure,
not generic differential geometry. The degrees of freedom are the 96 resonance classes.

Sectors:
- Unity/Klein (A0): Klein quartet class at unity
- Mirror (Graph): Involution μ (bit-7 flip) preservation
- Conservation (A1): Sum constraints
- Smoothness: Graph Laplacian on 96-vertex adjacency
- Budget (A3): β = 0 tracking
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple

from action_framework.core.atlas_structure import (
    AtlasField, R96_CANONICAL_BYTES, get_klein_class_index,
    ByteStructure
)


@dataclass
class AtlasSectorWeights:
    """Sector weights for Atlas-specific action."""
    # Unity sector (A0): Klein class at unity
    lambda_unity: float = 100.0

    # Mirror sector: Symmetry under bit-7 flip
    lambda_mirror: float = 10.0

    # Conservation sector (A1)
    lambda_conservation: float = 10.0

    # Graph smoothness (adjacency-based)
    kappa_graph: float = 1.0

    # Budget tracking (A3)
    track_budget: bool = True


class UnitySector:
    """
    Unity sector (Generator axiom A0).

    The Klein quartet class (index 0) should have amplitude 1+0j (unity).
    This enforces the R96 alphabet structure where certain positions
    evaluate to 1.

    Energy: λ·|ψ₀ - 1|²
    """

    def __init__(self, lambda_unity: float = 100.0):
        self.lambda_unity = lambda_unity
        self.klein_idx = get_klein_class_index()

    def energy(self, psi: AtlasField) -> float:
        """Penalty for Klein class deviation from unity."""
        if self.lambda_unity == 0:
            return 0.0

        deviation = abs(psi[self.klein_idx] - complex(1, 0))**2
        return self.lambda_unity * deviation

    def gradient(self, psi: AtlasField) -> AtlasField:
        """
        Gradient: ∂E/∂ψᵢ = 2λ·(ψ₀ - 1) for i=0, else 0
        """
        grad = AtlasField()
        if self.lambda_unity > 0:
            grad[self.klein_idx] = 2.0 * self.lambda_unity * (psi[self.klein_idx] - 1.0)
        return grad

    def project(self, psi: AtlasField) -> AtlasField:
        """Hard projection: set Klein class to exactly 1."""
        psi_proj = psi.copy()
        psi_proj[self.klein_idx] = complex(1, 0)
        return psi_proj


class MirrorSector:
    """
    Mirror involution sector.

    The involution μ is bit-7 flip: b ↦ b ⊕ 128.
    This pairs the 96 vertices into 48 mirror pairs.

    For consistency, we want ψ[μ(i)] = ψ̄[i] (complex conjugate)
    or some other mirror relationship.

    Energy: λ·Σᵢ |ψ[i] - ψ̄[μ(i)]|²
    """

    def __init__(self, lambda_mirror: float = 10.0):
        self.lambda_mirror = lambda_mirror
        self._build_mirror_pairs()

    def _build_mirror_pairs(self):
        """Build the 48 mirror pairs from bit-7 flip."""
        self.mirror_pairs: List[Tuple[int, int]] = []

        for i, byte_i in enumerate(R96_CANONICAL_BYTES):
            # Apply bit-7 flip
            byte_j = byte_i ^ 128

            # Find its canonical representative
            canonical_j = ByteStructure(byte_j).canonical_representative()

            # Find index in R96
            try:
                j = R96_CANONICAL_BYTES.index(canonical_j)
            except ValueError:
                # If not in R96, skip (shouldn't happen for canonical bytes)
                continue

            # Only add each pair once
            if i < j:
                self.mirror_pairs.append((i, j))

        assert len(self.mirror_pairs) == 48, f"Should have 48 mirror pairs, got {len(self.mirror_pairs)}"

    def energy(self, psi: AtlasField) -> float:
        """
        Mirror symmetry penalty.

        For now, enforce ψ[i] = ψ[μ(i)] (same value, not conjugate).
        This makes the quotient well-defined.
        """
        if self.lambda_mirror == 0:
            return 0.0

        violation = 0.0
        for i, j in self.mirror_pairs:
            diff = abs(psi[i] - psi[j])**2
            violation += diff

        return self.lambda_mirror * violation

    def gradient(self, psi: AtlasField) -> AtlasField:
        """
        Gradient of mirror penalty.

        ∂E/∂ψᵢ = 2λ·Σⱼ (ψᵢ - ψⱼ) where j = μ(i)
        """
        grad = AtlasField()

        if self.lambda_mirror > 0:
            for i, j in self.mirror_pairs:
                diff_i = psi[i] - psi[j]
                diff_j = psi[j] - psi[i]

                grad[i] += 2.0 * self.lambda_mirror * diff_i
                grad[j] += 2.0 * self.lambda_mirror * diff_j

        return grad


class ConservationSector:
    """
    Conservation sector (Generator axiom A1).

    Total sum of amplitudes should be conserved.
    For simplicity, enforce Σᵢ ψᵢ = 96 (one unit per class).

    Energy: λ·|Σᵢ ψᵢ - 96|²
    """

    def __init__(self, lambda_conservation: float = 10.0):
        self.lambda_conservation = lambda_conservation

    def energy(self, psi: AtlasField) -> float:
        """Conservation penalty."""
        if self.lambda_conservation == 0:
            return 0.0

        total_sum = np.sum(psi.amplitudes)
        target = complex(96, 0)  # One unit per class
        deviation = abs(total_sum - target)**2

        return self.lambda_conservation * deviation

    def gradient(self, psi: AtlasField) -> AtlasField:
        """
        Gradient: ∂E/∂ψᵢ = 2λ·(Σⱼ ψⱼ - 96) for all i
        """
        grad = AtlasField()

        if self.lambda_conservation > 0:
            total_sum = np.sum(psi.amplitudes)
            target = complex(96, 0)
            factor = 2.0 * self.lambda_conservation * (total_sum - target)

            # Gradient is same for all components
            grad.amplitudes[:] = factor

        return grad


class GraphSmoothnessSector:
    """
    Graph smoothness sector.

    Uses the Atlas polytope's 1-skeleton (edge structure) to define
    a graph Laplacian. This penalizes differences between adjacent vertices.

    Energy: (κ/2)·Σ_{i~j} |ψᵢ - ψⱼ|²

    This is the correct notion of "smoothness" for the Atlas polytope,
    not the generic torus Laplacian we used before.
    """

    def __init__(self, kappa_graph: float = 1.0):
        self.kappa_graph = kappa_graph
        self._build_adjacency()

    def _build_adjacency(self):
        """
        Build adjacency matrix for the 96-vertex Atlas polytope.

        Adjacency is determined by Hamming-1 flips that stay within
        the canonical set after canonicalization.
        """
        self.edges: List[Tuple[int, int]] = []

        for i, byte_i in enumerate(R96_CANONICAL_BYTES):
            # Try flipping each bit
            for bit_pos in range(8):
                byte_j = byte_i ^ (1 << bit_pos)

                # Canonicalize
                canonical_j = ByteStructure(byte_j).canonical_representative()

                # Find in R96
                try:
                    j = R96_CANONICAL_BYTES.index(canonical_j)
                except ValueError:
                    continue  # Not adjacent within R96

                # Add edge (once)
                if i < j:
                    self.edges.append((i, j))

        print(f"  Atlas graph: 96 vertices, {len(self.edges)} edges")

    def energy(self, psi: AtlasField) -> float:
        """Graph Dirichlet energy."""
        if self.kappa_graph == 0:
            return 0.0

        energy = 0.0
        for i, j in self.edges:
            diff = abs(psi[i] - psi[j])**2
            energy += diff

        return (self.kappa_graph / 2.0) * energy

    def gradient(self, psi: AtlasField) -> AtlasField:
        """
        Graph Laplacian gradient.

        ∂E/∂ψᵢ = κ·Σⱼ~ᵢ (ψᵢ - ψⱼ) = κ·(deg(i)·ψᵢ - Σⱼ~ᵢ ψⱼ)
        """
        grad = AtlasField()

        if self.kappa_graph > 0:
            degree = np.zeros(96)
            neighbor_sum = np.zeros(96, dtype=complex)

            for i, j in self.edges:
                degree[i] += 1
                degree[j] += 1
                neighbor_sum[i] += psi[j]
                neighbor_sum[j] += psi[i]

            # Laplacian: deg(i)·ψᵢ - Σⱼ ψⱼ
            grad.amplitudes = self.kappa_graph * (degree * psi.amplitudes - neighbor_sum)

        return grad


class AtlasAction:
    """Total action for Atlas 96-vertex system."""

    def __init__(self, weights: AtlasSectorWeights):
        self.weights = weights
        self.unity = UnitySector(weights.lambda_unity)
        self.mirror = MirrorSector(weights.lambda_mirror)
        self.conservation = ConservationSector(weights.lambda_conservation)
        self.smoothness = GraphSmoothnessSector(weights.kappa_graph)

        # Budget tracking
        self.budget = 0.0

    def energy(self, psi: AtlasField) -> float:
        """Total energy."""
        E = 0.0

        if self.weights.lambda_unity > 0:
            E += self.unity.energy(psi)

        if self.weights.lambda_mirror > 0:
            E += self.mirror.energy(psi)

        if self.weights.lambda_conservation > 0:
            E += self.conservation.energy(psi)

        if self.weights.kappa_graph > 0:
            E += self.smoothness.energy(psi)

        return E

    def gradient(self, psi: AtlasField) -> AtlasField:
        """Total gradient."""
        grad = AtlasField()

        if self.weights.lambda_unity > 0:
            grad = grad + self.unity.gradient(psi)

        if self.weights.lambda_mirror > 0:
            grad = grad + self.mirror.gradient(psi)

        if self.weights.lambda_conservation > 0:
            grad = grad + self.conservation.gradient(psi)

        if self.weights.kappa_graph > 0:
            grad = grad + self.smoothness.gradient(psi)

        return grad

    def compute_budget(self, psi_initial: AtlasField, psi_final: AtlasField) -> float:
        """
        Compute budget β for a transformation.

        For now, simple: β = violation of all constraints.
        """
        violations = []

        # Unity violation
        klein_dev = abs(psi_final[self.unity.klein_idx] - 1.0)**2
        violations.append(klein_dev)

        # Mirror violation
        mirror_viol = 0.0
        for i, j in self.mirror.mirror_pairs:
            mirror_viol += abs(psi_final[i] - psi_final[j])**2
        violations.append(mirror_viol)

        # Conservation violation
        total_sum = abs(np.sum(psi_final.amplitudes) - 96.0)**2
        violations.append(total_sum)

        # Total budget
        beta = sum(violations)
        return beta


if __name__ == '__main__':
    print("Testing Atlas-specific Lagrangian sectors...")

    # Create action
    weights = AtlasSectorWeights()
    action = AtlasAction(weights)

    # Test field
    psi = AtlasField()
    for i in range(96):
        psi[i] = complex(1, 0)  # Start with all unity

    E = action.energy(psi)
    print(f"  Energy of unity field: {E:.6e}")

    # Compute gradient
    grad = action.gradient(psi)
    print(f"  Gradient norm: {grad.norm():.6e}")

    # Budget
    beta = action.compute_budget(psi, psi)
    print(f"  Budget β: {beta:.6e}")

    print("✓ Atlas Lagrangian sectors implemented")
