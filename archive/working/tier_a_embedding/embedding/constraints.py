"""
Embedding constraint handling.

This module manages constraints for the embedding search, including unity constraints,
degree filtering, and mirror pairing.
"""
from fractions import Fraction
from typing import List, Set, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Root

class UnityConstraint:
    """Manages unity constraint for embedding search."""

    def __init__(self, unity_indices: List[int], roots: List[Root]):
        """
        Initialize unity constraint.

        Args:
            unity_indices: Indices of unity vertices in atlas
            roots: List of E8 roots
        """
        self.unity_indices = set(unity_indices) if unity_indices else set()
        self.roots = roots
        self.dimension = len(roots[0]) if roots else 8
        self.unity_sum = [Fraction(0, 1)] * self.dimension
        self.placed_count = 0

    def add_unity_vertex(self, root_idx: int) -> None:
        """
        Add a unity vertex's root to the running sum.

        Args:
            root_idx: Index of the root to add
        """
        root = self.roots[root_idx]
        for k in range(self.dimension):
            self.unity_sum[k] += root[k]
        self.placed_count += 1

    def remove_unity_vertex(self, root_idx: int) -> None:
        """
        Remove a unity vertex's root from the running sum.

        Args:
            root_idx: Index of the root to remove
        """
        root = self.roots[root_idx]
        for k in range(self.dimension):
            self.unity_sum[k] -= root[k]
        self.placed_count -= 1

    def is_satisfied(self) -> bool:
        """
        Check if unity constraint is satisfied (sum = 0).

        Returns:
            True if all unity vertices have been placed and sum to zero
        """
        if self.placed_count != len(self.unity_indices):
            return False
        return all(x == 0 for x in self.unity_sum)

    def is_feasible(self) -> bool:
        """
        Check if the current partial sum could lead to a valid solution.

        This is a heuristic check - we can't definitively rule out solutions
        in general, but we can check for obvious impossibilities.

        Returns:
            True if the constraint might still be satisfiable
        """
        # If all unity vertices are placed, check exact satisfaction
        if self.placed_count == len(self.unity_indices):
            return self.is_satisfied()

        # Otherwise, we can't easily determine feasibility
        # (would need to check if remaining vertices can sum to negative of current sum)
        return True

class DegreeConstraint:
    """Manages degree constraints for embedding."""

    def __init__(self, atlas_degrees: List[int], e8_degrees: List[int]):
        """
        Initialize degree constraint.

        Args:
            atlas_degrees: Degree of each atlas vertex
            e8_degrees: Degree of each E8 root
        """
        self.atlas_degrees = atlas_degrees
        self.e8_degrees = e8_degrees

    def is_compatible(self, atlas_idx: int, root_idx: int) -> bool:
        """
        Check if an atlas vertex can map to a root based on degree.

        The root must have at least as many neighbors as the atlas vertex.

        Args:
            atlas_idx: Atlas vertex index
            root_idx: Root index

        Returns:
            True if degrees are compatible
        """
        return self.e8_degrees[root_idx] >= self.atlas_degrees[atlas_idx]

    def filter_candidates(self, atlas_idx: int, candidate_roots: List[int]) -> List[int]:
        """
        Filter root candidates based on degree constraint.

        Args:
            atlas_idx: Atlas vertex index
            candidate_roots: List of candidate root indices

        Returns:
            Filtered list of compatible roots
        """
        needed_degree = self.atlas_degrees[atlas_idx]
        return [r for r in candidate_roots if self.e8_degrees[r] >= needed_degree]

class MirrorConstraint:
    """Manages mirror pairing constraint."""

    def __init__(self, tau: List[int], negation_table: List[int]):
        """
        Initialize mirror constraint.

        Args:
            tau: Atlas mirror pairing
            negation_table: E8 root negation table
        """
        self.tau = tau
        self.negation_table = negation_table

    def get_forced_image(self, atlas_idx: int, mapping: List[int]) -> Optional[int]:
        """
        Get the forced root image if the mirror pair is already mapped.

        Args:
            atlas_idx: Atlas vertex index
            mapping: Current partial mapping

        Returns:
            Forced root index, or None if not forced
        """
        tau_mate = self.tau[atlas_idx]
        if mapping[tau_mate] != -1:
            return self.negation_table[mapping[tau_mate]]
        return None

    def is_available_pair(self, root_idx: int, used: List[bool]) -> bool:
        """
        Check if a root and its negation are both available.

        Args:
            root_idx: Root index
            used: Boolean array of used roots

        Returns:
            True if both root and its negation are available
        """
        neg_idx = self.negation_table[root_idx]
        return not used[root_idx] and not used[neg_idx]

class EdgeConstraint:
    """Manages edge preservation constraint."""

    def __init__(self, atlas_adj: List[Set[int]], e8_adj: List[Set[int]]):
        """
        Initialize edge constraint.

        Args:
            atlas_adj: Atlas adjacency list
            e8_adj: E8 adjacency list
        """
        self.atlas_adj = atlas_adj
        self.e8_adj = e8_adj

    def is_consistent(self, atlas_idx: int, root_idx: int, mapping: List[int]) -> bool:
        """
        Check if mapping a vertex to a root is consistent with already mapped neighbors.

        Args:
            atlas_idx: Atlas vertex index
            root_idx: Candidate root index
            mapping: Current partial mapping

        Returns:
            True if the mapping preserves all edges to mapped neighbors
        """
        for neighbor in self.atlas_adj[atlas_idx]:
            neighbor_root = mapping[neighbor]
            if neighbor_root != -1:
                if neighbor_root not in self.e8_adj[root_idx]:
                    return False
        return True

    def count_satisfied_edges(self, mapping: List[int]) -> int:
        """
        Count how many atlas edges are satisfied by the current mapping.

        Args:
            mapping: Current mapping

        Returns:
            Number of satisfied edges
        """
        count = 0
        for i, neighbors in enumerate(self.atlas_adj):
            if mapping[i] == -1:
                continue
            ri = mapping[i]
            for j in neighbors:
                if mapping[j] != -1:
                    rj = mapping[j]
                    if rj in self.e8_adj[ri]:
                        count += 1
        # Each edge is counted twice, so divide by 2
        return count // 2

class EmbeddingConstraints:
    """Aggregates all embedding constraints."""

    def __init__(
        self,
        unity_indices: Optional[List[int]],
        atlas_degrees: List[int],
        e8_degrees: List[int],
        tau: List[int],
        negation_table: List[int],
        atlas_adj: List[Set[int]],
        e8_adj: List[Set[int]],
        roots: List[Root]
    ):
        """
        Initialize all constraints.

        Args:
            unity_indices: Unity vertex indices
            atlas_degrees: Atlas vertex degrees
            e8_degrees: E8 root degrees
            tau: Atlas mirror pairing
            negation_table: E8 negation table
            atlas_adj: Atlas adjacency
            e8_adj: E8 adjacency
            roots: E8 roots
        """
        self.unity = UnityConstraint(unity_indices or [], roots) if unity_indices else None
        self.degree = DegreeConstraint(atlas_degrees, e8_degrees)
        self.mirror = MirrorConstraint(tau, negation_table)
        self.edge = EdgeConstraint(atlas_adj, e8_adj)

    def get_candidates(
        self,
        atlas_idx: int,
        mapping: List[int],
        used: List[bool]
    ) -> List[int]:
        """
        Get all valid candidate roots for an atlas vertex.

        Args:
            atlas_idx: Atlas vertex index
            mapping: Current partial mapping
            used: Boolean array of used roots

        Returns:
            List of valid root indices
        """
        # Check if mirror constraint forces a specific root
        forced = self.mirror.get_forced_image(atlas_idx, mapping)
        if forced is not None:
            if used[forced]:
                return []  # Cannot reuse root
            # Check other constraints
            if not self.degree.is_compatible(atlas_idx, forced):
                return []
            if not self.edge.is_consistent(atlas_idx, forced, mapping):
                return []
            return [forced]

        # Otherwise, find all compatible roots
        candidates = []
        for root_idx in range(len(used)):
            # Check availability (including mirror pair)
            if not self.mirror.is_available_pair(root_idx, used):
                continue

            # Check degree constraint
            if not self.degree.is_compatible(atlas_idx, root_idx):
                continue

            # Check edge consistency
            if not self.edge.is_consistent(atlas_idx, root_idx, mapping):
                continue

            candidates.append(root_idx)

        return candidates