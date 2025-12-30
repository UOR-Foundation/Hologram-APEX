"""
Embedding result data structure.

This module defines the EmbeddingResult class used to store embedding search results.
"""
from dataclasses import dataclass
from typing import List, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import EmbeddingResult as BaseEmbeddingResult

@dataclass
class EmbeddingResult(BaseEmbeddingResult):
    """
    Result of an embedding search.

    Attributes:
        mapping: List mapping atlas vertex indices to root indices
        used_root: Boolean list indicating which roots are used
    """

    def get_used_root_indices(self) -> List[int]:
        """
        Get the indices of all used roots.

        Returns:
            List of root indices that are used in the mapping
        """
        return [i for i, used in enumerate(self.used_root) if used]

    def get_mapped_roots(self) -> Set[int]:
        """
        Get the set of root indices in the mapping.

        Returns:
            Set of root indices
        """
        return set(self.mapping)

    def verify_mirror_pairing(self, tau: List[int], negation_table: List[int]) -> bool:
        """
        Verify that the mapping respects mirror pairing.

        Args:
            tau: Mirror pairing for atlas vertices
            negation_table: Negation table for roots

        Returns:
            True if f(tau(v)) = -f(v) for all vertices v
        """
        for i in range(len(self.mapping)):
            ri = self.mapping[i]
            r_tau = self.mapping[tau[i]]
            if negation_table[ri] != r_tau:
                return False
        return True

    def verify_edge_preservation(
        self,
        atlas_adj: List[Set[int]],
        e8_adj: List[Set[int]]
    ) -> bool:
        """
        Verify that the mapping preserves edges.

        Args:
            atlas_adj: Atlas graph adjacency list
            e8_adj: E8 graph adjacency list

        Returns:
            True if all atlas edges map to E8 edges
        """
        for i, neighbors in enumerate(atlas_adj):
            ri = self.mapping[i]
            for j in neighbors:
                rj = self.mapping[j]
                if rj not in e8_adj[ri]:
                    return False
        return True

    def compute_unity_sum(self, unity_indices: List[int], roots: List) -> List:
        """
        Compute the sum of unity vertices' images.

        Args:
            unity_indices: Indices of unity vertices
            roots: List of root vectors

        Returns:
            Sum vector
        """
        if not unity_indices:
            return []

        # Initialize with zeros of the same type as root components
        sum_vec = [type(roots[0][0])(0)] * len(roots[0])

        for u in unity_indices:
            root = roots[self.mapping[u]]
            for k in range(len(root)):
                sum_vec[k] += root[k]

        return sum_vec

    def is_unity_balanced(self, unity_indices: List[int], roots: List) -> bool:
        """
        Check if unity vertices sum to zero.

        Args:
            unity_indices: Indices of unity vertices
            roots: List of root vectors

        Returns:
            True if unity sum is zero vector
        """
        unity_sum = self.compute_unity_sum(unity_indices, roots)
        return all(x == 0 for x in unity_sum)

    def count_integer_roots(self, roots: List) -> int:
        """
        Count how many integer roots are used in the mapping.

        Args:
            roots: List of root vectors

        Returns:
            Number of integer roots in the mapping
        """
        count = 0
        for idx in self.mapping:
            if all(x.denominator == 1 for x in roots[idx]):
                count += 1
        return count

    def get_statistics(self) -> dict:
        """
        Get statistics about the embedding.

        Returns:
            Dictionary with embedding statistics
        """
        return {
            'num_vertices': len(self.mapping),
            'num_used_roots': self.num_used_roots,
            'is_injective': self.is_injective(),
            'mapped_roots': len(self.get_mapped_roots())
        }