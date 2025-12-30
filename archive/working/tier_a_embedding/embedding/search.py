"""
Embedding search algorithm.

This module implements the backtracking search for finding Tier-A embeddings.
"""
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Root


@dataclass
class EmbeddingConstraints:
    """Simple constraint configuration for embedding search."""
    max_solutions: int = 1
    target_signs: Optional[int] = None
    required_mapping: Optional[Dict[int, int]] = None
    verbose: bool = False


class EmbeddingSearch:
    """Backtracking search for Tier-A embeddings."""

    def __init__(self, atlas_graph, e8_system):
        """
        Initialize the embedding search.

        Args:
            atlas_graph: AtlasGraph instance
            e8_system: E8RootSystem instance
        """
        self.atlas = atlas_graph
        self.e8 = e8_system

        # Get unity indices from atlas
        from atlas import derive_unity_indices
        self.unity_indices = derive_unity_indices(atlas_graph.labels)

        # Build E8 adjacency matrix
        from e8.geometry import compute_adjacency_graph
        self.e8_edges, self.e8_adjacency = compute_adjacency_graph(e8_system.roots)

        # Initialize search state
        self.mapping = [-1] * len(self.atlas.labels)
        self.used_roots = [False] * len(self.e8.roots)
        self.solutions = []
        self.available_at = [set(range(len(self.e8.roots))) for _ in range(len(self.atlas.labels))]

    def search(self, constraints: EmbeddingConstraints) -> List[List[int]]:
        """
        Run the backtracking search with given constraints.

        Args:
            constraints: Search constraints

        Returns:
            List of found embeddings (each as a mapping list)
        """
        self.constraints = constraints
        self.solutions = []
        self.mapping = [-1] * len(self.atlas.labels)
        self.used_roots = [False] * len(self.e8.roots)

        # Apply any required mappings
        if constraints.required_mapping:
            for v, r in constraints.required_mapping.items():
                self.mapping[v] = r
                self.used_roots[r] = True

        # Start search
        if constraints.verbose:
            print(f"Starting search (max solutions: {constraints.max_solutions})")

        self._backtrack(0)

        if constraints.verbose:
            print(f"Search complete. Found {len(self.solutions)} solutions.")

        return self.solutions

    def _backtrack(self, pos: int) -> None:
        """
        Recursive backtracking search.

        Args:
            pos: Current vertex position
        """
        # Check if we've found enough solutions
        if len(self.solutions) >= self.constraints.max_solutions:
            return

        # Check if we've mapped all vertices
        if pos == len(self.atlas.labels):
            # Check unity constraint
            if not self._check_unity_constraint():
                return

            # Check sign class constraint
            if self.constraints.target_signs:
                sign_count = self._count_sign_classes()
                if sign_count != self.constraints.target_signs:
                    return

            # Found a valid embedding
            self.solutions.append(self.mapping.copy())
            if self.constraints.verbose:
                print(f"  Found embedding #{len(self.solutions)}")
            return

        # Skip if already assigned (from required_mapping)
        if self.mapping[pos] != -1:
            self._backtrack(pos + 1)
            return

        # Get candidates for this vertex
        candidates = self._get_candidates(pos)

        # Try each candidate
        for root_idx in candidates:
            if self._is_valid(pos, root_idx):
                # Make assignment
                self.mapping[pos] = root_idx
                self.used_roots[root_idx] = True

                # Also assign mirror vertex
                mirror_pos = self.atlas.tau[pos]
                neg_root = self.e8.negation_table[root_idx]
                self.mapping[mirror_pos] = neg_root
                self.used_roots[neg_root] = True

                # Recurse
                self._backtrack(pos + 1)

                # Undo assignment
                self.mapping[pos] = -1
                self.used_roots[root_idx] = False
                self.mapping[mirror_pos] = -1
                self.used_roots[neg_root] = False

    def _get_candidates(self, vertex: int) -> List[int]:
        """Get candidate roots for a vertex."""
        candidates = []
        for r in range(len(self.e8.roots)):
            if not self.used_roots[r] and self._is_valid(vertex, r):
                candidates.append(r)
        return candidates

    def _is_valid(self, vertex: int, root: int) -> bool:
        """Check if assigning root to vertex is valid."""
        if self.used_roots[root]:
            return False

        # Check mirror constraint
        mirror = self.atlas.tau[vertex]
        neg_root = self.e8.negation_table[root]
        if self.used_roots[neg_root]:
            return False

        # Check edges to already mapped vertices
        for neighbor in self.atlas.adjacency[vertex]:
            if self.mapping[neighbor] != -1:
                neighbor_root = self.mapping[neighbor]
                if neighbor_root not in self.e8_adjacency[root]:
                    return False

        return True

    def _check_mirror_constraint(self, vertex: int, root: int) -> bool:
        """Check if mirror constraint is satisfied."""
        mirror = self.atlas.tau[vertex]
        if self.mapping[mirror] != -1:
            expected = self.e8.negation_table[root]
            return self.mapping[mirror] == expected
        return True

    def _check_unity_constraint(self) -> bool:
        """Check if unity constraint is satisfied."""
        if not self.unity_indices:
            return True

        # Sum unity vertices
        sum_vec = [0] * 8
        for idx in self.unity_indices:
            root = self.e8.roots[self.mapping[idx]]
            for i in range(8):
                sum_vec[i] += root[i]

        # Check if sum is zero
        return all(x == 0 for x in sum_vec)

    def _count_sign_classes(self) -> int:
        """Count the number of sign classes used."""
        sign_classes = set()
        for root_idx in self.mapping:
            if root_idx >= 0:
                neg_idx = self.e8.negation_table[root_idx]
                sign_classes.add(min(root_idx, neg_idx))
        return len(sign_classes)


def find_embedding(
    atlas_graph,
    e8_system,
    unity_indices: Optional[List[int]] = None,
    max_solutions: int = 1
) -> List[List[int]]:
    """
    Find Tier-A embeddings.

    Args:
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance
        unity_indices: Optional unity vertex indices (not used - derived automatically)
        max_solutions: Maximum number of solutions to find

    Returns:
        List of found embeddings (as mapping lists)
    """
    search = EmbeddingSearch(atlas_graph, e8_system)
    constraints = EmbeddingConstraints(max_solutions=max_solutions)
    return search.search(constraints)


def tierA_search(atlas_graph, e8_system, max_solutions: int = 1) -> List[List[int]]:
    """
    Legacy interface for Tier-A search.

    Args:
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance
        max_solutions: Maximum number of solutions

    Returns:
        List of found embeddings
    """
    return find_embedding(atlas_graph, e8_system, None, max_solutions)