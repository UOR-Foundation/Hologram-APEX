"""
S4 automorphism operations for canonicalization.

This module handles the S4 group action on the Atlas graph for canonicalization.
"""
from itertools import permutations
from typing import List, Tuple, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Label, S4_PERMUTABLE_BITS

class S4Automorphism:
    """Represents an S4 automorphism on the Atlas graph."""

    def __init__(self, permutation: Tuple[int, int, int, int]):
        """
        Initialize with a permutation of (e1, e2, e3, e6) indices.

        Args:
            permutation: Permutation of (0, 1, 2, 4) indices
        """
        assert len(permutation) == 4 and set(permutation) == set(S4_PERMUTABLE_BITS), \
            "Permutation must be a valid permutation of (0, 1, 2, 4)"
        self.permutation = permutation

    def apply_to_label(self, label: Label) -> Label:
        """
        Apply automorphism to a label.

        Args:
            label: Atlas label

        Returns:
            Transformed label
        """
        e1, e2, e3, d45, e6, e7 = label
        vec = [e1, e2, e3, d45, e6, e7]
        old = vec[:]

        # Apply permutation to the four permutable positions
        src_positions = S4_PERMUTABLE_BITS
        for k, dest_abs in enumerate(src_positions):
            src_abs = self.permutation[k]
            vec[dest_abs] = old[src_abs]

        return tuple(vec)

    def apply_to_mapping(
        self,
        mapping: List[int],
        labels: List[Label]
    ) -> List[int]:
        """
        Apply automorphism to a vertex mapping.

        Args:
            mapping: Current vertex to root mapping
            labels: Atlas labels

        Returns:
            Transformed mapping
        """
        # Create label index
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        # Apply automorphism to all labels
        new_labels = [self.apply_to_label(lab) for lab in labels]

        # Create mapping from old to new vertex indices
        vertex_permutation = [label_to_idx[new_lab] for new_lab in new_labels]

        # Apply to mapping: new_mapping[new_idx] = mapping[old_idx]
        new_mapping = [0] * len(mapping)
        for new_idx, old_idx in enumerate(vertex_permutation):
            new_mapping[new_idx] = mapping[old_idx]

        return new_mapping

    def __eq__(self, other):
        return self.permutation == other.permutation

    def __hash__(self):
        return hash(self.permutation)

class S4Group:
    """The S4 automorphism group of the Atlas graph."""

    def __init__(self):
        """Initialize the S4 group with all 24 elements."""
        self.elements = [
            S4Automorphism(perm)
            for perm in permutations(S4_PERMUTABLE_BITS)
        ]
        assert len(self.elements) == 24, "S4 must have 24 elements"

        # Find identity element
        self.identity = S4Automorphism(S4_PERMUTABLE_BITS)

    def orbit(self, mapping: List[int], labels: List[Label]) -> List[List[int]]:
        """
        Compute the orbit of a mapping under S4 action.

        Args:
            mapping: Vertex to root mapping
            labels: Atlas labels

        Returns:
            List of all mappings in the orbit
        """
        orbit_set = set()
        orbit_list = []

        for g in self.elements:
            transformed = g.apply_to_mapping(mapping, labels)
            transformed_tuple = tuple(transformed)

            if transformed_tuple not in orbit_set:
                orbit_set.add(transformed_tuple)
                orbit_list.append(transformed)

        return orbit_list

    def stabilizer_size(self, mapping: List[int], labels: List[Label]) -> int:
        """
        Compute the size of the stabilizer subgroup.

        Args:
            mapping: Vertex to root mapping
            labels: Atlas labels

        Returns:
            Size of stabilizer
        """
        mapping_tuple = tuple(mapping)
        stabilizer_count = 0

        for g in self.elements:
            transformed = g.apply_to_mapping(mapping, labels)
            if tuple(transformed) == mapping_tuple:
                stabilizer_count += 1

        return stabilizer_count

    def find_automorphism(
        self,
        mapping1: List[int],
        mapping2: List[int],
        labels: List[Label]
    ) -> S4Automorphism:
        """
        Find automorphism g such that g(mapping1) = mapping2.

        Args:
            mapping1: First mapping
            mapping2: Second mapping
            labels: Atlas labels

        Returns:
            Automorphism if found, None otherwise
        """
        mapping2_tuple = tuple(mapping2)

        for g in self.elements:
            transformed = g.apply_to_mapping(mapping1, labels)
            if tuple(transformed) == mapping2_tuple:
                return g

        return None

def apply_automorphism_to_edges(
    edges: Set[Tuple[int, int]],
    automorphism: S4Automorphism,
    labels: List[Label]
) -> Set[Tuple[int, int]]:
    """
    Apply automorphism to edge set.

    Args:
        edges: Set of edges
        automorphism: S4 automorphism
        labels: Atlas labels

    Returns:
        Transformed edge set
    """
    # Create label mapping
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    new_labels = [automorphism.apply_to_label(lab) for lab in labels]
    vertex_perm = [label_to_idx[new_lab] for new_lab in new_labels]

    # Create inverse permutation
    inv_perm = [0] * len(vertex_perm)
    for new_idx, old_idx in enumerate(vertex_perm):
        inv_perm[old_idx] = new_idx

    # Apply to edges
    new_edges = set()
    for a, b in edges:
        new_a = inv_perm[a]
        new_b = inv_perm[b]
        new_edges.add((min(new_a, new_b), max(new_a, new_b)))

    return new_edges

def verify_automorphism_preserves_structure(
    automorphism: S4Automorphism,
    labels: List[Label],
    edges: Set[Tuple[int, int]]
) -> bool:
    """
    Verify that an automorphism preserves graph structure.

    Args:
        automorphism: S4 automorphism
        labels: Atlas labels
        edges: Atlas edges

    Returns:
        True if structure is preserved
    """
    new_edges = apply_automorphism_to_edges(edges, automorphism, labels)
    return new_edges == edges