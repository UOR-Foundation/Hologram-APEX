"""
Atlas symmetry and automorphism operations.

This module handles mirror symmetry and the S4 automorphism group acting on the Atlas graph.
"""
from typing import List, Tuple, Set
from itertools import permutations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Label, LabelIndex, S4_GROUP_SIZE, S4_PERMUTABLE_BITS
from .labels import create_label_index

def apply_mirror_symmetry(lab: Label) -> Label:
    """
    Apply mirror transformation to a label (flip e7).

    Args:
        lab: A canonical label

    Returns:
        Mirrored label with e7 flipped
    """
    e1, e2, e3, d45, e6, e7 = lab
    return (e1, e2, e3, d45, e6, 1 - e7)

def permute_label_bits(lab: Label, perm: Tuple[int, int, int, int]) -> Label:
    """
    Permute (e1, e2, e3, e6) positions by a given permutation.

    d45 and e7 remain unchanged. Indices in `perm` refer to positions within
    the tuple (e1, e2, e3, d45, e6, e7): 0, 1, 2, 4.

    Args:
        lab: A canonical label
        perm: Permutation of absolute indices (0, 1, 2, 4)

    Returns:
        Label with permuted bits
    """
    # Validate perm is a permutation of the four absolute indices
    assert len(perm) == 4 and set(perm) == set(S4_PERMUTABLE_BITS), \
        "perm must be a permutation of (0, 1, 2, 4)"

    e1, e2, e3, d45, e6, e7 = lab
    vec = [e1, e2, e3, d45, e6, e7]
    old = vec[:]  # Snapshot before writing
    src_positions = S4_PERMUTABLE_BITS  # The four bits we permute

    # For each destination slot in src_positions, pull from the source absolute index given by perm[k]
    for k, dest_abs in enumerate(src_positions):
        src_abs = perm[k]
        vec[dest_abs] = old[src_abs]

    return (vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])

def generate_s4_automorphism_group(labels: List[Label]) -> List[Tuple[int, ...]]:
    """
    Generate automorphisms of G_A as permutations of vertex indices.

    The S4 action on {e1, e2, e3, e6} gives 24 automorphisms.

    Args:
        labels: List of canonical labels

    Returns:
        List of vertex permutations (each as a tuple of vertex indices)
    """
    index = create_label_index(labels)
    autos: List[Tuple[int, ...]] = []

    for perm in permutations(S4_PERMUTABLE_BITS):
        permuted = [permute_label_bits(lab, perm) for lab in labels]
        idx_map = tuple(index[p] for p in permuted)
        autos.append(idx_map)

    return autos

def apply_vertex_permutation(
    mapping: List[int],
    tau: List[int],
    labels: List[Label],
    perm_idx_map: Tuple[int, ...]
) -> Tuple[List[int], List[int], List[Label]]:
    """
    Apply an automorphism given as a permutation of vertex indices.

    Args:
        mapping: Current vertex mapping
        tau: Current tau (mirror) pairing
        labels: Current labels
        perm_idx_map: Permutation of vertex indices

    Returns:
        Tuple of (new_mapping, new_tau, new_labels) in permuted order
    """
    n = len(labels)
    assert len(mapping) == n and len(tau) == n and len(perm_idx_map) == n, \
        "apply_vertex_permutation requires full-size inputs"

    new_labels: List[Label] = [None] * n  # type: ignore
    new_mapping = [0] * n

    # Build inverse permutation: where did old i go?
    inv = [0] * n
    for new_i, old_i in enumerate(perm_idx_map):
        inv[old_i] = new_i

    # Remap labels and mapping
    for new_i, old_i in enumerate(perm_idx_map):
        new_labels[new_i] = labels[old_i]
        new_mapping[new_i] = mapping[old_i]

    # Recompute tau under new labels
    index_new = create_label_index(new_labels)  # Safe: labels distinct
    new_tau = [index_new[apply_mirror_symmetry(lab)] for lab in new_labels]

    return new_mapping, new_tau, new_labels

def verify_automorphism_preserves_edges(
    edges: Set[Tuple[int, int]],
    permutation: Tuple[int, ...],
    n_vertices: int
) -> bool:
    """
    Verify that an automorphism preserves the edge set.

    Args:
        edges: Set of edges
        permutation: Vertex permutation
        n_vertices: Number of vertices

    Returns:
        True if the permutation preserves edges
    """
    # Build inverse permutation
    inv = [0] * n_vertices
    for new_i, old_i in enumerate(permutation):
        inv[old_i] = new_i

    # Map edges under permutation
    edge_image = {tuple(sorted((inv[a], inv[b]))) for (a, b) in edges}

    return edge_image == edges

def get_automorphism_group_size() -> int:
    """
    Return the size of the S4 automorphism group.

    Returns:
        24 (the order of S4)
    """
    return S4_GROUP_SIZE

class AtlasSymmetry:
    """Encapsulates symmetry operations on the Atlas graph."""

    def __init__(self, labels: List[Label]):
        """
        Initialize with the list of labels.

        Args:
            labels: List of canonical labels
        """
        self.labels = labels
        self.automorphisms = generate_s4_automorphism_group(labels)

    def apply_mirror(self, label: Label) -> Label:
        """
        Apply mirror symmetry to a label.

        Args:
            label: A canonical label

        Returns:
            Mirrored label
        """
        return apply_mirror_symmetry(label)

    def get_automorphism_count(self) -> int:
        """
        Get the number of automorphisms.

        Returns:
            Size of the automorphism group (24 for S4)
        """
        return len(self.automorphisms)

    def get_automorphisms(self) -> List[Tuple[int, ...]]:
        """
        Get all automorphisms as vertex permutations.

        Returns:
            List of vertex permutations
        """
        return self.automorphisms