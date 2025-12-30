"""
Atlas graph module.

This module provides the Atlas graph structure and operations.
"""

from .graph import AtlasGraph, build_GA
from .labels import (
    generate_canonical_labels,
    create_label_index,
    compute_neighbor_labels,
    derive_unity_indices,
    flip_d45_by_e4,
    flip_d45_by_e5,
)
from .symmetry import (
    apply_mirror_symmetry,
    permute_label_bits,
    generate_s4_automorphism_group,
    apply_vertex_permutation,
    AtlasSymmetry,
)

__all__ = [
    # Graph
    "AtlasGraph",
    "build_GA",
    # Labels
    "generate_canonical_labels",
    "create_label_index",
    "compute_neighbor_labels",
    "derive_unity_indices",
    "flip_d45_by_e4",
    "flip_d45_by_e5",
    # Symmetry
    "apply_mirror_symmetry",
    "permute_label_bits",
    "generate_s4_automorphism_group",
    "apply_vertex_permutation",
    "AtlasSymmetry",
]