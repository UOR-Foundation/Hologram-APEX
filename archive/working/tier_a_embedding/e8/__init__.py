"""
E8 root system module.

This module provides the E8 root system generation and geometric operations.
"""

from .roots import (
    E8RootSystem,
    generate_e8_roots,
    generate_integer_roots,
    generate_half_integer_roots,
    create_root_index,
    compute_negation_table,
    is_integer_root,
    is_half_integer_root,
    e8_roots,
)
from .geometry import (
    E8Geometry,
    dot_product,
    dot,
    norm_squared,
    are_adjacent,
    compute_adjacency_graph,
    build_GE8,
)

__all__ = [
    # Roots
    "E8RootSystem",
    "generate_e8_roots",
    "generate_integer_roots",
    "generate_half_integer_roots",
    "create_root_index",
    "compute_negation_table",
    "is_integer_root",
    "is_half_integer_root",
    "e8_roots",
    # Geometry
    "E8Geometry",
    "dot_product",
    "dot",
    "norm_squared",
    "are_adjacent",
    "compute_adjacency_graph",
    "build_GE8",
]