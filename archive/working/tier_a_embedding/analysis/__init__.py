"""
Analysis module for studying embedding structure.

This module provides tools for analyzing the graph structure
of Tier-A embeddings, including neighbor extraction, quotient
structures, and 1-skeleton analysis.
"""

from .neighbor_extraction import (
    extract_neighbor_map,
    get_sign_class_adjacency,
    analyze_degree_distribution,
    NeighborAnalyzer,
)

from .quotient_structure import (
    build_quotient_graph,
    compute_quotient_adjacency,
    analyze_quotient_properties,
    QuotientGraph,
    QuotientAnalyzer,
)

from .skeleton import (
    extract_1_skeleton,
    compute_skeleton_properties,
    SkeletonAnalysis,
)

__all__ = [
    # Neighbor extraction
    "extract_neighbor_map",
    "get_sign_class_adjacency",
    "analyze_degree_distribution",
    "NeighborAnalyzer",
    # Quotient structure
    "build_quotient_graph",
    "compute_quotient_adjacency",
    "analyze_quotient_properties",
    "QuotientGraph",
    "QuotientAnalyzer",
    # Skeleton
    "extract_1_skeleton",
    "compute_skeleton_properties",
    "SkeletonAnalysis",
]