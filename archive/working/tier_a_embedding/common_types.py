"""
Common type definitions for the tier_a_embedding package.

This module contains type aliases and data classes used throughout the package.
"""
from fractions import Fraction
from typing import Tuple, Set, List, Dict, Optional
from dataclasses import dataclass

# Type aliases for clarity and reusability
Label = Tuple[int, int, int, int, int, int]  # (e1, e2, e3, d45, e6, e7), d45 in {-1, 0, +1}
Root = Tuple[Fraction, ...]  # Vector in 8-dimensional space, length 8
EdgeSet = Set[Tuple[int, int]]  # Set of edges as pairs of vertex indices
AdjacencyList = List[Set[int]]  # Adjacency list representation of a graph

# Mapping types
LabelIndex = Dict[Label, int]  # Maps labels to their indices
RootIndex = Dict[Root, int]  # Maps roots to their indices
VertexMapping = List[int]  # Maps vertex indices to root indices

@dataclass
class GraphStructure:
    """Encapsulates a graph's structure."""
    vertices: int
    edges: EdgeSet
    adjacency: AdjacencyList

    @property
    def num_vertices(self) -> int:
        return self.vertices

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def degree(self, vertex: int) -> int:
        """Return the degree of a vertex."""
        return len(self.adjacency[vertex])

    def neighbors(self, vertex: int) -> Set[int]:
        """Return the neighbors of a vertex."""
        return self.adjacency[vertex]

@dataclass
class EmbeddingResult:
    """Result of an embedding search."""
    mapping: List[int]  # atlas_index -> root_index
    used_root: List[bool]  # which roots are used

    @property
    def num_used_roots(self) -> int:
        return sum(self.used_root)

    def is_injective(self) -> bool:
        """Check if the mapping is injective."""
        return len(set(self.mapping)) == len(self.mapping)

@dataclass
class CertificateData:
    """Data structure for embedding certificates."""
    version: str
    atlas_labels: List[str]
    unity_indices: List[int]
    roots: Dict[str, List[str]]
    mapping: Dict[str, str]
    canonical_key: List[int]
    integer_roots_used: int
    sign_classes_used: int
    automorphism_group: str

# Constants
ATLAS_VERTEX_COUNT = 96
E8_ROOT_COUNT = 240
E8_INTEGER_ROOT_COUNT = 112
E8_HALF_INTEGER_ROOT_COUNT = 128
ATLAS_DEGREE_MIN = 5
ATLAS_DEGREE_MAX = 6
E8_ROOT_DEGREE = 56
S4_GROUP_SIZE = 24
SIGN_CLASSES_USED = 48
UNITY_INDEX_COUNT = 2

# Klein group bits that can be permuted by S4
S4_PERMUTABLE_BITS = (0, 1, 2, 4)  # Indices for (e1, e2, e3, e6)