"""
Atlas graph construction and operations.

This module builds the Atlas core graph G_A with its 96 vertices and edges.
"""
from typing import List, Set, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Label, EdgeSet, AdjacencyList, GraphStructure, ATLAS_VERTEX_COUNT
from .labels import generate_canonical_labels, create_label_index, compute_neighbor_labels

class AtlasGraph:
    """
    Encapsulates the Atlas graph structure and operations.

    The Atlas graph G_A has 96 vertices with degrees 5-6, representing the
    canonical labels in a specific 8-bit encoding scheme.
    """

    def __init__(self):
        """Initialize the Atlas graph by building its structure."""
        self.labels: List[Label] = generate_canonical_labels()
        self.label_index = create_label_index(self.labels)
        self.edges: EdgeSet = set()
        self.adjacency: AdjacencyList = [set() for _ in range(len(self.labels))]
        self.tau: List[int] = []  # Mirror pairing

        self._build_graph()
        self._compute_tau()
        self._verify_invariants()

    def _build_graph(self) -> None:
        """Build the graph edges from label neighbor relationships."""
        for i, lab in enumerate(self.labels):
            for nlab in compute_neighbor_labels(lab):
                j = self.label_index[nlab]
                a, b = (i, j) if i < j else (j, i)
                if a != b:
                    self.edges.add((a, b))

        # Build adjacency list from edges
        for a, b in self.edges:
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

    def _compute_tau(self) -> None:
        """Compute the mirror pairing (tau map) for vertices."""
        self.tau = [self.label_index[self._mirror_tau(lab)] for lab in self.labels]

    def _mirror_tau(self, lab: Label) -> Label:
        """
        Apply mirror transformation to a label (flip e7).

        Args:
            lab: A canonical label

        Returns:
            Mirrored label with e7 flipped
        """
        e1, e2, e3, d45, e6, e7 = lab
        return (e1, e2, e3, d45, e6, 1 - e7)

    def _verify_invariants(self) -> None:
        """Verify critical graph invariants."""
        # Check vertex count
        assert len(self.labels) == ATLAS_VERTEX_COUNT, f"G_A must have {ATLAS_VERTEX_COUNT} vertices"

        # Check degree range
        degrees = [self.degree(i) for i in range(self.num_vertices)]
        min_deg, max_deg = min(degrees), max(degrees)
        assert min_deg >= 5 and max_deg <= 6, f"Unexpected degree range: {min_deg}..{max_deg}"

        # Ensure no tau-pair is an edge (mirror is a symmetry, not an edge)
        for i, t in enumerate(self.tau):
            edge = (min(i, t), max(i, t))
            assert edge not in self.edges, "Spec invariant: tau-pairs must not be edges in G_A core"

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        return len(self.labels)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def degree(self, vertex: int) -> int:
        """
        Return the degree of a vertex.

        Args:
            vertex: Vertex index

        Returns:
            Number of neighbors
        """
        return len(self.adjacency[vertex])

    def neighbors(self, vertex: int) -> Set[int]:
        """
        Return the neighbors of a vertex.

        Args:
            vertex: Vertex index

        Returns:
            Set of neighbor indices
        """
        return self.adjacency[vertex]

    def get_mirror_pair(self, vertex: int) -> int:
        """
        Get the mirror pair of a vertex under tau.

        Args:
            vertex: Vertex index

        Returns:
            Index of the mirror pair vertex
        """
        return self.tau[vertex]

    def is_mirror_pair(self, v1: int, v2: int) -> bool:
        """
        Check if two vertices form a mirror pair.

        Args:
            v1: First vertex index
            v2: Second vertex index

        Returns:
            True if v1 and v2 are mirror pairs
        """
        return self.tau[v1] == v2 or self.tau[v2] == v1

    def get_label(self, vertex: int) -> Label:
        """
        Get the label of a vertex.

        Args:
            vertex: Vertex index

        Returns:
            The canonical label
        """
        return self.labels[vertex]

    def get_graph_structure(self) -> GraphStructure:
        """
        Return a GraphStructure object encapsulating the graph.

        Returns:
            GraphStructure with vertices, edges, and adjacency
        """
        return GraphStructure(
            vertices=self.num_vertices,
            edges=self.edges,
            adjacency=self.adjacency
        )

def build_GA() -> Tuple[List[Label], EdgeSet, AdjacencyList, List[int]]:
    """
    Build the Atlas graph (backward compatibility function).

    Returns:
        Tuple of (labels, edges, adjacency, tau)
    """
    graph = AtlasGraph()
    return graph.labels, graph.edges, graph.adjacency, graph.tau