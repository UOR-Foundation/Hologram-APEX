"""
Neighbor extraction and analysis for Tier-A embeddings.

This module extracts and analyzes the neighbor relationships
in embeddings, revealing how sign classes connect.
"""
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NeighborData:
    """Container for neighbor analysis data."""
    vertex_to_root: List[int]  # Atlas vertex -> E8 root mapping
    atlas_adjacency: List[Set[int]]  # Atlas graph adjacency
    e8_adjacency: List[Set[int]]  # E8 root adjacency
    sign_classes: Dict[int, int]  # Root -> sign class representative


class NeighborAnalyzer:
    """Analyzes neighbor relationships in embeddings."""

    def __init__(self, mapping: List[int], atlas_graph, e8_system):
        """
        Initialize analyzer with embedding data.

        Args:
            mapping: Vertex to root mapping
            atlas_graph: AtlasGraph instance
            e8_system: E8RootSystem instance
        """
        self.mapping = mapping
        self.atlas = atlas_graph
        self.e8 = e8_system

        # Build E8 adjacency if needed
        from e8.geometry import compute_adjacency_graph
        self.e8_edges, self.e8_adjacency = compute_adjacency_graph(e8_system.roots)

        # Compute sign classes
        self.sign_classes = self._compute_sign_classes()

    def _compute_sign_classes(self) -> Dict[int, int]:
        """
        Compute sign class representatives for all roots.

        Returns:
            Dictionary mapping each root to its sign class representative
        """
        sign_classes = {}
        for i in range(len(self.e8.roots)):
            neg_idx = self.e8.negation_table[i]
            rep = min(i, neg_idx)
            sign_classes[i] = rep
        return sign_classes

    def extract_neighbor_map(self) -> Dict[int, Set[int]]:
        """
        Extract neighbor relationships for mapped vertices.

        Returns:
            Dictionary mapping Atlas vertex to its E8 root neighbors
        """
        neighbor_map = {}

        for v in range(len(self.atlas.labels)):
            root = self.mapping[v]
            # Get Atlas neighbors
            atlas_neighbors = self.atlas.adjacency[v]
            # Get their mapped roots
            root_neighbors = {self.mapping[n] for n in atlas_neighbors}
            neighbor_map[v] = root_neighbors

        return neighbor_map

    def get_sign_class_adjacency(self) -> Dict[int, Set[int]]:
        """
        Compute adjacency between sign classes.

        Returns:
            Dictionary mapping sign class to adjacent sign classes
        """
        class_adjacency = defaultdict(set)

        # For each vertex in the embedding
        for v in range(len(self.mapping)):
            root = self.mapping[v]
            sign_class = self.sign_classes[root]

            # Check neighbors
            for neighbor_v in self.atlas.adjacency[v]:
                neighbor_root = self.mapping[neighbor_v]
                neighbor_class = self.sign_classes[neighbor_root]

                # Add to adjacency (undirected)
                if neighbor_class != sign_class:
                    class_adjacency[sign_class].add(neighbor_class)
                    class_adjacency[neighbor_class].add(sign_class)

        return dict(class_adjacency)

    def analyze_degree_distribution(self) -> Dict[str, any]:
        """
        Analyze degree distribution in the quotient graph.

        Returns:
            Analysis results including degree statistics
        """
        class_adj = self.get_sign_class_adjacency()

        # Get degrees
        degrees = [len(neighbors) for neighbors in class_adj.values()]

        # Compute statistics
        analysis = {
            "num_classes": len(class_adj),
            "degrees": degrees,
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "degree_distribution": {}
        }

        # Count degree frequencies
        for d in degrees:
            if d not in analysis["degree_distribution"]:
                analysis["degree_distribution"][d] = 0
            analysis["degree_distribution"][d] += 1

        return analysis

    def get_induced_e8_subgraph(self) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Get the subgraph of E8 induced by the embedding.

        Returns:
            Tuple of (vertices, edges) in the induced subgraph
        """
        vertices = set(self.mapping)
        edges = set()

        for i, root_i in enumerate(self.mapping):
            for j in range(i + 1, len(self.mapping)):
                root_j = self.mapping[j]
                if root_j in self.e8_adjacency[root_i]:
                    edges.add((min(root_i, root_j), max(root_i, root_j)))

        return vertices, edges

    def verify_edge_preservation(self) -> bool:
        """
        Verify that all Atlas edges are preserved in E8.

        Returns:
            True if all edges are preserved
        """
        for v1 in range(len(self.atlas.labels)):
            root1 = self.mapping[v1]
            for v2 in self.atlas.adjacency[v1]:
                if v2 > v1:  # Check each edge once
                    root2 = self.mapping[v2]
                    if root2 not in self.e8_adjacency[root1]:
                        return False
        return True

    def get_class_representatives(self) -> Dict[int, List[int]]:
        """
        Get vertices grouped by their sign class.

        Returns:
            Dictionary mapping sign class to list of vertices
        """
        class_members = defaultdict(list)

        for v, root in enumerate(self.mapping):
            sign_class = self.sign_classes[root]
            class_members[sign_class].append(v)

        return dict(class_members)


def extract_neighbor_map(mapping: List[int], atlas_graph, e8_system) -> Dict[int, Set[int]]:
    """
    Convenience function to extract neighbor map.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Neighbor map dictionary
    """
    analyzer = NeighborAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.extract_neighbor_map()


def get_sign_class_adjacency(mapping: List[int], atlas_graph, e8_system) -> Dict[int, Set[int]]:
    """
    Convenience function to get sign class adjacency.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Sign class adjacency dictionary
    """
    analyzer = NeighborAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.get_sign_class_adjacency()


def analyze_degree_distribution(mapping: List[int], atlas_graph, e8_system) -> Dict[str, any]:
    """
    Convenience function to analyze degree distribution.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Degree distribution analysis
    """
    analyzer = NeighborAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.analyze_degree_distribution()