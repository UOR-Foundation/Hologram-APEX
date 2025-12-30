"""
Quotient structure analysis for Tier-A embeddings.

This module analyzes the quotient graph structure when
vertices are identified by sign classes.
"""
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QuotientGraph:
    """Represents the quotient graph modulo sign classes."""
    vertices: List[int]  # Sign class representatives (sorted)
    edges: List[Tuple[int, int]]  # Edges between classes
    adjacency_matrix: List[List[bool]]  # Adjacency matrix
    degree_sequence: List[int]  # Degree of each vertex
    class_sizes: Dict[int, int]  # Size of each equivalence class


class QuotientAnalyzer:
    """Analyzes quotient structures in embeddings."""

    def __init__(self, mapping: List[int], atlas_graph, e8_system):
        """
        Initialize quotient analyzer.

        Args:
            mapping: Vertex to root mapping
            atlas_graph: AtlasGraph instance
            e8_system: E8RootSystem instance
        """
        self.mapping = mapping
        self.atlas = atlas_graph
        self.e8 = e8_system

        # Compute sign class representatives
        self.sign_map = {}  # vertex -> sign class
        self.class_members = defaultdict(list)  # sign class -> vertices

        for v, root in enumerate(mapping):
            neg_root = e8_system.negation_table[root]
            sign_class = min(root, neg_root)
            self.sign_map[v] = sign_class
            self.class_members[sign_class].append(v)

    def build_quotient_graph(self) -> QuotientGraph:
        """
        Build the quotient graph structure.

        Returns:
            QuotientGraph with complete structure
        """
        # Get sorted list of sign classes
        vertices = sorted(self.class_members.keys())
        vertex_idx = {v: i for i, v in enumerate(vertices)}
        n = len(vertices)

        # Initialize adjacency matrix
        adj_matrix = [[False] * n for _ in range(n)]
        edges = []

        # Build adjacency
        for v1 in range(len(self.atlas.labels)):
            class1 = self.sign_map[v1]
            idx1 = vertex_idx[class1]

            for v2 in self.atlas.adjacency[v1]:
                class2 = self.sign_map[v2]
                idx2 = vertex_idx[class2]

                if idx1 != idx2 and not adj_matrix[idx1][idx2]:
                    adj_matrix[idx1][idx2] = True
                    adj_matrix[idx2][idx1] = True
                    if idx1 < idx2:
                        edges.append((class1, class2))

        # Compute degree sequence
        degree_sequence = [sum(row) for row in adj_matrix]

        # Get class sizes
        class_sizes = {cls: len(members) for cls, members in self.class_members.items()}

        return QuotientGraph(
            vertices=vertices,
            edges=edges,
            adjacency_matrix=adj_matrix,
            degree_sequence=degree_sequence,
            class_sizes=class_sizes
        )

    def compute_quotient_adjacency(self) -> Dict[int, Set[int]]:
        """
        Compute adjacency lists for the quotient graph.

        Returns:
            Dictionary mapping sign class to adjacent classes
        """
        adjacency = defaultdict(set)

        for v1 in range(len(self.atlas.labels)):
            class1 = self.sign_map[v1]

            for v2 in self.atlas.adjacency[v1]:
                class2 = self.sign_map[v2]

                if class1 != class2:
                    adjacency[class1].add(class2)
                    adjacency[class2].add(class1)

        return dict(adjacency)

    def analyze_quotient_properties(self) -> Dict:
        """
        Analyze detailed properties of the quotient structure.

        Returns:
            Dictionary with quotient properties and statistics
        """
        quotient = self.build_quotient_graph()
        adjacency = self.compute_quotient_adjacency()

        # Check if quotient preserves E8 structure
        e8_preservation = self._check_e8_preservation()

        # Compute graph properties
        properties = {
            "num_vertices": len(quotient.vertices),
            "num_edges": len(quotient.edges),
            "density": 2 * len(quotient.edges) / (len(quotient.vertices) * (len(quotient.vertices) - 1))
                      if len(quotient.vertices) > 1 else 0,
            "degree_sequence": sorted(quotient.degree_sequence, reverse=True),
            "min_degree": min(quotient.degree_sequence) if quotient.degree_sequence else 0,
            "max_degree": max(quotient.degree_sequence) if quotient.degree_sequence else 0,
            "avg_degree": sum(quotient.degree_sequence) / len(quotient.degree_sequence)
                         if quotient.degree_sequence else 0,
            "class_sizes": quotient.class_sizes,
            "min_class_size": min(quotient.class_sizes.values()) if quotient.class_sizes else 0,
            "max_class_size": max(quotient.class_sizes.values()) if quotient.class_sizes else 0,
            "is_regular": len(set(quotient.degree_sequence)) == 1,
            "e8_structure_preserved": e8_preservation,
            "connectivity": self._analyze_connectivity(quotient),
            "clustering_coefficient": self._compute_clustering(quotient),
        }

        return properties

    def _check_e8_preservation(self) -> bool:
        """
        Check if E8 adjacency is preserved in quotient.

        Returns:
            True if E8 structure is preserved
        """
        # For each edge in Atlas
        for v1 in range(len(self.atlas.labels)):
            root1 = self.mapping[v1]
            for v2 in self.atlas.adjacency[v1]:
                root2 = self.mapping[v2]

                # Check E8 adjacency
                from e8.geometry import are_adjacent
                if not are_adjacent(self.e8.roots[root1], self.e8.roots[root2]):
                    return False

        return True

    def _analyze_connectivity(self, quotient: QuotientGraph) -> Dict:
        """Analyze connectivity properties of the quotient."""
        n = len(quotient.vertices)
        if n == 0:
            return {"is_connected": False, "components": 0}

        # BFS to find components
        visited = [False] * n
        components = 0

        for start in range(n):
            if not visited[start]:
                components += 1
                queue = [start]
                visited[start] = True

                while queue:
                    v = queue.pop(0)
                    for u in range(n):
                        if quotient.adjacency_matrix[v][u] and not visited[u]:
                            visited[u] = True
                            queue.append(u)

        return {
            "is_connected": components == 1,
            "components": components
        }

    def _compute_clustering(self, quotient: QuotientGraph) -> float:
        """
        Compute average clustering coefficient.

        Returns:
            Average clustering coefficient
        """
        n = len(quotient.vertices)
        if n < 3:
            return 0.0

        total_coeff = 0.0
        valid_nodes = 0

        for i in range(n):
            neighbors = [j for j in range(n) if quotient.adjacency_matrix[i][j]]
            k = len(neighbors)

            if k >= 2:
                # Count triangles
                triangles = 0
                for j1_idx in range(len(neighbors)):
                    for j2_idx in range(j1_idx + 1, len(neighbors)):
                        j1 = neighbors[j1_idx]
                        j2 = neighbors[j2_idx]
                        if quotient.adjacency_matrix[j1][j2]:
                            triangles += 1

                # Clustering coefficient for node i
                coeff = 2.0 * triangles / (k * (k - 1))
                total_coeff += coeff
                valid_nodes += 1

        return total_coeff / valid_nodes if valid_nodes > 0 else 0.0

    def get_quotient_distance_matrix(self) -> Optional[List[List[int]]]:
        """
        Compute all-pairs shortest path distances in quotient.

        Returns:
            Distance matrix, or None if not connected
        """
        quotient = self.build_quotient_graph()
        n = len(quotient.vertices)

        if n == 0:
            return None

        # Initialize distance matrix
        dist = [[float('inf')] * n for _ in range(n)]

        # Set distances for edges
        for i in range(n):
            dist[i][i] = 0
            for j in range(n):
                if quotient.adjacency_matrix[i][j]:
                    dist[i][j] = 1

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # Check if connected
        for i in range(n):
            for j in range(n):
                if dist[i][j] == float('inf'):
                    return None  # Not connected

        return [[int(d) for d in row] for row in dist]


def build_quotient_graph(mapping: List[int], atlas_graph, e8_system) -> QuotientGraph:
    """
    Convenience function to build quotient graph.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        QuotientGraph structure
    """
    analyzer = QuotientAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.build_quotient_graph()


def compute_quotient_adjacency(mapping: List[int], atlas_graph, e8_system) -> Dict[int, Set[int]]:
    """
    Convenience function to compute quotient adjacency.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Adjacency dictionary for quotient
    """
    analyzer = QuotientAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.compute_quotient_adjacency()


def analyze_quotient_properties(mapping: List[int], atlas_graph, e8_system) -> Dict:
    """
    Convenience function to analyze quotient properties.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Dictionary with quotient properties
    """
    analyzer = QuotientAnalyzer(mapping, atlas_graph, e8_system)
    return analyzer.analyze_quotient_properties()