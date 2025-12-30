"""
1-skeleton extraction and analysis for embeddings.

This module extracts the 1-skeleton (edge graph) structure
from embeddings and analyzes its properties.
"""
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class SkeletonData:
    """Container for 1-skeleton data."""
    vertices: Set[int]  # Sign class representatives
    edges: Set[Tuple[int, int]]  # Edges between sign classes
    degree_map: Dict[int, int]  # Vertex degrees
    adjacency: Dict[int, Set[int]]  # Adjacency lists


class SkeletonAnalysis:
    """Analyzes the 1-skeleton structure of embeddings."""

    def __init__(self, mapping: List[int], atlas_graph, e8_system):
        """
        Initialize skeleton analysis.

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

        # Compute sign class mapping
        self.sign_classes = self._compute_sign_class_mapping()
        self.class_to_vertices = self._group_by_class()

    def _compute_sign_class_mapping(self) -> Dict[int, int]:
        """Map each vertex to its sign class representative."""
        sign_map = {}
        for v, root in enumerate(self.mapping):
            neg_root = self.e8.negation_table[root]
            sign_map[v] = min(root, neg_root)
        return sign_map

    def _group_by_class(self) -> Dict[int, List[int]]:
        """Group vertices by their sign class."""
        groups = {}
        for v, cls in self.sign_classes.items():
            if cls not in groups:
                groups[cls] = []
            groups[cls].append(v)
        return groups

    def extract_1_skeleton(self) -> SkeletonData:
        """
        Extract the 1-skeleton graph structure.

        Returns:
            SkeletonData containing vertices, edges, and properties
        """
        vertices = set(self.sign_classes.values())
        edges = set()
        adjacency = {v: set() for v in vertices}

        # Build edges between sign classes
        for v1 in range(len(self.atlas.labels)):
            class1 = self.sign_classes[v1]
            for v2 in self.atlas.adjacency[v1]:
                if v2 > v1:  # Process each edge once
                    class2 = self.sign_classes[v2]
                    if class1 != class2:  # Inter-class edge
                        edge = (min(class1, class2), max(class1, class2))
                        edges.add(edge)
                        adjacency[class1].add(class2)
                        adjacency[class2].add(class1)

        # Compute degrees
        degree_map = {v: len(adjacency[v]) for v in vertices}

        return SkeletonData(
            vertices=vertices,
            edges=edges,
            degree_map=degree_map,
            adjacency=adjacency
        )

    def compute_skeleton_properties(self) -> Dict:
        """
        Compute detailed properties of the 1-skeleton.

        Returns:
            Dictionary with skeleton properties
        """
        skeleton = self.extract_1_skeleton()

        # Count internal edges (within same sign class)
        internal_edges = 0
        for cls, vertices in self.class_to_vertices.items():
            for i, v1 in enumerate(vertices):
                for v2 in vertices[i+1:]:
                    if v2 in self.atlas.adjacency[v1]:
                        internal_edges += 1

        # Count total Atlas edges
        total_atlas_edges = sum(len(adj) for adj in self.atlas.adjacency) // 2

        # Compute statistics
        properties = {
            "num_sign_classes": len(skeleton.vertices),
            "num_inter_class_edges": len(skeleton.edges),
            "num_internal_edges": internal_edges,
            "total_atlas_edges": total_atlas_edges,
            "edge_preservation_ratio": len(skeleton.edges) / total_atlas_edges if total_atlas_edges > 0 else 0,
            "degrees": list(skeleton.degree_map.values()),
            "min_degree": min(skeleton.degree_map.values()) if skeleton.degree_map else 0,
            "max_degree": max(skeleton.degree_map.values()) if skeleton.degree_map else 0,
            "avg_degree": sum(skeleton.degree_map.values()) / len(skeleton.degree_map) if skeleton.degree_map else 0,
            "is_connected": self._check_connectivity(skeleton),
            "diameter": self._compute_diameter(skeleton) if self._check_connectivity(skeleton) else -1,
        }

        return properties

    def _check_connectivity(self, skeleton: SkeletonData) -> bool:
        """Check if the skeleton graph is connected."""
        if not skeleton.vertices:
            return False

        visited = set()
        start = next(iter(skeleton.vertices))
        queue = [start]
        visited.add(start)

        while queue:
            v = queue.pop(0)
            for neighbor in skeleton.adjacency[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(skeleton.vertices)

    def _compute_diameter(self, skeleton: SkeletonData) -> int:
        """Compute the diameter of the skeleton graph."""
        if not skeleton.vertices:
            return 0

        max_dist = 0
        for start in skeleton.vertices:
            # BFS from start
            distances = {start: 0}
            queue = [start]

            while queue:
                v = queue.pop(0)
                for neighbor in skeleton.adjacency[v]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[v] + 1
                        queue.append(neighbor)

            # Update max distance
            if distances:
                max_dist = max(max_dist, max(distances.values()))

        return max_dist

    def get_class_edge_matrix(self) -> List[List[int]]:
        """
        Get edge count matrix between sign classes.

        Returns:
            Matrix where entry (i,j) is number of edges from class i to class j
        """
        skeleton = self.extract_1_skeleton()
        class_list = sorted(skeleton.vertices)
        class_idx = {cls: i for i, cls in enumerate(class_list)}

        # Initialize matrix
        matrix = [[0] * len(class_list) for _ in range(len(class_list))]

        # Count edges between classes
        for v1 in range(len(self.atlas.labels)):
            class1 = self.sign_classes[v1]
            idx1 = class_idx[class1]
            for v2 in self.atlas.adjacency[v1]:
                class2 = self.sign_classes[v2]
                idx2 = class_idx[class2]
                matrix[idx1][idx2] += 1

        return matrix


def extract_1_skeleton(mapping: List[int], atlas_graph, e8_system) -> SkeletonData:
    """
    Convenience function to extract 1-skeleton.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        SkeletonData with graph structure
    """
    analyzer = SkeletonAnalysis(mapping, atlas_graph, e8_system)
    return analyzer.extract_1_skeleton()


def compute_skeleton_properties(mapping: List[int], atlas_graph, e8_system) -> Dict:
    """
    Convenience function to compute skeleton properties.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Dictionary with skeleton properties
    """
    analyzer = SkeletonAnalysis(mapping, atlas_graph, e8_system)
    return analyzer.compute_skeleton_properties()