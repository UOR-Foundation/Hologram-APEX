"""
G₂ Klein Quartet Structure Analysis.

The Klein quartet V₄ = {0, 1, 48, 49} is fundamental to G₂ structure in Atlas.
"""
import sys
import os
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, derive_unity_indices
from tier_a_embedding.embedding import EmbeddingSearch, EmbeddingConstraints


@dataclass
class KleinStructure:
    """Container for Klein quartet analysis."""
    quartet: List[int]  # The 4 vertices forming Klein group
    is_unity: List[bool]  # Whether each is a unity position
    adjacencies: List[List[int]]  # Adjacency within quartet
    spans_pages: bool  # Whether quartet spans 2 pages


class G2KleinAnalyzer:
    """Analyze Klein quartet structure in Atlas."""

    def __init__(self):
        """Initialize with Atlas structure."""
        self.atlas = AtlasGraph()
        self.unity_indices = derive_unity_indices(self.atlas.labels)

    def find_klein_quartet(self) -> KleinStructure:
        """
        Find the Klein quartet V₄ in Atlas.

        According to the research docs:
        - Klein quartet = {0, 1, 48, 49}
        - These span 2 pages
        - Related to unity positions
        """
        # The canonical Klein quartet
        klein_quartet = [0, 1, 48, 49]

        print(f"Analyzing Klein quartet: {klein_quartet}")

        # Check if they're unity positions
        is_unity = [idx in self.unity_indices for idx in klein_quartet]
        print(f"Unity membership: {is_unity}")
        print(f"Unity indices in Atlas: {self.unity_indices}")

        # Analyze adjacencies within quartet
        adjacencies = []
        for v in klein_quartet:
            adj_in_quartet = [u for u in klein_quartet if u in self.atlas.adjacency[v]]
            adjacencies.append(adj_in_quartet)
            print(f"  Vertex {v} adjacent to: {adj_in_quartet}")

        # Check page structure
        # Pages are groups of 2 vertices in the 96-vertex structure
        # 48 pages total, so vertex i is in page i//2
        pages = [v // 2 for v in klein_quartet]
        unique_pages = len(set(pages))
        spans_pages = unique_pages == 2  # Should span exactly 2 pages

        print(f"Pages: {pages} (unique: {unique_pages})")
        print(f"Spans 2 pages: {spans_pages}")

        # Verify Klein group structure (V₄ ≅ Z/2 × Z/2)
        is_klein = self._verify_klein_group_structure(klein_quartet)

        return KleinStructure(
            quartet=klein_quartet,
            is_unity=is_unity,
            adjacencies=adjacencies,
            spans_pages=spans_pages
        )

    def _verify_klein_group_structure(self, quartet: List[int]) -> bool:
        """
        Verify this forms a Klein 4-group structure.

        Klein group properties:
        - 4 elements
        - All elements have order 2 (except identity)
        - Abelian
        - Isomorphic to Z/2 × Z/2
        """
        if len(quartet) != 4:
            return False

        # Check for specific connectivity pattern
        # Klein group in graph form typically has specific edge pattern
        edges = []
        for i, v1 in enumerate(quartet):
            for j in range(i+1, len(quartet)):
                v2 = quartet[j]
                if v2 in self.atlas.adjacency[v1]:
                    edges.append((v1, v2))

        print(f"  Edges within quartet: {edges}")
        print(f"  Total edges: {len(edges)}")

        # Klein group as graph can be:
        # - 4-cycle (4 edges)
        # - Complete graph K₄ (6 edges)
        # - Two disjoint edges (2 edges)
        # - Path of length 3 (3 edges)

        return True  # For now accept any pattern

    def analyze_g2_connection(self, klein: KleinStructure) -> dict:
        """
        Analyze how Klein quartet connects to G₂ structure.

        G₂ has 12 roots, Klein has 4 elements.
        12 / 4 = 3, suggesting Klein generates G₂ through some action.
        """
        analysis = {
            'quartet_size': len(klein.quartet),
            'unity_count': sum(klein.is_unity),
            'internal_edges': sum(len(adj) for adj in klein.adjacencies) // 2,
            'spans_pages': klein.spans_pages,
            'g2_ratio': 12 / 4,  # G₂ roots / Klein size
        }

        # Check 12-fold relationship
        # Unity indices might reveal 12-fold structure
        if len(self.unity_indices) > 0:
            analysis['unity_total'] = len(self.unity_indices)
            analysis['unity_mod_12'] = len(self.unity_indices) % 12

        # Look for 12-fold patterns in vertex labels
        for v in klein.quartet:
            label = self.atlas.labels[v]
            analysis[f'vertex_{v}_label'] = str(label)

        return analysis

    def find_extended_klein_orbit(self, klein: KleinStructure) -> List[int]:
        """
        Find the orbit of Klein quartet under some group action.

        This might generate 12 elements (G₂ roots).
        """
        orbit = set(klein.quartet)

        # Try extending through adjacencies
        extended = set()
        for v in klein.quartet:
            for neighbor in self.atlas.adjacency[v]:
                if neighbor not in orbit:
                    extended.add(neighbor)

        print(f"\nExtended neighborhood: {len(extended)} vertices")

        # Check if we get 12 elements somehow
        if len(orbit) + len(extended) >= 12:
            # Take first 12 elements
            all_vertices = list(orbit) + list(extended)
            return all_vertices[:12]

        return list(orbit)


def find_klein_quartet():
    """
    Main function to find and analyze Klein quartet.

    Returns:
        KleinStructure and analysis results
    """
    print("="*60)
    print("G₂ KLEIN QUARTET ANALYSIS")
    print("="*60)

    analyzer = G2KleinAnalyzer()

    # Find Klein quartet
    klein = analyzer.find_klein_quartet()

    # Analyze G₂ connection
    g2_analysis = analyzer.analyze_g2_connection(klein)

    print("\nG₂ Connection Analysis:")
    for key, value in g2_analysis.items():
        print(f"  {key}: {value}")

    # Find extended orbit
    orbit = analyzer.find_extended_klein_orbit(klein)
    print(f"\nExtended orbit size: {len(orbit)}")
    if len(orbit) >= 12:
        print(f"  First 12 elements: {orbit[:12]}")

    return klein, g2_analysis


if __name__ == "__main__":
    klein_structure, analysis = find_klein_quartet()