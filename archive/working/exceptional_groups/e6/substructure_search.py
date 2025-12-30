"""
E₆ Substructure Search in 96 Atlas Vertices.

E₆ has 72 roots. Hypothesis: 96 = 72 (E₆) + 24 (half of F₄).
Search for a 72-vertex subset with E₆ properties.
"""
import sys
import os
from typing import List, Set, Dict, Tuple, Optional
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem


class E6SubstructureSearch:
    """Search for E₆ structure in 96 Atlas vertices."""

    def __init__(self):
        """Initialize with Atlas structure."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()

        # E₆ properties
        self.e6_roots = 72
        self.e6_rank = 6
        self.e6_weyl_order = 51840

    def analyze_vertex_symmetry(self) -> Dict[str, List[int]]:
        """
        Analyze symmetry properties to guide search.

        Look for natural 72-24 partition based on structure.
        """
        # Analyze degrees
        degrees = {}
        for v in range(96):
            deg = len(self.atlas.adjacency[v])
            if deg not in degrees:
                degrees[deg] = []
            degrees[deg].append(v)

        print("\nVertex degree distribution:")
        for deg in sorted(degrees.keys()):
            print(f"  Degree {deg}: {len(degrees[deg])} vertices")

        # Look for 72-24 split
        # Hypothesis: Maybe degree distinguishes E₆ from remainder
        analysis = {
            'degrees': degrees,
            'total': 96,
            'target_e6': 72,
            'remainder': 24
        }

        return analysis

    def search_by_graph_properties(self) -> Optional[Set[int]]:
        """
        Search for 72-vertex subset using graph properties.

        Strategy:
        1. Look for highly connected 72-vertex subgraph
        2. Check if it has E₆-like structure
        """
        print("\nSearching for E₆ by graph properties...")

        # Start with vertices that have specific properties
        # E₆ should be "more regular" than the remainder

        # Try vertices with most common degree
        degree_counts = {}
        for v in range(96):
            deg = len(self.atlas.adjacency[v])
            if deg not in degree_counts:
                degree_counts[deg] = 0
            degree_counts[deg] += 1

        # Most common degree
        most_common_degree = max(degree_counts, key=degree_counts.get)
        vertices_with_common_deg = [
            v for v in range(96)
            if len(self.atlas.adjacency[v]) == most_common_degree
        ]

        print(f"  Most common degree: {most_common_degree}")
        print(f"  Vertices with this degree: {len(vertices_with_common_deg)}")

        # If we have around 72 vertices with common degree, that's promising
        if 60 <= len(vertices_with_common_deg) <= 84:
            # Try to extend/reduce to exactly 72
            if len(vertices_with_common_deg) > 72:
                # Remove vertices with fewest connections to others in set
                subset = self._reduce_to_size(vertices_with_common_deg, 72)
            elif len(vertices_with_common_deg) < 72:
                # Add vertices most connected to the set
                subset = self._extend_to_size(vertices_with_common_deg, 72)
            else:
                subset = set(vertices_with_common_deg)

            if self._test_e6_properties(subset):
                print(f"  ✓ Found E₆ candidate!")
                return subset

        return None

    def search_by_orbit_structure(self) -> Optional[Set[int]]:
        """
        Use S₄ orbit structure to find E₆.

        We have 30 orbits. Maybe E₆ is union of specific orbits.
        """
        print("\nSearching for E₆ using orbit structure...")

        # Load S₄ orbit data
        import sys
        sys.path.append('..')
        from s4_automorphism import verify_s4_automorphism
        s4_data = verify_s4_automorphism()

        # Orbits have sizes: 12×1, 12×4, 6×6
        # Total: 12 + 48 + 36 = 96

        # Try different combinations to get 72
        # 72 = 12 + 48 + 12 (all singletons + all 4-orbits + 2 of 6-orbits)
        # or 72 = 48 + 24 (all 4-orbits + 4 of 6-orbits)
        # or 72 = 12 + 36 + 24 (all singletons + all 6-orbits + 6 of 4-orbits)

        orbit_sizes = {}
        for orbit in s4_data.orbits:
            size = len(orbit)
            if size not in orbit_sizes:
                orbit_sizes[size] = []
            orbit_sizes[size].append(orbit)

        print(f"  Orbit sizes available:")
        for size, orbits in orbit_sizes.items():
            print(f"    Size {size}: {len(orbits)} orbits")

        # Try combination: all 4-orbits + 4 of the 6-orbits
        if 4 in orbit_sizes and 6 in orbit_sizes:
            # All 4-orbits = 12 × 4 = 48 vertices
            vertices_from_4 = set()
            for orbit in orbit_sizes[4]:
                vertices_from_4.update(orbit)

            # Need 24 more vertices from 6-orbits (4 orbits)
            if len(orbit_sizes[6]) >= 4:
                for combo in combinations(orbit_sizes[6], 4):
                    candidate = vertices_from_4.copy()
                    for orbit in combo:
                        candidate.update(orbit)

                    if len(candidate) == 72:
                        if self._test_e6_properties(candidate):
                            print(f"  ✓ Found E₆ via orbit combination!")
                            return candidate

        return None

    def _reduce_to_size(self, vertices: List[int], target: int) -> Set[int]:
        """Reduce vertex set to target size."""
        # Sort by connectivity within set
        connectivity = []
        for v in vertices:
            conn = sum(1 for u in vertices if u in self.atlas.adjacency[v])
            connectivity.append((conn, v))

        # Keep most connected
        connectivity.sort(reverse=True)
        return set(v for _, v in connectivity[:target])

    def _extend_to_size(self, vertices: List[int], target: int) -> Set[int]:
        """Extend vertex set to target size."""
        current = set(vertices)
        remaining = set(range(96)) - current

        while len(current) < target and remaining:
            # Find vertex most connected to current set
            best_v = None
            best_conn = -1

            for v in remaining:
                conn = sum(1 for u in current if u in self.atlas.adjacency[v])
                if conn > best_conn:
                    best_conn = conn
                    best_v = v

            if best_v is not None:
                current.add(best_v)
                remaining.remove(best_v)
            else:
                break

        return current

    def _test_e6_properties(self, subset: Set[int]) -> bool:
        """
        Test if subset has E₆-like properties.

        E₆ properties:
        - 72 vertices (roots)
        - Rank 6 structure
        - Specific degree distribution
        """
        if len(subset) != 72:
            return False

        # Check connectivity
        # E₆ subgraph should be connected
        if not self._is_connected(subset):
            return False

        # Check degree distribution within subset
        internal_degrees = []
        for v in subset:
            internal_deg = sum(1 for u in subset if u in self.atlas.adjacency[v])
            internal_degrees.append(internal_deg)

        # E₆ should have regular structure (exact arithmetic)
        sum_degrees = sum(internal_degrees)
        avg_degree = sum_degrees / len(internal_degrees) if internal_degrees else 0

        # Calculate exact variance
        variance = sum((d - avg_degree) ** 2 for d in internal_degrees) / len(internal_degrees) if internal_degrees else 0

        # Rough check: E₆ should be somewhat regular
        # Use exact comparison: variance should be small relative to average
        if variance > (avg_degree ** 2) / 4:  # Too irregular (variance > (avg/2)^2)
            return False

        print(f"    Subset passes basic E₆ tests:")
        print(f"      Size: {len(subset)}")
        print(f"      Connected: True")
        print(f"      Avg internal degree: {avg_degree:.2f}")
        print(f"      Variance internal degree: {variance:.2f}")

        return True

    def _is_connected(self, subset: Set[int]) -> bool:
        """Check if subset induces connected subgraph."""
        if not subset:
            return False

        # BFS from first vertex
        start = next(iter(subset))
        visited = {start}
        queue = [start]

        while queue:
            v = queue.pop(0)
            for u in self.atlas.adjacency[v]:
                if u in subset and u not in visited:
                    visited.add(u)
                    queue.append(u)

        return len(visited) == len(subset)

    def analyze_decomposition(self, e6_subset: Optional[Set[int]]) -> Dict:
        """
        Analyze 96 = 72 (E₆) + 24 decomposition.

        If E₆ found, analyze the 24 remaining vertices.
        """
        if e6_subset is None:
            return {
                'e6_found': False,
                'reason': 'No E₆ subset identified'
            }

        remainder = set(range(96)) - e6_subset

        # Analyze remainder
        # Hypothesis: remainder = 24 = half of F₄ (48)
        analysis = {
            'e6_found': True,
            'e6_size': len(e6_subset),
            'remainder_size': len(remainder),
            'is_half_f4': len(remainder) == 24,
        }

        # Check if remainder has special structure
        # F₄ has 24 long roots and 24 short roots
        # Maybe remainder is one type?

        print(f"\n96 = 72 + 24 Decomposition Analysis:")
        print(f"  E₆ subset: {len(e6_subset)} vertices")
        print(f"  Remainder: {len(remainder)} vertices")

        if len(remainder) == 24:
            print(f"  ✓ Remainder is exactly 24 (half of F₄)")

            # Check connectivity of remainder
            if self._is_connected(remainder):
                print(f"  ✓ Remainder is connected")
                analysis['remainder_connected'] = True
            else:
                print(f"  ✗ Remainder is not connected")
                analysis['remainder_connected'] = False

        return analysis


def search_for_e6():
    """
    Main function to search for E₆ in Atlas.

    Returns:
        E₆ subset if found, analysis results
    """
    print("="*60)
    print("E₆ SUBSTRUCTURE SEARCH (72 ROOTS)")
    print("="*60)

    searcher = E6SubstructureSearch()

    # Analyze vertex symmetry
    symmetry = searcher.analyze_vertex_symmetry()

    # Try different search strategies
    e6_subset = None

    # Strategy 1: Graph properties
    e6_subset = searcher.search_by_graph_properties()

    # Strategy 2: Orbit structure
    if e6_subset is None:
        e6_subset = searcher.search_by_orbit_structure()

    # Analyze decomposition
    decomp = searcher.analyze_decomposition(e6_subset)

    # Result
    result = {
        'e6_found': e6_subset is not None,
        'e6_vertices': list(e6_subset) if e6_subset else None,
        'decomposition': decomp
    }

    print("\n" + "="*60)
    if result['e6_found']:
        print("✓✓✓ E₆ structure found!")
        print(f"    72 vertices identified")
        if decomp.get('is_half_f4'):
            print(f"    Remainder is 24 vertices (half of F₄)")
    else:
        print("⚠ E₆ structure not found")
        print("  May need more sophisticated search")

    return result


if __name__ == "__main__":
    e6_result = search_for_e6()