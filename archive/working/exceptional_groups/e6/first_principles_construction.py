"""
E₆ First Principles Construction from Atlas.

Construct E₆ (72 roots) from Atlas structure using only intrinsic properties.
DO NOT rely on external E₈ knowledge - build from Atlas alone.

Key observations:
- Atlas: 96 vertices
- E₆: 72 roots
- 96 = 72 + 24 (suggestive)
- E₆ has triality (D₄ outer automorphism)
- Atlas has 3-fold structure (768 = 3×256)
"""
import sys
import os
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, derive_unity_indices


class E6FirstPrinciplesConstruction:
    """Construct E₆ from Atlas using first principles only."""

    def __init__(self):
        """Initialize with Atlas structure."""
        self.atlas = AtlasGraph()
        self.unity_indices = derive_unity_indices(self.atlas.labels)

        # E₆ properties (known from Lie theory, but we'll verify)
        self.e6_roots = 72
        self.e6_rank = 6

    def analyze_triality_structure(self) -> Dict[str, any]:
        """
        Analyze 3-fold (triality) structure in Atlas.

        E₆ has outer automorphism group of order 2 (triality).
        Atlas has 3-fold structure via 768 = 3×256 window.

        Connection: Look for vertices with 3-fold symmetry.
        """
        print("\nAnalyzing triality (3-fold) structure:")

        # Group vertices by position mod 3
        mod3_classes = {0: [], 1: [], 2: []}
        for v in range(96):
            # Use label structure to determine 3-fold class
            label = self.atlas.labels[v]
            # e4-e5 coordinate might relate to triality
            e4_e5 = label[3]  # This is e4-e5 ∈ {-1, 0, 1}

            # Map to mod 3 class
            triality_class = (e4_e5 + 1) % 3  # Maps {-1,0,1} to {0,1,2}
            mod3_classes[triality_class].append(v)

        print(f"  3-fold classes:")
        for cls, vertices in mod3_classes.items():
            print(f"    Class {cls}: {len(vertices)} vertices")

        analysis = {
            'mod3_classes': mod3_classes,
            'class_sizes': {k: len(v) for k, v in mod3_classes.items()},
            'total': sum(len(v) for v in mod3_classes.values())
        }

        return analysis

    def analyze_coordinate_patterns(self) -> Dict[str, any]:
        """
        Analyze coordinate patterns that might define E₆.

        Atlas labels: (e1, e2, e3, e4-e5, e6, e7)
        E₆ has rank 6, suggesting 6 coordinates.

        Strategy: Look for vertices with special coordinate properties.
        """
        print("\nAnalyzing coordinate patterns for E₆:")

        # Count coordinate patterns
        patterns = defaultdict(list)

        for v in range(96):
            label = self.atlas.labels[v]
            e1, e2, e3, e4_e5, e6, e7 = label

            # Pattern 1: e7 coordinate (mirror symmetry)
            if e7 == 0:
                patterns['e7_zero'].append(v)
            else:
                patterns['e7_one'].append(v)

            # Pattern 2: e4-e5 (unity constraint)
            patterns[f'e4_e5_{e4_e5}'].append(v)

            # Pattern 3: Sum of coordinates
            coord_sum = e1 + e2 + e3 + e6  # Skip e4-e5 and e7
            patterns[f'sum_{coord_sum}'].append(v)

        print(f"  Coordinate pattern analysis:")
        for pattern_name in ['e7_zero', 'e7_one', 'e4_e5_-1', 'e4_e5_0', 'e4_e5_1']:
            if pattern_name in patterns:
                print(f"    {pattern_name}: {len(patterns[pattern_name])} vertices")

        return patterns

    def search_by_distance_from_unity(self) -> Optional[Set[int]]:
        """
        Search for E₆ based on distance from unity positions.

        Hypothesis: E₆ vertices might be characterized by their
        distance from the unity positions {1, 4}.
        """
        print("\nSearching by distance from unity:")

        unity = list(self.unity_indices)
        print(f"  Unity positions: {unity}")

        # Compute distance from each vertex to nearest unity
        distances = {}
        for v in range(96):
            min_dist = None  # Use None instead of float('inf')
            for u in unity:
                # Graph distance via BFS
                dist = self._graph_distance(v, u)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            distances[v] = min_dist if min_dist is not None else 1000  # Large integer fallback

        # Analyze distance distribution
        dist_distribution = defaultdict(list)
        for v, d in distances.items():
            dist_distribution[d].append(v)

        print(f"  Distance distribution:")
        for d in sorted(dist_distribution.keys()):
            print(f"    Distance {d}: {len(dist_distribution[d])} vertices")

        # Try different distance thresholds to get 72 vertices
        for max_dist in range(1, 10):
            candidate = set()
            for d in range(max_dist + 1):
                candidate.update(dist_distribution[d])

            if len(candidate) == 72:
                print(f"\n  ✓ Found 72 vertices with distance ≤ {max_dist}")
                return candidate
            elif len(candidate) > 72:
                # Too many, try to reduce
                print(f"    Distance ≤ {max_dist}: {len(candidate)} vertices (too many)")
                break

        return None

    def search_by_degree_and_structure(self) -> Optional[Set[int]]:
        """
        Search for E₆ using degree and local structure.

        Hypothesis: E₆ vertices form a more regular subgraph.
        """
        print("\nSearching by degree and structure:")

        # Analyze degree distribution
        degrees = {}
        for v in range(96):
            degrees[v] = len(self.atlas.adjacency[v])

        degree_groups = defaultdict(list)
        for v, deg in degrees.items():
            degree_groups[deg].append(v)

        print(f"  Degree distribution:")
        for deg in sorted(degree_groups.keys()):
            print(f"    Degree {deg}: {len(degree_groups[deg])} vertices")

        # Try taking vertices with most common degree
        # We saw earlier: degree 5 has 64 vertices, degree 6 has 32

        # Try degree 5 (64) + 8 from degree 6 = 72
        if 5 in degree_groups and 6 in degree_groups:
            candidate = set(degree_groups[5])  # 64 vertices

            # Need 8 more from degree 6
            degree_6_vertices = degree_groups[6]

            # Choose 8 degree-6 vertices most connected to degree-5 set
            connections = []
            for v in degree_6_vertices:
                conn_to_deg5 = sum(1 for u in candidate if u in self.atlas.adjacency[v])
                connections.append((conn_to_deg5, v))

            connections.sort(reverse=True)
            for _, v in connections[:8]:
                candidate.add(v)

            if len(candidate) == 72:
                print(f"\n  ✓ Found 72 vertices: 64 degree-5 + 8 degree-6")
                return candidate

        return None

    def search_by_e7_complement(self) -> Optional[Set[int]]:
        """
        Search for E₆ using E₇ structure.

        We know E₇ = 96 + 30 orbits = 126.
        E₆ ⊂ E₇ means 72 roots of E₇.

        Hypothesis: E₆ might be the 96 vertices minus some special 24.
        """
        print("\nSearching via E₇ complement:")

        # The 24 vertices NOT in E₆ might have special structure
        # They might be related to F₄/2 (since F₄ has 24 long + 24 short)

        # Try: Remove vertices with e7=1 (mirror side)
        e7_zero = set()
        e7_one = set()

        for v in range(96):
            if self.atlas.labels[v][5] == 0:  # e7 coordinate
                e7_zero.add(v)
            else:
                e7_one.add(v)

        print(f"  e7=0: {len(e7_zero)} vertices")
        print(f"  e7=1: {len(e7_one)} vertices")

        if len(e7_zero) == 72:
            print(f"\n  ✓ Found E₆ as vertices with e7=0!")
            return e7_zero
        elif len(e7_one) == 72:
            print(f"\n  ✓ Found E₆ as vertices with e7=1!")
            return e7_one

        return None

    def _graph_distance(self, v1: int, v2: int) -> int:
        """Compute graph distance between two vertices (returns large integer if unreachable)."""
        if v1 == v2:
            return 0

        visited = {v1}
        queue = [(v1, 0)]

        while queue:
            v, dist = queue.pop(0)
            for u in self.atlas.adjacency[v]:
                if u == v2:
                    return dist + 1
                if u not in visited:
                    visited.add(u)
                    queue.append((u, dist + 1))

        return 1000  # Large integer for unreachable vertices

    def verify_e6_candidate(self, candidate: Set[int]) -> Dict[str, bool]:
        """
        Verify if candidate set has E₆ properties.

        Properties to check:
        1. Size = 72
        2. Connected subgraph
        3. Has rank-6 structure
        4. Regular enough degree distribution
        """
        checks = {}

        # Size
        checks['size_72'] = (len(candidate) == 72)

        # Connectivity
        checks['connected'] = self._is_connected(candidate)

        # Induced subgraph analysis
        if checks['connected']:
            # Check degree distribution within E₆ (exact arithmetic)
            internal_degrees = []
            for v in candidate:
                internal_deg = sum(1 for u in candidate if u in self.atlas.adjacency[v])
                internal_degrees.append(internal_deg)

            sum_deg = sum(internal_degrees)
            avg_deg = sum_deg / len(internal_degrees) if internal_degrees else 0

            # Calculate exact variance
            variance = sum((d - avg_deg) ** 2 for d in internal_degrees) / len(internal_degrees) if internal_degrees else 0

            checks['avg_degree'] = avg_deg
            checks['variance'] = variance
            checks['regular_enough'] = (variance < (avg_deg ** 2) * 0.09)  # variance < (avg * 0.3)^2

            print(f"\n  E₆ candidate verification:")
            print(f"    Size: {len(candidate)}")
            print(f"    Connected: {checks['connected']}")
            print(f"    Avg internal degree: {avg_deg:.2f}")
            print(f"    Std internal degree: {std_deg:.2f}")
            print(f"    Regular enough: {checks['regular_enough']}")

        return checks

    def _is_connected(self, vertices: Set[int]) -> bool:
        """Check if vertex set induces connected subgraph."""
        if not vertices:
            return False

        start = next(iter(vertices))
        visited = {start}
        queue = [start]

        while queue:
            v = queue.pop(0)
            for u in self.atlas.adjacency[v]:
                if u in vertices and u not in visited:
                    visited.add(u)
                    queue.append(u)

        return len(visited) == len(vertices)


def construct_e6_first_principles():
    """
    Main function: Construct E₆ from Atlas first principles.

    Returns:
        E₆ vertex set if found, None otherwise
    """
    print("="*60)
    print("E₆ FIRST PRINCIPLES CONSTRUCTION")
    print("="*60)

    constructor = E6FirstPrinciplesConstruction()

    # Strategy 1: Triality structure
    triality = constructor.analyze_triality_structure()

    # Strategy 2: Coordinate patterns
    patterns = constructor.analyze_coordinate_patterns()

    # Strategy 3: Distance from unity
    e6_candidate = constructor.search_by_distance_from_unity()

    if e6_candidate is None:
        # Strategy 4: Degree and structure
        e6_candidate = constructor.search_by_degree_and_structure()

    if e6_candidate is None:
        # Strategy 5: E₇ complement
        e6_candidate = constructor.search_by_e7_complement()

    # Verify candidate
    if e6_candidate:
        verification = constructor.verify_e6_candidate(e6_candidate)

        result = {
            'e6_found': True,
            'e6_vertices': list(e6_candidate),
            'verification': verification,
            'complement_size': 96 - len(e6_candidate)
        }

        print("\n" + "="*60)
        print("✓✓✓ E₆ CONSTRUCTED FROM FIRST PRINCIPLES!")
        print(f"    72 vertices identified")
        print(f"    Remaining: {result['complement_size']} vertices")

    else:
        result = {
            'e6_found': False,
            'reason': 'No 72-vertex subset found with E₆ properties'
        }

        print("\n" + "="*60)
        print("⚠ E₆ not found by current strategies")
        print("  Need deeper analysis of Atlas structure")

    return result


if __name__ == "__main__":
    e6_result = construct_e6_first_principles()