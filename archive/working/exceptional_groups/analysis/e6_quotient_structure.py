"""
Analyze E₆ as a quotient structure.

We found E₆ = 72 vertices = 36 complete mirror pairs.
Can we define an E₆ quotient graph on these 36 pairs?
"""
import sys
import os
import json
from typing import Set, Dict, Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def build_e6_quotient_graph(atlas: AtlasGraph, e6_vertices: Set[int]) -> Dict:
    """
    Build quotient graph for E₆ by taking one representative from each mirror pair.

    E₆ = 72 vertices = 36 mirror pairs
    → E₆ quotient = 36 sign classes
    """
    # Find mirror pairs within E₆
    mirror_pairs = []
    seen = set()

    for v in e6_vertices:
        if v in seen:
            continue

        mirror = atlas.get_mirror_pair(v)
        seen.add(v)
        seen.add(mirror)

        if mirror in e6_vertices:
            # Complete pair
            mirror_pairs.append((min(v, mirror), max(v, mirror)))

    print(f"E₆ quotient construction:")
    print(f"  E₆ vertices: {len(e6_vertices)}")
    print(f"  Mirror pairs: {len(mirror_pairs)}")
    print(f"  Quotient size: {len(mirror_pairs)}")

    # Build quotient graph
    # Representatives: one from each pair (take the smaller index)
    representatives = [pair[0] for pair in mirror_pairs]

    # Build adjacency for quotient
    # Two sign classes are adjacent if any representative of one class
    # is adjacent to any representative of the other class
    n = len(representatives)
    quotient_adj = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Get both vertices in each pair
            v1_a, v1_b = mirror_pairs[i]
            v2_a, v2_b = mirror_pairs[j]

            # Check if any are adjacent in Atlas
            if (v2_a in atlas.neighbors(v1_a) or
                v2_a in atlas.neighbors(v1_b) or
                v2_b in atlas.neighbors(v1_a) or
                v2_b in atlas.neighbors(v1_b)):
                quotient_adj[i][j] = True

    # Check connectivity
    visited = set()
    queue = [0]
    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)
        for u in range(n):
            if quotient_adj[v][u] and u not in visited:
                queue.append(u)

    is_connected = (len(visited) == n)

    # Count edges
    edges = sum(sum(row) for row in quotient_adj) // 2

    # Degree sequence
    degrees = [sum(row) for row in quotient_adj]

    return {
        'quotient_size': n,
        'representatives': representatives,
        'adjacency_matrix': quotient_adj,
        'edges': edges,
        'connected': is_connected,
        'degrees': degrees,
        'avg_degree': sum(degrees) / n if n > 0 else 0,
    }


def compare_quotients(f4_quotient: Dict, e6_quotient: Dict) -> Dict:
    """
    Compare F₄ and E₆ quotient structures.

    Classical: F₄ ⊂ E₆ means F₄ is a sub-root-system.
    In quotient space: Does F₄ quotient embed in E₆ quotient?
    """
    print(f"\n{'='*70}")
    print("QUOTIENT COMPARISON")
    print(f"{'='*70}")

    print(f"\nF₄ quotient:")
    print(f"  Size: {f4_quotient['size']}")
    print(f"  Connected: {f4_quotient['connected']}")
    print(f"  Edges: {f4_quotient['edges']}")
    print(f"  Avg degree: {f4_quotient['avg_degree']:.2f}")

    print(f"\nE₆ quotient:")
    print(f"  Size: {e6_quotient['quotient_size']}")
    print(f"  Connected: {e6_quotient['connected']}")
    print(f"  Edges: {e6_quotient['edges']}")
    print(f"  Avg degree: {e6_quotient['avg_degree']:.2f}")

    # Can F₄ (48) be a subset of E₆ (36)?
    # NO! 48 > 36

    print(f"\nInclusion analysis:")
    print(f"  F₄ quotient size: {f4_quotient['size']}")
    print(f"  E₆ quotient size: {e6_quotient['quotient_size']}")
    print(f"  F₄ ⊂ E₆ possible: {f4_quotient['size'] <= e6_quotient['quotient_size']}")

    if f4_quotient['size'] > e6_quotient['quotient_size']:
        print(f"\n⚠ F₄ quotient (48) CANNOT be subset of E₆ quotient (36)")
        print(f"  → F₄ has more equivalence classes than E₆!")
        print(f"  → This is a FUNDAMENTAL structural issue")

    return {
        'f4_size': f4_quotient['size'],
        'e6_size': e6_quotient['quotient_size'],
        'inclusion_possible': f4_quotient['size'] <= e6_quotient['quotient_size'],
    }


def load_f4_quotient() -> Dict:
    """Load F₄ quotient structure."""
    from exceptional_groups.f4.sign_class_analysis import F4SignClassAnalyzer

    analyzer = F4SignClassAnalyzer()
    f4 = analyzer.extract_sign_classes()

    # Check connectivity
    n = len(f4.adjacency_matrix)
    visited = set()
    queue = [0]
    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)
        for u in range(n):
            if f4.adjacency_matrix[v][u] and u not in visited:
                queue.append(u)

    is_connected = (len(visited) == n)

    edges = sum(sum(row) for row in f4.adjacency_matrix) // 2
    degrees = [sum(row) for row in f4.adjacency_matrix]

    return {
        'size': len(f4.sign_classes),
        'adjacency_matrix': f4.adjacency_matrix,
        'edges': edges,
        'connected': is_connected,
        'degrees': degrees,
        'avg_degree': sum(degrees) / len(degrees),
    }


def main():
    """Main analysis."""
    print("="*70)
    print("E₆ AS QUOTIENT STRUCTURE")
    print("="*70)

    atlas = AtlasGraph()

    # Load E₆
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    # Build E₆ quotient
    e6_quotient = build_e6_quotient_graph(atlas, e6_vertices)

    print(f"\nE₆ quotient graph:")
    print(f"  Vertices: {e6_quotient['quotient_size']}")
    print(f"  Edges: {e6_quotient['edges']}")
    print(f"  Connected: {e6_quotient['connected']} {'✓' if e6_quotient['connected'] else '✗'}")
    print(f"  Average degree: {e6_quotient['avg_degree']:.2f}")

    # Load F₄ quotient
    f4_quotient = load_f4_quotient()

    # Compare
    comparison = compare_quotients(f4_quotient, e6_quotient)

    # KEY INSIGHT
    print(f"\n{'='*70}")
    print("KEY STRUCTURAL INSIGHT")
    print(f"{'='*70}")

    print(f"\nAtlas (96 vertices) = 48 mirror pairs")
    print(f"  → Atlas quotient = 48 sign classes")
    print(f"  → This gives F₄!")

    print(f"\nE₆ (72 vertices) = 36 mirror pairs")
    print(f"  → E₆ quotient = 36 sign classes")
    print(f"  → This is E₆ in quotient space")

    print(f"\nF₄ ⊂ E₆ problem:")
    print(f"  48 sign classes ⊄ 36 sign classes")
    print(f"  → F₄ quotient CANNOT be a subset of E₆ quotient!")

    print(f"\nConclusion:")
    print(f"  The classical F₄ ⊂ E₆ embedding is NOT a simple subset inclusion.")
    print(f"  It requires Lie algebra structure (Cartan matrices, root systems).")
    print(f"  Our categorical constructions (quotient vs filtration) don't compose!")

    return {
        'e6_quotient': e6_quotient,
        'f4_quotient': f4_quotient,
        'comparison': comparison,
    }


if __name__ == "__main__":
    results = main()
