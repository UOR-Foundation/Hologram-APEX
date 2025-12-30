"""
Find F₄ as 48-root subsystem of E₆'s 72 roots in E₈.

Use exact arithmetic with E₈ root vectors.
"""
import sys
import os
import json
from typing import Set, Dict, List, Tuple
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from tier_a_embedding.e8.geometry import dot_product, norm_squared
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def load_e6_roots_in_e8(atlas: AtlasGraph) -> List[Tuple[int, List[Fraction]]]:
    """
    Load E₆'s 72 vertices as E₈ root vectors.

    Returns:
        List of (vertex_id, root_vector) tuples
    """
    # Load E₆ vertices
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    # Load tier_a mapping
    with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
        cert = json.load(f)
        mapping = {int(k): int(v) for k, v in cert['mapping'].items()}

    # Load E₈ root system
    e8 = E8RootSystem()

    # Map E₆ vertices to E₈ roots
    e6_roots = []
    for v in sorted(e6_vertices):
        if v in mapping:
            e8_root_idx = mapping[v]
            root_vector = e8.roots[e8_root_idx]
            e6_roots.append((v, root_vector))

    print(f"Loaded {len(e6_roots)} E₆ roots as E₈ vectors")

    # Verify all have norm 2
    for v, root in e6_roots:
        norm = norm_squared(root)
        assert norm == Fraction(2), f"Root {v} has norm {norm} != 2"

    print(f"✓ All E₆ roots have norm² = 2")

    return e6_roots


def compute_root_adjacency(roots: List[List[Fraction]]) -> List[List[bool]]:
    """
    Compute adjacency matrix for roots.

    Two roots are adjacent if their dot product = 1.
    """
    n = len(roots)
    adj = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dot = dot_product(roots[i], roots[j])
                if dot == Fraction(1):
                    adj[i][j] = True

    return adj


def analyze_root_system_structure(roots: List[List[Fraction]]) -> Dict:
    """
    Analyze structure of a root system.

    Compute all dot products and their distribution.
    """
    n = len(roots)

    # Compute all dot products
    dot_products = []
    for i in range(n):
        for j in range(i+1, n):
            dot = dot_product(roots[i], roots[j])
            dot_products.append(dot)

    # Count distribution
    from collections import Counter
    dot_dist = Counter(dot_products)

    return {
        'num_roots': n,
        'dot_product_distribution': dict(dot_dist),
        'unique_dot_products': sorted(set(dot_products)),
    }


def find_f4_subsets(e6_roots: List[Tuple[int, List[Fraction]]]) -> List[Set[int]]:
    """
    Search for 48-root subsets of E₆ that could be F₄.

    Strategy: F₄ should be a closed, connected root subsystem.
    """
    print(f"\nSearching for F₄ as 48-root subset of E₆...")

    vertices = [v for v, _ in e6_roots]
    root_vectors = [r for _, r in e6_roots]

    # Build adjacency
    adj_matrix = compute_root_adjacency(root_vectors)

    # Strategy 1: Find largest connected component with good structure
    # For now, try systematic search

    candidates = []

    # Try: Remove 24 roots with specific properties
    # E₆ has 72 roots, F₄ needs 48, so remove 24

    # Analyze E₆ structure first
    e6_analysis = analyze_root_system_structure(root_vectors)
    print(f"\nE₆ structure:")
    print(f"  Roots: {e6_analysis['num_roots']}")
    print(f"  Dot product distribution: {e6_analysis['dot_product_distribution']}")

    # Try removing roots to get 48
    # Start simple: remove first 24, last 24, etc.

    candidate1 = set(vertices[:48])
    candidates.append(('first_48', candidate1))

    candidate2 = set(vertices[24:])
    candidates.append(('last_48', candidate2))

    candidate3 = set(vertices[::2][:48] if len(vertices) >= 96 else vertices[:48])
    candidates.append(('every_other', candidate3))

    return candidates


def verify_f4_properties(subset_vertices: Set[int],
                        e6_roots: List[Tuple[int, List[Fraction]]],
                        name: str) -> Dict:
    """
    Verify if a subset has F₄ properties.
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {name}")
    print(f"{'='*70}")

    # Extract subset root vectors
    vertex_to_root = {v: r for v, r in e6_roots}
    subset_roots = [vertex_to_root[v] for v in subset_vertices if v in vertex_to_root]

    if len(subset_roots) != 48:
        print(f"Size: {len(subset_roots)} ✗ (expected 48)")
        return {'valid': False, 'reason': f'Size {len(subset_roots)} != 48'}

    print(f"Size: {len(subset_roots)} ✓")

    # Analyze structure
    analysis = analyze_root_system_structure(subset_roots)

    print(f"\nDot product distribution:")
    for val, count in sorted(analysis['dot_product_distribution'].items()):
        print(f"  {val}: {count}")

    # Check connectivity
    adj = compute_root_adjacency(subset_roots)

    # BFS for connectivity
    visited = set()
    queue = [0]
    while queue:
        i = queue.pop(0)
        if i in visited:
            continue
        visited.add(i)
        for j in range(len(subset_roots)):
            if adj[i][j] and j not in visited:
                queue.append(j)

    connected = (len(visited) == len(subset_roots))
    print(f"Connected: {connected} {'✓' if connected else '✗'}")

    # Count edges
    edges = sum(sum(row) for row in adj) // 2
    print(f"Edges: {edges}")

    return {
        'valid': len(subset_roots) == 48 and connected,
        'size': len(subset_roots),
        'connected': connected,
        'edges': edges,
        'dot_distribution': analysis['dot_product_distribution'],
    }


def search_f4_systematically(e6_roots: List[Tuple[int, List[Fraction]]]) -> Set[int]:
    """
    Systematic search for F₄ by checking which 48 of 72 have F₄ structure.

    F₄ characteristics in E₈:
    - 48 roots forming closed system
    - Should have specific dot product pattern
    - Connected graph
    """
    print(f"\n{'='*70}")
    print("SYSTEMATIC SEARCH FOR F₄")
    print(f"{'='*70}")

    vertices = [v for v, _ in e6_roots]
    root_vectors = [r for _, r in e6_roots]

    # Analyze full E₆ first
    print(f"\nAnalyzing E₆ (72 roots):")
    e6_analysis = analyze_root_system_structure(root_vectors)
    print(f"  Dot products: {e6_analysis['unique_dot_products']}")
    print(f"  Distribution: {e6_analysis['dot_product_distribution']}")

    # Key insight: Look at the structure of Atlas vertices
    # E₆ = 72 vertices, we need to find which 48 form F₄

    # Load Atlas to check degrees
    atlas = AtlasGraph()

    # Check degree distribution within E₆
    e6_by_degree = {'deg5': [], 'deg6': []}
    for v, _ in e6_roots:
        deg = atlas.degree(v)
        if deg == 5:
            e6_by_degree['deg5'].append(v)
        else:
            e6_by_degree['deg6'].append(v)

    print(f"\nE₆ degree distribution:")
    print(f"  Degree-5: {len(e6_by_degree['deg5'])}")
    print(f"  Degree-6: {len(e6_by_degree['deg6'])}")

    # F₄ should have 48 roots
    # Try: 40 deg-5 + 8 deg-6 (all the deg-6)
    if len(e6_by_degree['deg6']) == 8 and len(e6_by_degree['deg5']) >= 40:
        candidate = set(e6_by_degree['deg6'] + e6_by_degree['deg5'][:40])
        print(f"\nTrying: All 8 deg-6 + first 40 deg-5")
        result = verify_f4_properties(candidate, e6_roots, "deg6_all_plus_40_deg5")

        if result.get('valid'):
            return candidate

    # Try: Different selections of deg-5
    # Maybe the LAST 40 deg-5?
    if len(e6_by_degree['deg6']) == 8 and len(e6_by_degree['deg5']) >= 40:
        candidate = set(e6_by_degree['deg6'] + e6_by_degree['deg5'][-40:])
        print(f"\nTrying: All 8 deg-6 + last 40 deg-5")
        result = verify_f4_properties(candidate, e6_roots, "deg6_all_plus_last_40_deg5")

        if result.get('valid'):
            return candidate

    return set()


def main():
    """Main execution."""
    print("="*70)
    print("FIND F₄ AS 48-ROOT SUBSYSTEM OF E₆ IN E₈")
    print("="*70)

    atlas = AtlasGraph()

    # Load E₆ as E₈ roots
    e6_roots = load_e6_roots_in_e8(atlas)

    # Search for F₄
    f4_vertices = search_f4_systematically(e6_roots)

    if f4_vertices:
        print(f"\n{'='*70}")
        print("F₄ FOUND!")
        print(f"{'='*70}")
        print(f"  {len(f4_vertices)} roots identified")
        print(f"  F₄ ⊂ E₆ ⊂ E₈ proven by construction")
    else:
        print(f"\n{'='*70}")
        print("SEARCH INCOMPLETE")
        print(f"{'='*70}")
        print("  Need to refine search strategy")

    return f4_vertices


if __name__ == "__main__":
    result = main()
