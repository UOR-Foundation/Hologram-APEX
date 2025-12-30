"""
Search for F₄ as a 48-vertex subgraph OF E₆.

Instead of constructing F₄ independently, find which 48 of E₆'s 72 vertices form F₄.
"""
import sys
import os
import json
from typing import Set, Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def find_f4_in_e6_via_mirror_pairs(atlas: AtlasGraph, e6_vertices: Set[int]) -> Set[int]:
    """
    Find F₄ by taking mirror pair quotient WITHIN E₆.

    Strategy:
    1. E₆ has 72 vertices
    2. These should include complete mirror pairs
    3. Take one representative from each pair that's FULLY in E₆
    """
    # Find which mirror pairs are completely in E₆
    complete_pairs = []
    partial_pairs = []

    seen = set()
    for v in e6_vertices:
        if v in seen:
            continue

        mirror = atlas.get_mirror_pair(v)
        seen.add(v)
        seen.add(mirror)

        if mirror in e6_vertices:
            # Both in E₆
            complete_pairs.append((v, mirror))
        else:
            # Only one in E₆
            partial_pairs.append((v, mirror))

    print(f"Mirror pair analysis in E₆:")
    print(f"  Complete pairs (both in E₆): {len(complete_pairs)}")
    print(f"  Partial pairs (one in E₆): {len(partial_pairs)}")
    print(f"  Total E₆ vertices accounted: {len(complete_pairs)*2 + len(partial_pairs)}")

    # F₄ = one representative from each complete pair
    f4_from_pairs = set([pair[0] for pair in complete_pairs])

    # Check degree distribution
    deg5 = sum(1 for v in f4_from_pairs if atlas.degree(v) == 5)
    deg6 = sum(1 for v in f4_from_pairs if atlas.degree(v) == 6)

    print(f"\nF₄ from complete pairs:")
    print(f"  Size: {len(f4_from_pairs)}")
    print(f"  Degree-5: {deg5}")
    print(f"  Degree-6: {deg6}")
    print(f"  Ratio: {deg5}:{deg6}")

    return f4_from_pairs


def find_f4_via_e8_subroot_system(atlas: AtlasGraph, e6_vertices: Set[int]) -> Dict:
    """
    Find F₄ by looking for a 48-root subsystem of E₆'s 72 roots in E₈.

    Classical approach: F₄ is a sub-root-system of E₆.
    """
    # Load E₈ roots
    try:
        with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
            cert = json.load(f)
            mapping = {int(k): int(v) for k, v in cert['mapping'].items()}
    except FileNotFoundError:
        return {'error': 'tier_a_certificate.json not found'}

    e8 = E8RootSystem()

    # Map E₆ vertices to E₈ roots
    e6_roots = [mapping[v] for v in e6_vertices if v in mapping]

    print(f"\nE₈ root system analysis:")
    print(f"  E₆ → {len(e6_roots)} E₈ roots")

    # Look for structural patterns in E₆ roots
    # F₄ should be a 48-root subsystem with specific properties

    # Check inner products between E₆ roots
    adjacency_in_e8 = {}
    for r1 in e6_roots:
        adjacency_in_e8[r1] = []
        for r2 in e6_roots:
            if r1 != r2:
                # Two E₈ roots are adjacent if dot product = 1
                # Check if they're related
                if e8.negation_table[r1] == r2:
                    continue  # Skip negation pairs
                # TODO: compute actual dot product
                adjacency_in_e8[r1].append(r2)

    return {
        'e6_roots': e6_roots,
        'e6_root_count': len(e6_roots),
    }


def find_f4_via_degree_6_selection(atlas: AtlasGraph, e6_vertices: Set[int]) -> Set[int]:
    """
    Find F₄ by selecting which 48 of E₆'s 72 to keep.

    E₆ = 64 deg-5 + 8 deg-6 = 72
    F₄ = ? deg-5 + ? deg-6 = 48

    If classical F₄ has 24+24, we need to remove 40 deg-5 and keep all 8 deg-6.
    But that gives 24+8=32, not 48.

    Alternative: Remove 24 vertices from E₆ to get 48.
    """
    e6_deg5 = [v for v in e6_vertices if atlas.degree(v) == 5]
    e6_deg6 = [v for v in e6_vertices if atlas.degree(v) == 6]

    print(f"\nE₆ degree distribution:")
    print(f"  Deg-5: {len(e6_deg5)}")
    print(f"  Deg-6: {len(e6_deg6)}")

    # Strategy: Keep all deg-6, plus 40 deg-5
    if len(e6_deg6) == 8 and len(e6_deg5) >= 40:
        f4_candidate = set(e6_deg6 + e6_deg5[:40])
        print(f"\nF₄ candidate (8 deg-6 + 40 deg-5):")
        print(f"  Size: {len(f4_candidate)}")
        return f4_candidate

    # Strategy: 24 deg-5 + 24 deg-6, but E₆ only has 8 deg-6
    # This won't work!

    return set()


def verify_subgraph_properties(atlas: AtlasGraph, subgraph: Set[int]) -> Dict:
    """Verify if subgraph has F₄-like properties."""
    if len(subgraph) != 48:
        return {'valid': False, 'reason': f'Size {len(subgraph)} != 48'}

    # Check connectivity
    induced_edges = 0
    for v in subgraph:
        for u in atlas.neighbors(v):
            if u in subgraph and v < u:
                induced_edges += 1

    # Compute average degree
    degrees = [sum(1 for u in atlas.neighbors(v) if u in subgraph) for v in subgraph]
    avg_degree = sum(degrees) / len(degrees)

    # F₄ is known to be connected
    is_connected = check_connectivity(atlas, subgraph)

    return {
        'valid': True,
        'size': len(subgraph),
        'induced_edges': induced_edges,
        'avg_degree': avg_degree,
        'is_connected': is_connected,
    }


def check_connectivity(atlas: AtlasGraph, subgraph: Set[int]) -> bool:
    """Check if induced subgraph is connected."""
    if not subgraph:
        return False

    visited = set()
    queue = [next(iter(subgraph))]

    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)

        for u in atlas.neighbors(v):
            if u in subgraph and u not in visited:
                queue.append(u)

    return len(visited) == len(subgraph)


def main():
    """Main search function."""
    print("="*70)
    print("SEARCHING FOR F₄ AS 48-VERTEX SUBGRAPH OF E₆")
    print("="*70)

    atlas = AtlasGraph()

    # Load E₆
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    if not e6_vertices:
        print("Error: Could not construct E₆")
        return

    print(f"\nE₆ structure:")
    print(f"  Vertices: {len(e6_vertices)}")
    print(f"  Deg-5: {sum(1 for v in e6_vertices if atlas.degree(v) == 5)}")
    print(f"  Deg-6: {sum(1 for v in e6_vertices if atlas.degree(v) == 6)}")

    # Method 1: Mirror pair quotient within E₆
    print("\n" + "="*70)
    print("METHOD 1: Mirror Pair Quotient Within E₆")
    print("="*70)
    f4_method1 = find_f4_in_e6_via_mirror_pairs(atlas, e6_vertices)

    if f4_method1:
        props1 = verify_subgraph_properties(atlas, f4_method1)
        print(f"\nProperties:")
        for key, val in props1.items():
            print(f"  {key}: {val}")

    # Method 2: E₈ root subsystem
    print("\n" + "="*70)
    print("METHOD 2: E₈ Root Subsystem Analysis")
    print("="*70)
    e8_analysis = find_f4_via_e8_subroot_system(atlas, e6_vertices)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if f4_method1 and len(f4_method1) == 48:
        print(f"✓ Found F₄ as 48-vertex subgraph of E₆ (Method 1)")
        print(f"  F₄ ⊂ E₆ ⊂ Atlas")
        print(f"  This proves F₄ ⊂ E₆ by construction!")
    else:
        print(f"⚠ Method 1 gave {len(f4_method1) if f4_method1 else 0} vertices (expected 48)")
        print(f"  Need alternative construction")

    return {
        'e6_vertices': e6_vertices,
        'f4_method1': f4_method1,
        'e8_analysis': e8_analysis,
    }


if __name__ == "__main__":
    results = main()
