"""
Find F₄ as a 48-vertex connected subgraph of Atlas (not necessarily subset of E₆).

Key insight: Maybe F₄ uses vertices from BOTH E₆ and its complement.
"""
import sys
import os
import json
from typing import Set, Dict, List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def find_maximal_connected_subgraphs(atlas: AtlasGraph, size: int = 48) -> List[Set[int]]:
    """
    Find connected subgraphs of specific size via greedy search.

    Start from different seed vertices and grow connected components.
    """
    candidates = []

    # Try starting from different vertices
    for seed in range(min(20, atlas.num_vertices)):  # Try first 20 vertices as seeds
        subgraph = {seed}
        frontier = {seed}

        while len(subgraph) < size and frontier:
            # Find vertex in frontier with most neighbors not yet in subgraph
            best_expand = None
            best_new_neighbors = 0

            for v in frontier:
                new_neighbors = [u for u in atlas.neighbors(v) if u not in subgraph]
                if len(new_neighbors) > best_new_neighbors:
                    best_new_neighbors = len(new_neighbors)
                    best_expand = v

            if best_expand is None or best_new_neighbors == 0:
                break

            # Add neighbors of best vertex
            for u in atlas.neighbors(best_expand):
                if u not in subgraph and len(subgraph) < size:
                    subgraph.add(u)
                    frontier.add(u)

            frontier.discard(best_expand)

        if len(subgraph) == size:
            # Check if connected
            if is_connected(atlas, subgraph):
                # Check if we already have this one
                is_new = True
                for existing in candidates:
                    if existing == subgraph:
                        is_new = False
                        break
                if is_new:
                    candidates.append(subgraph)

    return candidates


def is_connected(atlas: AtlasGraph, vertices: Set[int]) -> bool:
    """Check if induced subgraph is connected."""
    if not vertices:
        return False

    visited = set()
    queue = [next(iter(vertices))]

    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)

        for u in atlas.neighbors(v):
            if u in vertices and u not in visited:
                queue.append(u)

    return len(visited) == len(vertices)


def analyze_f4_from_sign_classes(atlas: AtlasGraph) -> Dict:
    """Analyze our existing F₄ from sign classes."""
    from exceptional_groups.f4.sign_class_analysis import F4SignClassAnalyzer

    analyzer = F4SignClassAnalyzer()
    f4 = analyzer.extract_sign_classes()

    # sign_classes are E₈ root indices
    # Need to map back to Atlas vertices

    try:
        with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
            cert = json.load(f)
            mapping = {int(k): int(v) for k, v in cert['mapping'].items()}
            reverse_mapping = {v: k for k, v in mapping.items()}
    except:
        return {'error': 'Cannot load mapping'}

    # Map sign class representatives to Atlas vertices
    f4_vertices = set()
    for sign_class_root in f4.sign_classes:
        if sign_class_root in reverse_mapping:
            f4_vertices.add(reverse_mapping[sign_class_root])

    # Check if connected in Atlas
    connected = is_connected(atlas, f4_vertices)

    deg5 = sum(1 for v in f4_vertices if atlas.degree(v) == 5)
    deg6 = sum(1 for v in f4_vertices if atlas.degree(v) == 6)

    return {
        'vertices': f4_vertices,
        'size': len(f4_vertices),
        'connected': connected,
        'deg5': deg5,
        'deg6': deg6,
    }


def check_overlap_with_e6(f4_vertices: Set[int], e6_vertices: Set[int]) -> Dict:
    """Check how F₄ overlaps with E₆."""
    overlap = f4_vertices & e6_vertices
    f4_only = f4_vertices - e6_vertices
    e6_only = e6_vertices - f4_vertices

    return {
        'overlap': len(overlap),
        'f4_only': len(f4_only),
        'e6_only': len(e6_only),
        'f4_in_e6': (f4_only == set()),
        'overlap_pct': len(overlap) / len(f4_vertices) * 100 if f4_vertices else 0,
    }


def main():
    """Main analysis."""
    print("="*70)
    print("F₄ AS CONNECTED SUBGRAPH OF ATLAS")
    print("="*70)

    atlas = AtlasGraph()

    # Load E₆
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()
    complement = set(range(96)) - e6_vertices

    print(f"\nAtlas structure:")
    print(f"  Total vertices: 96")
    print(f"  E₆ vertices: {len(e6_vertices)}")
    print(f"  Complement: {len(complement)}")

    # Analyze our known F₄ from sign classes
    print(f"\n{'='*70}")
    print("KNOWN F₄ FROM SIGN CLASSES")
    print(f"{'='*70}")

    f4_known = analyze_f4_from_sign_classes(atlas)

    if 'error' not in f4_known:
        print(f"Size: {f4_known['size']}")
        print(f"Connected: {f4_known['connected']}")
        print(f"Degrees: {f4_known['deg5']} deg-5, {f4_known['deg6']} deg-6")

        # Check overlap with E₆
        overlap_info = check_overlap_with_e6(f4_known['vertices'], e6_vertices)
        print(f"\nOverlap with E₆:")
        print(f"  In both F₄ and E₆: {overlap_info['overlap']}")
        print(f"  F₄ only: {overlap_info['f4_only']}")
        print(f"  E₆ only: {overlap_info['e6_only']}")
        print(f"  F₄ ⊂ E₆: {overlap_info['f4_in_e6']}")
        print(f"  Overlap: {overlap_info['overlap_pct']:.1f}%")

        # Check where F₄-only vertices are
        if overlap_info['f4_only'] > 0:
            f4_only_vertices = f4_known['vertices'] - e6_vertices
            in_complement = sum(1 for v in f4_only_vertices if v in complement)
            print(f"\nF₄-only vertices ({overlap_info['f4_only']}):")
            print(f"  In complement: {in_complement}/{overlap_info['f4_only']}")

            # Check degrees
            deg5_only = sum(1 for v in f4_only_vertices if atlas.degree(v) == 5)
            deg6_only = sum(1 for v in f4_only_vertices if atlas.degree(v) == 6)
            print(f"  Degrees: {deg5_only} deg-5, {deg6_only} deg-6")

    else:
        print(f"Error: {f4_known['error']}")

    # Search for alternative connected F₄ subgraphs
    print(f"\n{'='*70}")
    print("SEARCHING FOR CONNECTED 48-VERTEX SUBGRAPHS")
    print(f"{'='*70}")
    print("(This may take a moment...)")

    candidates = find_maximal_connected_subgraphs(atlas, size=48)

    print(f"\nFound {len(candidates)} connected 48-vertex subgraphs")

    for i, candidate in enumerate(candidates[:5], 1):  # Show first 5
        print(f"\nCandidate {i}:")
        deg5 = sum(1 for v in candidate if atlas.degree(v) == 5)
        deg6 = sum(1 for v in candidate if atlas.degree(v) == 6)
        print(f"  Degrees: {deg5} deg-5, {deg6} deg-6")

        overlap_info = check_overlap_with_e6(candidate, e6_vertices)
        print(f"  Overlap with E₆: {overlap_info['overlap']}/48 ({overlap_info['overlap_pct']:.1f}%)")
        print(f"  F₄ ⊂ E₆: {overlap_info['f4_in_e6']}")

    # Summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    if 'vertices' in f4_known:
        if f4_known['connected']:
            print(f"✓ Sign class F₄ IS connected in Atlas")
        else:
            print(f"✗ Sign class F₄ is NOT connected in Atlas")

        overlap_info = check_overlap_with_e6(f4_known['vertices'], e6_vertices)
        if overlap_info['f4_in_e6']:
            print(f"✓ F₄ ⊂ E₆ (all 48 vertices)")
        else:
            print(f"⚠ F₄ ⊄ E₆ ({overlap_info['f4_only']} vertices outside E₆)")
            print(f"  → F₄ uses vertices from both E₆ AND its complement")

    return {
        'f4_known': f4_known,
        'e6_vertices': e6_vertices,
        'complement': complement,
        'candidates': candidates,
    }


if __name__ == "__main__":
    results = main()
