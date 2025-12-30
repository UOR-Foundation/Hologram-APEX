"""
Atlas Structure Analysis.

Analyze Atlas to understand F₄ and E₆ constructions.
"""
import sys
import os
import json
from typing import Set, Dict, List, Tuple
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem


def analyze_degree_distribution(atlas: AtlasGraph) -> Dict:
    """Analyze degree distribution in Atlas."""
    deg5_vertices = []
    deg6_vertices = []

    for v in range(atlas.num_vertices):
        deg = atlas.degree(v)
        if deg == 5:
            deg5_vertices.append(v)
        elif deg == 6:
            deg6_vertices.append(v)

    return {
        'deg5_vertices': deg5_vertices,
        'deg6_vertices': deg6_vertices,
        'deg5_count': len(deg5_vertices),
        'deg6_count': len(deg6_vertices),
    }


def analyze_mirror_pairing(atlas: AtlasGraph) -> Dict:
    """Analyze mirror pairing structure."""
    mirror_pairs = []
    seen = set()

    for v in range(atlas.num_vertices):
        mirror = atlas.get_mirror_pair(v)
        if v not in seen:
            mirror_pairs.append((v, mirror))
            seen.add(v)
            seen.add(mirror)

    # Check degree distribution of mirror pairs
    both_deg5 = 0
    both_deg6 = 0
    mixed = 0

    for v1, v2 in mirror_pairs:
        d1, d2 = atlas.degree(v1), atlas.degree(v2)
        if d1 == 5 and d2 == 5:
            both_deg5 += 1
        elif d1 == 6 and d2 == 6:
            both_deg6 += 1
        else:
            mixed += 1

    return {
        'mirror_pairs': mirror_pairs,
        'num_pairs': len(mirror_pairs),
        'both_deg5': both_deg5,
        'both_deg6': both_deg6,
        'mixed_degree': mixed,
    }


def analyze_e8_mapping(atlas: AtlasGraph) -> Dict:
    """Analyze how Atlas vertices map to E₈ roots."""
    # Load tier_a embedding
    try:
        with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
            cert = json.load(f)
            mapping = {int(k): int(v) for k, v in cert['mapping'].items()}
    except FileNotFoundError:
        return {'error': 'tier_a_certificate.json not found'}

    e8 = E8RootSystem()

    # Check if mirror pairs map to negation pairs in E₈
    mirror_to_negation = 0
    mirror_not_negation = 0

    for v in range(atlas.num_vertices):
        mirror_v = atlas.get_mirror_pair(v)

        e8_root_v = mapping[v]
        e8_root_mirror = mapping[mirror_v]
        e8_negation_v = e8.negation_table[e8_root_v]

        if e8_root_mirror == e8_negation_v:
            mirror_to_negation += 1
        else:
            mirror_not_negation += 1

    return {
        'atlas_mirror_equals_e8_negation': mirror_to_negation == atlas.num_vertices,
        'mirror_to_negation_count': mirror_to_negation,
        'mirror_not_negation_count': mirror_not_negation,
    }


def find_f4_subgraphs(atlas: AtlasGraph) -> List[Dict]:
    """
    Search for 48-vertex subgraphs that could be F₄.

    Classical F₄ has 24 short + 24 long roots.
    We need a 48-vertex induced subgraph with the right properties.
    """
    deg_dist = analyze_degree_distribution(atlas)

    # Try different selections
    candidates = []

    # Candidate 1: 24 deg-5 + 24 deg-6 (balanced)
    if len(deg_dist['deg5_vertices']) >= 24 and len(deg_dist['deg6_vertices']) >= 24:
        # Take first 24 of each
        candidate1 = set(deg_dist['deg5_vertices'][:24] + deg_dist['deg6_vertices'][:24])
        candidates.append({
            'name': 'Balanced 24+24',
            'vertices': candidate1,
            'deg5_count': 24,
            'deg6_count': 24,
        })

    # Candidate 2: Use mirror pairing - one from each pair
    mirror_info = analyze_mirror_pairing(atlas)
    if len(mirror_info['mirror_pairs']) == 48:
        # Take one representative from each mirror pair
        candidate2 = set([pair[0] for pair in mirror_info['mirror_pairs']])
        deg5_in_c2 = sum(1 for v in candidate2 if atlas.degree(v) == 5)
        deg6_in_c2 = sum(1 for v in candidate2 if atlas.degree(v) == 6)
        candidates.append({
            'name': 'Mirror quotient (one per pair)',
            'vertices': candidate2,
            'deg5_count': deg5_in_c2,
            'deg6_count': deg6_in_c2,
        })

    # Candidate 3: Connected deg-5 heavy subgraph
    # Start from a deg-5 vertex and grow
    if deg_dist['deg5_vertices']:
        candidate3 = grow_connected_subgraph(atlas, deg_dist['deg5_vertices'][0], target_size=48)
        deg5_in_c3 = sum(1 for v in candidate3 if atlas.degree(v) == 5)
        deg6_in_c3 = sum(1 for v in candidate3 if atlas.degree(v) == 6)
        candidates.append({
            'name': 'Connected growth from deg-5',
            'vertices': candidate3,
            'deg5_count': deg5_in_c3,
            'deg6_count': deg6_in_c3,
        })

    return candidates


def grow_connected_subgraph(atlas: AtlasGraph, start: int, target_size: int) -> Set[int]:
    """Grow a connected subgraph by BFS."""
    subgraph = {start}
    frontier = {start}

    while len(subgraph) < target_size and frontier:
        new_frontier = set()
        for v in frontier:
            for neighbor in atlas.neighbors(v):
                if neighbor not in subgraph:
                    subgraph.add(neighbor)
                    new_frontier.add(neighbor)
                    if len(subgraph) >= target_size:
                        return subgraph
        frontier = new_frontier

    return subgraph


def check_subgraph_in_e6(subgraph: Set[int], e6_vertices: Set[int]) -> Dict:
    """Check how many vertices of subgraph are in E₆."""
    overlap = subgraph & e6_vertices
    missing = subgraph - e6_vertices

    return {
        'overlap_count': len(overlap),
        'missing_count': len(missing),
        'overlap_percentage': len(overlap) / len(subgraph) * 100 if subgraph else 0,
        'is_subset': missing == set(),
    }


def main():
    """Main analysis function."""
    print("="*70)
    print("ATLAS STRUCTURE ANALYSIS FOR F₄ ⊂ E₆")
    print("="*70)

    atlas = AtlasGraph()

    # 1. Degree distribution
    print("\n1. DEGREE DISTRIBUTION")
    print("-"*70)
    deg_dist = analyze_degree_distribution(atlas)
    print(f"Degree-5 vertices: {deg_dist['deg5_count']}")
    print(f"Degree-6 vertices: {deg_dist['deg6_count']}")
    print(f"Total: {deg_dist['deg5_count'] + deg_dist['deg6_count']}")

    # 2. Mirror pairing
    print("\n2. MIRROR PAIRING STRUCTURE")
    print("-"*70)
    mirror_info = analyze_mirror_pairing(atlas)
    print(f"Number of mirror pairs: {mirror_info['num_pairs']}")
    print(f"Both deg-5: {mirror_info['both_deg5']}")
    print(f"Both deg-6: {mirror_info['both_deg6']}")
    print(f"Mixed degree: {mirror_info['mixed_degree']}")

    # 3. E₈ mapping
    print("\n3. E₈ MAPPING ANALYSIS")
    print("-"*70)
    e8_info = analyze_e8_mapping(atlas)
    if 'error' not in e8_info:
        print(f"Atlas mirror = E₈ negation: {e8_info['atlas_mirror_equals_e8_negation']}")
        print(f"Match count: {e8_info['mirror_to_negation_count']}/96")
        print(f"Mismatch count: {e8_info['mirror_not_negation_count']}/96")
    else:
        print(f"Error: {e8_info['error']}")

    # 4. Load E₆
    print("\n4. E₆ STRUCTURE")
    print("-"*70)
    from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    if e6_vertices:
        deg5_in_e6 = sum(1 for v in e6_vertices if atlas.degree(v) == 5)
        deg6_in_e6 = sum(1 for v in e6_vertices if atlas.degree(v) == 6)
        print(f"E₆ vertices: {len(e6_vertices)}")
        print(f"  Degree-5: {deg5_in_e6}")
        print(f"  Degree-6: {deg6_in_e6}")

    # 5. F₄ candidates
    print("\n5. F₄ SUBGRAPH CANDIDATES")
    print("-"*70)
    candidates = find_f4_subgraphs(atlas)

    for i, candidate in enumerate(candidates, 1):
        print(f"\nCandidate {i}: {candidate['name']}")
        print(f"  Size: {len(candidate['vertices'])}")
        print(f"  Degree-5: {candidate['deg5_count']}")
        print(f"  Degree-6: {candidate['deg6_count']}")

        # Check overlap with E₆
        if e6_vertices:
            overlap_info = check_subgraph_in_e6(candidate['vertices'], e6_vertices)
            print(f"  Overlap with E₆: {overlap_info['overlap_count']}/{len(candidate['vertices'])}")
            print(f"  Subset of E₆: {overlap_info['is_subset']}")
            print(f"  Percentage: {overlap_info['overlap_percentage']:.1f}%")

    # 6. Key findings
    print("\n6. KEY FINDINGS")
    print("-"*70)
    print(f"✓ Atlas has {atlas.num_vertices} vertices (48 mirror pairs)")
    print(f"✓ Degree distribution: {deg_dist['deg5_count']} deg-5, {deg_dist['deg6_count']} deg-6")

    if e8_info.get('atlas_mirror_equals_e8_negation'):
        print(f"✓ Atlas mirror = E₈ negation (quotient is well-defined)")
    else:
        print(f"⚠ Atlas mirror ≠ E₈ negation (may cause issues)")

    print(f"\n→ Need to find the correct 48-vertex subset for F₄")
    print(f"→ Classical F₄ has 24 short + 24 long (1:1 ratio)")
    print(f"→ Current quotient gives 32+16 (2:1 ratio) - mismatch!")

    return {
        'degree_distribution': deg_dist,
        'mirror_pairing': mirror_info,
        'e8_mapping': e8_info,
        'e6_vertices': e6_vertices,
        'f4_candidates': candidates,
    }


if __name__ == "__main__":
    results = main()
