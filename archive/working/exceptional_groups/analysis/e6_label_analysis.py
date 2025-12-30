"""
Analyze E₆ vertex labels to find F₄ structure.

Since E₆ = 72 vertices (36 pairs) and F₄ needs 48 vertices,
F₄ cannot be a quotient. It must be a direct subset.

Look for structural patterns in labels.
"""
import sys
import os
from typing import Set, Dict, List
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def analyze_labels(atlas: AtlasGraph, vertices: Set[int]) -> Dict:
    """Analyze the structure of vertex labels."""
    labels = [atlas.get_label(v) for v in vertices]

    # Labels are (e1, e2, e3, d45, e6, e7)
    e1_dist = Counter([lab[0] for lab in labels])
    e2_dist = Counter([lab[1] for lab in labels])
    e3_dist = Counter([lab[2] for lab in labels])
    d45_dist = Counter([lab[3] for lab in labels])
    e6_dist = Counter([lab[4] for lab in labels])
    e7_dist = Counter([lab[5] for lab in labels])

    return {
        'e1': dict(e1_dist),
        'e2': dict(e2_dist),
        'e3': dict(e3_dist),
        'd45': dict(d45_dist),
        'e6': dict(e6_dist),
        'e7': dict(e7_dist),
    }


def find_f4_by_coordinate_filter(atlas: AtlasGraph, e6_vertices: Set[int]) -> List[Dict]:
    """
    Find 48-vertex subsets of E₆ by filtering on coordinates.

    Try various coordinate conditions.
    """
    candidates = []

    e6_list = list(e6_vertices)
    labels = [atlas.get_label(v) for v in e6_list]

    # E7 coordinate splits Atlas into two halves
    e7_0 = set([v for v in e6_list if atlas.get_label(v)[5] == 0])
    e7_1 = set([v for v in e6_list if atlas.get_label(v)[5] == 1])

    print(f"E7 coordinate split:")
    print(f"  e7=0: {len(e7_0)} vertices")
    print(f"  e7=1: {len(e7_1)} vertices")

    # E6 coordinate
    e6_0 = set([v for v in e6_list if atlas.get_label(v)[4] == 0])
    e6_1 = set([v for v in e6_list if atlas.get_label(v)[4] == 1])

    print(f"\nE6 coordinate split:")
    print(f"  e6=0: {len(e6_0)} vertices")
    print(f"  e6=1: {len(e6_1)} vertices")

    # Try combinations
    candidates.append({
        'name': 'e7=0',
        'vertices': e7_0,
        'size': len(e7_0),
    })

    candidates.append({
        'name': 'e7=1',
        'vertices': e7_1,
        'size': len(e7_1),
    })

    candidates.append({
        'name': 'e6=0',
        'vertices': e6_0,
        'size': len(e6_0),
    })

    candidates.append({
        'name': 'e6=1',
        'vertices': e6_1,
        'size': len(e6_1),
    })

    # Try intersection/union
    e6_0_e7_0 = e6_0 & e7_0
    e6_0_e7_1 = e6_0 & e7_1
    e6_1_e7_0 = e6_1 & e7_0
    e6_1_e7_1 = e6_1 & e7_1

    print(f"\nCombinations:")
    print(f"  e6=0 ∧ e7=0: {len(e6_0_e7_0)}")
    print(f"  e6=0 ∧ e7=1: {len(e6_0_e7_1)}")
    print(f"  e6=1 ∧ e7=0: {len(e6_1_e7_0)}")
    print(f"  e6=1 ∧ e7=1: {len(e6_1_e7_1)}")

    candidates.append({'name': 'e6=0 ∧ e7=0', 'vertices': e6_0_e7_0, 'size': len(e6_0_e7_0)})
    candidates.append({'name': 'e6=0 ∧ e7=1', 'vertices': e6_0_e7_1, 'size': len(e6_0_e7_1)})
    candidates.append({'name': 'e6=1 ∧ e7=0', 'vertices': e6_1_e7_0, 'size': len(e6_1_e7_0)})
    candidates.append({'name': 'e6=1 ∧ e7=1', 'vertices': e6_1_e7_1, 'size': len(e6_1_e7_1)})

    # Union to get 48?
    union1 = e6_0_e7_0 | e6_0_e7_1
    union2 = e6_1_e7_0 | e6_1_e7_1

    print(f"\nUnions:")
    print(f"  (e6=0 ∧ e7=0) ∪ (e6=0 ∧ e7=1) = e6=0: {len(union1)}")
    print(f"  (e6=1 ∧ e7=0) ∪ (e6=1 ∧ e7=1) = e6=1: {len(union2)}")

    return [c for c in candidates if c['size'] == 48]


def find_f4_by_removing_vertices(atlas: AtlasGraph, e6_vertices: Set[int]) -> List[Dict]:
    """
    Find F₄ by removing 24 vertices from E₆.

    E₆ = 72 vertices
    F₄ = 48 vertices
    Need to remove: 24 vertices

    What structure do the 24 removed vertices have?
    """
    candidates = []

    e6_list = list(e6_vertices)

    # Strategy 1: Remove vertices with specific coordinate pattern
    # Remove smallest d45 values?
    sorted_by_d45 = sorted(e6_list, key=lambda v: atlas.get_label(v)[3])

    # Remove first 24
    f4_keep_high_d45 = set(sorted_by_d45[24:])
    candidates.append({
        'name': 'Remove 24 lowest d45',
        'vertices': f4_keep_high_d45,
        'size': len(f4_keep_high_d45),
    })

    # Remove last 24
    f4_keep_low_d45 = set(sorted_by_d45[:48])
    candidates.append({
        'name': 'Remove 24 highest d45',
        'vertices': f4_keep_low_d45,
        'size': len(f4_keep_low_d45),
    })

    # Strategy 2: Remove by degree
    # E₆ has 64 deg-5 + 8 deg-6
    # To get 48: maybe 40 deg-5 + 8 deg-6?
    deg5_in_e6 = [v for v in e6_vertices if atlas.degree(v) == 5]
    deg6_in_e6 = [v for v in e6_vertices if atlas.degree(v) == 6]

    if len(deg5_in_e6) >= 40 and len(deg6_in_e6) >= 8:
        f4_40_8 = set(deg5_in_e6[:40] + deg6_in_e6[:8])
        candidates.append({
            'name': '40 deg-5 + 8 deg-6',
            'vertices': f4_40_8,
            'size': len(f4_40_8),
        })

    # Strategy 3: Remove by label symmetry
    # Remove vertices where e1=1?
    e1_0 = set([v for v in e6_vertices if atlas.get_label(v)[0] == 0])
    candidates.append({
        'name': 'e1=0 only',
        'vertices': e1_0,
        'size': len(e1_0),
    })

    return [c for c in candidates if c['size'] == 48]


def main():
    """Main analysis."""
    print("="*70)
    print("E₆ LABEL ANALYSIS TO FIND F₄")
    print("="*70)

    atlas = AtlasGraph()

    # Load E₆
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    print(f"\nE₆: {len(e6_vertices)} vertices")

    # Analyze label distribution
    print("\n" + "-"*70)
    print("LABEL COORDINATE DISTRIBUTION IN E₆")
    print("-"*70)
    label_dist = analyze_labels(atlas, e6_vertices)
    for coord, dist in label_dist.items():
        print(f"{coord}: {dist}")

    # Method 1: Coordinate filters
    print("\n" + "="*70)
    print("METHOD 1: Coordinate Filtering")
    print("="*70)
    coord_candidates = find_f4_by_coordinate_filter(atlas, e6_vertices)

    print(f"\nFound {len(coord_candidates)} candidates with size 48:")
    for c in coord_candidates:
        print(f"  - {c['name']}: {c['size']} vertices")
        deg5 = sum(1 for v in c['vertices'] if atlas.degree(v) == 5)
        deg6 = sum(1 for v in c['vertices'] if atlas.degree(v) == 6)
        print(f"    Degrees: {deg5} deg-5, {deg6} deg-6")

    # Method 2: Removal strategies
    print("\n" + "="*70)
    print("METHOD 2: Remove 24 from E₆")
    print("="*70)
    removal_candidates = find_f4_by_removing_vertices(atlas, e6_vertices)

    print(f"\nFound {len(removal_candidates)} candidates with size 48:")
    for c in removal_candidates:
        print(f"  - {c['name']}: {c['size']} vertices")
        deg5 = sum(1 for v in c['vertices'] if atlas.degree(v) == 5)
        deg6 = sum(1 for v in c['vertices'] if atlas.degree(v) == 6)
        print(f"    Degrees: {deg5} deg-5, {deg6} deg-6")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_candidates = coord_candidates + removal_candidates

    if all_candidates:
        print(f"✓ Found {len(all_candidates)} candidate F₄ constructions")
        print(f"  All are 48-vertex subsets of E₆'s 72 vertices")
        print(f"  Need to verify which has correct F₄ properties")
    else:
        print(f"⚠ No 48-vertex subsets found by these methods")

    return {
        'e6_vertices': e6_vertices,
        'label_distribution': label_dist,
        'candidates': all_candidates,
    }


if __name__ == "__main__":
    results = main()
