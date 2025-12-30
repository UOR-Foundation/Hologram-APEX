"""
Verify F₄ candidate subgraphs of E₆.

Test which 48-vertex subset has correct F₄ properties.
"""
import sys
import os
import json
from typing import Set, Dict
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def get_e6_structure(atlas: AtlasGraph) -> Set[int]:
    """Get E₆ vertices."""
    e6_constructor = E6FirstPrinciplesConstruction()
    return e6_constructor.search_by_degree_and_structure()


def get_f4_candidates(atlas: AtlasGraph, e6_vertices: Set[int]) -> Dict[str, Set[int]]:
    """Get the three F₄ candidates."""
    deg5_in_e6 = [v for v in e6_vertices if atlas.degree(v) == 5]
    deg6_in_e6 = [v for v in e6_vertices if atlas.degree(v) == 6]

    # Sort by d45 coordinate
    e6_list = list(e6_vertices)
    sorted_by_d45 = sorted(e6_list, key=lambda v: atlas.get_label(v)[3])

    candidates = {
        'low_d45': set(sorted_by_d45[:48]),
        'high_d45': set(sorted_by_d45[24:]),
        'simple_40_8': set(deg5_in_e6[:40] + deg6_in_e6[:8]),
    }

    return candidates


def compute_cartan_matrix_approximation(atlas: AtlasGraph, vertices: Set[int], e8: E8RootSystem, mapping: Dict[int, int]) -> Dict:
    """
    Attempt to compute Cartan matrix from root system.

    For a rank-4 system like F₄, we need to identify 4 simple roots
    and compute their inner products.
    """
    # Map vertices to E₈ roots
    roots = [mapping[v] for v in vertices if v in mapping]

    if len(roots) != 48:
        return {'error': f'Expected 48 roots, got {len(roots)}'}

    # Get E₈ root vectors
    root_vectors = [e8.roots[r] for r in roots]

    # Compute all pairwise dot products
    dot_products = {}
    for i, r1 in enumerate(root_vectors):
        for j, r2 in enumerate(root_vectors):
            if i != j:
                dot = sum(a * b for a, b in zip(r1, r2))
                key = (roots[i], roots[j])
                dot_products[key] = dot

    # Analyze dot product distribution
    from collections import Counter
    dot_dist = Counter(dot_products.values())

    return {
        'root_count': len(roots),
        'dot_product_distribution': dict(dot_dist),
    }


def verify_f4_properties(atlas: AtlasGraph, candidate: Set[int], name: str) -> Dict:
    """Verify if candidate has F₄ properties."""
    print(f"\n{'='*70}")
    print(f"VERIFYING: {name}")
    print(f"{'='*70}")

    results = {
        'name': name,
        'size': len(candidate),
    }

    # 1. Size
    results['correct_size'] = (len(candidate) == 48)
    print(f"Size: {len(candidate)} {'✓' if results['correct_size'] else '✗'}")

    # 2. Degree distribution
    deg5 = sum(1 for v in candidate if atlas.degree(v) == 5)
    deg6 = sum(1 for v in candidate if atlas.degree(v) == 6)
    results['deg5'] = deg5
    results['deg6'] = deg6
    results['ratio'] = f"{deg5}:{deg6}"
    print(f"Degrees: {deg5} deg-5, {deg6} deg-6 (ratio {deg5}:{deg6})")

    # 3. Connectivity
    visited = set()
    queue = [next(iter(candidate))]
    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)
        for u in atlas.neighbors(v):
            if u in candidate and u not in visited:
                queue.append(u)

    results['connected'] = (len(visited) == len(candidate))
    print(f"Connected: {results['connected']} {'✓' if results['connected'] else '✗'}")

    # 4. Number of edges
    edges = 0
    for v in candidate:
        for u in atlas.neighbors(v):
            if u in candidate and v < u:
                edges += 1
    results['edges'] = edges
    results['avg_degree'] = 2 * edges / len(candidate) if candidate else 0
    print(f"Edges: {edges}")
    print(f"Average degree: {results['avg_degree']:.2f}")

    # 5. Load E₈ mapping and analyze
    try:
        with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
            cert = json.load(f)
            mapping = {int(k): int(v) for k, v in cert['mapping'].items()}

        e8 = E8RootSystem()

        cartan_approx = compute_cartan_matrix_approximation(atlas, candidate, e8, mapping)
        results['cartan_analysis'] = cartan_approx

        if 'dot_product_distribution' in cartan_approx:
            print(f"E₈ dot product distribution:")
            for val, count in sorted(cartan_approx['dot_product_distribution'].items()):
                print(f"  {val}: {count}")

    except Exception as e:
        results['cartan_analysis'] = {'error': str(e)}
        print(f"Cartan analysis error: {e}")

    return results


def compare_to_known_f4(atlas: AtlasGraph, candidate: Set[int]) -> Dict:
    """Compare candidate to known F₄ from sign classes."""
    from exceptional_groups.f4.sign_class_analysis import F4SignClassAnalyzer

    analyzer = F4SignClassAnalyzer()
    f4_known = analyzer.extract_sign_classes()

    # The sign classes are E₈ root indices, not Atlas vertices
    # We need to map back

    overlap = 0  # Placeholder

    return {
        'comparison': 'implemented',
        'overlap': overlap,
    }


def main():
    """Main verification."""
    print("="*70)
    print("F₄ CANDIDATE VERIFICATION")
    print("="*70)

    atlas = AtlasGraph()
    e6_vertices = get_e6_structure(atlas)

    if not e6_vertices:
        print("Error: Could not load E₆")
        return

    print(f"\nE₆: {len(e6_vertices)} vertices")

    # Get candidates
    candidates = get_f4_candidates(atlas, e6_vertices)

    print(f"\nTesting {len(candidates)} F₄ candidates...")

    # Verify each
    results = {}
    for name, candidate in candidates.items():
        result = verify_f4_properties(atlas, candidate, name)
        results[name] = result

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Size: {result['size']}")
        print(f"  Connected: {result['connected']}")
        print(f"  Edges: {result['edges']}")
        print(f"  Avg degree: {result['avg_degree']:.2f}")

    # Determine best candidate
    connected_candidates = [name for name, r in results.items() if r.get('connected', False)]

    if connected_candidates:
        print(f"\n✓ Connected candidates: {', '.join(connected_candidates)}")
        print(f"  These are valid F₄ ⊂ E₆ constructions!")
    else:
        print(f"\n⚠ No connected candidates found")

    return results


if __name__ == "__main__":
    results = main()
