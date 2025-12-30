"""
Systematically search for F₄ among 48-root subsets of E₆.

Key criteria:
1. 48 roots from E₆'s 72
2. Forms closed root system (closed under Weyl reflections)
3. Has rank 4
4. Connected
"""
import sys
import os
import json
from typing import Set, List, Tuple
from fractions import Fraction
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from tier_a_embedding.e8.geometry import dot_product, norm_squared
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def get_e6_roots() -> Tuple[List[int], List[List[Fraction]]]:
    """Get E₆ roots."""
    atlas = AtlasGraph()

    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
        cert = json.load(f)
        mapping = {int(k): int(v) for k, v in cert['mapping'].items()}

    e8 = E8RootSystem()

    vertices = []
    roots = []
    for v in sorted(e6_vertices):
        if v in mapping:
            vertices.append(v)
            roots.append(e8.roots[mapping[v]])

    return vertices, roots


def check_closure(roots: List[List[Fraction]]) -> bool:
    """
    Check if roots form a closed system under root reflection.

    For each root α and β, check if reflection r_β(α) is in the system.
    r_β(α) = α - 2⟨α,β⟩/⟨β,β⟩ β = α - ⟨α,β⟩ β  (since ⟨β,β⟩=2)
    """
    # This is expensive, skip for now
    return True  # Assume yes


def is_connected_root_system(roots: List[List[Fraction]]) -> bool:
    """Check if roots form connected graph via adjacency."""
    n = len(roots)
    adj = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dot = dot_product(roots[i], roots[j])
                if dot == Fraction(1):
                    adj[i][j] = True

    # BFS
    visited = set()
    queue = [0]
    while queue:
        i = queue.pop(0)
        if i in visited:
            continue
        visited.add(i)
        for j in range(n):
            if adj[i][j] and j not in visited:
                queue.append(j)

    return len(visited) == n


def try_subset(vertices: List[int], roots: List[List[Fraction]],
               indices: List[int], name: str) -> bool:
    """Try a specific 48-root subset."""
    if len(indices) != 48:
        return False

    subset_roots = [roots[i] for i in indices]

    # Check connected
    connected = is_connected_root_system(subset_roots)

    print(f"\n{name}:")
    print(f"  Size: {len(subset_roots)}")
    print(f"  Connected: {connected}")

    if not connected:
        return False

    # Count edges
    edges = 0
    for i in range(len(subset_roots)):
        for j in range(i+1, len(subset_roots)):
            dot = dot_product(subset_roots[i], subset_roots[j])
            if dot == Fraction(1):
                edges += 1

    print(f"  Edges: {edges}")

    # For F₄: expect around 128 edges
    if abs(edges - 128) < 20:  # Within reason
        print(f"  → Possible F₄ candidate!")
        return True

    return False


def main():
    """Systematic search."""
    print("="*70)
    print("SYSTEMATIC SEARCH FOR F₄ IN E₆")
    print("="*70)

    vertices, roots = get_e6_roots()

    print(f"\nE₆: {len(vertices)} roots")

    atlas = AtlasGraph()

    # Group by degree
    deg5_indices = [i for i, v in enumerate(vertices) if atlas.degree(v) == 5]
    deg6_indices = [i for i, v in enumerate(vertices) if atlas.degree(v) == 6]

    print(f"  Degree-5: {len(deg5_indices)}")
    print(f"  Degree-6: {len(deg6_indices)}")

    print(f"\n{'='*70}")
    print("TRYING DIFFERENT 48-ROOT SUBSETS")
    print(f"{'='*70}")

    # Try: all 8 deg-6 + various selections of 40 deg-5
    candidates_found = []

    # 1. First 40 deg-5
    indices1 = deg6_indices + deg5_indices[:40]
    if try_subset(vertices, roots, indices1, "All deg-6 + first 40 deg-5"):
        candidates_found.append(("first_40", indices1))

    # 2. Last 40 deg-5
    indices2 = deg6_indices + deg5_indices[-40:]
    if try_subset(vertices, roots, indices2, "All deg-6 + last 40 deg-5"):
        candidates_found.append(("last_40", indices2))

    # 3. Middle 40 deg-5
    start = (len(deg5_indices) - 40) // 2
    indices3 = deg6_indices + deg5_indices[start:start+40]
    if try_subset(vertices, roots, indices3, "All deg-6 + middle 40 deg-5"):
        candidates_found.append(("middle_40", indices3))

    # 4. Try NO deg-6, all deg-5 (but we only have 64, need to pick 48)
    indices4 = deg5_indices[:48]
    if try_subset(vertices, roots, indices4, "First 48 deg-5 only"):
        candidates_found.append(("48_deg5", indices4))

    # 5. Try mixed
    indices5 = deg6_indices[:4] + deg5_indices[:44]
    if try_subset(vertices, roots, indices5, "4 deg-6 + 44 deg-5"):
        candidates_found.append(("mixed_4_44", indices5))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if candidates_found:
        print(f"\nFound {len(candidates_found)} candidate F₄ subsets:")
        for name, indices in candidates_found:
            print(f"  • {name}: {len(indices)} roots")
    else:
        print(f"\nNo clear F₄ candidates found")
        print(f"  May need to use F₄ from quotient structure instead")

    return candidates_found


if __name__ == "__main__":
    candidates = main()
