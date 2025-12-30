"""
Verify the 48-root subset is actually F₄ by checking Cartan matrix.
"""
import sys
import os
import json
from typing import Set, List, Tuple
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph, E8RootSystem
from tier_a_embedding.e8.geometry import dot_product, norm_squared
from exceptional_groups.e6.first_principles_construction import E6FirstPrinciplesConstruction


def get_f4_subset() -> Tuple[Set[int], List[List[Fraction]]]:
    """Get the F₄ subset we identified."""
    atlas = AtlasGraph()

    # Load E₆
    e6_constructor = E6FirstPrinciplesConstruction()
    e6_vertices = e6_constructor.search_by_degree_and_structure()

    # Load mapping
    with open('/workspaces/Hologram/working/tier_a_embedding/tier_a_certificate.json', 'r') as f:
        cert = json.load(f)
        mapping = {int(k): int(v) for k, v in cert['mapping'].items()}

    e8 = E8RootSystem()

    # Get E₆ roots by degree
    e6_deg5 = []
    e6_deg6 = []
    e6_root_vectors = {}

    for v in sorted(e6_vertices):
        if v in mapping:
            e8_root_idx = mapping[v]
            root_vector = e8.roots[e8_root_idx]
            e6_root_vectors[v] = root_vector

            deg = atlas.degree(v)
            if deg == 5:
                e6_deg5.append(v)
            elif deg == 6:
                e6_deg6.append(v)

    # F₄ = all 8 deg-6 + first 40 deg-5
    f4_vertices = set(e6_deg6 + e6_deg5[:40])
    f4_roots = [e6_root_vectors[v] for v in sorted(f4_vertices)]

    print(f"F₄ subset:")
    print(f"  Vertices: {len(f4_vertices)}")
    print(f"  Deg-6: {len([v for v in f4_vertices if atlas.degree(v) == 6])}")
    print(f"  Deg-5: {len([v for v in f4_vertices if atlas.degree(v) == 5])}")

    return f4_vertices, f4_roots


def find_simple_roots(roots: List[List[Fraction]]) -> List[int]:
    """
    Find simple roots by searching for F₄ Cartan matrix.

    Try combinations of 4 roots that give the F₄ Cartan matrix.
    """
    from itertools import combinations

    n = len(roots)

    f4_cartan = [
        [ 2, -1,  0,  0],
        [-1,  2, -2,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ]

    print(f"  Searching {n} roots for F₄ Cartan pattern...")

    # Try combinations of 4 roots
    # C(48,4) = 194,580 combinations - manageable
    count = 0
    total = n * (n-1) * (n-2) * (n-3) // 24  # C(n,4)
    print(f"    Total combinations to test: {total}")

    for combo in combinations(range(n), 4):
        count += 1
        if count % 10000 == 0:
            print(f"    Tested {count}/{total} combinations ({100*count//total}%)...")

        # Compute Cartan matrix for this combination
        cartan = compute_cartan_matrix(roots, list(combo))

        # Check if it matches F₄ (with any permutation)
        if matches_f4_cartan(cartan, f4_cartan):
            print(f"  ✓ Found F₄ Cartan at indices {combo}")
            return list(combo)

    print(f"  ⚠ Tested all {count} combinations, none match F₄ Cartan")
    print(f"    This suggests the 48-root subset may not be standard F₄")
    return []


def matches_f4_cartan(cartan: List[List[int]], f4_cartan: List[List[int]]) -> bool:
    """Check if cartan matrix matches F₄ (up to permutation)."""
    from itertools import permutations

    # Direct match
    if cartan == f4_cartan:
        return True

    # Try permutations of rows/columns
    for perm in permutations(range(4)):
        # Permute rows and columns
        permuted = [[cartan[perm[i]][perm[j]] for j in range(4)] for i in range(4)]
        if permuted == f4_cartan:
            return True

    return False


def compute_cartan_matrix(roots: List[List[Fraction]], simple_indices: List[int]) -> List[List[int]]:
    """
    Compute Cartan matrix from simple roots.

    Cartan matrix: C_ij = 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩

    For E₈ roots (all norm 2): C_ij = 2⟨α_i, α_j⟩ / 2 = ⟨α_i, α_j⟩
    """
    k = len(simple_indices)
    cartan = [[0] * k for _ in range(k)]

    simple_roots = [roots[i] for i in simple_indices]

    for i in range(k):
        for j in range(k):
            # Cartan entry
            dot_ij = dot_product(simple_roots[i], simple_roots[j])
            norm_j = norm_squared(simple_roots[j])

            # C_ij = 2 * dot_ij / norm_j
            c_ij = 2 * dot_ij / norm_j

            # Should be integer
            cartan[i][j] = int(c_ij)

    return cartan


def verify_f4_cartan(cartan: List[List[int]]) -> bool:
    """
    Verify if Cartan matrix matches F₄.

    F₄ Cartan matrix:
    [[ 2, -1,  0,  0],
     [-1,  2, -2,  0],
     [ 0, -1,  2, -1],
     [ 0,  0, -1,  2]]
    """
    f4_cartan = [
        [ 2, -1,  0,  0],
        [-1,  2, -2,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ]

    # Check if matrices match (up to permutation of simple roots)
    if cartan == f4_cartan:
        return True

    # Try other permutations
    # For now, just check shape and diagonal
    if len(cartan) != 4:
        return False

    # Diagonal should be all 2's
    if not all(cartan[i][i] == 2 for i in range(4)):
        return False

    # Should be symmetric
    for i in range(4):
        for j in range(4):
            if cartan[i][j] != cartan[j][i]:
                # Cartan matrices are NOT symmetric in general!
                # They satisfy: C_ij * C_ji = 0, 1, 2, 3, or 4
                pass

    return False  # Need exact match


def main():
    """Main verification."""
    print("="*70)
    print("VERIFY F₄ STRUCTURE VIA CARTAN MATRIX")
    print("="*70)

    # Get F₄ subset
    f4_vertices, f4_roots = get_f4_subset()

    print(f"\nF₄ subset has {len(f4_roots)} roots")

    # Find simple roots
    print(f"\nFinding simple roots...")
    simple_indices = find_simple_roots(f4_roots)

    print(f"Found {len(simple_indices)} simple roots: {simple_indices}")

    if len(simple_indices) != 4:
        print(f"⚠ Expected 4 simple roots for F₄, found {len(simple_indices)}")
        return

    # Compute Cartan matrix
    print(f"\nComputing Cartan matrix...")
    cartan = compute_cartan_matrix(f4_roots, simple_indices)

    print(f"\nCartan matrix:")
    for row in cartan:
        print(f"  {row}")

    # Check against F₄
    print(f"\nF₄ Cartan matrix:")
    f4_cartan = [
        [ 2, -1,  0,  0],
        [-1,  2, -2,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ]
    for row in f4_cartan:
        print(f"  {row}")

    is_f4 = verify_f4_cartan(cartan)

    print(f"\n{'='*70}")
    if is_f4:
        print("✓✓✓ VERIFIED: This is F₄!")
        print(f"  Cartan matrix matches F₄")
    else:
        print("⚠ Cartan matrix doesn't match F₄ exactly")
        print("  May need different simple root selection")

        # Show differences
        print(f"\nDifferences:")
        for i in range(4):
            for j in range(4):
                if cartan[i][j] != f4_cartan[i][j]:
                    print(f"  [{i},{j}]: got {cartan[i][j]}, expected {f4_cartan[i][j]}")
    print(f"{'='*70}")

    return cartan


if __name__ == "__main__":
    cartan = main()
