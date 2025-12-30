"""
Find F₄ ⊂ E₆ ⊂ E₈ by searching E₈'s 240 roots directly.

Strategy:
1. Load all 240 E₈ roots
2. Find 72-root closed subsystem → E₆
3. Find 48-root closed subsystem within E₆ → F₄
4. Verify Cartan matrices
"""
import sys
import os
from typing import List, Set, Tuple, Dict
from fractions import Fraction
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import E8RootSystem
from tier_a_embedding.e8.geometry import dot_product, norm_squared


def analyze_e8_structure():
    """Analyze E₈ root system structure."""
    print("="*70)
    print("ANALYZING E₈ ROOT SYSTEM (240 roots)")
    print("="*70)

    e8 = E8RootSystem()

    print(f"\nTotal roots: {len(e8.roots)}")

    # Verify all roots have norm 2
    for i, root in enumerate(e8.roots):
        norm = norm_squared(root)
        if norm != Fraction(2):
            print(f"  ⚠ Root {i} has norm {norm} != 2")
            return None

    print(f"✓ All {len(e8.roots)} roots have norm² = 2 (simply-laced)")

    # Compute dot product distribution
    print(f"\nComputing dot products between all pairs...")
    dot_products = []
    for i in range(len(e8.roots)):
        for j in range(i+1, len(e8.roots)):
            dot = dot_product(e8.roots[i], e8.roots[j])
            dot_products.append(dot)

    dot_dist = Counter(dot_products)
    print(f"\nDot product distribution:")
    for val in sorted(dot_dist.keys()):
        print(f"  {val}: {dot_dist[val]} pairs")

    # Count adjacencies (dot product = 1)
    adjacency_count = dot_dist.get(Fraction(1), 0)
    print(f"\nAdjacent pairs (dot=1): {adjacency_count}")
    print(f"Expected for E₈: each root has 56 neighbors → 240×56/2 = 6,720")

    return e8


def find_e6_in_e8(e8: E8RootSystem) -> List[int]:
    """
    Find E₆ as 72-root subsystem of E₈.

    Strategy: E₆ roots should form a closed root system.
    Classical approach: Use known E₆ embedding pattern.
    """
    print(f"\n{'='*70}")
    print("SEARCHING FOR E₆ (72 roots) IN E₈")
    print(f"{'='*70}")

    # Classical fact: E₆ can be embedded by taking roots with last 2 coordinates = 0
    # E₈ roots in standard basis: (±1,±1,0,0,0,0,0,0) and permutations
    # E₆: restrict to first 6 coordinates

    # Try: find roots where last 2 coordinates are 0
    e6_candidates = []

    for i, root in enumerate(e8.roots):
        # Check if last 2 coordinates are 0
        if root[6] == Fraction(0) and root[7] == Fraction(0):
            e6_candidates.append(i)

    print(f"\nRoots with last 2 coords = 0: {len(e6_candidates)}")

    if len(e6_candidates) == 72:
        print(f"✓ Found exactly 72 roots!")
        return e6_candidates

    # Try different patterns
    # E₆ might be embedded differently

    # Try: roots where sum of last 2 coords = 0
    e6_candidates2 = []
    for i, root in enumerate(e8.roots):
        if root[6] + root[7] == Fraction(0):
            e6_candidates2.append(i)

    print(f"Roots with coord[6] + coord[7] = 0: {len(e6_candidates2)}")

    if len(e6_candidates2) == 72:
        print(f"✓ Found exactly 72 roots!")
        return e6_candidates2

    # Try: Search by checking closure
    # This is expensive but systematic
    print(f"\n⚠ Standard embeddings didn't give 72 roots")
    print(f"  Need to search for closed 72-root subsystems...")

    return []


def verify_closed_root_system(e8: E8RootSystem, indices: List[int]) -> bool:
    """
    Verify that a subset forms a closed root system.

    A root system is closed if for any two roots α, β:
    - If ⟨α,β⟩ = -1, then α+β is also in the system
    - If ⟨α,β⟩ = -2, then α is a multiple of β (same up to sign)
    """
    roots = [e8.roots[i] for i in indices]
    root_set = set(tuple(r) for r in roots)  # For O(1) lookup

    for i, alpha in enumerate(roots):
        for j, beta in enumerate(roots):
            if i == j:
                continue

            dot = dot_product(alpha, beta)

            # If dot = -1, then alpha + beta should be in system
            if dot == Fraction(-1):
                alpha_plus_beta = tuple(a + b for a, b in zip(alpha, beta))
                if alpha_plus_beta not in root_set:
                    return False

    return True


def find_simple_roots_e6(e8: E8RootSystem, indices: List[int]) -> List[int]:
    """Find 6 simple roots for E₆."""
    from itertools import combinations

    # E₆ has rank 6, so we need 6 simple roots
    # Simple roots should have specific dot product pattern

    E6_CARTAN = [
        [ 2, -1,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0],
        [ 0, -1,  2, -1,  0, -1],
        [ 0,  0, -1,  2, -1,  0],
        [ 0,  0,  0, -1,  2,  0],
        [ 0,  0, -1,  0,  0,  2]
    ]

    roots = [e8.roots[i] for i in indices]

    print(f"  Searching for 6 simple roots in {len(roots)} roots...")

    # Try combinations of 6 roots
    count = 0
    for combo in combinations(range(min(len(roots), 20)), 6):
        count += 1
        if count % 1000 == 0:
            print(f"    Tested {count} combinations...")

        # Compute Cartan matrix
        cartan = [[0]*6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                dot_ij = dot_product(roots[combo[i]], roots[combo[j]])
                cartan[i][j] = int(dot_ij)  # Since all norms are 2

        # Check if matches E₆ Cartan
        if cartan == E6_CARTAN:
            print(f"  ✓ Found E₆ Cartan at indices {combo}")
            return list(combo)

    print(f"  ⚠ No E₆ Cartan found in {count} combinations")
    return []


def main():
    """Main execution."""
    print("="*70)
    print("FIND F₄ ⊂ E₆ ⊂ E₈ IN E₈ ROOT SYSTEM")
    print("="*70)

    # Step 1: Analyze E₈
    e8 = analyze_e8_structure()
    if e8 is None:
        return

    # Step 2: Find E₆
    e6_indices = find_e6_in_e8(e8)

    if not e6_indices:
        print(f"\n⚠ Could not find E₆ in E₈")
        print(f"  This requires more sophisticated search")
        return

    # Step 3: Verify E₆ is closed
    print(f"\nVerifying E₆ is closed root system...")
    is_closed = verify_closed_root_system(e8, e6_indices)
    print(f"  Closed: {is_closed} {'✓' if is_closed else '✗'}")

    if not is_closed:
        print(f"  ⚠ E₆ candidate is not a closed root system")
        return

    # Step 4: Find simple roots for E₆
    e6_simple = find_simple_roots_e6(e8, e6_indices)

    if e6_simple:
        print(f"\n✓ E₆ verified in E₈")
    else:
        print(f"\n⚠ E₆ structure not fully verified")

    # TODO: Find F₄ within E₆

    return {
        'e8_roots': len(e8.roots),
        'e6_indices': e6_indices,
        'e6_simple': e6_simple,
    }


if __name__ == "__main__":
    result = main()
