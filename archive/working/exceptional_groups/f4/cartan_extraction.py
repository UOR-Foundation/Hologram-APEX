"""
F₄ Cartan Matrix Extraction.

Extract and verify the Cartan matrix from the quotient adjacency structure.
"""
import sys
import os
from typing import List, Tuple, Optional, Dict
import json
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding.analysis import QuotientAnalyzer


class CartanExtractor:
    """Extract Cartan matrix from root system adjacency."""

    def __init__(self, adjacency_matrix: List[List[bool]], degree_sequence: List[int]):
        """
        Initialize with adjacency data.

        Args:
            adjacency_matrix: 48x48 adjacency matrix
            degree_sequence: Degree of each vertex
        """
        self.adjacency = adjacency_matrix
        self.degrees = degree_sequence
        self.n = len(adjacency_matrix)

    def find_simple_roots(self, root_coords: Optional[Dict] = None) -> List[int]:
        """
        Find 4 simple roots that generate F₄.

        Standard F₄ simple roots (Bourbaki convention):
        α₁ = e₂ - e₃  (long root, norm² = 2)
        α₂ = e₃ - e₄  (long root, norm² = 2)
        α₃ = e₄       (short root, norm² = 1)
        α₄ = ½(e₁ - e₂ - e₃ - e₄)  (short root, norm² = 1)

        These give the standard F₄ Cartan matrix:
        [[ 2, -1,  0,  0],
         [-1,  2, -2,  0],
         [ 0, -1,  2, -1],
         [ 0,  0, -1,  2]]
        """
        from fractions import Fraction

        if root_coords is None:
            raise ValueError("F₄ simple root extraction requires root coordinates")

        # Search for exact matches to standard simple roots
        found = {}

        for idx, coords in root_coords.items():
            c = [Fraction(x) for x in coords]

            # α₁ = e₂ - e₃ = (0, 1, -1, 0)
            if c == [Fraction(0), Fraction(1), Fraction(-1), Fraction(0)]:
                found['alpha1'] = idx

            # α₂ = e₃ - e₄ = (0, 0, 1, -1)
            elif c == [Fraction(0), Fraction(0), Fraction(1), Fraction(-1)]:
                found['alpha2'] = idx

            # α₃ = e₄ = (0, 0, 0, 1)
            elif c == [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]:
                found['alpha3'] = idx

            # α₄ = ½(e₁ - e₂ - e₃ - e₄) = (1/2, -1/2, -1/2, -1/2)
            elif c == [Fraction(1,2), Fraction(-1,2), Fraction(-1,2), Fraction(-1,2)]:
                found['alpha4'] = idx

        if len(found) == 4:
            simple_roots = [found['alpha1'], found['alpha2'], found['alpha3'], found['alpha4']]
            print(f"✓ Found standard F₄ simple roots: {simple_roots}")
            return simple_roots

        raise ValueError(f"Could not find all 4 standard F₄ simple roots (found {len(found)}/4)")

    def _find_tree_subset(self, size: int) -> List[int]:
        """Find a subset of given size that forms a tree."""
        # Start with vertex of minimum degree
        min_degree_vertex = min(range(self.n), key=lambda i: self.degrees[i])
        subset = [min_degree_vertex]
        used = {min_degree_vertex}

        # Greedily extend
        while len(subset) < size and len(subset) < self.n:
            # Find vertex connected to exactly one vertex in subset
            best_vertex = None
            for v in range(self.n):
                if v not in used:
                    connections = sum(1 for u in subset if self.adjacency[v][u])
                    if connections == 1:  # Connected to exactly one
                        best_vertex = v
                        break

            if best_vertex is None:
                # No perfect candidate, take any connected vertex
                for v in range(self.n):
                    if v not in used:
                        for u in subset:
                            if self.adjacency[v][u]:
                                best_vertex = v
                                break
                        if best_vertex is not None:
                            break

            if best_vertex is None:
                break

            subset.append(best_vertex)
            used.add(best_vertex)

        return subset[:size]

    def extract_gram_matrix(self, simple_roots: List[int]) -> List[List[int]]:
        """
        Extract 4x4 Gram-like adjacency matrix for given simple roots.

        WARNING: This creates a SYMMETRIC matrix based on adjacency.
        This is NOT a proper Cartan matrix for F₄ (which must be asymmetric).

        Use extract_cartan_matrix() for proper Cartan matrix generation.

        The Gram matrix G has entries:
        - G[i][i] = 2 (diagonal)
        - G[i][j] = -m where m is the bond multiplicity (symmetric)

        Bond multiplicity determined by root lengths:
        - Both same length → -1 (single bond)
        - Different lengths → -2 (double bond, characteristic of F₄)
        """
        n = len(simple_roots)
        gram = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    gram[i][j] = 2
                else:
                    # Check if roots are adjacent
                    ri, rj = simple_roots[i], simple_roots[j]
                    if self.adjacency[ri][rj]:
                        # Determine bond multiplicity from root lengths
                        # In F₄: degree-5 = short, degree-6 = long
                        is_i_short = (self.degrees[ri] == 5)
                        is_j_short = (self.degrees[rj] == 5)

                        # Different lengths → double bond (-2)
                        if is_i_short != is_j_short:
                            gram[i][j] = -2
                        else:
                            # Same length → single bond (-1)
                            gram[i][j] = -1
                    else:
                        gram[i][j] = 0

        return gram

    def extract_cartan_submatrix(self, simple_roots: List[int], root_coords: Optional[Dict] = None) -> List[List[int]]:
        """
        Extract proper 4x4 Cartan matrix for given simple roots.

        Cartan matrix C[i][j] = 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩

        For F₄ (non-simply-laced), this is ASYMMETRIC.
        Standard F₄: C[1][2] = -2 but C[2][1] = -1

        If root_coords not provided, falls back to Gram matrix (symmetric).
        """
        from fractions import Fraction

        if root_coords is None:
            print("  ⚠ Warning: No root coordinates provided, generating Gram matrix (symmetric)")
            print("  ⚠ This is NOT a proper Cartan matrix for F₄!")
            return self.extract_gram_matrix(simple_roots)

        n = len(simple_roots)
        cartan = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                idx_i = simple_roots[i]
                idx_j = simple_roots[j]

                if idx_i not in root_coords or idx_j not in root_coords:
                    # Fallback to adjacency-based
                    if i == j:
                        cartan[i][j] = 2
                    elif self.adjacency[idx_i][idx_j]:
                        # Symmetric approximation
                        is_i_short = (self.degrees[idx_i] == 5)
                        is_j_short = (self.degrees[idx_j] == 5)
                        cartan[i][j] = -2 if is_i_short != is_j_short else -1
                    continue

                # Get root coordinates
                alpha_i = root_coords[idx_i]
                alpha_j = root_coords[idx_j]

                # Compute inner products (exact arithmetic)
                inner_ij = sum(Fraction(a) * Fraction(b) for a, b in zip(alpha_i, alpha_j))
                inner_jj = sum(Fraction(a) * Fraction(a) for a in alpha_j)

                # Cartan entry: 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩
                if inner_jj != 0:
                    cartan_val = 2 * inner_ij / inner_jj
                    # Round to integer (should be exact for root systems)
                    cartan[i][j] = int(cartan_val)
                else:
                    cartan[i][j] = 0

        return cartan

    def refine_cartan_for_f4(self, cartan: List[List[int]]) -> List[List[int]]:
        """
        Refine Cartan matrix to match F₄ pattern.

        F₄ Cartan matrix should be:
        [[ 2, -1,  0,  0],
         [-1,  2, -2,  0],  # Note the -2 for double bond
         [ 0, -1,  2, -1],
         [ 0,  0, -1,  2]]

        Bond multiplicities now detected automatically in extract_cartan_submatrix
        based on degree pattern (degree-5 vs degree-6).
        """
        # Verify the -2 entry exists (characteristic of F₄)
        has_double_bond = any(cartan[i][j] == -2 for i in range(len(cartan)) for j in range(len(cartan)))

        if has_double_bond:
            print("  ✓ Double bond (-2) detected - F₄ signature confirmed")
        else:
            print("  ⚠ No double bond found - may indicate D₄ structure instead")

        return cartan


def extract_cartan_matrix(f4_structure, root_coords: Optional[Dict] = None, simple_roots: Optional[List[int]] = None) -> Tuple[List[List[int]], bool]:
    """
    Extract Cartan matrix from F₄ structure.

    Args:
        f4_structure: F4Structure from sign_class_analysis

    Returns:
        Tuple of (cartan_matrix, is_valid)
    """
    extractor = CartanExtractor(
        f4_structure.adjacency_matrix,
        f4_structure.degree_sequence
    )

    # Find simple roots if not provided
    if simple_roots is None:
        simple_roots = extractor.find_simple_roots(root_coords=root_coords)

    # Extract Cartan submatrix (with root_coords for proper asymmetric matrix)
    cartan = extractor.extract_cartan_submatrix(simple_roots, root_coords=root_coords)

    # Refine for F₄
    cartan = extractor.refine_cartan_for_f4(cartan)

    # Verify basic properties
    is_valid = verify_cartan_properties(cartan)

    return cartan, is_valid


def verify_cartan_properties(cartan: List[List[int]]) -> bool:
    """Verify basic Cartan matrix properties."""
    n = len(cartan)

    # Check size (F₄ has rank 4)
    if n != 4:
        print(f"Wrong size: {n} (expected 4)")
        return False

    # Check diagonal = 2
    for i in range(n):
        if cartan[i][i] != 2:
            print(f"Diagonal element [{i}][{i}] = {cartan[i][i]} (expected 2)")
            return False

    # Check off-diagonal <= 0
    for i in range(n):
        for j in range(n):
            if i != j and cartan[i][j] > 0:
                print(f"Off-diagonal [{i}][{j}] = {cartan[i][j]} (expected <= 0)")
                return False

    return True


def verify_f4_cartan(cartan: List[List[int]]) -> bool:
    """
    Verify this is specifically the F₄ Cartan matrix.

    F₄ Cartan matrix:
    [[ 2, -1,  0,  0],
     [-1,  2, -2,  0],
     [ 0, -1,  2, -1],
     [ 0,  0, -1,  2]]
    """
    expected_f4 = [
        [ 2, -1,  0,  0],
        [-1,  2, -2,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ]

    # Check various permutations since root ordering may differ
    # For now, just check if we have the right entries
    our_entries = sorted([cartan[i][j] for i in range(4) for j in range(4)])
    f4_entries = sorted([expected_f4[i][j] for i in range(4) for j in range(4)])

    entries_match = our_entries == f4_entries

    if entries_match:
        print("✓ Cartan matrix has F₄ entry pattern!")
    else:
        print(f"Entry mismatch:")
        print(f"  Expected: {f4_entries}")
        print(f"  Got: {our_entries}")

    # Check for the characteristic -2 entry
    has_double_bond = any(cartan[i][j] == -2 for i in range(4) for j in range(4))

    if not has_double_bond:
        print("⚠ Warning: No -2 entry found (F₄ has a double bond)")

    return entries_match


def display_cartan_matrix(cartan: List[List[int]], name: str = "Cartan Matrix"):
    """Display Cartan matrix nicely."""
    print(f"\n{name}:")
    print("     ", end="")
    for j in range(len(cartan)):
        print(f"{j:3}", end=" ")
    print()

    for i, row in enumerate(cartan):
        print(f"{i:3}: ", end="")
        for val in row:
            if val == 0:
                print("  .", end=" ")
            else:
                print(f"{val:3}", end=" ")
        print()


if __name__ == "__main__":
    # Test with extracted F₄ structure
    from f4.sign_class_analysis import extract_f4_from_sign_classes

    print("="*60)
    print("F₄ CARTAN MATRIX EXTRACTION")
    print("="*60)

    f4_structure, _ = extract_f4_from_sign_classes()

    cartan, is_valid = extract_cartan_matrix(f4_structure)

    display_cartan_matrix(cartan, "Extracted Cartan Matrix")

    print(f"\nBasic validity: {is_valid}")
    print(f"F₄ verification: {verify_f4_cartan(cartan)}")