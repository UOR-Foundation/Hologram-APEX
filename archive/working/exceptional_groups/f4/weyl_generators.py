"""
F₄ Weyl Group Generators.

Generate the Weyl group of F₄ (order 1152) from simple reflections.
F₄ Dynkin diagram: o---o==>o---o (with double bond).
"""
import sys
import os
from typing import List, Set, Tuple, Dict
from fractions import Fraction
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph
from f4.sign_class_analysis import extract_f4_from_sign_classes


class ExactMatrix:
    """Matrix with exact rational entries."""
    def __init__(self, data: List[List[Fraction]]):
        self.data = [[Fraction(x) for x in row] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @staticmethod
    def eye(n: int) -> 'ExactMatrix':
        """Identity matrix."""
        data = [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
        return ExactMatrix(data)

    def __matmul__(self, other: 'ExactMatrix') -> 'ExactMatrix':
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions")
        result = [[Fraction(0) for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                result[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
        return ExactMatrix(result)

    def __eq__(self, other: 'ExactMatrix') -> bool:
        """Exact equality check."""
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return all(self.data[i][j] == other.data[i][j]
                  for i in range(self.rows) for j in range(self.cols))

    def to_tuple(self) -> Tuple:
        """Convert to tuple for hashing."""
        return tuple(tuple(row) for row in self.data)

    def __hash__(self):
        return hash(self.to_tuple())


class ExactVector:
    """Vector with exact rational coordinates."""
    def __init__(self, coords: List):
        self.coords = [Fraction(c) for c in coords]
        self.dim = len(coords)

    def dot(self, other: 'ExactVector') -> Fraction:
        """Exact dot product."""
        if self.dim != other.dim:
            raise ValueError("Incompatible dimensions")
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def __sub__(self, other: 'ExactVector') -> 'ExactVector':
        """Vector subtraction."""
        return ExactVector([a - b for a, b in zip(self.coords, other.coords)])

    def __mul__(self, scalar: Fraction) -> 'ExactVector':
        """Scalar multiplication."""
        scalar = Fraction(scalar)
        return ExactVector([scalar * c for c in self.coords])

    def __rmul__(self, scalar: Fraction) -> 'ExactVector':
        """Right scalar multiplication."""
        return self.__mul__(scalar)


class F4WeylGroup:
    """Generate and analyze F₄ Weyl group."""

    def __init__(self):
        """Initialize F₄ Weyl group structure."""
        self.rank = 4
        self.expected_order = 1152
        self.simple_roots = None
        self.cartan_matrix = None
        self.generators = []
        self.elements = set()

    def get_f4_cartan_matrix(self) -> List[List[int]]:
        """
        Get the standard F₄ Cartan matrix (exact integers).

        F₄ Dynkin: o---o==>o---o
        Double bond between roots 2 and 3.
        """
        cartan = [
            [ 2, -1,  0,  0],
            [-1,  2, -2,  0],  # -2 indicates double bond
            [ 0, -1,  2, -1],
            [ 0,  0, -1,  2]
        ]
        self.cartan_matrix = cartan
        return cartan

    def generate_simple_roots(self) -> List[ExactVector]:
        """
        Generate F₄ simple roots with exact rational coordinates.

        F₄ has 4 simple roots with specific length relationships.
        Using standard basis where roots satisfy Cartan matrix.
        """
        # Standard F₄ simple roots (exact rational coordinates)
        # Using the explicit F₄ root system with half-integers
        alpha1 = ExactVector([0, 1, -1, 0])
        alpha2 = ExactVector([0, 0, 1, -1])
        alpha3 = ExactVector([0, 0, 0, 1])
        alpha4 = ExactVector([Fraction(1,2), Fraction(-1,2), Fraction(-1,2), Fraction(-1,2)])

        self.simple_roots = [alpha1, alpha2, alpha3, alpha4]
        return self.simple_roots

    def simple_reflection(self, root: ExactVector) -> ExactMatrix:
        """
        Create exact reflection matrix for a root.

        Reflection through hyperplane perpendicular to root:
        s_α(v) = v - 2⟨v,α⟩/⟨α,α⟩ * α
        """
        root_norm_sq = root.dot(root)
        if root_norm_sq == 0:
            raise ValueError("Cannot reflect through zero root")

        # Create reflection matrix: I - 2(α ⊗ α)/⟨α,α⟩
        n = root.dim
        reflection_data = []
        for i in range(n):
            row = []
            for j in range(n):
                # I[i,j] - 2*α[i]*α[j]/⟨α,α⟩
                delta = Fraction(1) if i == j else Fraction(0)
                entry = delta - 2 * root.coords[i] * root.coords[j] / root_norm_sq
                row.append(entry)
            reflection_data.append(row)
        return ExactMatrix(reflection_data)

    def generate_weyl_group(self, max_length: int = 30) -> Set[ExactMatrix]:
        """
        Generate Weyl group elements up to given word length (exact arithmetic).

        Uses breadth-first search from identity.
        Each element is an ExactMatrix (exact rational entries).
        """
        if self.simple_roots is None:
            self.generate_simple_roots()

        # Generate reflection matrices for simple roots
        reflections = []
        for root in self.simple_roots:
            reflections.append(self.simple_reflection(root))
        self.generators = reflections

        # Start with identity
        identity = ExactMatrix.eye(4)
        elements = {identity: identity}  # Use matrix as key (hashable)
        frontier = [identity]

        print(f"Generating F₄ Weyl group (expected order: {self.expected_order})")

        # BFS to generate group - stop when no new elements found
        for length in range(1, max_length + 1):
            new_frontier = []
            prev_count = len(elements)

            for elem in frontier:
                for refl in reflections:
                    # Apply reflection (exact matrix multiplication)
                    new_elem = refl @ elem

                    # Check if new (exact equality)
                    if new_elem not in elements:
                        elements[new_elem] = new_elem
                        new_frontier.append(new_elem)

            frontier = new_frontier

            print(f"  Length {length}: Found {len(elements)} elements")

            # Check if we've generated full group
            if len(elements) == self.expected_order:
                print(f"✓ Generated full F₄ Weyl group!")
                break
            elif len(frontier) == 0 or len(elements) == prev_count:
                # No new elements found - group is complete
                print(f"⚠ Generation stopped at {len(elements)} elements (expected {self.expected_order})")
                break

        self.elements = elements
        return elements

    def verify_group_properties(self) -> Dict[str, bool]:
        """Verify F₄ Weyl group properties (exact arithmetic)."""
        checks = {}

        # Check order
        checks['correct_order'] = (len(self.elements) == self.expected_order)

        # Check generators have order 2 (reflections)
        gens_order_2 = True
        identity = ExactMatrix.eye(4)
        for gen in self.generators:
            gen_squared = gen @ gen
            if gen_squared != identity:  # Exact equality
                gens_order_2 = False
                break
        checks['generators_order_2'] = gens_order_2

        # Check Cartan matrix consistency (exact)
        if self.cartan_matrix is not None and self.simple_roots is not None:
            cartan_consistent = True
            n = len(self.simple_roots)
            for i in range(n):
                for j in range(n):
                    alpha_i = self.simple_roots[i]
                    alpha_j = self.simple_roots[j]

                    # Cartan entry: 2⟨α_i, α_j⟩/⟨α_j, α_j⟩ (exact rational)
                    computed = 2 * alpha_i.dot(alpha_j) / alpha_j.dot(alpha_j)
                    expected = Fraction(self.cartan_matrix[i][j])

                    if computed != expected:  # Exact equality
                        cartan_consistent = False
                        print(f"  Cartan mismatch at [{i},{j}]: {computed} vs {expected}")
        else:
            cartan_consistent = False
        checks['cartan_consistent'] = cartan_consistent

        return checks

    def find_longest_element(self) -> Tuple[ExactMatrix, int]:
        """
        Find the longest element of F₄ Weyl group.

        This is the unique element of maximal length.
        In F₄, it has length 24.
        """
        max_length = 0
        longest = None

        for elem in self.elements:
            # Compute trace (exact)
            trace = sum(elem.data[i][i] for i in range(elem.rows))

            # In F₄, longest element has trace -4 (exact)
            if trace == Fraction(-4):
                longest = elem
                max_length = 24  # Known length
                break

        return longest, max_length


def verify_f4_weyl_group():
    """
    Main function to generate and verify F₄ Weyl group (exact arithmetic).

    Returns:
        Dictionary with Weyl group data and verification results
    """
    print("="*60)
    print("F₄ WEYL GROUP GENERATION (EXACT ARITHMETIC)")
    print("="*60)

    # Create F₄ Weyl group
    weyl = F4WeylGroup()

    # Get Cartan matrix
    cartan = weyl.get_f4_cartan_matrix()
    print("\nF₄ Cartan matrix (exact integers):")
    for row in cartan:
        print(f"  {row}")

    # Generate simple roots
    simple_roots = weyl.generate_simple_roots()
    print(f"\nGenerated {len(simple_roots)} simple roots (exact rationals)")

    # Generate full Weyl group
    elements = weyl.generate_weyl_group()

    # Verify properties
    print("\nVerifying F₄ Weyl group properties:")
    checks = weyl.verify_group_properties()
    for prop, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {prop}")

    # Find longest element
    longest, length = weyl.find_longest_element()
    if longest is not None:
        print(f"\n✓ Found longest element (length {length})")

    # Summary
    result = {
        'order': len(elements),
        'expected_order': weyl.expected_order,
        'rank': weyl.rank,
        'num_generators': len(weyl.generators),
        'cartan_matrix': cartan,  # Already a list
        'verification': checks,
        'has_longest_element': longest is not None
    }

    if len(elements) == weyl.expected_order:
        print(f"\n✓✓✓ Successfully generated F₄ Weyl group with {len(elements)} elements!")
    else:
        print(f"\n⚠ Generated {len(elements)} elements (expected {weyl.expected_order})")

    return result


if __name__ == "__main__":
    f4_weyl_data = verify_f4_weyl_group()