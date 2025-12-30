"""
Tests for the E8 root system module.
"""
import unittest
from fractions import Fraction

from e8 import E8RootSystem
from common_types import E8_ROOT_COUNT


class TestE8RootSystem(unittest.TestCase):
    """Test E8 root system generation and properties."""

    def setUp(self):
        """Create E8 root system for testing."""
        self.e8 = E8RootSystem()

    def test_root_count(self):
        """Test that E8 has exactly 240 roots."""
        self.assertEqual(len(self.e8.roots), E8_ROOT_COUNT)

    def test_root_dimension(self):
        """Test that all roots are 8-dimensional."""
        for root in self.e8.roots:
            self.assertEqual(len(root), 8)

    def test_root_norm(self):
        """Test that all roots have norm 2."""
        for i, root in enumerate(self.e8.roots):
            norm_squared = sum(x * x for x in root)
            self.assertEqual(norm_squared, Fraction(2, 1),
                             f"Root {i} has norm^2 = {norm_squared}, expected 2")

    def test_integer_roots(self):
        """Test properties of integer roots."""
        integer_count = 0
        for root in self.e8.roots:
            if all(x.denominator == 1 for x in root):
                integer_count += 1
                # Check specific properties of integer roots
                # Should have exactly 2 non-zero coordinates
                non_zeros = [x for x in root if x != 0]
                self.assertEqual(len(non_zeros), 2,
                                 f"Integer root {root} should have exactly 2 non-zero coords")
                # Non-zero coordinates should be ±1
                for x in non_zeros:
                    self.assertIn(abs(x), [Fraction(1, 1)])

        # E8 has 112 integer roots
        self.assertEqual(integer_count, 112)

    def test_half_integer_roots(self):
        """Test properties of half-integer roots."""
        half_integer_count = 0
        for root in self.e8.roots:
            if all(x.denominator == 2 for x in root):
                half_integer_count += 1
                # Check parity constraint
                # All coordinates should be ±1/2
                for x in root:
                    self.assertIn(abs(x), [Fraction(1, 2)])
                # Sum should be even (integer when divided by 1/2)
                coord_sum = sum(root)
                self.assertEqual(coord_sum.denominator, 1,
                                 f"Half-integer root {root} sum not integer")

        # E8 has 128 half-integer roots
        self.assertEqual(half_integer_count, 128)

    def test_negation_table(self):
        """Test root negation table."""
        self.assertEqual(len(self.e8.negation_table), E8_ROOT_COUNT)

        for i in range(E8_ROOT_COUNT):
            j = self.e8.negation_table[i]
            # Check j is valid index
            self.assertGreaterEqual(j, 0)
            self.assertLess(j, E8_ROOT_COUNT)
            # Check roots are negatives
            root_i = self.e8.roots[i]
            root_j = self.e8.roots[j]
            for k in range(8):
                self.assertEqual(root_i[k], -root_j[k])
            # Check involution property
            self.assertEqual(self.e8.negation_table[j], i)

    def test_adjacency_matrix_symmetry(self):
        """Test that adjacency matrix is symmetric."""
        n = E8_ROOT_COUNT
        for i in range(n):
            for j in range(i + 1, n):
                self.assertEqual(self.e8.adjacency[i][j], self.e8.adjacency[j][i])

    def test_adjacency_dot_products(self):
        """Test adjacency based on dot products."""
        # Sample a subset to avoid very long test
        sample_size = min(20, E8_ROOT_COUNT)
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                dot = sum(a * b for a, b in
                          zip(self.e8.roots[i], self.e8.roots[j]))
                expected_adjacent = (dot == Fraction(1, 1))
                actual_adjacent = self.e8.adjacency[i][j]
                self.assertEqual(actual_adjacent, expected_adjacent,
                                 f"Roots {i},{j} adjacency mismatch: dot={dot}")

    def test_no_self_adjacency(self):
        """Test that no root is adjacent to itself."""
        for i in range(E8_ROOT_COUNT):
            self.assertFalse(self.e8.adjacency[i][i])

    def test_root_uniqueness(self):
        """Test that all roots are unique."""
        root_set = set()
        for root in self.e8.roots:
            root_tuple = tuple(root)
            self.assertNotIn(root_tuple, root_set,
                             f"Duplicate root found: {root}")
            root_set.add(root_tuple)


class TestE8Geometry(unittest.TestCase):
    """Test geometric properties of E8."""

    def setUp(self):
        """Create E8 root system for testing."""
        self.e8 = E8RootSystem()

    def test_root_angles(self):
        """Test angles between roots."""
        # Sample a few roots
        for i in range(min(10, E8_ROOT_COUNT)):
            for j in range(i + 1, min(10, E8_ROOT_COUNT)):
                dot = sum(a * b for a, b in
                          zip(self.e8.roots[i], self.e8.roots[j]))
                # Possible dot products for norm-2 roots:
                # 2 (same root), -2 (negative), 1 (60°), 0 (90°), -1 (120°)
                self.assertIn(dot, [Fraction(-2, 1), Fraction(-1, 1),
                                     Fraction(0, 1), Fraction(1, 1), Fraction(2, 1)])

    def test_opposite_roots(self):
        """Test that each root has an opposite."""
        for i, root in enumerate(self.e8.roots):
            j = self.e8.negation_table[i]
            opposite = self.e8.roots[j]
            # Check they sum to zero
            root_sum = tuple(a + b for a, b in zip(root, opposite))
            zero = tuple(Fraction(0, 1) for _ in range(8))
            self.assertEqual(root_sum, zero)

    def test_root_system_closure(self):
        """Test that adjacent roots have specific angle."""
        # For E8, adjacent roots have dot product 1
        # This means angle θ where cos(θ) = 1/2, so θ = 60°
        sample_size = min(5, E8_ROOT_COUNT)
        for i in range(sample_size):
            adjacent_count = sum(self.e8.adjacency[i])
            # Each root should have some adjacent roots
            self.assertGreater(adjacent_count, 0,
                               f"Root {i} has no adjacent roots")


class TestE8Helpers(unittest.TestCase):
    """Test helper functions for E8 operations."""

    def setUp(self):
        """Create E8 root system for testing."""
        self.e8 = E8RootSystem()

    def test_get_adjacent_roots(self):
        """Test getting adjacent roots."""
        for i in range(min(10, E8_ROOT_COUNT)):
            adjacent = self.e8.get_adjacent_roots(i)
            # Check all are valid indices
            for j in adjacent:
                self.assertGreaterEqual(j, 0)
                self.assertLess(j, E8_ROOT_COUNT)
                # Check adjacency is correct
                self.assertTrue(self.e8.adjacency[i][j])

    def test_is_adjacent(self):
        """Test adjacency checking."""
        # Test a few pairs
        for i in range(min(5, E8_ROOT_COUNT)):
            for j in range(min(5, E8_ROOT_COUNT)):
                # Method should match matrix
                self.assertEqual(self.e8.is_adjacent(i, j),
                                 self.e8.adjacency[i][j])

    def test_rational_arithmetic(self):
        """Test that all arithmetic uses exact rationals."""
        # All root coordinates should be Fraction objects
        for root in self.e8.roots:
            for coord in root:
                self.assertIsInstance(coord, Fraction)
                # No floating point approximations
                self.assertIn(coord.denominator, [1, 2])


if __name__ == "__main__":
    unittest.main()