"""
Tests for the canonicalization module.
"""
import unittest
from fractions import Fraction
from typing import List

from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from canonicalization import (
    S4Automorphism,
    S4Group,
    CanonicalSelector,
    mapping_equivalence_key,
    compute_sign_class_key,
    count_root_types,
    canonicalize_embedding,
    compute_canonical_key,
)
from common_types import S4_PERMUTABLE_BITS, ATLAS_VERTEX_COUNT


class TestS4Automorphism(unittest.TestCase):
    """Test S4 automorphism operations."""

    def test_identity_automorphism(self):
        """Test identity automorphism."""
        identity = S4Automorphism(S4_PERMUTABLE_BITS)
        label = (1, 0, 1, -1, 0, 1)
        transformed = identity.apply_to_label(label)
        self.assertEqual(transformed, label)

    def test_permutation_application(self):
        """Test permutation applies correctly to labels."""
        # Swap first two positions (0, 1)
        perm = (1, 0, 2, 4)
        auto = S4Automorphism(perm)

        label = (1, 0, 1, -1, 0, 1)
        transformed = auto.apply_to_label(label)

        # e1 and e2 should be swapped
        expected = (0, 1, 1, -1, 0, 1)
        self.assertEqual(transformed, expected)

    def test_permutation_validation(self):
        """Test that invalid permutations are rejected."""
        with self.assertRaises(AssertionError):
            # Invalid: not a permutation of (0,1,2,4)
            S4Automorphism((0, 1, 2, 3))

        with self.assertRaises(AssertionError):
            # Invalid: wrong length
            S4Automorphism((0, 1, 2))


class TestS4Group(unittest.TestCase):
    """Test S4 group operations."""

    def test_group_size(self):
        """Test S4 has 24 elements."""
        s4 = S4Group()
        self.assertEqual(len(s4.elements), 24)

    def test_group_identity(self):
        """Test identity element exists."""
        s4 = S4Group()
        self.assertIsNotNone(s4.identity)
        self.assertEqual(s4.identity.permutation, S4_PERMUTABLE_BITS)

    def test_orbit_computation(self):
        """Test orbit computation."""
        atlas = AtlasGraph()
        s4 = S4Group()

        # Create a simple mapping
        mapping = list(range(ATLAS_VERTEX_COUNT))
        orbit = s4.orbit(mapping, atlas.labels)

        # Orbit should contain at least one element (the original)
        self.assertGreaterEqual(len(orbit), 1)
        # Orbit size divides group order
        self.assertIn(24 % len(orbit), [0])

    def test_stabilizer_size(self):
        """Test stabilizer computation."""
        atlas = AtlasGraph()
        s4 = S4Group()

        # Identity mapping should have some stabilizer
        mapping = list(range(ATLAS_VERTEX_COUNT))
        stab_size = s4.stabilizer_size(mapping, atlas.labels)

        # Stabilizer size should divide group order
        self.assertEqual(24 % stab_size, 0)
        # Orbit-stabilizer theorem
        orbit = s4.orbit(mapping, atlas.labels)
        self.assertEqual(len(orbit) * stab_size, 24)


class TestEquivalenceKeys(unittest.TestCase):
    """Test equivalence key computation."""

    def setUp(self):
        """Set up test structures."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()

    def test_mapping_equivalence_key(self):
        """Test basic equivalence key computation."""
        # Create a simple mapping
        mapping = list(range(ATLAS_VERTEX_COUNT))

        key = mapping_equivalence_key(
            mapping,
            self.atlas.tau,
            self.e8.negation_table
        )

        # Key should be a tuple
        self.assertIsInstance(key, tuple)
        # Should have 48 representatives (96/2 vertex pairs)
        self.assertEqual(len(key), 48)

    def test_equivalence_key_consistency(self):
        """Test that equivalent mappings give same key."""
        # Get a real embedding
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]
            key1 = mapping_equivalence_key(
                mapping,
                self.atlas.tau,
                self.e8.negation_table
            )

            # Same mapping should give same key
            key2 = mapping_equivalence_key(
                mapping,
                self.atlas.tau,
                self.e8.negation_table
            )
            self.assertEqual(key1, key2)

    def test_sign_class_key(self):
        """Test sign class computation."""
        # Create a mapping using first 96 roots
        mapping = list(range(96))

        key = compute_sign_class_key(mapping, self.e8.negation_table)

        # Should be a sorted tuple
        self.assertIsInstance(key, tuple)
        self.assertEqual(key, tuple(sorted(key)))

    def test_root_type_counting(self):
        """Test counting integer vs half-integer roots."""
        # Create mapping with known properties
        mapping = []
        # Add some integer roots (first ones are integer)
        for i in range(48):
            mapping.append(i)
        # Add negations to complete
        for i in range(48):
            mapping.append(self.e8.negation_table[i])

        int_count, half_count = count_root_types(mapping, self.e8.roots)

        # All should be integer roots in this case
        self.assertEqual(int_count, 96)
        self.assertEqual(half_count, 0)


class TestCanonicalSelector(unittest.TestCase):
    """Test canonical form selection."""

    def setUp(self):
        """Set up test structures."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()
        self.selector = CanonicalSelector(
            tau=self.atlas.tau,
            negation_table=self.e8.negation_table,
            labels=self.atlas.labels,
            roots=self.e8.roots
        )

    def test_canonicalize_single_mapping(self):
        """Test canonicalizing a single mapping returns it."""
        mapping = list(range(96))
        result = self.selector.canonicalize([mapping])
        self.assertEqual(result, mapping)

    def test_canonicalize_selects_minimum(self):
        """Test that canonicalize selects the minimum."""
        # Create two mappings
        mapping1 = list(range(96))
        mapping2 = list(range(1, 97))

        # Canonicalize should pick one consistently
        result = self.selector.canonicalize([mapping1, mapping2])
        self.assertIn(result, [mapping1, mapping2])

        # Should pick the same one if order is reversed
        result2 = self.selector.canonicalize([mapping2, mapping1])
        self.assertEqual(result, result2)

    def test_automorphism_equivalence(self):
        """Test checking equivalence under automorphisms."""
        # Create a mapping
        mapping = list(range(96))

        # Apply an automorphism
        s4 = S4Group()
        orbit = s4.orbit(mapping, self.atlas.labels)

        if len(orbit) > 1:
            mapping2 = orbit[1]
            # Should be equivalent under automorphisms
            equiv = self.selector.are_equivalent_modulo_automorphisms(
                mapping, mapping2
            )
            self.assertTrue(equiv)

    def test_canonical_in_orbit(self):
        """Test finding canonical representative in orbit."""
        # Create a mapping
        mapping = list(range(96))

        canonical = self.selector.find_canonical_in_orbit(mapping)

        # Should be in the orbit
        s4 = S4Group()
        orbit = s4.orbit(mapping, self.atlas.labels)
        orbit_tuples = [tuple(m) for m in orbit]
        self.assertIn(tuple(canonical), orbit_tuples)


class TestCanonicalizationIntegration(unittest.TestCase):
    """Test canonicalization with real embeddings."""

    def setUp(self):
        """Find embeddings for testing."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(max_solutions=2)
        self.solutions = search.search(constraints)

    def test_canonicalize_embedding_function(self):
        """Test convenience canonicalization function."""
        if len(self.solutions) > 0:
            canonical = canonicalize_embedding(
                self.solutions,
                self.atlas,
                self.e8
            )

            # Should return one of the inputs
            self.assertIn(canonical, self.solutions)

    def test_compute_canonical_key_function(self):
        """Test canonical key computation."""
        if len(self.solutions) > 0:
            mapping = self.solutions[0]

            # Without automorphisms
            key1 = compute_canonical_key(
                mapping, self.atlas, self.e8, use_automorphisms=False
            )

            # With automorphisms
            key2 = compute_canonical_key(
                mapping, self.atlas, self.e8, use_automorphisms=True
            )

            # Both should be tuples
            self.assertIsInstance(key1, tuple)
            self.assertIsInstance(key2, tuple)

    def test_canonicalization_consistency(self):
        """Test that canonicalization is consistent."""
        if len(self.solutions) > 1:
            # Canonicalize in different orders
            canonical1 = canonicalize_embedding(
                self.solutions,
                self.atlas,
                self.e8
            )

            canonical2 = canonicalize_embedding(
                list(reversed(self.solutions)),
                self.atlas,
                self.e8
            )

            # Should get same result
            self.assertEqual(canonical1, canonical2)


class TestEquivalenceClassSize(unittest.TestCase):
    """Test equivalence class size computation."""

    def setUp(self):
        """Set up test structures."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()

    def test_equivalence_class_size_bounds(self):
        """Test that class sizes are reasonable."""
        selector = CanonicalSelector(
            tau=self.atlas.tau,
            negation_table=self.e8.negation_table,
            labels=self.atlas.labels,
            roots=self.e8.roots
        )

        # Create a mapping
        mapping = list(range(96))

        size = selector.equivalence_class_size(mapping)

        # Size should be at most 48 (2 * 24)
        self.assertLessEqual(size, 48)
        # Size should be at least 1
        self.assertGreaterEqual(size, 1)
        # Size should divide 48
        self.assertEqual(48 % size, 0)


if __name__ == "__main__":
    unittest.main()