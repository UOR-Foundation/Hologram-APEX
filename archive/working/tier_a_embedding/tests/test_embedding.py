"""
Tests for the embedding search module.
"""
import unittest
from typing import List

from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from common_types import ATLAS_VERTEX_COUNT, E8_ROOT_COUNT


class TestEmbeddingConstraints(unittest.TestCase):
    """Test embedding constraint validation."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = EmbeddingConstraints()
        self.assertEqual(constraints.max_solutions, 1)
        self.assertIsNone(constraints.target_signs)
        self.assertIsNone(constraints.required_mapping)
        self.assertFalse(constraints.verbose)

    def test_custom_constraints(self):
        """Test custom constraint creation."""
        constraints = EmbeddingConstraints(
            max_solutions=10,
            target_signs=48,
            required_mapping={0: 5},
            verbose=True
        )
        self.assertEqual(constraints.max_solutions, 10)
        self.assertEqual(constraints.target_signs, 48)
        self.assertEqual(constraints.required_mapping, {0: 5})
        self.assertTrue(constraints.verbose)


class TestEmbeddingSearch(unittest.TestCase):
    """Test embedding search functionality."""

    def setUp(self):
        """Create components for testing."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()
        self.search = EmbeddingSearch(self.atlas, self.e8)

    def test_initialization(self):
        """Test search initialization."""
        self.assertIsNotNone(self.search.atlas)
        self.assertIsNotNone(self.search.e8)
        self.assertEqual(len(self.search.mapping), ATLAS_VERTEX_COUNT)
        self.assertEqual(len(self.search.used_roots), E8_ROOT_COUNT)
        self.assertEqual(len(self.search.available_at), ATLAS_VERTEX_COUNT)

    def test_mapping_initialization(self):
        """Test that mapping starts unassigned."""
        for i in range(ATLAS_VERTEX_COUNT):
            self.assertEqual(self.search.mapping[i], -1)

    def test_unity_constraint_setup(self):
        """Test unity constraint initialization."""
        # Unity indices should be set
        self.assertGreater(len(self.search.unity_indices), 0)
        # All should be valid indices
        for idx in self.search.unity_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, ATLAS_VERTEX_COUNT)

    def test_is_valid_checks(self):
        """Test validity checking methods."""
        # Initially no mapping, so checks should be permissive
        # Try assigning a valid root
        self.search.mapping[0] = 0
        self.search.used_roots[0] = True

        # Check that we can't reuse the same root
        self.assertFalse(self.search._is_valid(1, 0))

        # Reset for other tests
        self.search.mapping[0] = -1
        self.search.used_roots[0] = False

    def test_adjacency_preservation(self):
        """Test that adjacency checking works."""
        # Get first vertex and its neighbors
        v = 0
        neighbors = list(self.atlas.adjacency[v])
        if len(neighbors) > 0:
            neighbor = neighbors[0]

            # Find two adjacent roots
            root1, root2 = -1, -1
            for i in range(E8_ROOT_COUNT):
                for j in range(i + 1, E8_ROOT_COUNT):
                    if self.e8.adjacency[i][j]:
                        root1, root2 = i, j
                        break
                if root1 >= 0:
                    break

            # Assign adjacent roots to adjacent vertices
            self.search.mapping[v] = root1
            self.search.used_roots[root1] = True
            self.search.mapping[neighbor] = root2
            self.search.used_roots[root2] = True

            # Check they are recognized as adjacent
            self.assertTrue(self.e8.adjacency[root1][root2])

            # Clean up
            self.search.mapping[v] = -1
            self.search.mapping[neighbor] = -1
            self.search.used_roots[root1] = False
            self.search.used_roots[root2] = False

    def test_mirror_pairing_check(self):
        """Test mirror pairing validation."""
        # Get a vertex and its mirror
        v = 0
        mirror = self.atlas.tau[v]

        # Assign a root to v
        root_v = 0
        neg_root = self.e8.negation_table[root_v]

        self.search.mapping[v] = root_v
        # Mirror must map to negation
        valid = self.search._check_mirror_constraint(mirror, neg_root)
        self.assertTrue(valid, "Mirror pairing should be valid")

        # Wrong pairing should fail
        wrong_root = (neg_root + 1) % E8_ROOT_COUNT
        if wrong_root != neg_root:  # Make sure it's different
            valid = self.search._check_mirror_constraint(mirror, wrong_root)
            self.assertFalse(valid, "Wrong mirror pairing should be invalid")

        # Clean up
        self.search.mapping[v] = -1


class TestEmbeddingSolution(unittest.TestCase):
    """Test properties of found embeddings."""

    def setUp(self):
        """Create and run limited search."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()

    def test_search_finds_solution(self):
        """Test that search can find at least one embedding."""
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        self.assertGreaterEqual(len(solutions), 1, "Should find at least one embedding")

        if len(solutions) > 0:
            solution = solutions[0]
            self._validate_solution(solution)

    def _validate_solution(self, mapping: List[int]):
        """Validate a solution satisfies all constraints."""
        # Check injectivity
        used = set()
        for root_idx in mapping:
            self.assertNotIn(root_idx, used, "Mapping not injective")
            used.add(root_idx)

        # Check mirror pairing
        for i in range(ATLAS_VERTEX_COUNT):
            j = self.atlas.tau[i]
            self.assertEqual(
                self.e8.negation_table[mapping[i]],
                mapping[j],
                f"Mirror pairing violated at vertex {i}"
            )

        # Check edge preservation
        for a, b in self.atlas.edges:
            root_a = mapping[a]
            root_b = mapping[b]
            self.assertTrue(
                self.e8.adjacency[root_a][root_b],
                f"Edge ({a},{b}) not preserved"
            )

        # Check unity constraint
        unity_indices = self.atlas.unity_indices
        if unity_indices:
            sum_vec = [0] * 8
            for idx in unity_indices:
                root = self.e8.roots[mapping[idx]]
                for k in range(8):
                    sum_vec[k] += root[k]
            for k in range(8):
                self.assertEqual(sum_vec[k], 0, f"Unity sum not zero at dimension {k}")

    def test_sign_class_counting(self):
        """Test sign class computation."""
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            target_signs=48  # Look for specific sign count
        )
        solutions = search.search(constraints)

        if len(solutions) > 0:
            solution = solutions[0]
            # Count sign classes
            sign_classes = set()
            for root_idx in solution:
                neg_idx = self.e8.negation_table[root_idx]
                sign_classes.add(min(root_idx, neg_idx))

            self.assertEqual(len(sign_classes), 48,
                             "Solution should use exactly 48 sign classes")


class TestBacktrackingOptimizations(unittest.TestCase):
    """Test search optimizations and constraint propagation."""

    def setUp(self):
        """Create components for testing."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()

    def test_constraint_propagation(self):
        """Test that constraints properly propagate."""
        search = EmbeddingSearch(self.atlas, self.e8)

        # Assign first vertex
        search.mapping[0] = 0
        search.used_roots[0] = True

        # Check mirror constraint propagates
        mirror = self.atlas.tau[0]
        neg_root = self.e8.negation_table[0]

        # Mirror vertex should be constrained
        available_before = len([r for r in range(E8_ROOT_COUNT)
                                 if search._is_valid(mirror, r)])
        self.assertEqual(available_before, 1,
                         "Mirror vertex should have exactly one valid option")

    def test_available_roots_tracking(self):
        """Test available roots are properly tracked."""
        search = EmbeddingSearch(self.atlas, self.e8)

        # Initially all roots available
        for v in range(ATLAS_VERTEX_COUNT):
            initial_available = len([r for r in range(E8_ROOT_COUNT)
                                     if not search.used_roots[r]])
            self.assertEqual(initial_available, E8_ROOT_COUNT)

        # Assign some roots and check availability decreases
        for i in range(5):
            search.mapping[i] = i
            search.used_roots[i] = True

        remaining = len([r for r in range(E8_ROOT_COUNT)
                         if not search.used_roots[r]])
        self.assertEqual(remaining, E8_ROOT_COUNT - 5)


if __name__ == "__main__":
    unittest.main()