"""
Tests for the Atlas graph module.
"""
import unittest
from typing import Set, Tuple

from atlas import AtlasGraph
from common_types import Label, ATLAS_VERTEX_COUNT


class TestAtlasGraph(unittest.TestCase):
    """Test Atlas graph construction and properties."""

    def setUp(self):
        """Create Atlas graph for testing."""
        self.graph = AtlasGraph()

    def test_vertex_count(self):
        """Test that Atlas has exactly 96 vertices."""
        self.assertEqual(len(self.graph.labels), ATLAS_VERTEX_COUNT)

    def test_label_uniqueness(self):
        """Test that all labels are unique."""
        label_set = set(self.graph.labels)
        self.assertEqual(len(label_set), ATLAS_VERTEX_COUNT)

    def test_label_format(self):
        """Test that labels have correct format."""
        for label in self.graph.labels:
            # Check it's a tuple of 6 integers
            self.assertIsInstance(label, tuple)
            self.assertEqual(len(label), 6)
            for val in label:
                self.assertIsInstance(val, int)

            # Check specific bit constraints
            e1, e2, e3, d45, e6, e7 = label
            self.assertIn(e1, [0, 1])
            self.assertIn(e2, [0, 1])
            self.assertIn(e3, [0, 1])
            self.assertIn(d45, [-1, 0, 1])
            self.assertIn(e6, [0, 1])
            self.assertIn(e7, [0, 1])

    def test_tau_pairing(self):
        """Test mirror pairing properties."""
        # Check tau is defined
        self.assertEqual(len(self.graph.tau), ATLAS_VERTEX_COUNT)

        # Check tau is an involution (tau(tau(i)) = i)
        for i in range(ATLAS_VERTEX_COUNT):
            j = self.graph.tau[i]
            self.assertEqual(self.graph.tau[j], i)

        # Check tau pairs differ only in e7
        for i, label in enumerate(self.graph.labels):
            j = self.graph.tau[i]
            mirror_label = self.graph.labels[j]

            # First 5 components should be same
            self.assertEqual(label[:5], mirror_label[:5])
            # e7 should be flipped
            self.assertEqual(label[5] + mirror_label[5], 1)

    def test_edge_symmetry(self):
        """Test edge set is symmetric."""
        for a, b in self.graph.edges:
            # Edges should be stored as (smaller, larger)
            self.assertLess(a, b)
            # Reverse edge should not be stored separately
            self.assertNotIn((b, a), self.graph.edges)

    def test_adjacency_consistency(self):
        """Test adjacency list matches edge set."""
        # Build adjacency from edges
        expected_adj = [set() for _ in range(ATLAS_VERTEX_COUNT)]
        for a, b in self.graph.edges:
            expected_adj[a].add(b)
            expected_adj[b].add(a)

        # Compare with graph's adjacency
        for i in range(ATLAS_VERTEX_COUNT):
            self.assertEqual(self.graph.adjacency[i], expected_adj[i])

    def test_vertex_degrees(self):
        """Test that all vertices have degree 5 or 6."""
        degrees = self.graph.degrees
        self.assertEqual(len(degrees), ATLAS_VERTEX_COUNT)

        for i, degree in enumerate(degrees):
            self.assertIn(degree, [5, 6],
                          f"Vertex {i} has degree {degree}, expected 5 or 6")

    def test_degree_sum(self):
        """Test that sum of degrees equals 2 * number of edges."""
        total_degree = sum(self.graph.degrees)
        expected = 2 * len(self.graph.edges)
        self.assertEqual(total_degree, expected)

    def test_edge_preservation_under_tau(self):
        """Test that tau preserves edge structure."""
        for a, b in self.graph.edges:
            # If (a,b) is an edge, then (tau(a), tau(b)) should also be
            ta = self.graph.tau[a]
            tb = self.graph.tau[b]

            # Check edge exists (in either order)
            edge_exists = (min(ta, tb), max(ta, tb)) in self.graph.edges
            self.assertTrue(edge_exists,
                            f"Edge ({a},{b}) maps to ({ta},{tb}) which is not an edge")

    def test_unity_indices_valid(self):
        """Test unity indices are valid vertex indices."""
        for idx in self.graph.unity_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, ATLAS_VERTEX_COUNT)

    def test_unity_indices_property(self):
        """Test unity indices have expected properties."""
        # Unity vertices should be 48 vertices with e7=0
        unity_labels = [self.graph.labels[i] for i in self.graph.unity_indices]

        for label in unity_labels:
            e7 = label[5]
            self.assertEqual(e7, 0, "Unity vertices should have e7=0")

    def test_neighbor_computation(self):
        """Test neighbor computation for sample vertices."""
        # Test a few specific vertices
        for i in range(min(10, ATLAS_VERTEX_COUNT)):
            label = self.graph.labels[i]
            neighbors = self.graph.adjacency[i]

            # Each neighbor should be valid
            for j in neighbors:
                self.assertGreaterEqual(j, 0)
                self.assertLess(j, ATLAS_VERTEX_COUNT)
                self.assertNotEqual(i, j)  # No self-loops

                # Check symmetric adjacency
                self.assertIn(i, self.graph.adjacency[j])


class TestAtlasConnectivity(unittest.TestCase):
    """Test connectivity properties of Atlas graph."""

    def setUp(self):
        """Create Atlas graph for testing."""
        self.graph = AtlasGraph()

    def test_graph_connected(self):
        """Test that the Atlas graph is connected."""
        # BFS from vertex 0
        visited = set([0])
        queue = [0]

        while queue:
            v = queue.pop(0)
            for u in self.graph.adjacency[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)

        # Should visit all vertices
        self.assertEqual(len(visited), ATLAS_VERTEX_COUNT,
                         "Graph is not connected")

    def test_no_self_loops(self):
        """Test that there are no self-loops."""
        for a, b in self.graph.edges:
            self.assertNotEqual(a, b)

    def test_no_multi_edges(self):
        """Test that there are no multi-edges."""
        # Edge set should have no duplicates by construction
        edge_list = list(self.graph.edges)
        edge_set = set(edge_list)
        self.assertEqual(len(edge_list), len(edge_set))


if __name__ == "__main__":
    unittest.main()