"""
End-to-end integration tests for the Tier-A embedding package.
"""
import unittest
import json
import time
from typing import List

from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from certificate import create_certificate, verify_certificate
from canonicalization import canonicalize_embedding, CanonicalSelector


class TestCompleteEmbeddingPipeline(unittest.TestCase):
    """Test the complete embedding search and verification pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline from search to certificate."""
        # 1. Create structures
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # 2. Search for embedding
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            target_signs=48,
            verbose=False
        )
        solutions = search.search(constraints)

        self.assertGreaterEqual(len(solutions), 1,
                                "Should find at least one embedding")

        # 3. Generate certificate
        mapping = solutions[0]
        cert_json = create_certificate(mapping, atlas, e8)

        self.assertIsInstance(cert_json, str)
        cert_dict = json.loads(cert_json)
        self.assertEqual(cert_dict["sign_classes_used"], 48)

        # 4. Verify certificate independently
        is_valid = verify_certificate(cert_json)
        self.assertTrue(is_valid, "Certificate should be valid")

    def test_multiple_solutions_canonicalization(self):
        """Test finding and canonicalizing multiple solutions."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Search for multiple solutions
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=3,
            target_signs=48,
            verbose=False
        )
        solutions = search.search(constraints)

        if len(solutions) > 1:
            # Canonicalize solutions
            canonical = canonicalize_embedding(solutions, atlas, e8)

            # Verify canonical is valid
            cert_json = create_certificate(canonical, atlas, e8)
            is_valid = verify_certificate(cert_json)
            self.assertTrue(is_valid)

            # Check it's one of the originals
            self.assertIn(canonical, solutions)

    def test_constrained_search(self):
        """Test search with specific constraints."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Search with required mapping
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            required_mapping={0: 0},  # Fix first vertex to first root
            verbose=False
        )
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]
            # Check constraint is satisfied
            self.assertEqual(mapping[0], 0)

            # Verify solution is still valid
            cert_json = create_certificate(mapping, atlas, e8)
            is_valid = verify_certificate(cert_json)
            self.assertTrue(is_valid)


class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance characteristics."""

    def test_search_time_reasonable(self):
        """Test that search completes in reasonable time."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        start_time = time.time()

        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            verbose=False
        )
        solutions = search.search(constraints)

        elapsed = time.time() - start_time

        # Should find solution quickly (under 10 seconds)
        self.assertLess(elapsed, 10.0,
                        f"Search took {elapsed:.2f}s, expected < 10s")

        # Should find at least one solution
        self.assertGreaterEqual(len(solutions), 1)

    def test_certificate_generation_speed(self):
        """Test certificate generation is fast."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Get a solution first
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]

            start_time = time.time()
            cert_json = create_certificate(mapping, atlas, e8)
            elapsed = time.time() - start_time

            # Should be very fast (under 1 second)
            self.assertLess(elapsed, 1.0,
                            f"Certificate generation took {elapsed:.2f}s")

    def test_verification_speed(self):
        """Test certificate verification is fast."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Get a certificate
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]
            cert_json = create_certificate(mapping, atlas, e8)

            start_time = time.time()
            is_valid = verify_certificate(cert_json)
            elapsed = time.time() - start_time

            # Should be very fast (under 0.5 seconds)
            self.assertLess(elapsed, 0.5,
                            f"Verification took {elapsed:.2f}s")
            self.assertTrue(is_valid)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity throughout the pipeline."""

    def test_no_floating_point_errors(self):
        """Test that all computations use exact rational arithmetic."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Run a search
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]

            # Generate certificate
            cert_json = create_certificate(mapping, atlas, e8)
            cert_dict = json.loads(cert_json)

            # Check all root coordinates are exact fractions
            for i in range(240):
                root_strs = cert_dict["roots"][str(i)]
                for coord_str in root_strs:
                    # Should be integer or fraction format
                    self.assertRegex(coord_str,
                                     r'^-?\d+(/\d+)?$',
                                     f"Coordinate {coord_str} not in fraction format")

    def test_certificate_consistency(self):
        """Test that certificates are consistent across runs."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Search with fixed constraint for reproducibility
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            required_mapping={0: 0}  # Fix first mapping
        )
        solutions1 = search.search(constraints)

        if len(solutions1) > 0:
            # Run search again
            search2 = EmbeddingSearch(atlas, e8)
            solutions2 = search2.search(constraints)

            # Should find same solution
            self.assertEqual(len(solutions1), len(solutions2))
            self.assertEqual(solutions1[0], solutions2[0])


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_certificate_rejection(self):
        """Test that invalid certificates are rejected."""
        # Test various invalid certificates
        test_cases = [
            "not json",
            "{}",
            '{"version": "wrong"}',
            '{"mapping": "not a dict"}',
        ]

        for invalid_cert in test_cases:
            is_valid = verify_certificate(invalid_cert, verbose=False)
            self.assertFalse(is_valid,
                             f"Should reject invalid certificate: {invalid_cert[:50]}")

    def test_impossible_constraints(self):
        """Test handling of impossible constraints."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        search = EmbeddingSearch(atlas, e8)

        # Impossible: map adjacent vertices to same root
        constraints = EmbeddingConstraints(
            max_solutions=1,
            required_mapping={0: 0, 1: 0}  # Adjacent vertices to same root
        )
        solutions = search.search(constraints)

        # Should find no solutions
        self.assertEqual(len(solutions), 0)

    def test_empty_solutions_handling(self):
        """Test handling of empty solution sets."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        # Try to canonicalize empty set
        selector = CanonicalSelector(
            tau=atlas.tau,
            negation_table=e8.negation_table,
            labels=atlas.labels,
            roots=e8.roots
        )

        with self.assertRaises(ValueError):
            selector.canonicalize([])


class TestModuleInterfaces(unittest.TestCase):
    """Test that module interfaces work correctly together."""

    def test_all_modules_importable(self):
        """Test that all modules can be imported."""
        # These imports should all work
        import atlas
        import e8
        import embedding
        import certificate
        import canonicalization

        # Check main classes exist
        self.assertTrue(hasattr(atlas, 'AtlasGraph'))
        self.assertTrue(hasattr(e8, 'E8RootSystem'))
        self.assertTrue(hasattr(embedding, 'EmbeddingSearch'))
        self.assertTrue(hasattr(certificate, 'CertificateGenerator'))
        self.assertTrue(hasattr(canonicalization, 'CanonicalSelector'))

    def test_public_api_complete(self):
        """Test that public APIs are complete."""
        # Check __all__ exports
        import atlas
        import e8
        import embedding
        import certificate
        import canonicalization

        # Each module should export its main components
        self.assertIn('AtlasGraph', atlas.__all__)
        self.assertIn('E8RootSystem', e8.__all__)
        self.assertIn('EmbeddingSearch', embedding.__all__)
        self.assertIn('CertificateGenerator', certificate.__all__)
        self.assertIn('CanonicalSelector', canonicalization.__all__)


class TestRegressionTests(unittest.TestCase):
    """Regression tests to ensure refactoring preserved behavior."""

    def test_finds_known_embedding(self):
        """Test that we can still find known embeddings."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()

        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(
            max_solutions=1,
            target_signs=48
        )
        solutions = search.search(constraints)

        # Should find embedding with 48 sign classes
        self.assertGreaterEqual(len(solutions), 1)

        if len(solutions) > 0:
            mapping = solutions[0]

            # Count sign classes
            sign_classes = set()
            for root_idx in mapping:
                neg_idx = e8.negation_table[root_idx]
                sign_classes.add(min(root_idx, neg_idx))

            self.assertEqual(len(sign_classes), 48,
                             "Should use exactly 48 sign classes")


if __name__ == "__main__":
    unittest.main()