"""
Tests for certificate generation and verification.
"""
import unittest
import json
from fractions import Fraction

from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from certificate import (
    CertificateGenerator,
    CertificateVerifier,
    CertificateFormat,
    CERTIFICATE_VERSION,
    validate_certificate_format,
    format_root,
    parse_root,
    create_certificate,
    verify_certificate,
)


class TestCertificateFormat(unittest.TestCase):
    """Test certificate format utilities."""

    def test_format_root(self):
        """Test root formatting."""
        root = (Fraction(1, 1), Fraction(-1, 2), Fraction(0, 1),
                Fraction(1, 2), Fraction(0, 1), Fraction(0, 1),
                Fraction(0, 1), Fraction(0, 1))
        formatted = format_root(root)

        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 8)
        self.assertEqual(formatted[0], "1")
        self.assertEqual(formatted[1], "-1/2")
        self.assertEqual(formatted[2], "0")
        self.assertEqual(formatted[3], "1/2")

    def test_parse_root(self):
        """Test root parsing."""
        root_strings = ["1", "-1/2", "0", "1/2", "0", "0", "0", "0"]
        root = parse_root(root_strings)

        self.assertIsInstance(root, tuple)
        self.assertEqual(len(root), 8)
        self.assertEqual(root[0], Fraction(1, 1))
        self.assertEqual(root[1], Fraction(-1, 2))
        self.assertEqual(root[2], Fraction(0, 1))
        self.assertEqual(root[3], Fraction(1, 2))

    def test_format_parse_roundtrip(self):
        """Test format and parse are inverses."""
        original = (Fraction(1, 1), Fraction(-1, 2), Fraction(0, 1),
                    Fraction(1, 2), Fraction(3, 2), Fraction(-5, 2),
                    Fraction(7, 1), Fraction(-4, 1))
        formatted = format_root(original)
        parsed = parse_root(formatted)
        self.assertEqual(parsed, original)

    def test_certificate_format_validation(self):
        """Test certificate format validation."""
        # Valid minimal certificate
        valid_cert = {
            "version": CERTIFICATE_VERSION,
            "algorithm": "backtrack",
            "timestamp": "2024-01-01T00:00:00Z",
            "atlas_vertex_count": 96,
            "e8_root_count": 240,
            "atlas_labels": ["(0, 0, 0, -1, 0, 0)"] * 96,
            "roots": {str(i): ["0"] * 8 for i in range(240)},
            "mapping": {str(i): str(i) for i in range(96)},
            "unity_indices": [0, 1, 2],
            "sign_classes_used": 48,
        }

        is_valid, msg = validate_certificate_format(valid_cert)
        self.assertTrue(is_valid, f"Valid certificate rejected: {msg}")

        # Invalid version
        invalid_cert = valid_cert.copy()
        invalid_cert["version"] = "0.0.0"
        is_valid, msg = validate_certificate_format(invalid_cert)
        self.assertFalse(is_valid)

        # Missing field
        invalid_cert = valid_cert.copy()
        del invalid_cert["mapping"]
        is_valid, msg = validate_certificate_format(invalid_cert)
        self.assertFalse(is_valid)


class TestCertificateGenerator(unittest.TestCase):
    """Test certificate generation."""

    def setUp(self):
        """Create components and find an embedding."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)
        self.assertGreater(len(solutions), 0, "Need at least one solution to test")
        self.mapping = solutions[0]

    def test_generator_creates_certificate(self):
        """Test that generator creates valid certificate."""
        generator = CertificateGenerator(self.atlas, self.e8)
        cert_dict = generator.generate(self.mapping)

        # Check basic structure
        self.assertIn("version", cert_dict)
        self.assertIn("mapping", cert_dict)
        self.assertIn("roots", cert_dict)
        self.assertIn("atlas_labels", cert_dict)

        # Check counts
        self.assertEqual(cert_dict["atlas_vertex_count"], 96)
        self.assertEqual(cert_dict["e8_root_count"], 240)
        self.assertEqual(len(cert_dict["mapping"]), 96)
        self.assertEqual(len(cert_dict["roots"]), 240)

    def test_certificate_json_serializable(self):
        """Test certificate can be serialized to JSON."""
        generator = CertificateGenerator(self.atlas, self.e8)
        cert_dict = generator.generate(self.mapping)

        # Should serialize without error
        cert_json = json.dumps(cert_dict, indent=2)
        self.assertIsInstance(cert_json, str)

        # Should deserialize correctly
        parsed = json.loads(cert_json)
        self.assertEqual(parsed["version"], cert_dict["version"])

    def test_create_certificate_function(self):
        """Test convenience function."""
        cert_json = create_certificate(self.mapping, self.atlas, self.e8)
        self.assertIsInstance(cert_json, str)

        # Should be valid JSON
        cert_dict = json.loads(cert_json)
        self.assertIn("version", cert_dict)


class TestCertificateVerifier(unittest.TestCase):
    """Test certificate verification."""

    def setUp(self):
        """Create components and generate a certificate."""
        self.atlas = AtlasGraph()
        self.e8 = E8RootSystem()
        search = EmbeddingSearch(self.atlas, self.e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)
        self.assertGreater(len(solutions), 0, "Need at least one solution to test")
        self.mapping = solutions[0]
        self.cert_json = create_certificate(self.mapping, self.atlas, self.e8)

    def test_verify_valid_certificate(self):
        """Test verification of valid certificate."""
        verifier = CertificateVerifier()
        is_valid, msg = verifier.verify(self.cert_json)
        self.assertTrue(is_valid, f"Valid certificate rejected: {msg}")

    def test_verify_catches_invalid_json(self):
        """Test verification rejects invalid JSON."""
        verifier = CertificateVerifier()
        is_valid, msg = verifier.verify("not valid json")
        self.assertFalse(is_valid)
        self.assertIn("JSON", msg)

    def test_verify_catches_wrong_format(self):
        """Test verification rejects wrong format."""
        verifier = CertificateVerifier()
        invalid_cert = json.dumps({"wrong": "format"})
        is_valid, msg = verifier.verify(invalid_cert)
        self.assertFalse(is_valid)
        self.assertIn("Format", msg)

    def test_verify_catches_invalid_mapping(self):
        """Test verification catches invalid mappings."""
        cert_dict = json.loads(self.cert_json)

        # Break injectivity
        cert_dict["mapping"]["0"] = cert_dict["mapping"]["1"]
        broken_json = json.dumps(cert_dict)

        verifier = CertificateVerifier()
        is_valid, msg = verifier.verify(broken_json)
        self.assertFalse(is_valid)
        self.assertIn("inject", msg.lower())

    def test_verify_catches_broken_mirror(self):
        """Test verification catches broken mirror pairing."""
        cert_dict = json.loads(self.cert_json)

        # Break mirror pairing by swapping two mappings
        if "0" in cert_dict["mapping"] and "1" in cert_dict["mapping"]:
            cert_dict["mapping"]["0"], cert_dict["mapping"]["1"] = \
                cert_dict["mapping"]["1"], cert_dict["mapping"]["0"]
        broken_json = json.dumps(cert_dict)

        verifier = CertificateVerifier()
        is_valid, msg = verifier.verify(broken_json)
        self.assertFalse(is_valid)
        # Should catch either mirror or edge issue

    def test_verify_convenience_function(self):
        """Test convenience verification function."""
        is_valid = verify_certificate(self.cert_json)
        self.assertTrue(is_valid)


class TestCertificateIndependence(unittest.TestCase):
    """Test that verifier is independent of generator."""

    def test_verifier_reconstructs_structures(self):
        """Test verifier rebuilds all structures from certificate."""
        # Create minimal certificate
        atlas = AtlasGraph()
        e8 = E8RootSystem()
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(max_solutions=1)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            mapping = solutions[0]
            cert_json = create_certificate(mapping, atlas, e8)

            # Create new verifier with no access to original structures
            verifier = CertificateVerifier()

            # Verification should work independently
            is_valid, msg = verifier.verify(cert_json)
            self.assertTrue(is_valid, f"Independent verification failed: {msg}")


class TestCertificateProperties(unittest.TestCase):
    """Test specific properties captured in certificates."""

    def setUp(self):
        """Create test certificate."""
        atlas = AtlasGraph()
        e8 = E8RootSystem()
        search = EmbeddingSearch(atlas, e8)
        constraints = EmbeddingConstraints(max_solutions=1, target_signs=48)
        solutions = search.search(constraints)

        if len(solutions) > 0:
            self.mapping = solutions[0]
            self.cert_json = create_certificate(self.mapping, atlas, e8)
            self.cert_dict = json.loads(self.cert_json)
        else:
            self.skipTest("No embedding found for testing")

    def test_certificate_records_sign_classes(self):
        """Test that certificate records sign class count."""
        self.assertIn("sign_classes_used", self.cert_dict)
        self.assertIsInstance(self.cert_dict["sign_classes_used"], int)
        self.assertGreaterEqual(self.cert_dict["sign_classes_used"], 1)
        self.assertLessEqual(self.cert_dict["sign_classes_used"], 48)

    def test_certificate_records_unity_constraint(self):
        """Test that certificate records unity indices."""
        self.assertIn("unity_indices", self.cert_dict)
        self.assertIsInstance(self.cert_dict["unity_indices"], list)
        # Unity indices should be for e7=0 vertices
        self.assertEqual(len(self.cert_dict["unity_indices"]), 48)

    def test_certificate_metadata(self):
        """Test certificate includes metadata."""
        self.assertIn("timestamp", self.cert_dict)
        self.assertIn("algorithm", self.cert_dict)
        self.assertEqual(self.cert_dict["algorithm"], "backtrack")


if __name__ == "__main__":
    unittest.main()