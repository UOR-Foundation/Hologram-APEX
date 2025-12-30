"""
Certificate generation module.

This module generates certificates for valid Tier-A embeddings.
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .format import CertificateFormat, format_root, CERTIFICATE_VERSION

class CertificateGenerator:
    """Generates certificates for embeddings."""

    def __init__(self, atlas_graph, e8_system):
        """
        Initialize certificate generator.

        Args:
            atlas_graph: AtlasGraph instance
            e8_system: E8RootSystem instance
        """
        self.atlas = atlas_graph
        self.e8 = e8_system

    def generate(
        self,
        mapping,
        unity_indices: list,
        include_timestamp: bool = True,
        include_canonical_key: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a certificate for an embedding.

        Args:
            mapping: Vertex to root mapping (list of root indices)
            unity_indices: Unity vertex indices
            include_timestamp: Whether to include timestamp
            include_canonical_key: Whether to compute and include canonical key
            metadata: Optional metadata to include

        Returns:
            JSON certificate string
        """
        cert = CertificateFormat()

        # Set basic fields
        cert.version = CERTIFICATE_VERSION
        cert.atlas_labels = [str(lab) for lab in self.atlas.labels]
        cert.unity_indices = unity_indices

        # Format roots
        cert.roots = {
            str(i): format_root(self.e8.roots[i])
            for i in range(self.e8.num_roots)
        }

        # Format mapping
        cert.mapping = {
            str(i): str(mapping[i])
            for i in range(len(mapping))
        }

        # Compute statistics
        # Count integer roots
        cert.integer_roots_used = sum(
            1 for root_idx in mapping
            if all(coord.denominator == 1 for coord in self.e8.roots[root_idx])
        )
        # Count sign classes
        sign_classes = set()
        for root_idx in mapping:
            neg_idx = self.e8.negation_table[root_idx]
            sign_classes.add(min(root_idx, neg_idx))
        cert.sign_classes_used = len(sign_classes)

        # Set automorphism group
        cert.automorphism_group = "S4 on {e1,e2,e3,e6}"

        # Optional fields
        if include_timestamp:
            cert.timestamp = datetime.now().isoformat()

        if include_canonical_key:
            cert.canonical_key = self._compute_canonical_key(mapping)

        if metadata:
            cert.metadata = metadata

        # Convert to JSON
        return json.dumps(cert.to_dict(), indent=2)

    def _compute_canonical_key(self, mapping) -> list:
        """
        Compute canonical key for the embedding.

        Args:
            mapping: Vertex to root mapping

        Returns:
            Canonical key as list of integers
        """
        # This would use the canonicalization module once implemented
        # For now, return a simple key based on sign classes
        neg_table = self.e8.negation_table

        # Group vertices by their sign class representative
        sign_reps = []
        for i in range(len(mapping)):
            root_idx = mapping[i]
            sign_rep = min(root_idx, neg_table[root_idx])
            sign_reps.append(sign_rep)

        return sorted(set(sign_reps))

    def generate_from_dict(self, data: dict) -> str:
        """
        Generate certificate from raw data dictionary.

        Args:
            data: Dictionary with certificate data

        Returns:
            JSON certificate string
        """
        cert = CertificateFormat.from_dict(data)
        return json.dumps(cert.to_dict(), indent=2)

def create_certificate(
    mapping,
    atlas_graph,
    e8_system,
    **kwargs
) -> str:
    """
    Convenience function to create a certificate.

    Args:
        mapping: Vertex to root mapping (list of root indices)
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance
        **kwargs: Additional options for generation

    Returns:
        JSON certificate string
    """
    # Get unity indices from atlas
    from atlas import derive_unity_indices
    unity_indices = derive_unity_indices(atlas_graph.labels)

    gen = CertificateGenerator(atlas_graph, e8_system)
    return gen.generate(mapping, unity_indices, **kwargs)