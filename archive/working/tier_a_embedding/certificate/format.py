"""
Certificate format and schema definitions.

This module defines the structure and validation rules for embedding certificates.
"""
from typing import Dict, List, Any
from dataclasses import dataclass, field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import CertificateData

# Certificate version
CERTIFICATE_VERSION = "A2"

# Required fields for a valid certificate
REQUIRED_FIELDS = {
    "version",
    "atlas_labels",
    "unity_indices",
    "roots",
    "mapping",
    "integer_roots_used",
    "sign_classes_used",
    "automorphism_group"
}

# Optional fields
OPTIONAL_FIELDS = {
    "canonical_key",
    "timestamp",
    "metadata"
}

@dataclass
class CertificateFormat:
    """
    Defines the structure of an embedding certificate.
    """
    version: str = CERTIFICATE_VERSION
    atlas_labels: List[str] = field(default_factory=list)
    unity_indices: List[int] = field(default_factory=list)
    roots: Dict[str, List[str]] = field(default_factory=dict)
    mapping: Dict[str, str] = field(default_factory=dict)
    canonical_key: List[int] = field(default_factory=list)
    integer_roots_used: int = 0
    sign_classes_used: int = 0
    automorphism_group: str = "S4 on {e1,e2,e3,e6}"
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "atlas_labels": self.atlas_labels,
            "unity_indices": self.unity_indices,
            "roots": self.roots,
            "mapping": self.mapping,
            "integer_roots_used": self.integer_roots_used,
            "sign_classes_used": self.sign_classes_used,
            "automorphism_group": self.automorphism_group
        }

        # Add optional fields if present
        if self.canonical_key:
            result["canonical_key"] = self.canonical_key
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'CertificateFormat':
        """Create from dictionary."""
        return cls(
            version=data.get("version", CERTIFICATE_VERSION),
            atlas_labels=data.get("atlas_labels", []),
            unity_indices=data.get("unity_indices", []),
            roots=data.get("roots", {}),
            mapping=data.get("mapping", {}),
            canonical_key=data.get("canonical_key", []),
            integer_roots_used=data.get("integer_roots_used", 0),
            sign_classes_used=data.get("sign_classes_used", 0),
            automorphism_group=data.get("automorphism_group", "S4 on {e1,e2,e3,e6}"),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {})
        )

def validate_certificate_format(cert_dict: dict) -> tuple[bool, str]:
    """
    Validate certificate format.

    Args:
        cert_dict: Certificate dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check version
    if cert_dict.get("version") != CERTIFICATE_VERSION:
        return False, f"Invalid version: expected {CERTIFICATE_VERSION}"

    # Check required fields
    missing_fields = REQUIRED_FIELDS - set(cert_dict.keys())
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Check atlas_labels
    labels = cert_dict.get("atlas_labels", [])
    if len(labels) != 96:
        return False, f"Expected 96 atlas labels, got {len(labels)}"

    # Check unity_indices
    unity = cert_dict.get("unity_indices", [])
    if len(unity) != 2:
        return False, f"Expected 2 unity indices, got {len(unity)}"

    # Check roots
    roots = cert_dict.get("roots", {})
    if len(roots) != 240:
        return False, f"Expected 240 roots, got {len(roots)}"

    # Check mapping
    mapping = cert_dict.get("mapping", {})
    if len(mapping) != 96:
        return False, f"Expected 96 mappings, got {len(mapping)}"

    # Check sign_classes_used
    sign_classes = cert_dict.get("sign_classes_used", 0)
    if sign_classes != 48:
        return False, f"Expected 48 sign classes used, got {sign_classes}"

    return True, ""

def format_root(root) -> List[str]:
    """
    Format a root vector for certificate.

    Args:
        root: Root vector with Fraction components

    Returns:
        List of strings in "numerator/denominator" format
    """
    return [f"{x.numerator}/{x.denominator}" for x in root]

def parse_root(root_strings: List[str]):
    """
    Parse a root from certificate format.

    Args:
        root_strings: List of "numerator/denominator" strings

    Returns:
        Root vector with Fraction components
    """
    from fractions import Fraction
    return tuple(Fraction(s) for s in root_strings)