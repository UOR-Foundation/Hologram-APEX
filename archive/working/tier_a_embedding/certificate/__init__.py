"""
Certificate module for Tier-A embeddings.

This module provides certificate generation and verification functionality.
"""

from .format import (
    CertificateFormat,
    CERTIFICATE_VERSION,
    validate_certificate_format,
    format_root,
    parse_root,
)
from .generator import (
    CertificateGenerator,
    create_certificate,
)
from .verifier import (
    CertificateVerifier,
    verify_certificate,
)

__all__ = [
    # Format
    "CertificateFormat",
    "CERTIFICATE_VERSION",
    "validate_certificate_format",
    "format_root",
    "parse_root",
    # Generator
    "CertificateGenerator",
    "create_certificate",
    # Verifier
    "CertificateVerifier",
    "verify_certificate",
]