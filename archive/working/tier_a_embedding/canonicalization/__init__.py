"""
Canonicalization module for selecting unique embedding representatives.

This module provides functionality to:
- Compute equivalence keys for embeddings
- Apply S4 automorphisms to the Atlas graph
- Select canonical representatives from equivalence classes
"""

from .equivalence import (
    mapping_equivalence_key,
    compute_sign_class_key,
    count_root_types,
    equivalence_class_size,
    EquivalenceChecker,
)
from .automorphism import (
    S4Automorphism,
    S4Group,
    apply_automorphism_to_edges,
    verify_automorphism_preserves_structure,
)
from .canonical import (
    CanonicalSelector,
    canonicalize_embedding,
    compute_canonical_key,
)

__all__ = [
    # Equivalence
    "mapping_equivalence_key",
    "compute_sign_class_key",
    "count_root_types",
    "equivalence_class_size",
    "EquivalenceChecker",
    # Automorphism
    "S4Automorphism",
    "S4Group",
    "apply_automorphism_to_edges",
    "verify_automorphism_preserves_structure",
    # Canonical
    "CanonicalSelector",
    "canonicalize_embedding",
    "compute_canonical_key",
]