"""
Canonical form selection for embeddings.

This module selects canonical representatives from equivalence classes.
"""
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Label, Root
from .equivalence import mapping_equivalence_key, count_root_types
from .automorphism import S4Group

class CanonicalSelector:
    """Selects canonical forms for embeddings."""

    def __init__(
        self,
        tau: List[int],
        negation_table: List[int],
        labels: List[Label],
        roots: List[Root]
    ):
        """
        Initialize canonical selector.

        Args:
            tau: Mirror pairing
            negation_table: Root negation table
            labels: Atlas labels
            roots: E8 roots
        """
        self.tau = tau
        self.negation_table = negation_table
        self.labels = labels
        self.roots = roots
        self.s4_group = S4Group()

    def canonicalize(
        self,
        mappings: List[List[int]],
        use_automorphisms: bool = True
    ) -> List[int]:
        """
        Select canonical representative from mappings.

        Primary ordering:
        1. Minimal automorphism equivalence key
        2. Maximum integer roots (minimal denominators)
        3. Lexicographically minimal mapping

        Args:
            mappings: List of mappings to select from
            use_automorphisms: Whether to consider S4 automorphisms

        Returns:
            The canonical mapping
        """
        if not mappings:
            raise ValueError("No mappings provided")

        if len(mappings) == 1:
            return mappings[0]

        # Compute scores for each mapping
        scored_mappings = []
        for mapping in mappings:
            if use_automorphisms:
                key = self._automorphism_equivalence_key(mapping)
            else:
                key = mapping_equivalence_key(mapping, self.tau, self.negation_table)

            int_count, half_count = count_root_types(mapping, self.roots)

            score = (
                key,                # Minimize equivalence key
                -int_count,        # Maximize integer roots
                tuple(mapping)     # Lexicographic tiebreaker
            )
            scored_mappings.append((score, mapping))

        # Select minimum by score
        scored_mappings.sort(key=lambda x: x[0])
        return scored_mappings[0][1]

    def _automorphism_equivalence_key(self, mapping: List[int]) -> Tuple[int, ...]:
        """
        Compute minimal equivalence key over S4 automorphisms.

        Args:
            mapping: Vertex to root mapping

        Returns:
            Minimal equivalence key
        """
        orbit = self.s4_group.orbit(mapping, self.labels)
        keys = []

        for transformed in orbit:
            key = mapping_equivalence_key(transformed, self.tau, self.negation_table)
            keys.append(key)

        return min(keys)

    def are_equivalent_modulo_automorphisms(
        self,
        mapping1: List[int],
        mapping2: List[int]
    ) -> bool:
        """
        Check if two mappings are equivalent under S4 and tau/sign.

        Args:
            mapping1: First mapping
            mapping2: Second mapping

        Returns:
            True if equivalent
        """
        key1 = self._automorphism_equivalence_key(mapping1)
        key2 = self._automorphism_equivalence_key(mapping2)
        return key1 == key2

    def equivalence_class_size(self, mapping: List[int]) -> int:
        """
        Compute the size of the equivalence class.

        Args:
            mapping: Vertex to root mapping

        Returns:
            Size of equivalence class
        """
        orbit = self.s4_group.orbit(mapping, self.labels)
        stabilizer_size = self.s4_group.stabilizer_size(mapping, self.labels)

        # Orbit-stabilizer theorem: |G| = |orbit| * |stabilizer|
        # So |orbit| = |G| / |stabilizer| = 24 / stabilizer_size

        # Also account for tau/sign symmetry
        tau_sign_factor = 2  # Usually 2, unless self-inverse

        return len(orbit) * tau_sign_factor

    def find_canonical_in_orbit(self, mapping: List[int]) -> List[int]:
        """
        Find the canonical representative in the orbit of a mapping.

        Args:
            mapping: Vertex to root mapping

        Returns:
            Canonical representative in the orbit
        """
        orbit = self.s4_group.orbit(mapping, self.labels)
        return self.canonicalize(orbit, use_automorphisms=False)

def canonicalize_embedding(
    mappings: List[List[int]],
    atlas_graph,
    e8_system
) -> List[int]:
    """
    Convenience function to canonicalize embeddings.

    Args:
        mappings: List of mappings
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance

    Returns:
        Canonical mapping
    """
    selector = CanonicalSelector(
        tau=atlas_graph.tau,
        negation_table=e8_system.negation_table,
        labels=atlas_graph.labels,
        roots=e8_system.roots
    )
    return selector.canonicalize(mappings)

def compute_canonical_key(
    mapping: List[int],
    atlas_graph,
    e8_system,
    use_automorphisms: bool = True
) -> Tuple[int, ...]:
    """
    Compute canonical key for a mapping.

    Args:
        mapping: Vertex to root mapping
        atlas_graph: AtlasGraph instance
        e8_system: E8RootSystem instance
        use_automorphisms: Whether to use S4 automorphisms

    Returns:
        Canonical key
    """
    selector = CanonicalSelector(
        tau=atlas_graph.tau,
        negation_table=e8_system.negation_table,
        labels=atlas_graph.labels,
        roots=e8_system.roots
    )

    if use_automorphisms:
        return selector._automorphism_equivalence_key(mapping)
    else:
        return mapping_equivalence_key(
            mapping,
            atlas_graph.tau,
            e8_system.negation_table
        )