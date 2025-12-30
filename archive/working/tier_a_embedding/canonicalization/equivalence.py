"""
Mapping equivalence key computation.

This module computes canonical keys for embeddings modulo symmetries.
"""
from typing import Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Root

def mapping_equivalence_key(
    mapping: List[int],
    tau: List[int],
    negation_table: List[int]
) -> Tuple[int, ...]:
    """
    Compute canonical key modulo tau/sign pairing.

    This collapses each {v, tau(v)} vertex pair to the smaller index
    of their {rho, -rho} root pair.

    Args:
        mapping: Vertex to root mapping
        tau: Mirror pairing of vertices
        negation_table: Table of root negations

    Returns:
        Canonical key as sorted tuple of representative root indices
    """
    n = len(mapping)
    seen = [False] * n
    representatives: List[int] = []

    for i in range(n):
        if seen[i]:
            continue

        # Get mirror pair vertex
        j = tau[i]
        seen[i] = True
        seen[j] = True

        # Get their root mappings
        ri = mapping[i]
        rj = mapping[j]

        # Verify mirror pairing is satisfied
        assert rj == negation_table[ri], \
            f"Mapping must satisfy mirror pairing: f(tau({i})) = -f({i})"

        # Use the smaller root index as representative
        representatives.append(min(ri, rj))

    return tuple(sorted(representatives))

def compute_sign_class_key(
    mapping: List[int],
    negation_table: List[int]
) -> Tuple[int, ...]:
    """
    Compute key based on sign equivalence classes.

    Args:
        mapping: Vertex to root mapping
        negation_table: Table of root negations

    Returns:
        Sorted tuple of sign class representatives
    """
    sign_classes = set()

    for root_idx in mapping:
        # Representative is the smaller of root and its negation
        rep = min(root_idx, negation_table[root_idx])
        sign_classes.add(rep)

    return tuple(sorted(sign_classes))

def count_root_types(
    mapping: List[int],
    roots: List[Root]
) -> Tuple[int, int]:
    """
    Count integer and half-integer roots in mapping.

    Args:
        mapping: Vertex to root mapping
        roots: List of all roots

    Returns:
        Tuple of (integer_count, half_integer_count)
    """
    integer_count = 0
    half_integer_count = 0

    for root_idx in mapping:
        root = roots[root_idx]
        # Check if all coordinates are integers
        if all(x.denominator == 1 for x in root):
            integer_count += 1
        # Check if all coordinates are half-integers
        elif all(x.denominator == 2 for x in root):
            half_integer_count += 1

    return integer_count, half_integer_count

def equivalence_class_size(
    mapping: List[int],
    tau: List[int],
    automorphism_group_size: int = 24
) -> int:
    """
    Compute the size of the equivalence class.

    The equivalence class size is bounded by:
    - Factor of 2 from global sign/tau symmetry
    - Factor of automorphism_group_size from graph automorphisms

    Args:
        mapping: Vertex to root mapping
        tau: Mirror pairing
        automorphism_group_size: Size of automorphism group (24 for S4)

    Returns:
        Upper bound on equivalence class size
    """
    # Check if mapping is self-inverse under tau/sign
    self_inverse = True
    for i, j in enumerate(tau):
        if mapping[i] != mapping[j]:
            self_inverse = False
            break

    # If self-inverse, the sign symmetry doesn't double the class
    sign_factor = 1 if self_inverse else 2

    # Maximum size is product of symmetry factors
    return sign_factor * automorphism_group_size

class EquivalenceChecker:
    """Check equivalence between embeddings."""

    def __init__(self, tau: List[int], negation_table: List[int]):
        """
        Initialize equivalence checker.

        Args:
            tau: Mirror pairing
            negation_table: Root negation table
        """
        self.tau = tau
        self.negation_table = negation_table

    def are_equivalent(
        self,
        mapping1: List[int],
        mapping2: List[int],
        check_automorphisms: bool = False
    ) -> bool:
        """
        Check if two mappings are equivalent.

        Args:
            mapping1: First mapping
            mapping2: Second mapping
            check_automorphisms: Whether to check under automorphisms

        Returns:
            True if mappings are equivalent
        """
        # First check simple tau/sign equivalence
        key1 = mapping_equivalence_key(mapping1, self.tau, self.negation_table)
        key2 = mapping_equivalence_key(mapping2, self.tau, self.negation_table)

        if key1 == key2:
            return True

        if check_automorphisms:
            # Would need automorphism group to check this fully
            # For now, just check keys match
            pass

        return False

    def canonical_representative(
        self,
        mappings: List[List[int]]
    ) -> List[int]:
        """
        Select canonical representative from equivalent mappings.

        Args:
            mappings: List of equivalent mappings

        Returns:
            The canonical representative
        """
        if not mappings:
            raise ValueError("No mappings provided")

        # Sort by equivalence key, then by mapping itself
        def sort_key(m):
            return (
                mapping_equivalence_key(m, self.tau, self.negation_table),
                tuple(m)
            )

        return min(mappings, key=sort_key)