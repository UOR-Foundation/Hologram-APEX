"""
E8 root system generation.

This module generates the 240 roots of the E8 root system, consisting of
112 integer roots and 128 half-integer roots.
"""
from fractions import Fraction
from itertools import combinations, product
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Root, RootIndex, E8_ROOT_COUNT, E8_INTEGER_ROOT_COUNT, E8_HALF_INTEGER_ROOT_COUNT

def generate_integer_roots() -> List[Root]:
    """
    Generate the 112 integer roots of E8.

    Integer roots have the form ±ei ± ej for i ≠ j, where ei is the i-th standard basis vector.

    Returns:
        List of 112 integer roots
    """
    roots: List[Root] = []
    z = Fraction(0, 1)
    o = Fraction(1, 1)

    for i, j in combinations(range(8), 2):
        for s1, s2 in product([+1, -1], repeat=2):
            v = [z] * 8
            v[i] = o * s1
            v[j] = o * s2
            roots.append(tuple(v))

    assert len(roots) == E8_INTEGER_ROOT_COUNT, f"Expected {E8_INTEGER_ROOT_COUNT} integer roots"
    return roots

def generate_half_integer_roots() -> List[Root]:
    """
    Generate the 128 half-integer roots of E8.

    Half-integer roots have all coordinates ±1/2 with an even number of minus signs.

    Returns:
        List of 128 half-integer roots
    """
    roots: List[Root] = []
    h = Fraction(1, 2)

    for signs in product([+1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            roots.append(tuple(h * s for s in signs))

    assert len(roots) == E8_HALF_INTEGER_ROOT_COUNT, f"Expected {E8_HALF_INTEGER_ROOT_COUNT} half-integer roots"
    return roots

def generate_e8_roots() -> List[Root]:
    """
    Generate all 240 roots of the E8 root system.

    Returns:
        List of all 240 E8 roots
    """
    roots = generate_integer_roots() + generate_half_integer_roots()
    assert len(roots) == E8_ROOT_COUNT, f"Expected {E8_ROOT_COUNT} total roots"
    return roots

def create_root_index(roots: List[Root]) -> RootIndex:
    """
    Create a dictionary mapping roots to their indices.

    Args:
        roots: List of roots

    Returns:
        Dictionary mapping each root to its index
    """
    return {r: i for i, r in enumerate(roots)}

def compute_negation_table(roots: List[Root]) -> List[int]:
    """
    Compute a table mapping each root index to the index of its negation.

    Args:
        roots: List of roots

    Returns:
        List where index i contains the index of -roots[i]
    """
    tbl: List[int] = [-1] * len(roots)
    mp = create_root_index(roots)

    for i, r in enumerate(roots):
        neg = tuple(-x for x in r)
        tbl[i] = mp[neg]

    return tbl

def is_integer_root(r: Root) -> bool:
    """
    Check if a root has only integer coordinates (no 1/2).

    Args:
        r: A root vector

    Returns:
        True if the root has only integer coordinates
    """
    return all(x.denominator == 1 for x in r)

def is_half_integer_root(r: Root) -> bool:
    """
    Check if a root has all half-integer coordinates.

    Args:
        r: A root vector

    Returns:
        True if all coordinates are ±1/2
    """
    return all(x.denominator == 2 and abs(x.numerator) == 1 for x in r)

def get_sign_class_representative(root_idx: int, negation_table: List[int]) -> int:
    """
    Get the canonical representative of a sign class {r, -r}.

    Args:
        root_idx: Index of a root
        negation_table: Table mapping root indices to their negations

    Returns:
        The smaller of root_idx and its negation index
    """
    return min(root_idx, negation_table[root_idx])

def count_sign_classes(root_indices: List[int], negation_table: List[int]) -> int:
    """
    Count the number of distinct sign classes used.

    Args:
        root_indices: List of root indices
        negation_table: Table mapping root indices to their negations

    Returns:
        Number of distinct sign classes
    """
    representatives = {get_sign_class_representative(idx, negation_table) for idx in root_indices}
    return len(representatives)

class E8RootSystem:
    """Encapsulates the E8 root system and its properties."""

    def __init__(self):
        """Initialize the E8 root system."""
        self.roots = generate_e8_roots()
        self.root_index = create_root_index(self.roots)
        self.negation_table = compute_negation_table(self.roots)

        self._verify_invariants()

    def _verify_invariants(self) -> None:
        """Verify critical E8 root system invariants."""
        # Check counts
        assert len(self.roots) == E8_ROOT_COUNT, f"Must have {E8_ROOT_COUNT} roots"

        integer_roots = [r for r in self.roots if is_integer_root(r)]
        assert len(integer_roots) == E8_INTEGER_ROOT_COUNT, f"Must have {E8_INTEGER_ROOT_COUNT} integer roots"

        half_integer_roots = [r for r in self.roots if is_half_integer_root(r)]
        assert len(half_integer_roots) == E8_HALF_INTEGER_ROOT_COUNT, f"Must have {E8_HALF_INTEGER_ROOT_COUNT} half-integer roots"

        # Check that every root has a negation
        for i, r in enumerate(self.roots):
            neg_idx = self.negation_table[i]
            assert neg_idx != i, "No root equals its own negative"
            neg = tuple(-x for x in r)
            assert self.roots[neg_idx] == neg, "Negation table must be correct"

    @property
    def num_roots(self) -> int:
        """Return the number of roots."""
        return len(self.roots)

    def get_root(self, index: int) -> Root:
        """
        Get a root by its index.

        Args:
            index: Root index

        Returns:
            The root vector
        """
        return self.roots[index]

    def get_negation(self, index: int) -> int:
        """
        Get the index of the negation of a root.

        Args:
            index: Root index

        Returns:
            Index of -root
        """
        return self.negation_table[index]

    def get_sign_classes_used(self, root_indices: List[int]) -> int:
        """
        Count sign classes used by a set of roots.

        Args:
            root_indices: List of root indices

        Returns:
            Number of distinct sign classes
        """
        return count_sign_classes(root_indices, self.negation_table)

def e8_roots() -> List[Root]:
    """
    Generate all E8 roots (backward compatibility function).

    Returns:
        List of all 240 E8 roots
    """
    return generate_e8_roots()