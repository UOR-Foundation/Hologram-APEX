"""
Atlas label generation and manipulation functions.

This module handles the canonical labeling system for the Atlas graph vertices.
"""
from itertools import product
from typing import List, Dict, Set
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Label, LabelIndex

def generate_canonical_labels() -> List[Label]:
    """
    Generate all 96 canonical Atlas labels.

    Each label is a 6-tuple (e1, e2, e3, d45, e6, e7) where:
    - e1, e2, e3, e6, e7 are binary (0 or 1)
    - d45 is ternary (-1, 0, or +1) representing the canonical e4-e5 difference

    Returns:
        List of 96 canonical labels
    """
    labels: List[Label] = []
    for e1, e2, e3, e6, e7 in product([0, 1], repeat=5):
        for d45 in (-1, 0, +1):
            labels.append((e1, e2, e3, d45, e6, e7))
    return labels

def create_label_index(labels: List[Label]) -> LabelIndex:
    """
    Create a dictionary mapping labels to their indices.

    Args:
        labels: List of labels

    Returns:
        Dictionary mapping each label to its index
    """
    return {lab: i for i, lab in enumerate(labels)}

def flip_d45_by_e4(d: int) -> int:
    """
    Update d45 when e4 is flipped.

    Flip e4 only, then re-canonicalize (choose among (0,1),(0,0),(1,0)) by sign of e4-e5)
    Cases: -1->0, 0->+1, +1->0

    Args:
        d: Current d45 value (-1, 0, or +1)

    Returns:
        New d45 value after e4 flip
    """
    if d == -1:
        return 0
    if d == 0:
        return +1
    if d == +1:
        return 0
    raise ValueError("d45 must be -1, 0, or +1")

def flip_d45_by_e5(d: int) -> int:
    """
    Update d45 when e5 is flipped.

    Flip e5 only: -1->0, 0->-1, +1->0

    Args:
        d: Current d45 value (-1, 0, or +1)

    Returns:
        New d45 value after e5 flip
    """
    if d == -1:
        return 0
    if d == 0:
        return -1
    if d == +1:
        return 0
    raise ValueError("d45 must be -1, 0, or +1")

def compute_neighbor_labels(lab: Label) -> Set[Label]:
    """
    Compute all distinct canonical neighbor labels under Hamming-1 flips.

    This excludes bit-7 (mirror) and bit-0 flips, implementing the canonical projection
    implicitly by updating d45 for e4/e5 flips. This enforces the proof-first Path A:
    mirror is a global involution, not an edge.

    Args:
        lab: A canonical label

    Returns:
        Set of neighbor labels
    """
    e1, e2, e3, d45, e6, e7 = lab
    nbrs: Set[Label] = set()

    # Flip e1, e2, e3, e6 (INCLUDED)
    nbrs.add((1 - e1, e2, e3, d45, e6, e7))
    nbrs.add((e1, 1 - e2, e3, d45, e6, e7))
    nbrs.add((e1, e2, 1 - e3, d45, e6, e7))
    nbrs.add((e1, e2, e3, d45, 1 - e6, e7))

    # Flip e7 (EXCLUDED) — mirror is not an adjacency
    # nbrs.add((e1, e2, e3, d45, e6, 1 - e7))

    # Flip e4 or e5 (changes d45 via canonicalization)
    nbrs.add((e1, e2, e3, flip_d45_by_e4(d45), e6, e7))
    nbrs.add((e1, e2, e3, flip_d45_by_e5(d45), e6, e7))

    # Bit 0 flip is ignored (does not change canonical label)
    nbrs.discard(lab)

    return nbrs

def derive_unity_indices(labels: List[Label]) -> List[int]:
    """
    Derive the 12 unity indices from Atlas semantics (no arbitrary input).

    Unity requires only α4·α5 contributions (which are 1) and no other factors.
    In our canonical label model, that corresponds to:
      • d45 == 0  (balanced e4/e5 in canonicalization), and
      • e1 == e2 == e3 == e6 == 0 (no non-unity fields active),
      • e7 ∈ {0,1} (mirror pairs both included).

    This derivation yields 2 indices in the 96-vertex canonical slice. The full 12 unity
    positions in the 768-cycle are page-lifts of these canonical representatives.

    Args:
        labels: List of all canonical labels

    Returns:
        List of unity indices
    """
    idx = create_label_index(labels)
    res = []
    for lab in labels:
        e1, e2, e3, d45, e6, e7 = lab
        if d45 == 0 and e1 == 0 and e2 == 0 and e3 == 0 and e6 == 0:
            res.append(idx[lab])
    res = sorted(res)
    assert len(res) == 2 and labels[res[0]][5] == 0 and labels[res[1]][5] == 1, \
        "Unity derivation must produce exactly the τ-pair"
    return res