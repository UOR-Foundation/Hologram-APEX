"""
Certificate verification module.

This module provides independent verification of embedding certificates.
"""
import json
from fractions import Fraction
from typing import Tuple, List, Set, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .format import validate_certificate_format, parse_root

class CertificateVerifier:
    """Verifies embedding certificates independently."""

    def __init__(self, verbose: bool = False):
        """
        Initialize verifier.

        Args:
            verbose: Whether to print detailed verification steps
        """
        self.verbose = verbose

    def verify(self, cert_json: str) -> Tuple[bool, str]:
        """
        Verify a certificate JSON string.

        Args:
            cert_json: JSON certificate string

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            cert = json.loads(cert_json)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        # Check format
        valid, msg = validate_certificate_format(cert)
        if not valid:
            return False, f"Format validation failed: {msg}"

        # Rebuild structures from certificate
        labels = [eval(s) for s in cert["atlas_labels"]]
        roots = self._parse_roots(cert["roots"])
        mapping = [int(cert["mapping"][str(i)]) for i in range(96)]
        unity_indices = cert["unity_indices"]

        # Run verification checks
        checks = [
            ("Injectivity", self._check_injectivity(mapping)),
            ("Mirror pairing", self._check_mirror_pairing(mapping, labels, roots)),
            ("Edge preservation", self._check_edge_preservation(mapping, labels, roots)),
            ("Unity constraint", self._check_unity_constraint(mapping, unity_indices, roots)),
            ("Sign classes", self._check_sign_classes(mapping, roots, cert.get("sign_classes_used", 48)))
        ]

        for check_name, (passed, check_msg) in checks:
            if self.verbose:
                print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                return False, f"{check_name} failed: {check_msg}"

        return True, "Certificate verified successfully"

    def _parse_roots(self, roots_dict: dict) -> List:
        """Parse roots from certificate format."""
        roots = []
        for i in range(240):
            root_strings = roots_dict[str(i)]
            root = tuple(Fraction(s) for s in root_strings)
            roots.append(root)
        return roots

    def _check_injectivity(self, mapping: List[int]) -> Tuple[bool, str]:
        """Check if mapping is injective."""
        if len(set(mapping)) != len(mapping):
            return False, "Mapping is not injective"
        return True, ""

    def _check_mirror_pairing(
        self,
        mapping: List[int],
        labels: List,
        roots: List
    ) -> Tuple[bool, str]:
        """Check if mirror pairing is preserved."""
        # Compute tau (mirror) pairing
        label_index = {lab: i for i, lab in enumerate(labels)}
        tau = []
        for lab in labels:
            e1, e2, e3, d45, e6, e7 = lab
            mirror = (e1, e2, e3, d45, e6, 1 - e7)
            tau.append(label_index[mirror])

        # Compute root negation table
        neg_table = {}
        for i, r in enumerate(roots):
            neg = tuple(-x for x in r)
            for j, r2 in enumerate(roots):
                if r2 == neg:
                    neg_table[i] = j
                    break

        # Check mirror pairing
        for i in range(len(mapping)):
            ri = mapping[i]
            r_tau = mapping[tau[i]]
            if neg_table[ri] != r_tau:
                return False, f"Mirror pairing violated at vertex {i}"

        return True, ""

    def _check_edge_preservation(
        self,
        mapping: List[int],
        labels: List,
        roots: List
    ) -> Tuple[bool, str]:
        """Check if edges are preserved."""
        # Build atlas edges
        atlas_edges = self._build_atlas_edges(labels)

        # Check each atlas edge
        for (i, j) in atlas_edges:
            ri, rj = mapping[i], mapping[j]
            if not self._are_adjacent_roots(roots[ri], roots[rj]):
                return False, f"Edge ({i},{j}) not preserved"

        return True, ""

    def _build_atlas_edges(self, labels: List) -> Set[Tuple[int, int]]:
        """Build atlas graph edges from labels."""
        edges = set()
        label_index = {lab: i for i, lab in enumerate(labels)}

        for i, lab in enumerate(labels):
            neighbors = self._compute_neighbors(lab)
            for nlab in neighbors:
                if nlab in label_index:
                    j = label_index[nlab]
                    if i < j:
                        edges.add((i, j))

        return edges

    def _compute_neighbors(self, lab: tuple) -> Set[tuple]:
        """Compute neighbor labels."""
        e1, e2, e3, d45, e6, e7 = lab
        nbrs = set()

        # Flip e1, e2, e3, e6
        nbrs.add((1 - e1, e2, e3, d45, e6, e7))
        nbrs.add((e1, 1 - e2, e3, d45, e6, e7))
        nbrs.add((e1, e2, 1 - e3, d45, e6, e7))
        nbrs.add((e1, e2, e3, d45, 1 - e6, e7))

        # Flip e4 or e5 (changes d45)
        nbrs.add((e1, e2, e3, self._flip_d45_by_e4(d45), e6, e7))
        nbrs.add((e1, e2, e3, self._flip_d45_by_e5(d45), e6, e7))

        nbrs.discard(lab)
        return nbrs

    def _flip_d45_by_e4(self, d: int) -> int:
        """Flip d45 when e4 is flipped."""
        if d == -1:
            return 0
        if d == 0:
            return 1
        if d == 1:
            return 0
        return d

    def _flip_d45_by_e5(self, d: int) -> int:
        """Flip d45 when e5 is flipped."""
        if d == -1:
            return 0
        if d == 0:
            return -1
        if d == 1:
            return 0
        return d

    def _are_adjacent_roots(self, r1: tuple, r2: tuple) -> bool:
        """Check if two roots are adjacent (dot product = 1)."""
        dot = sum(a * b for a, b in zip(r1, r2))
        return dot == Fraction(1, 1)

    def _check_unity_constraint(
        self,
        mapping: List[int],
        unity_indices: List[int],
        roots: List
    ) -> Tuple[bool, str]:
        """Check if unity vertices sum to zero."""
        if not unity_indices:
            return True, ""

        sum_vec = [Fraction(0, 1)] * 8
        for u in unity_indices:
            root = roots[mapping[u]]
            for k in range(8):
                sum_vec[k] += root[k]

        if any(x != 0 for x in sum_vec):
            return False, f"Unity sum is not zero: {sum_vec}"

        return True, ""

    def _check_sign_classes(
        self,
        mapping: List[int],
        roots: List,
        expected: int
    ) -> Tuple[bool, str]:
        """Check sign class count."""
        # Build negation table
        neg_table = {}
        for i, r in enumerate(roots):
            neg = tuple(-x for x in r)
            for j, r2 in enumerate(roots):
                if r2 == neg:
                    neg_table[i] = j
                    break

        # Count sign classes
        sign_reps = set()
        for root_idx in mapping:
            rep = min(root_idx, neg_table[root_idx])
            sign_reps.add(rep)

        if len(sign_reps) != expected:
            return False, f"Expected {expected} sign classes, got {len(sign_reps)}"

        return True, ""

def verify_certificate(cert_json: str, verbose: bool = False) -> bool:
    """
    Convenience function to verify a certificate.

    Args:
        cert_json: JSON certificate string
        verbose: Whether to print details

    Returns:
        True if certificate is valid
    """
    verifier = CertificateVerifier(verbose)
    valid, msg = verifier.verify(cert_json)
    if verbose:
        print(f"Verification result: {msg}")
    return valid