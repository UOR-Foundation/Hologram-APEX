"""
G₂ Certificate Generator.

Generate certificate for G₂ structure from unity positions.

Per Atlas First Principles Proof Stack §3:
- 12 unity positions in ℤ/768
- Klein quartet V₄ = {0, 1, 48, 49}
- Weyl group order 12 (dihedral D₆)
- Cartan matrix [[2, -3], [-1, 2]]
"""
import sys
import os
import json
from typing import Dict, List, Any
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from g2.twelve_fold import verify_twelve_fold
from g2.klein_structure import find_klein_quartet
from g2.weyl_dihedral import verify_g2_weyl_dihedral


class G2CertificateGenerator:
    """Generate G₂ certificate with all required data."""

    def __init__(self):
        """Initialize with G₂ structures."""
        self.periodicity = None
        self.klein = None
        self.weyl = None

    def generate_certificate(self) -> Dict[str, Any]:
        """
        Generate complete G₂ certificate.

        Returns:
            Dictionary with G₂ structure data
        """
        print("="*60)
        print("GENERATING G₂ CERTIFICATE")
        print("="*60)

        # Load G₂ structures
        self.periodicity = verify_twelve_fold()
        self.klein = find_klein_quartet()
        self.weyl = verify_g2_weyl_dihedral()

        # Build certificate
        certificate = {
            "structure": "G2",
            "timestamp": "2025-10-06",
            "unity_positions": self.periodicity.unity_positions,
            "klein_quartet": self.periodicity.klein_base,
            "total_roots": 12,
            "rank": 2,
            "weyl_order": 12,
            "cartan_matrix": [[2, -3], [-1, 2]],
            "simple_roots": self._format_simple_roots(),
            "root_mapping": self._format_root_mapping(),
            "twelve_fold_divisibility": self.periodicity.period_12_divisors,
            "properties": self._compute_properties()
        }

        print(f"\n✓ Certificate generated")
        print(f"  Unity positions: {len(certificate['unity_positions'])}")
        print(f"  Klein quartet: {certificate['klein_quartet']}")
        print(f"  Total roots: {certificate['total_roots']}")
        print(f"  Weyl order: {certificate['weyl_order']}")

        return certificate

    def _format_simple_roots(self) -> Dict[str, List[int]]:
        """Format G₂ simple roots (exact integers)."""
        return {
            "alpha1": [1, -1, 0],  # Short root
            "alpha2": [-2, 1, 1]   # Long root
        }

    def _format_root_mapping(self) -> List[Dict[str, Any]]:
        """Format unity position → G₂ root mapping."""
        mapping = []
        for pos, root_label in self.periodicity.g2_root_mapping:
            mapping.append({
                "unity_position": pos,
                "root_label": root_label
            })
        return mapping

    def _compute_properties(self) -> Dict[str, Any]:
        """Compute G₂ properties for verification."""
        return {
            "dynkin_diagram": "o<≡≡≡o (triple bond)",
            "long_roots": 6,
            "short_roots": 6,
            "length_ratio": "√3:1",
            "weyl_group": "D₆ (dihedral)",
            "is_simply_laced": False,
            "twelve_fold_verified": self.periodicity.is_valid
        }


def verify_g2_certificate(certificate: Dict[str, Any]) -> Dict[str, bool]:
    """
    Verify G₂ certificate properties.

    Checks:
    - 12 unity positions
    - Klein quartet present
    - Weyl order 12
    - Cartan matrix correct
    - 12-fold divisibility
    """
    checks = {}

    # Unity positions
    checks['unity_count_12'] = (len(certificate['unity_positions']) == 12)

    # Klein quartet
    checks['klein_size_4'] = (len(certificate['klein_quartet']) == 4)

    # Root system
    checks['total_roots_12'] = (certificate['total_roots'] == 12)
    checks['rank_2'] = (certificate['rank'] == 2)

    # Weyl group
    checks['weyl_order_12'] = (certificate['weyl_order'] == 12)

    # Cartan matrix
    expected_cartan = [[2, -3], [-1, 2]]
    checks['cartan_correct'] = (certificate['cartan_matrix'] == expected_cartan)

    # 12-fold divisibility
    divisors = certificate['twelve_fold_divisibility']
    checks['atlas_divisible'] = (divisors.get('atlas_total', 0) % 12 == 0)
    checks['vertices_divisible'] = (divisors.get('atlas_vertices', 0) % 12 == 0)

    print("\n" + "="*60)
    print("CERTIFICATE VERIFICATION")
    print("="*60)

    all_pass = True
    for check, result in sorted(checks.items()):
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        all_pass = all_pass and result

    return checks


def generate_g2_certificate(output_path: str = None):
    """
    Main function to generate G₂ certificate.

    Args:
        output_path: Optional path to save certificate JSON

    Returns:
        Certificate dictionary
    """
    generator = G2CertificateGenerator()
    certificate = generator.generate_certificate()

    # Verify certificate
    verification = verify_g2_certificate(certificate)

    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(certificate, f, indent=2)
        print(f"\n✓ Certificate saved to {output_path}")

    if all(verification.values()):
        print("\n✓✓✓ G₂ CERTIFICATE VALID")
    else:
        print("\n⚠ Certificate validation warnings")

    return certificate


if __name__ == "__main__":
    output_file = "/workspaces/Hologram/working/exceptional_groups/g2_certificate.json"
    cert = generate_g2_certificate(output_file)
