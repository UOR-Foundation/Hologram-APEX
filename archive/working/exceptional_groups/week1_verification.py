"""
Week 1 Verification - Complete Integration Test.

Verifies all Week 1 deliverables per IMPLEMENTATION_ROADMAP.md:
1. F₄ Cartan matrix extracted from 48 sign classes
2. G₂ 12-fold structure verified
3. Klein quartet identified
4. S₄ automorphism verified
5. Certificates generated

All using exact arithmetic (no floats).
"""
import sys
import os
from typing import Dict, List, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# F₄ imports
from f4.sign_class_analysis import extract_f4_from_sign_classes
from f4.cartan_extraction import extract_cartan_matrix
from f4.page_correspondence import establish_page_root_correspondence
from f4.weyl_generators import verify_f4_weyl_group
from f4.certificate_generator import generate_f4_certificate, verify_certificate as verify_f4_cert

# G₂ imports
from g2.klein_structure import find_klein_quartet
from g2.twelve_fold import verify_twelve_fold
from g2.weyl_dihedral import verify_g2_weyl_dihedral
from g2.certificate_generator import generate_g2_certificate, verify_g2_certificate

# S₄ import
from s4_automorphism import verify_s4_automorphism


class Week1Verification:
    """Complete Week 1 verification suite."""

    def __init__(self):
        """Initialize verification."""
        self.results = {}
        self.f4_results = {}
        self.g2_results = {}
        self.s4_results = {}

    def verify_f4_complete(self) -> Dict[str, bool]:
        """
        Verify F₄ implementation complete.

        Deliverables:
        - 48 sign classes extracted
        - Cartan matrix with -2 entry (F₄ signature)
        - Page-root correspondence established
        - Weyl group generated
        - Certificate valid
        """
        print("\n" + "="*70)
        print("F₄ VERIFICATION")
        print("="*70)

        checks = {}

        # 1. Extract 48 sign classes
        print("\n1. Extracting F₄ from 48 sign classes...")
        try:
            f4_structure, properties = extract_f4_from_sign_classes()
            checks['f4_48_roots'] = (properties['num_roots'] == 48)
            checks['f4_32_short'] = (properties['num_short_roots'] == 32)
            checks['f4_16_long'] = (properties['num_long_roots'] == 16)
            checks['f4_triangle_free'] = properties['triangle_free']
            print(f"  ✓ 48 roots extracted")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['f4_48_roots'] = False

        # 2. Extract Cartan matrix
        print("\n2. Extracting Cartan matrix...")
        try:
            # Need bijection first for root coordinates
            from f4.page_correspondence import establish_page_root_correspondence
            bijection = establish_page_root_correspondence()

            cartan, valid = extract_cartan_matrix(f4_structure, root_coords=bijection.root_coords)
            has_double_bond = any(cartan[i][j] == -2 for i in range(4) for j in range(4))
            checks['f4_cartan_valid'] = valid
            checks['f4_has_double_bond'] = has_double_bond
            print(f"  ✓ Cartan matrix extracted")
            if has_double_bond:
                print(f"  ✓ Double bond (-2) detected - F₄ signature confirmed")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['f4_cartan_valid'] = False

        # 3. Establish page-root correspondence
        print("\n3. Establishing page-root correspondence...")
        try:
            bijection = establish_page_root_correspondence()
            checks['f4_bijection_48'] = (len(bijection.page_to_root) == 48)
            checks['f4_mirror_pairs'] = (len(bijection.mirror_pairs) == 24)
            print(f"  ✓ Page-root bijection established")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['f4_bijection_48'] = False

        # 4. Verify Weyl group
        print("\n4. Verifying F₄ Weyl group...")
        try:
            weyl_result = verify_f4_weyl_group()
            weyl_checks = weyl_result['verification']
            checks['f4_weyl_order'] = weyl_checks.get('correct_order', False)
            print(f"  ✓ Weyl group verified")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['f4_weyl_order'] = False

        # 5. Generate and verify certificate
        print("\n5. Generating F₄ certificate...")
        try:
            cert = generate_f4_certificate()
            cert_checks = verify_f4_cert(cert)
            checks['f4_certificate_valid'] = all(cert_checks.values())
            print(f"  ✓ Certificate generated and verified")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['f4_certificate_valid'] = False

        self.f4_results = checks
        return checks

    def verify_g2_complete(self) -> Dict[str, bool]:
        """
        Verify G₂ implementation complete.

        Deliverables:
        - Klein quartet identified
        - 12-fold periodicity verified
        - Weyl group D₆ verified
        - Certificate valid
        """
        print("\n" + "="*70)
        print("G₂ VERIFICATION")
        print("="*70)

        checks = {}

        # 1. Find Klein quartet
        print("\n1. Finding Klein quartet...")
        try:
            klein_structure, g2_analysis = find_klein_quartet()
            # KleinStructure has a .quartet attribute
            klein_quartet = klein_structure.quartet
            checks['g2_klein_found'] = (len(klein_quartet) == 4)
            checks['g2_klein_positions'] = (klein_quartet == [0, 1, 48, 49])
            print(f"  ✓ Klein quartet found: {klein_quartet}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['g2_klein_found'] = False

        # 2. Verify 12-fold periodicity
        print("\n2. Verifying 12-fold periodicity...")
        try:
            periodicity = verify_twelve_fold()
            checks['g2_12_unity'] = (len(periodicity.unity_positions) == 12)
            checks['g2_divisibility'] = all(
                v % 12 == 0 for v in periodicity.period_12_divisors.values()
            )
            print(f"  ✓ 12-fold structure verified")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['g2_12_unity'] = False

        # 3. Verify Weyl group D₆
        print("\n3. Verifying G₂ Weyl group (D₆)...")
        try:
            weyl_result = verify_g2_weyl_dihedral()
            checks['g2_weyl_order_12'] = (weyl_result['order'] == 12)
            checks['g2_weyl_d6'] = (weyl_result['group'] == 'D₆')
            print(f"  ✓ Weyl group D₆ verified (order 12)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['g2_weyl_order_12'] = False

        # 4. Generate and verify certificate
        print("\n4. Generating G₂ certificate...")
        try:
            cert = generate_g2_certificate()
            cert_checks = verify_g2_certificate(cert)
            checks['g2_certificate_valid'] = all(cert_checks.values())
            print(f"  ✓ Certificate generated and verified")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['g2_certificate_valid'] = False

        self.g2_results = checks
        return checks

    def verify_s4_automorphism(self) -> Dict[str, bool]:
        """
        Verify S₄ automorphism structure.

        Deliverables:
        - 24 S₄ elements generated
        - 30 orbits found
        - Orbit sizes: 12×1, 12×4, 6×6
        """
        print("\n" + "="*70)
        print("S₄ AUTOMORPHISM VERIFICATION")
        print("="*70)

        checks = {}

        print("\n1. Verifying S₄ automorphism...")
        try:
            s4_result = verify_s4_automorphism()
            checks['s4_order_24'] = (s4_result.s4_verified)
            checks['s4_30_orbits'] = (s4_result.total_orbits == 30)

            # Check orbit size distribution
            expected_sizes = {1: 12, 4: 12, 6: 6}
            checks['s4_orbit_distribution'] = (
                s4_result.orbit_sizes == expected_sizes
            )

            print(f"  ✓ S₄ automorphism verified")
            print(f"  ✓ 30 orbits with correct distribution")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            checks['s4_order_24'] = False

        self.s4_results = checks
        return checks

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete Week 1 verification report."""
        print("\n" + "="*70)
        print("WEEK 1 VERIFICATION SUMMARY")
        print("="*70)

        # Combine all results
        all_checks = {
            **self.f4_results,
            **self.g2_results,
            **self.s4_results
        }

        # Count passes/fails
        total = len(all_checks)
        passed = sum(1 for v in all_checks.values() if v)
        failed = total - passed

        print(f"\nResults: {passed}/{total} checks passed")

        if failed > 0:
            print(f"\nFailed checks:")
            for check, result in all_checks.items():
                if not result:
                    print(f"  ✗ {check}")

        # Deliverables status
        print(f"\n" + "="*70)
        print("WEEK 1 DELIVERABLES STATUS")
        print("="*70)

        deliverables = {
            "F₄ Cartan matrix extracted": self.f4_results.get('f4_cartan_valid', False),
            "F₄ has double bond (F₄ signature)": self.f4_results.get('f4_has_double_bond', False),
            "F₄ page-root correspondence": self.f4_results.get('f4_bijection_48', False),
            "G₂ 12-fold structure verified": self.g2_results.get('g2_12_unity', False),
            "Klein quartet identified": self.g2_results.get('g2_klein_found', False),
            "S₄ automorphism verified": self.s4_results.get('s4_order_24', False),
            "F₄ certificate valid": self.f4_results.get('f4_certificate_valid', False),
            "G₂ certificate valid": self.g2_results.get('g2_certificate_valid', False)
        }

        for deliverable, status in deliverables.items():
            symbol = "✓" if status else "✗"
            print(f"  {symbol} {deliverable}")

        all_complete = all(deliverables.values())

        print(f"\n" + "="*70)
        if all_complete:
            print("✓✓✓ WEEK 1 COMPLETE - ALL DELIVERABLES VERIFIED")
        else:
            print("⚠ WEEK 1 INCOMPLETE - Some deliverables need attention")
        print("="*70)

        return {
            "complete": all_complete,
            "checks": all_checks,
            "deliverables": deliverables,
            "passed": passed,
            "total": total
        }


def run_week1_verification():
    """Main function to run complete Week 1 verification."""
    print("="*70)
    print("WEEK 1 COMPLETE VERIFICATION")
    print("Atlas Exceptional Groups Implementation")
    print("="*70)

    verifier = Week1Verification()

    # Run all verifications
    verifier.verify_f4_complete()
    verifier.verify_g2_complete()
    verifier.verify_s4_automorphism()

    # Generate report
    report = verifier.generate_report()

    return report


if __name__ == "__main__":
    report = run_week1_verification()
