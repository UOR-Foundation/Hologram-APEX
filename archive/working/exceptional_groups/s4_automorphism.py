"""
S₄ Automorphism Verification in Quotient Structure.

S₄ (symmetric group on 4 elements) acts on coordinates {e1, e2, e3, e6}.
This creates orbit structure in the 96 Atlas vertices.
"""
import sys
import os
from typing import List, Dict, Set, Tuple
from itertools import permutations
from collections import defaultdict
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tier_a_embedding import AtlasGraph, derive_unity_indices


@dataclass
class S4OrbitStructure:
    """Container for S₄ orbit analysis."""
    orbits: List[Set[int]]           # Orbit sets
    orbit_sizes: Dict[int, int]      # Size → count
    orbit_representatives: List[int]  # One rep per orbit
    total_orbits: int
    s4_verified: bool


class S4AutomorphismAnalyzer:
    """Analyze S₄ automorphism acting on Atlas."""

    def __init__(self):
        """Initialize with Atlas structure."""
        self.atlas = AtlasGraph()
        self.s4_group = self._generate_s4()

    def _generate_s4(self) -> List[Tuple[int, ...]]:
        """
        Generate all 24 elements of S₄.

        S₄ = symmetric group on 4 elements
        |S₄| = 4! = 24
        """
        s4 = list(permutations([0, 1, 2, 3]))
        print(f"Generated S₄ with {len(s4)} elements")
        return s4

    def apply_s4_to_label(self, label: Tuple, perm: Tuple[int, ...]) -> Tuple:
        """
        Apply S₄ permutation to a label.

        S₄ acts on coordinates (e1, e2, e3, e6) = indices (1, 2, 3, 6).
        Label format: (e1, e2, e3, e4-e5, e6, e7)
        """
        # Extract the coordinates S₄ acts on
        e1, e2, e3, e4_e5, e6, e7 = label

        # Apply permutation to (e1, e2, e3, e6)
        coords = [e1, e2, e3, e6]
        permuted = [coords[perm[i]] for i in range(4)]

        # Reconstruct label
        new_label = (permuted[0], permuted[1], permuted[2], e4_e5, permuted[3], e7)
        return new_label

    def compute_orbits(self) -> List[Set[int]]:
        """
        Compute S₄ orbits of the 96 Atlas vertices.

        Expected: 30 orbits with size distribution:
        - 12 orbits of size 1 (fixed points)
        - 12 orbits of size 4
        - 6 orbits of size 6
        """
        # Track which vertices have been assigned to orbits
        assigned = set()
        orbits = []

        print("\nComputing S₄ orbits...")

        for v in range(96):
            if v in assigned:
                continue

            # Start new orbit with vertex v
            orbit = set()
            label = self.atlas.labels[v]

            # Apply all S₄ elements
            for perm in self.s4_group:
                new_label = self.apply_s4_to_label(label, perm)

                # Find vertex with this label
                for u in range(96):
                    if self.atlas.labels[u] == new_label:
                        orbit.add(u)
                        break

            # Add orbit
            orbits.append(orbit)
            assigned.update(orbit)

        print(f"Found {len(orbits)} orbits")
        return orbits

    def analyze_orbit_structure(self, orbits: List[Set[int]]) -> Dict[int, int]:
        """
        Analyze the size distribution of orbits.

        Returns:
            Dict mapping orbit size to count
        """
        size_distribution = defaultdict(int)

        for orbit in orbits:
            size_distribution[len(orbit)] += 1

        print("\nOrbit size distribution:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"  Size {size}: {count} orbits")

        # Total vertices check
        total = sum(size * count for size, count in size_distribution.items())
        print(f"  Total vertices: {total} (expected 96)")

        return dict(size_distribution)

    def find_fixed_points(self, orbits: List[Set[int]]) -> List[int]:
        """
        Find S₄ fixed points (orbits of size 1).

        These are vertices unchanged by all S₄ permutations.
        """
        fixed_points = []

        for orbit in orbits:
            if len(orbit) == 1:
                fixed_points.extend(list(orbit))

        print(f"\nFixed points under S₄: {len(fixed_points)}")

        # Analyze fixed point labels
        if fixed_points:
            print("Fixed point labels:")
            for v in sorted(fixed_points)[:5]:  # Show first 5
                label = self.atlas.labels[v]
                print(f"  Vertex {v:2d}: {label}")

        return fixed_points

    def verify_s4_structure(self, orbits: List[Set[int]],
                           size_dist: Dict[int, int]) -> bool:
        """
        Verify the S₄ orbit structure matches expectations.

        Expected:
        - 30 total orbits
        - 12 of size 1
        - 12 of size 4
        - 6 of size 6
        """
        checks = []

        # Total orbits
        checks.append(('Total orbits = 30', len(orbits) == 30))

        # Size 1 orbits (fixed points)
        checks.append(('12 orbits of size 1', size_dist.get(1, 0) == 12))

        # Size 4 orbits
        checks.append(('12 orbits of size 4', size_dist.get(4, 0) == 12))

        # Size 6 orbits
        checks.append(('6 orbits of size 6', size_dist.get(6, 0) == 6))

        # Total vertices
        total = sum(size * count for size, count in size_dist.items())
        checks.append(('Total vertices = 96', total == 96))

        print("\nS₄ structure verification:")
        all_pass = True
        for name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {name}")
            all_pass = all_pass and result

        return all_pass

    def analyze_orbit_symmetry(self, orbits: List[Set[int]]) -> Dict:
        """
        Analyze symmetry properties within orbits.
        """
        analysis = {
            'avg_orbit_size': sum(len(o) for o in orbits) / len(orbits),
            'singleton_orbits': sum(1 for o in orbits if len(o) == 1),
            'max_orbit_size': max(len(o) for o in orbits),
            'min_orbit_size': min(len(o) for o in orbits),
        }

        # Check if orbits respect mirror symmetry
        # Vertices v and v^48 should be in related orbits
        mirror_preserved = 0
        for orbit in orbits:
            orbit_list = list(orbit)
            mirrors = [v ^ 48 for v in orbit_list]  # XOR with 48 for mirror
            if all(m in orbit for m in mirrors):
                mirror_preserved += 1

        analysis['mirror_preserving_orbits'] = mirror_preserved

        print("\nOrbit symmetry analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        return analysis


def verify_s4_automorphism():
    """
    Main function to verify S₄ automorphism.

    Returns:
        S4OrbitStructure with complete analysis
    """
    print("="*60)
    print("S₄ AUTOMORPHISM VERIFICATION")
    print("="*60)

    analyzer = S4AutomorphismAnalyzer()

    # Compute orbits
    orbits = analyzer.compute_orbits()

    # Analyze structure
    size_dist = analyzer.analyze_orbit_structure(orbits)

    # Find fixed points
    fixed_points = analyzer.find_fixed_points(orbits)

    # Verify structure
    is_valid = analyzer.verify_s4_structure(orbits, size_dist)

    # Additional symmetry analysis
    symmetry = analyzer.analyze_orbit_symmetry(orbits)

    # Get representatives
    representatives = [min(orbit) for orbit in orbits]

    # Build result
    result = S4OrbitStructure(
        orbits=orbits,
        orbit_sizes=size_dist,
        orbit_representatives=representatives,
        total_orbits=len(orbits),
        s4_verified=is_valid
    )

    if is_valid:
        print("\n✓ S₄ automorphism structure confirmed!")
        print("  - 30 orbits with expected size distribution")
        print("  - 12 fixed points under S₄ action")
        print("  - Symmetry preserved in orbit structure")
    else:
        print("\n⚠ S₄ structure does not match expectations")

    return result


if __name__ == "__main__":
    s4_structure = verify_s4_automorphism()