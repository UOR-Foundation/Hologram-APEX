#!/usr/bin/env python3
"""
Atlas 96-Vertex Polytope Structure

The fundamental degrees of freedom are the 96 resonance classes (R96),
not the 12,288 boundary sites. Each class represents an equivalence class
of bytes under the unity constraint α₄α₅=1.

From atlas_polytope_comprehensive_formalization_v_1.md Appendix B:
96 canonical bytes determined by (e₁,e₂,e₃,e₄-e₅,e₆,e₇) where:
- e₄-e₅ ∈ {-1, 0, +1} (3 choices)
- e₁,e₂,e₃,e₆,e₇ ∈ {0, 1} (2⁵ = 32 choices)
- e₀ = 0 (fixed)
Total: 3 × 32 = 96 classes
"""

from typing import List, Tuple, Set, Union
from dataclasses import dataclass
from fractions import Fraction

from action_framework.core.exact_arithmetic import ComplexFraction


# Klein quartet - the fundamental unity positions
KLEIN_QUARTET = frozenset({0, 1, 48, 49})


# The 96 canonical byte representatives
# These are the even integers that represent distinct resonance classes
R96_CANONICAL_BYTES = [
    0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30,
   32,  34,  36,  38,  40,  42,  44,  46,  64,  66,  68,  70,  72,  74,  76,  78,
   80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110,
  128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158,
  160, 162, 164, 166, 168, 170, 172, 174, 192, 194, 196, 198, 200, 202, 204, 206,
  208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238
]

assert len(R96_CANONICAL_BYTES) == 96, "Must have exactly 96 canonical bytes"
assert all(b % 2 == 0 for b in R96_CANONICAL_BYTES), "All canonical bytes must be even (e₀=0)"


@dataclass
class ByteStructure:
    """
    Structure of a byte in terms of its 8 bits.

    Attributes:
        byte_value: Integer 0-255
        e0, e1, ..., e7: Individual bits
    """
    byte_value: int

    @property
    def e0(self) -> int:
        return (self.byte_value >> 0) & 1

    @property
    def e1(self) -> int:
        return (self.byte_value >> 1) & 1

    @property
    def e2(self) -> int:
        return (self.byte_value >> 2) & 1

    @property
    def e3(self) -> int:
        return (self.byte_value >> 3) & 1

    @property
    def e4(self) -> int:
        return (self.byte_value >> 4) & 1

    @property
    def e5(self) -> int:
        return (self.byte_value >> 5) & 1

    @property
    def e6(self) -> int:
        return (self.byte_value >> 6) & 1

    @property
    def e7(self) -> int:
        return (self.byte_value >> 7) & 1

    def resonance_class_label(self) -> Tuple[int, int, int, int, int, int]:
        """
        Return the 6-tuple (e₁, e₂, e₃, e₄-e₅, e₆, e₇) that determines resonance class.

        This is the equivalence class under unity constraint α₄α₅=1.
        """
        return (self.e1, self.e2, self.e3, self.e4 - self.e5, self.e6, self.e7)

    def canonical_representative(self) -> int:
        """
        Return the canonical byte in the same resonance class.

        Canonical choice:
        - e₀ = 0 (always)
        - (e₄, e₅) chosen as: (0,1) if e₄-e₅=-1
                               (0,0) if e₄-e₅=0
                               (1,0) if e₄-e₅=+1
        """
        e1, e2, e3, diff, e6, e7 = self.resonance_class_label()

        # Choose canonical (e4, e5) pair
        if diff == -1:
            e4, e5 = 0, 1
        elif diff == 0:
            e4, e5 = 0, 0
        else:  # diff == +1
            e4, e5 = 1, 0

        # Construct canonical byte with e0=0
        canonical = (e1 << 1) | (e2 << 2) | (e3 << 3) | (e4 << 4) | (e5 << 5) | (e6 << 6) | (e7 << 7)
        return canonical


class AtlasField:
    """
    A field on the Atlas 96-vertex polytope.

    This is the CORRECT representation: one complex amplitude per resonance class,
    not one per boundary site. The 12,288 boundary sites are derived from these
    96 fundamental values.

    This is a 96-dimensional complex vector space with EXACT arithmetic.
    """

    def __init__(self, amplitudes=None):
        """
        Initialize Atlas field with exact arithmetic.

        Args:
            amplitudes: Either None (zeros), or list of 96 ComplexFraction
        """
        if amplitudes is None:
            self.amplitudes: List[ComplexFraction] = [ComplexFraction.zero() for _ in range(96)]
        elif isinstance(amplitudes, list):
            assert len(amplitudes) == 96, f"Expected 96 amplitudes, got {len(amplitudes)}"
            assert all(isinstance(a, ComplexFraction) for a in amplitudes), "All amplitudes must be ComplexFraction"
            self.amplitudes = amplitudes
        else:
            raise TypeError(f"Invalid amplitudes type: {type(amplitudes)}")

    def __getitem__(self, class_index: int) -> ComplexFraction:
        """Get amplitude for resonance class by index 0-95."""
        assert 0 <= class_index < 96, f"Class index must be in [0,96), got {class_index}"
        return self.amplitudes[class_index]

    def __setitem__(self, class_index: int, value: ComplexFraction):
        """Set amplitude for resonance class by index 0-95."""
        assert 0 <= class_index < 96, f"Class index must be in [0,96), got {class_index}"
        assert isinstance(value, ComplexFraction), f"Value must be ComplexFraction, got {type(value)}"
        self.amplitudes[class_index] = value

    def get_by_byte(self, byte_value: int) -> ComplexFraction:
        """
        Get amplitude for the resonance class containing this byte.

        Args:
            byte_value: Any byte 0-255

        Returns:
            Amplitude of its resonance class
        """
        canonical = ByteStructure(byte_value).canonical_representative()
        class_index = R96_CANONICAL_BYTES.index(canonical)
        return self.amplitudes[class_index]

    def set_by_byte(self, byte_value: int, amplitude: ComplexFraction):
        """Set amplitude for resonance class containing this byte."""
        canonical = ByteStructure(byte_value).canonical_representative()
        class_index = R96_CANONICAL_BYTES.index(canonical)
        self.amplitudes[class_index] = amplitude

    def __add__(self, other: 'AtlasField') -> 'AtlasField':
        """Pointwise addition (exact)."""
        new_amplitudes = [a + b for a, b in zip(self.amplitudes, other.amplitudes)]
        return AtlasField(new_amplitudes)

    def __sub__(self, other: 'AtlasField') -> 'AtlasField':
        """Pointwise subtraction (exact)."""
        new_amplitudes = [a - b for a, b in zip(self.amplitudes, other.amplitudes)]
        return AtlasField(new_amplitudes)

    def __mul__(self, scalar: Union[int, Fraction, ComplexFraction]) -> 'AtlasField':
        """Scalar multiplication (exact)."""
        if isinstance(scalar, int):
            scalar = ComplexFraction(scalar, 0)
        elif isinstance(scalar, Fraction):
            scalar = ComplexFraction(scalar, 0)
        assert isinstance(scalar, ComplexFraction), f"Scalar must be ComplexFraction, got {type(scalar)}"
        new_amplitudes = [a * scalar for a in self.amplitudes]
        return AtlasField(new_amplitudes)

    def __rmul__(self, scalar: Union[int, Fraction, ComplexFraction]) -> 'AtlasField':
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def norm_squared(self) -> Fraction:
        """L² norm squared: ||ψ||² = Σ|ψᵢ|² (exact rational)."""
        return sum((a.norm_squared() for a in self.amplitudes), Fraction(0))

    def dot(self, other: 'AtlasField') -> ComplexFraction:
        """Inner product ⟨ψ, φ⟩ = Σ ψ̄ᵢ·φᵢ (exact)."""
        result = ComplexFraction.zero()
        for a, b in zip(self.amplitudes, other.amplitudes):
            result = result + a.conjugate() * b
        return result

    def copy(self) -> 'AtlasField':
        """Deep copy."""
        return AtlasField([a for a in self.amplitudes])


def get_klein_class_index() -> int:
    """
    Get the resonance class index for the Klein quartet.

    IMPORTANT: All four Klein quartet bytes {0, 1, 48, 49} belong to the
    SAME resonance class (the unity class). They differ in e₀ and in the
    (e₄,e₅) pair being (0,0) or (1,1), but both give R(b)=1 due to:
    - α₀ = 1 (so e₀ doesn't affect R)
    - α₄α₅ = 1 (so (0,0) and (1,1) both give contribution 1)

    The Klein quartet forms the Klein four-group V₄ acting on this class.

    Returns:
        Index 0 (the unity class, canonical byte 0)
    """
    # All Klein bytes canonicalize to byte 0
    canonical = ByteStructure(0).canonical_representative()
    assert canonical == 0, "Klein unity class should be canonical byte 0"
    idx = R96_CANONICAL_BYTES.index(canonical)
    assert idx == 0, "Klein class should be index 0"
    return idx


def test_atlas_structure():
    """Test Atlas structure implementation with EXACT arithmetic."""
    print("Testing Atlas 96-vertex structure (EXACT ARITHMETIC)...")

    # Test 1: All canonical bytes are distinct
    assert len(set(R96_CANONICAL_BYTES)) == 96, "Canonical bytes must be distinct"
    print(f"  ✓ 96 distinct canonical bytes")

    # Test 2: Klein quartet (all map to same class!)
    klein_idx = get_klein_class_index()
    print(f"  Klein quartet class index: {klein_idx}")
    print(f"  Klein quartet bytes {sorted(KLEIN_QUARTET)} → resonance class {klein_idx}")
    # Verify all Klein bytes map to the same class
    for byte_val in KLEIN_QUARTET:
        canonical = ByteStructure(byte_val).canonical_representative()
        assert canonical == 0, f"Klein byte {byte_val} should canonicalize to 0"
    print(f"  ✓ Klein quartet = unity class (V₄ symmetry on class 0)")

    # Test 3: Canonical representative computation
    test_bytes = [0, 1, 16, 17, 32, 33]  # Some test cases
    for b in test_bytes:
        bs = ByteStructure(b)
        canonical = bs.canonical_representative()
        assert canonical in R96_CANONICAL_BYTES, f"Canonical {canonical} for byte {b} not in R96"
    print(f"  ✓ Canonical representative computation correct")

    # Test 4: Atlas field operations (EXACT)
    psi = AtlasField()
    assert psi.norm_squared() == Fraction(0), "Zero field should have zero norm (exact)"

    psi[0] = ComplexFraction(1, 0)  # 1 + 0j
    psi[1] = ComplexFraction(0, 1)  # 0 + 1j
    norm_sq = psi.norm_squared()
    assert norm_sq == Fraction(2), f"Norm squared should be exactly 2, got {norm_sq}"

    phi = AtlasField()
    phi[0] = ComplexFraction(1, 0)
    inner = psi.dot(phi)
    assert inner == ComplexFraction(1, 0), f"Inner product should be exactly 1, got {inner}"
    print(f"  ✓ Atlas field operations correct (EXACT, no tolerances)")

    # Test 5: Klein quartet field (unity class)
    klein_field = AtlasField()
    klein_field[klein_idx] = ComplexFraction(1, 0)  # Set unity class to 1

    # Verify all Klein bytes access the same class (EXACT equality)
    for byte_val in KLEIN_QUARTET:
        val = klein_field.get_by_byte(byte_val)
        assert val == ComplexFraction(1, 0), f"Klein byte {byte_val} should access unity class (exact)"
    print(f"  ✓ Klein quartet field access (all access class 0, EXACT)")

    print("✓ All Atlas structure tests passed (EXACT ARITHMETIC)")
    print(f"\nAtlas field: 96-dimensional complex vector space")
    print(f"  Resonance classes: {len(R96_CANONICAL_BYTES)}")
    print(f"  Klein quartet: {sorted(KLEIN_QUARTET)}")
    print(f"  Dimension: 96 (not 12,288!)")
    print(f"  Arithmetic: EXACT (Fraction-based, no floats)")


if __name__ == '__main__':
    test_atlas_structure()
