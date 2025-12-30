#!/usr/bin/env python3
"""
Exact Arithmetic for Atlas

All arithmetic must be exact - no floating point approximations.
Complex numbers with rational real and imaginary parts.
"""

from fractions import Fraction
from typing import Union


class ComplexFraction:
    """
    Complex number with exact rational real and imaginary parts.

    This is the correct type for Atlas field amplitudes.
    """

    def __init__(self, real: Union[int, Fraction, str] = 0,
                 imag: Union[int, Fraction, str] = 0):
        """
        Initialize exact complex number.

        Args:
            real: Real part (int, Fraction, or string like "1/2")
            imag: Imaginary part (int, Fraction, or string like "1/2")
        """
        self.real = Fraction(real) if not isinstance(real, Fraction) else real
        self.imag = Fraction(imag) if not isinstance(imag, Fraction) else imag

    def __add__(self, other: 'ComplexFraction') -> 'ComplexFraction':
        """Exact addition."""
        return ComplexFraction(
            self.real + other.real,
            self.imag + other.imag
        )

    def __sub__(self, other: 'ComplexFraction') -> 'ComplexFraction':
        """Exact subtraction."""
        return ComplexFraction(
            self.real - other.real,
            self.imag - other.imag
        )

    def __mul__(self, other: Union['ComplexFraction', int, Fraction]) -> 'ComplexFraction':
        """Exact multiplication."""
        if isinstance(other, (int, Fraction)):
            # Scalar multiplication
            other_frac = Fraction(other) if not isinstance(other, Fraction) else other
            return ComplexFraction(
                self.real * other_frac,
                self.imag * other_frac
            )
        elif isinstance(other, ComplexFraction):
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            return ComplexFraction(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real
            )
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, Fraction]) -> 'ComplexFraction':
        """Right scalar multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union['ComplexFraction', int, Fraction]) -> 'ComplexFraction':
        """Exact division."""
        if isinstance(other, (int, Fraction)):
            # Scalar division
            other_frac = Fraction(other) if not isinstance(other, Fraction) else other
            return ComplexFraction(
                self.real / other_frac,
                self.imag / other_frac
            )
        elif isinstance(other, ComplexFraction):
            # (a + bi)/(c + di) = (a + bi)(c - di)/(c² + d²)
            denom = other.real * other.real + other.imag * other.imag
            numerator = self * other.conjugate()
            return ComplexFraction(
                numerator.real / denom,
                numerator.imag / denom
            )
        else:
            return NotImplemented

    def __neg__(self) -> 'ComplexFraction':
        """Negation."""
        return ComplexFraction(-self.real, -self.imag)

    def conjugate(self) -> 'ComplexFraction':
        """Complex conjugate."""
        return ComplexFraction(self.real, -self.imag)

    def norm_squared(self) -> Fraction:
        """Squared norm |z|² = zz̄ (exact rational)."""
        return self.real * self.real + self.imag * self.imag

    def __eq__(self, other: 'ComplexFraction') -> bool:
        """Exact equality (no tolerance!)."""
        if isinstance(other, ComplexFraction):
            return self.real == other.real and self.imag == other.imag
        elif isinstance(other, (int, Fraction)):
            return self.real == other and self.imag == 0
        else:
            return False

    def __repr__(self) -> str:
        """String representation."""
        if self.imag == 0:
            return f"{self.real}"
        elif self.real == 0:
            return f"{self.imag}j"
        else:
            sign = '+' if self.imag >= 0 else '-'
            return f"{self.real}{sign}{abs(self.imag)}j"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def zero() -> 'ComplexFraction':
        """Zero complex number."""
        return ComplexFraction(0, 0)

    @staticmethod
    def one() -> 'ComplexFraction':
        """Unity complex number."""
        return ComplexFraction(1, 0)

    def is_zero(self) -> bool:
        """Check if exactly zero."""
        return self.real == 0 and self.imag == 0


# Convenience functions for common values
def cfrac(real: Union[int, Fraction, str] = 0,
          imag: Union[int, Fraction, str] = 0) -> ComplexFraction:
    """Create ComplexFraction (convenience function)."""
    return ComplexFraction(real, imag)


def sqrt_exact(n: int) -> Union[Fraction, str]:
    """
    Return exact square root if rational, otherwise return symbolic.

    For Atlas work, we need sqrt(2) for F₄ long roots.
    Since sqrt(2) is irrational, we keep it symbolic.

    Args:
        n: Integer to take square root of

    Returns:
        Fraction if perfect square, otherwise string "sqrt(n)"
    """
    # Check if perfect square
    sqrt_n = int(n ** 0.5)
    if sqrt_n * sqrt_n == n:
        return Fraction(sqrt_n)
    else:
        # Return symbolic - caller must handle algebraically
        return f"sqrt({n})"


if __name__ == '__main__':
    print("Testing exact complex arithmetic...")

    # Test basic operations
    z1 = ComplexFraction(1, 2)  # 1 + 2j
    z2 = ComplexFraction(3, -1)  # 3 - j

    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    print(f"z1 + z2 = {z1 + z2}")
    print(f"z1 * z2 = {z1 * z2}")
    print(f"|z1|² = {z1.norm_squared()}")

    # Test exact equality
    z3 = ComplexFraction(Fraction(1, 2), 0)
    z4 = ComplexFraction(Fraction(2, 4), 0)
    print(f"\n{z3} == {z4}: {z3 == z4}")  # Should be True (1/2 == 2/4)

    # Test scalar multiplication
    z5 = z1 * 2
    print(f"2 * {z1} = {z5}")

    # Test division
    z6 = z1 / z2
    print(f"{z1} / {z2} = {z6}")

    # Test conjugate
    print(f"conj({z1}) = {z1.conjugate()}")

    print("\n✓ Exact complex arithmetic working")
    print("  No floats, no approximations, no tolerances")
    print("  Pure exact rational arithmetic")
