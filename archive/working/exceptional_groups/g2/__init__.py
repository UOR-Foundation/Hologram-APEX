"""
G₂ Exceptional Group Analysis.

G₂ is the smallest exceptional group with 12 roots.
It manifests in Atlas through 12-fold periodicity.
"""

from .klein_structure import G2KleinAnalyzer, find_klein_quartet
from .twelve_fold import G2Periodicity, verify_twelve_fold

__all__ = [
    'G2KleinAnalyzer',
    'find_klein_quartet',
    'G2Periodicity',
    'verify_twelve_fold'
]