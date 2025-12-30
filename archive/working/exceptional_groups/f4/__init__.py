"""
F₄ Exceptional Group Analysis.

F₄ has 48 roots - exactly matching our 48 sign classes from tier_a_embedding!
"""

from .sign_class_analysis import F4SignClassAnalyzer, extract_f4_from_sign_classes
from .cartan_extraction import extract_cartan_matrix, verify_f4_cartan

__all__ = [
    'F4SignClassAnalyzer',
    'extract_f4_from_sign_classes',
    'extract_cartan_matrix',
    'verify_f4_cartan'
]