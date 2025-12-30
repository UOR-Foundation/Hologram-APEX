"""
Exceptional Groups Analysis Module.

Building the exceptional ladder G₂ ⊂ F₄ ⊂ E₆ ⊂ E₇ ⊂ E₈ from Atlas structure.
"""

import sys
import os
# Add tier_a_embedding to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .f4 import *
from .g2 import *

__all__ = ['f4', 'g2']