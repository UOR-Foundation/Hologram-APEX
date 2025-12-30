"""
Embedding search module.

This module implements the backtracking search for Tier-A embeddings.
"""

from .result import EmbeddingResult
from .search import (
    EmbeddingSearch,
    EmbeddingConstraints,
    find_embedding,
    tierA_search,
)

__all__ = [
    # Result
    "EmbeddingResult",
    # Search
    "EmbeddingSearch",
    "EmbeddingConstraints",
    "find_embedding",
    "tierA_search",
]