"""
Tier-A Embedding Package.

A refactored implementation of the Atlas-to-E8 embedding search algorithm.
"""

# Import main types
from .common_types import (
    Label,
    Root,
    EdgeSet,
    AdjacencyList,
    GraphStructure,
    EmbeddingResult,
    CertificateData,
)

# Import main classes and functions
from .atlas.graph import AtlasGraph
from .atlas.labels import derive_unity_indices
from .e8.roots import E8RootSystem
from .embedding.search import find_embedding
from .main import main, search_embedding, verify_embedding_mapping

__version__ = "1.0.0"

__all__ = [
    # Types
    "Label",
    "Root",
    "EdgeSet",
    "AdjacencyList",
    "GraphStructure",
    "EmbeddingResult",
    "CertificateData",
    # Main components
    "AtlasGraph",
    "E8RootSystem",
    "derive_unity_indices",
    "find_embedding",
    # Pipeline functions
    "main",
    "search_embedding",
    "verify_embedding_mapping",
]