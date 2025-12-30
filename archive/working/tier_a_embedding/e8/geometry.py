"""
E8 geometric operations.

This module provides geometric operations on the E8 root system, including
dot products, adjacency computation, and norm calculations.
"""
from fractions import Fraction
from typing import List, Set, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_types import Root, EdgeSet, AdjacencyList, GraphStructure, E8_ROOT_DEGREE

def dot_product(u: Root, v: Root) -> Fraction:
    """
    Compute the dot product of two root vectors.

    Args:
        u: First root vector
        v: Second root vector

    Returns:
        The dot product as a Fraction
    """
    return sum(ui * vi for ui, vi in zip(u, v))

def norm_squared(r: Root) -> Fraction:
    """
    Compute the squared norm of a root vector.

    Args:
        r: Root vector

    Returns:
        The squared norm ||r||²
    """
    return dot_product(r, r)

def are_adjacent(u: Root, v: Root) -> bool:
    """
    Check if two roots are adjacent in the E8 root graph.

    Two roots are adjacent if their dot product equals 1.

    Args:
        u: First root
        v: Second root

    Returns:
        True if the roots are adjacent
    """
    return dot_product(u, v) == Fraction(1, 1)

def compute_adjacency_graph(roots: List[Root]) -> Tuple[EdgeSet, AdjacencyList]:
    """
    Compute the adjacency graph of the E8 root system.

    In the E8 root graph, two roots are connected if their dot product is 1.
    Each root has exactly 56 neighbors.

    Args:
        roots: List of E8 roots

    Returns:
        Tuple of (edges, adjacency_list)
    """
    edges: EdgeSet = set()
    adj: AdjacencyList = [set() for _ in range(len(roots))]

    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            if are_adjacent(roots[i], roots[j]):
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)

    return edges, adj

def verify_e8_geometry(roots: List[Root], adjacency: AdjacencyList) -> bool:
    """
    Verify the geometric invariants of the E8 root system.

    Args:
        roots: List of E8 roots
        adjacency: Adjacency list

    Returns:
        True if all invariants are satisfied
    """
    # Check that each root has norm squared = 2
    for r in roots:
        if norm_squared(r) != Fraction(2, 1):
            return False

    # Check that each root has degree 56
    for adj_set in adjacency:
        if len(adj_set) != E8_ROOT_DEGREE:
            return False

    return True

def find_adjacent_roots(root: Root, all_roots: List[Root]) -> List[int]:
    """
    Find all roots adjacent to a given root.

    Args:
        root: The root to find neighbors for
        all_roots: List of all roots

    Returns:
        List of indices of adjacent roots
    """
    adjacent_indices = []
    for i, other in enumerate(all_roots):
        if are_adjacent(root, other):
            adjacent_indices.append(i)
    return adjacent_indices

def compute_angle_cosine(u: Root, v: Root) -> Fraction:
    """
    Compute the cosine of the angle between two roots.

    cos(θ) = ⟨u, v⟩ / (||u|| ||v||)

    Args:
        u: First root
        v: Second root

    Returns:
        Cosine of the angle between u and v
    """
    dot = dot_product(u, v)
    norm_u = norm_squared(u)
    norm_v = norm_squared(v)

    # For E8 roots, norm_squared = 2, so ||u|| = ||v|| = √2
    # cos(θ) = ⟨u, v⟩ / 2
    return dot / 2

def is_orthogonal(u: Root, v: Root) -> bool:
    """
    Check if two roots are orthogonal.

    Args:
        u: First root
        v: Second root

    Returns:
        True if the roots are orthogonal
    """
    return dot_product(u, v) == Fraction(0, 1)

def is_opposite(u: Root, v: Root) -> bool:
    """
    Check if two roots are opposite (v = -u).

    Args:
        u: First root
        v: Second root

    Returns:
        True if v = -u
    """
    return all(vi == -ui for ui, vi in zip(u, v))

class E8Geometry:
    """Encapsulates geometric operations on the E8 root system."""

    def __init__(self, roots: List[Root]):
        """
        Initialize with E8 roots.

        Args:
            roots: List of E8 roots
        """
        self.roots = roots
        self.edges, self.adjacency = compute_adjacency_graph(roots)

        self._verify_invariants()

    def _verify_invariants(self) -> None:
        """Verify E8 geometric invariants."""
        # Check norm squared = 2 for all roots
        for r in self.roots:
            assert norm_squared(r) == Fraction(2, 1), "Each root must have norm squared = 2"

        # Check degree = 56 for all roots
        degrees = [len(adj) for adj in self.adjacency]
        assert all(d == E8_ROOT_DEGREE for d in degrees), f"Each E8 root must have degree {E8_ROOT_DEGREE}"

    def distance(self, i: int, j: int) -> int:
        """
        Compute graph distance between two roots.

        Args:
            i: First root index
            j: Second root index

        Returns:
            Shortest path distance in the root graph
        """
        if i == j:
            return 0

        # BFS to find shortest path
        from collections import deque
        visited = [False] * len(self.roots)
        queue = deque([(i, 0)])
        visited[i] = True

        while queue:
            curr, dist = queue.popleft()
            if curr == j:
                return dist

            for neighbor in self.adjacency[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))

        return -1  # Not connected (shouldn't happen in E8)

    def get_graph_structure(self) -> GraphStructure:
        """
        Return a GraphStructure object for the E8 root graph.

        Returns:
            GraphStructure with vertices, edges, and adjacency
        """
        return GraphStructure(
            vertices=len(self.roots),
            edges=self.edges,
            adjacency=self.adjacency
        )

def build_GE8() -> Tuple[List[Root], EdgeSet, AdjacencyList]:
    """
    Build the E8 root graph (backward compatibility function).

    Returns:
        Tuple of (roots, edges, adjacency)
    """
    from .roots import generate_e8_roots
    roots = generate_e8_roots()
    edges, adj = compute_adjacency_graph(roots)
    return roots, edges, adj

def dot(u: Root, v: Root) -> Fraction:
    """
    Compute dot product (backward compatibility function).

    Args:
        u: First root
        v: Second root

    Returns:
        Dot product
    """
    return dot_product(u, v)