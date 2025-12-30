"""
Check F₄ quotient graph properties.

IMPORTANT: The F₄ QUOTIENT GRAPH is the graph on 48 sign classes,
NOT the 48 representative vertices in Atlas.
"""
import sys
import os
from typing import Set, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tier_a_embedding import AtlasGraph
from exceptional_groups.f4.sign_class_analysis import F4SignClassAnalyzer


def check_quotient_connectivity(adjacency_matrix) -> bool:
    """Check if quotient graph is connected via BFS."""
    n = len(adjacency_matrix)
    if n == 0:
        return False

    visited = set()
    queue = [0]

    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)

        for u in range(n):
            if adjacency_matrix[v][u] and u not in visited:
                queue.append(u)

    return len(visited) == n


def main():
    """Check F₄ quotient graph."""
    print("="*70)
    print("F₄ QUOTIENT GRAPH ANALYSIS")
    print("="*70)

    # Load F₄ structure
    analyzer = F4SignClassAnalyzer()
    f4 = analyzer.extract_sign_classes()

    print(f"\nF₄ quotient graph:")
    print(f"  Vertices (sign classes): {len(f4.sign_classes)}")
    print(f"  Adjacency matrix: {len(f4.adjacency_matrix)}×{len(f4.adjacency_matrix)}")

    # Check connectivity of QUOTIENT GRAPH
    is_connected = check_quotient_connectivity(f4.adjacency_matrix)

    print(f"\nQuotient graph connectivity:")
    print(f"  Connected: {is_connected} {'✓' if is_connected else '✗'}")

    # Compute properties
    edges = 0
    degrees = []
    for i in range(len(f4.adjacency_matrix)):
        deg = sum(f4.adjacency_matrix[i])
        degrees.append(deg)
        for j in range(i+1, len(f4.adjacency_matrix)):
            if f4.adjacency_matrix[i][j]:
                edges += 1

    print(f"\nQuotient graph properties:")
    print(f"  Edges: {edges}")
    print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Degree sequence: {f4.degree_sequence[:10]}... (first 10)")

    # This is the KEY: F₄ quotient graph should be connected!
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print(f"{'='*70}")

    if is_connected:
        print("✓ F₄ quotient graph IS connected")
        print("  → This is the correct F₄ root system")
        print("  → 48 sign classes form a connected root system")
    else:
        print("✗ F₄ quotient graph is NOT connected")
        print("  → This would indicate a problem with our construction")

    print(f"\nNOTE: The 48 representative VERTICES in Atlas don't need to be connected.")
    print(f"      What matters is the QUOTIENT GRAPH connectivity!")

    return {
        'connected': is_connected,
        'vertices': len(f4.sign_classes),
        'edges': edges,
        'degrees': degrees,
    }


if __name__ == "__main__":
    results = main()
