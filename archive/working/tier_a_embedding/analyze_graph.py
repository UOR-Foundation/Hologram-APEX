#!/usr/bin/env python3
"""
Analyze the graph structure of Tier-A embeddings.

This script extracts and displays the 1-skeleton graph structure
and quotient properties of the embedding.
"""
import json
from typing import List, Dict

from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from analysis import (
    NeighborAnalyzer,
    SkeletonAnalysis,
    QuotientAnalyzer,
    extract_1_skeleton,
    analyze_quotient_properties
)


def print_separator(title: str = ""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def find_embedding() -> List[int]:
    """Find a Tier-A embedding with 48 sign classes."""
    print("Finding Tier-A embedding...")
    atlas = AtlasGraph()
    e8 = E8RootSystem()

    search = EmbeddingSearch(atlas, e8)
    constraints = EmbeddingConstraints(
        max_solutions=1,
        target_signs=48,
        verbose=False
    )

    solutions = search.search(constraints)
    if not solutions:
        raise RuntimeError("No embedding found")

    print(f"✓ Found embedding using {constraints.target_signs} sign classes")
    return solutions[0], atlas, e8


def analyze_neighbors(mapping: List[int], atlas, e8):
    """Analyze neighbor relationships."""
    print_separator("Neighbor Analysis")

    analyzer = NeighborAnalyzer(mapping, atlas, e8)

    # Verify edge preservation
    preserved = analyzer.verify_edge_preservation()
    print(f"Edge preservation verified: {preserved}")

    # Analyze degree distribution
    degree_info = analyzer.analyze_degree_distribution()
    print(f"Sign classes: {degree_info['num_classes']}")
    print(f"Degree range: {degree_info['min_degree']}-{degree_info['max_degree']}")
    print(f"Average degree: {degree_info['avg_degree']:.2f}")

    # Show degree distribution
    print("\nDegree distribution:")
    for degree, count in sorted(degree_info['degree_distribution'].items()):
        print(f"  Degree {degree}: {count} classes")

    # Get sign class adjacency
    class_adj = analyzer.get_sign_class_adjacency()

    # Show sample of adjacency (first 5 classes)
    print("\nSample adjacency (first 5 sign classes):")
    shown = 0
    for cls, neighbors in sorted(class_adj.items())[:5]:
        neighbors_list = sorted(list(neighbors))[:10]  # Show first 10 neighbors
        print(f"  Class {cls}: {len(neighbors)} neighbors - {neighbors_list[:5]}...")
        shown += 1

    return class_adj


def analyze_skeleton(mapping: List[int], atlas, e8):
    """Analyze 1-skeleton structure."""
    print_separator("1-Skeleton Analysis")

    analyzer = SkeletonAnalysis(mapping, atlas, e8)

    # Extract skeleton
    skeleton = analyzer.extract_1_skeleton()
    print(f"Vertices (sign classes): {len(skeleton.vertices)}")
    print(f"Edges (between classes): {len(skeleton.edges)}")

    # Compute properties
    properties = analyzer.compute_skeleton_properties()
    print(f"Internal edges (within classes): {properties['num_internal_edges']}")
    print(f"Total Atlas edges: {properties['total_atlas_edges']}")
    print(f"Inter-class edge ratio: {properties['edge_preservation_ratio']:.3f}")
    print(f"Connected: {properties['is_connected']}")
    if properties['is_connected']:
        print(f"Diameter: {properties['diameter']}")

    # Show degree statistics
    print(f"\nDegree statistics:")
    print(f"  Min: {properties['min_degree']}")
    print(f"  Max: {properties['max_degree']}")
    print(f"  Avg: {properties['avg_degree']:.2f}")

    return skeleton


def analyze_quotient(mapping: List[int], atlas, e8):
    """Analyze quotient structure."""
    print_separator("Quotient Structure")

    analyzer = QuotientAnalyzer(mapping, atlas, e8)

    # Build quotient graph
    quotient = analyzer.build_quotient_graph()
    print(f"Quotient vertices: {len(quotient.vertices)}")
    print(f"Quotient edges: {len(quotient.edges)}")

    # Analyze properties
    properties = analyzer.analyze_quotient_properties()
    print(f"Density: {properties['density']:.3f}")
    print(f"Regular: {properties['is_regular']}")
    print(f"E8 structure preserved: {properties['e8_structure_preserved']}")
    print(f"Connected: {properties['connectivity']['is_connected']}")
    print(f"Clustering coefficient: {properties['clustering_coefficient']:.3f}")

    # Show class sizes
    print(f"\nClass size distribution:")
    size_counts = {}
    for size in properties['class_sizes'].values():
        size_counts[size] = size_counts.get(size, 0) + 1
    for size, count in sorted(size_counts.items()):
        print(f"  Size {size}: {count} classes")

    # Show degree sequence
    print(f"\nDegree sequence (sorted):")
    deg_seq = properties['degree_sequence'][:10]  # First 10
    print(f"  {deg_seq}...")

    # Compute distance matrix
    distances = analyzer.get_quotient_distance_matrix()
    if distances:
        # Find diameter
        max_dist = max(max(row) for row in distances)
        print(f"\nGraph diameter: {max_dist}")

        # Show distance distribution
        dist_counts = {}
        for row in distances:
            for d in row:
                if d > 0:  # Skip self-distances
                    dist_counts[d] = dist_counts.get(d, 0) + 1

        print("Distance distribution:")
        for d in sorted(dist_counts.keys()):
            print(f"  Distance {d}: {dist_counts[d]//2} pairs")  # Divide by 2 for undirected

    return quotient, properties


def display_graph_representation(quotient, class_adj):
    """Display a text representation of the graph."""
    print_separator("Graph Structure Visualization")

    # Show adjacency matrix for small subset
    print("\nAdjacency matrix (first 8x8 subgraph):")
    print("    ", end="")
    for j in range(min(8, len(quotient.vertices))):
        print(f"{j:3}", end="")
    print()

    for i in range(min(8, len(quotient.vertices))):
        print(f"{i:3}: ", end="")
        for j in range(min(8, len(quotient.vertices))):
            if quotient.adjacency_matrix[i][j]:
                print(" ■ ", end="")
            else:
                print(" · ", end="")
        print()

    # Show edge list for first few vertices
    print("\nEdge list (first 10 edges):")
    for i, (v1, v2) in enumerate(quotient.edges[:10]):
        print(f"  {i+1:2}. Class {v1:3} -- Class {v2:3}")

    # Create simple ASCII visualization of a small subgraph
    print("\nSubgraph visualization (first 5 vertices):")
    if len(quotient.vertices) >= 5:
        # Simple star/mesh visualization
        for i in range(min(5, len(quotient.vertices))):
            v = quotient.vertices[i]
            neighbors = []
            for j in range(min(5, len(quotient.vertices))):
                if i != j and quotient.adjacency_matrix[i][j]:
                    neighbors.append(quotient.vertices[j])
            print(f"  [{v:3}] --> {neighbors}")


def save_results(mapping, quotient, properties, filename="graph_analysis.json"):
    """Save analysis results to JSON file."""
    results = {
        "summary": {
            "sign_classes_used": len(quotient.vertices),
            "quotient_edges": len(quotient.edges),
            "is_regular": properties["is_regular"],
            "e8_preserved": properties["e8_structure_preserved"],
            "clustering": properties["clustering_coefficient"],
            "diameter": properties.get("diameter", -1),
        },
        "degree_sequence": properties["degree_sequence"],
        "class_sizes": list(properties["class_sizes"].values()),
        "edges": [(int(a), int(b)) for a, b in quotient.edges[:100]],  # First 100 edges
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("Tier-A Embedding Graph Structure Analysis")
    print("="*60)

    # Find embedding
    mapping, atlas, e8 = find_embedding()

    # Analyze neighbors
    class_adj = analyze_neighbors(mapping, atlas, e8)

    # Analyze skeleton
    skeleton = analyze_skeleton(mapping, atlas, e8)

    # Analyze quotient
    quotient, properties = analyze_quotient(mapping, atlas, e8)

    # Display visualization
    display_graph_representation(quotient, class_adj)

    # Save results
    save_results(mapping, quotient, properties)

    print_separator()
    print("Analysis complete!")


if __name__ == "__main__":
    main()