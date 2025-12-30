#!/usr/bin/env python3
"""
Main execution pipeline for Tier-A embedding search.

This module provides the high-level interface for finding and verifying
Tier-A embeddings from the Atlas graph to the E8 root system.
"""
import json
from typing import Optional, List

# Import core components
from atlas import AtlasGraph, derive_unity_indices
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints
from certificate import create_certificate, verify_certificate, CertificateGenerator
from canonicalization import canonicalize_embedding


def run_tests() -> bool:
    """
    Run basic verification tests.

    Returns:
        True if all tests pass
    """
    print("Running tests...")

    # Test Atlas graph
    print("  Testing Atlas graph...")
    atlas = AtlasGraph()
    assert len(atlas.labels) == 96, f"Expected 96 vertices, got {len(atlas.labels)}"
    assert len(atlas.edges) > 0, "Atlas graph has no edges"

    # Test E8 root system
    print("  Testing E8 root system...")
    e8 = E8RootSystem()
    assert len(e8.roots) == 240, f"Expected 240 roots, got {len(e8.roots)}"

    # Test unity indices
    print("  Testing unity derivation...")
    unity = derive_unity_indices(atlas.labels)
    assert len(unity) == 2, f"Expected 2 unity indices, got {len(unity)}"

    print("OK: All tests passed")
    return True


def search_embedding(
    target_signs: Optional[int] = None,
    max_solutions: int = 1,
    verbose: bool = True
) -> List:
    """
    Search for Tier-A embeddings.

    Args:
        target_signs: Target number of sign classes to use (e.g., 48)
        max_solutions: Maximum number of solutions to find
        verbose: Whether to print progress

    Returns:
        List of embedding mappings
    """
    # Build structures
    if verbose:
        print("Building Atlas graph...")
    atlas = AtlasGraph()

    if verbose:
        print("Building E8 root system...")
    e8 = E8RootSystem()

    # Create search constraints
    constraints = EmbeddingConstraints(
        max_solutions=max_solutions,
        target_signs=target_signs,
        verbose=verbose
    )

    # Search for embeddings
    if verbose:
        print(f"Searching for embeddings (max {max_solutions})...")
    search = EmbeddingSearch(atlas, e8)
    results = search.search(constraints)

    if verbose:
        if results:
            print(f"Found {len(results)} embedding(s)")
        else:
            print("No embeddings found")

    return results


def verify_embedding_mapping(mapping: List[int], atlas: AtlasGraph, e8: E8RootSystem) -> bool:
    """
    Verify an embedding mapping.

    Args:
        mapping: Vertex to root mapping
        atlas: AtlasGraph instance
        e8: E8RootSystem instance

    Returns:
        True if the embedding is valid
    """
    # Generate certificate and verify it
    cert_json = create_certificate(mapping, atlas, e8)
    is_valid = verify_certificate(cert_json, verbose=True)
    return is_valid


def main(
    target_signs: Optional[int] = 48,
    max_solutions: int = 1,
    run_verification: bool = True,
    use_canonicalization: bool = True
):
    """
    Main execution pipeline.

    Args:
        target_signs: Target number of sign classes (e.g., 48)
        max_solutions: Maximum number of solutions to find
        run_verification: Whether to run verification tests first
        use_canonicalization: Whether to canonicalize found embeddings
    """
    # Run tests if requested
    if run_verification:
        if not run_tests():
            print("Tests failed, aborting")
            return

    # Search for embeddings
    results = search_embedding(target_signs, max_solutions)

    if not results:
        print("No Tier-A embedding found")
        return

    # Build structures for verification
    atlas = AtlasGraph()
    e8 = E8RootSystem()

    # Canonicalize if requested and multiple solutions found
    if use_canonicalization and len(results) > 1:
        print(f"\nCanonicalizing {len(results)} embeddings...")
        mapping = canonicalize_embedding(results, atlas, e8)
        print("Selected canonical representative")
    else:
        mapping = results[0]

    # Verify the embedding
    if not verify_embedding_mapping(mapping, atlas, e8):
        print("Embedding verification failed")
        return

    # Generate certificate
    cert_json = create_certificate(mapping, atlas, e8)
    print("\nCertificate generated successfully")

    # Parse and print statistics
    cert_dict = json.loads(cert_json)
    print(f"\nStatistics:")
    print(f"  Vertices mapped: {len(mapping)}")
    print(f"  Sign classes used: {cert_dict['sign_classes_used']}")

    # Count root types
    int_count = sum(1 for idx in mapping if all(x.denominator == 1 for x in e8.roots[idx]))
    print(f"  Integer roots used: {int_count}")
    print(f"  Half-integer roots used: {len(mapping) - int_count}")

    # Save certificate to file
    with open("tier_a_certificate.json", "w") as f:
        f.write(cert_json)
    print("\nCertificate saved to tier_a_certificate.json")


if __name__ == "__main__":
    main()