# Tier-A Embedding Package

A Python implementation for finding and verifying Tier-A embeddings from the Atlas graph to the E8 root system.

## Overview

This package implements a backtracking search algorithm to find injective mappings from the 96-vertex Atlas graph to the 240-root E8 root system that preserve:
- Edge adjacency relationships
- Mirror symmetry (tau pairing)
- Unity constraint (selected vertices sum to zero)
- Sign class constraints

The implementation uses exact rational arithmetic (fractions.Fraction) throughout to ensure mathematical correctness.

## Package Structure

```
tier_a_embedding/
├── atlas/              # Atlas graph module
│   ├── graph.py       # Graph construction and properties
│   ├── labels.py      # Label generation and operations
│   └── symmetry.py    # Mirror and S4 automorphism symmetries
├── e8/                # E8 root system module
│   ├── roots.py       # Root generation and properties
│   └── geometry.py    # Adjacency and geometric operations
├── embedding/         # Embedding search module
│   ├── search.py      # Backtracking search algorithm
│   ├── constraints.py # Constraint checking
│   └── result.py      # Result representation
├── certificate/       # Certificate generation/verification
│   ├── generator.py   # Generate verifiable certificates
│   ├── verifier.py    # Independent certificate verification
│   └── format.py      # Certificate schema and formatting
├── canonicalization/  # Canonical form selection
│   ├── canonical.py   # Select unique representatives
│   ├── equivalence.py # Equivalence key computation
│   └── automorphism.py # S4 group operations
├── tests/             # Test suite
│   ├── test_atlas.py
│   ├── test_e8.py
│   ├── test_embedding.py
│   ├── test_certificate.py
│   ├── test_canonicalization.py
│   └── test_integration.py
├── common_types.py    # Shared type definitions
└── main.py           # Main execution pipeline
```

## Key Components

### Atlas Graph
- 96 vertices labeled by 6-bit tuples (e1, e2, e3, d45, e6, e7)
- Mirror symmetry: tau(e7) = 1 - e7
- S4 automorphism group acting on {e1, e2, e3, e6}
- Unity vertices: 2 special vertices that must map to roots summing to zero

### E8 Root System
- 240 roots in 8-dimensional space
- 112 integer roots (coordinates in Z)
- 128 half-integer roots (coordinates in Z/2)
- Adjacency based on dot product = 1
- All roots have norm 2

### Embedding Search
- Complete backtracking search with constraint propagation
- Mirror pairing constraint reduces search space by half
- Early termination when target sign classes reached
- Support for required mappings and multiple solutions

### Certificates
- JSON format for independently verifiable proofs
- Contains complete mapping and validation data
- Independent verification without search code
- Includes statistics and metadata

### Canonicalization
- Selects unique representative from equivalence classes
- Handles S4 automorphisms and tau/sign symmetries
- Orders by: automorphism key, integer root count, lexicographic

## Usage

### Basic Search
```python
from atlas import AtlasGraph
from e8 import E8RootSystem
from embedding import EmbeddingSearch, EmbeddingConstraints

# Build structures
atlas = AtlasGraph()
e8 = E8RootSystem()

# Configure search
constraints = EmbeddingConstraints(
    max_solutions=1,
    target_signs=48,  # Use exactly 48 sign classes
    verbose=True
)

# Run search
search = EmbeddingSearch(atlas, e8)
solutions = search.search(constraints)

if solutions:
    print(f"Found {len(solutions)} embedding(s)")
```

### Generate Certificate
```python
from certificate import create_certificate

# Generate certificate for verification
mapping = solutions[0]
cert_json = create_certificate(mapping, atlas, e8)

# Save to file
with open("certificate.json", "w") as f:
    f.write(cert_json)
```

### Verify Certificate
```python
from certificate import verify_certificate

# Verify independently
is_valid = verify_certificate(cert_json, verbose=True)
print(f"Certificate valid: {is_valid}")
```

### Canonicalize Multiple Solutions
```python
from canonicalization import canonicalize_embedding

# Select canonical representative
if len(solutions) > 1:
    canonical = canonicalize_embedding(solutions, atlas, e8)
    print("Selected canonical representative")
```

### Command Line Usage
```bash
# Run complete pipeline
python main.py

# The pipeline will:
# 1. Run verification tests
# 2. Search for embeddings with 48 sign classes
# 3. Canonicalize if multiple solutions found
# 4. Generate and verify certificate
# 5. Save certificate to tier_a_certificate.json
```

## Mathematical Background

The Tier-A embedding problem involves finding structure-preserving maps between two mathematical objects:

1. **Atlas Graph**: A specific 96-vertex graph with rich symmetry
2. **E8 Root System**: The 240 roots of the exceptional Lie group E8

The embedding must preserve:
- **Adjacency**: Adjacent vertices map to adjacent roots
- **Mirror Symmetry**: tau(v) maps to -f(v)
- **Unity Constraint**: Special vertices sum to zero vector

The existence of such embeddings with exactly 48 sign classes (using 96 of 120 root pairs) has deep connections to:
- Exceptional Lie groups and their representations
- Lattice theory and sphere packings
- String theory and quantum gravity

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m unittest tests.test_atlas

# Run with verbose output
python -m pytest tests/ -v
```

## Certificate Format

Generated certificates have the following structure:
```json
{
  "version": "1.0.0",
  "algorithm": "backtrack",
  "timestamp": "2024-01-01T00:00:00",
  "atlas_vertex_count": 96,
  "e8_root_count": 240,
  "atlas_labels": ["(0,0,0,-1,0,0)", ...],
  "roots": {
    "0": ["1", "1", "0", "0", "0", "0", "0", "0"],
    ...
  },
  "mapping": {
    "0": "112",
    ...
  },
  "unity_indices": [1, 4],
  "sign_classes_used": 48,
  "integer_roots_used": 58,
  "automorphism_group": "S4 on {e1,e2,e3,e6}"
}
```

## Performance

- **Search Time**: Typically < 1 second for first solution
- **Memory Usage**: ~10 MB for structures
- **Certificate Size**: ~100 KB JSON
- **Verification Time**: < 100 ms

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)
- Uses fractions.Fraction for exact rational arithmetic

## Implementation Notes

- **Exact Arithmetic**: All calculations use Fraction to avoid floating-point errors
- **Constraint Propagation**: Mirror pairing reduces search space significantly
- **Memory Efficient**: Avoids storing full adjacency matrices when possible
- **Independent Verification**: Certificates can be verified without search code
- **Modular Design**: Clean separation between Atlas, E8, and embedding logic

## References

- Atlas of Finite Groups (Conway et al.)
- E8 root system and exceptional Lie groups
- Sphere packings and lattices in dimension 8
- Unified theories and mathematical physics

## License

[Specify license]

## Contributing

[Contribution guidelines]