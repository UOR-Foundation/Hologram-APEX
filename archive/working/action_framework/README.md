# Action Framework - Status and Usage

## Current State

### ✓ VALID (Exact Arithmetic)

Core data structures with exact rational arithmetic:
- `core/exact_arithmetic.py` - ComplexFraction class
- `core/atlas_structure.py` - AtlasField (96-dimensional, exact)
- `core/quotient_field.py` - F4QuotientField, G2QuotientField (exact)

### ✗ INVALID (Numerical Optimization - Violates Atlas Principles)

The following components use numpy and numerical methods, **violating CLAUDE.md**:
- `sectors/lagrangians.py` - Generic numerical Lagrangians
- `sectors/atlas_lagrangians.py` - Atlas sectors with numpy
- `sectors/f4_root_action.py` - F₄ action with numpy
- `solvers/minimizer.py` - Gradient descent (numerical optimization)
- All `tests/test_*.py` - Numerical tests with tolerances

**Why invalid:**
- Numerical gradient descent is approximate (floats, tolerances)
- Cannot distinguish true algebraic critical points from numerical artifacts
- Accumulated rounding errors destroy exact root relationships
- Violates requirement for exact rational arithmetic

## Correct Usage of Action Hypothesis

The user's hypothesis: *"unified action is the **function signature** of exceptional group embeddings"*

This means:
1. Action **characterizes** the structure, doesn't optimize to find it
2. F₄ roots (from first-principles construction) **are** a critical point
3. Different groups have different sector weight "signatures"
4. Verification is exact: ∂S/∂ψ = 0 (no tolerances)

## Path Forward

### What To Do:

1. **Use existing exact F₄/G₂ roots** from `exceptional_groups/` directory
2. **Define action algebraically** with rational sector weights
3. **Compute ∂S/∂ψ symbolically** at root configurations
4. **Verify exact stationarity**: ∂S/∂ψ == ComplexFraction.zero()

### What NOT To Do:

1. ❌ Gradient descent to "find" roots
2. ❌ Numerical optimization with tolerances
3. ❌ Float-based energy minimization
4. ❌ Approximate and "clean up" to integers

## Refactoring Plan

To make sectors usable for exact verification:
1. Remove numpy from sector classes
2. Compute energies and gradients exactly (Fraction arithmetic)
3. Use for verification: `action.gradient(known_f4_roots) == zero`
4. Delete numerical minimization code

## Files to Remove/Deprecate

- `solvers/minimizer.py` - Delete (numerical optimization invalid)
- `tests/test_g2_emergence.py` - Delete (numerical)
- `tests/test_f4_roots.py` - Delete (numerical)
- `tests/test_laplacian_correctness.py` - Delete (numerical)
- `core/boundary_torus.py` - Deprecated (wrong dimensional space)

## Files to Keep and Refactor

- `core/exact_arithmetic.py` - ✓ Keep as-is
- `core/atlas_structure.py` - ✓ Keep as-is (now exact)
- `core/quotient_field.py` - ✓ Keep as-is (now exact)
- `sectors/atlas_lagrangians.py` - Refactor to exact arithmetic
- `sectors/f4_root_action.py` - Refactor to exact arithmetic
