# CLAUDE.md - Working Directory

## Atlas Implementation Principles

**CRITICAL: Mathematical Rigor Required**

### PROHIBITED in ALL Atlas work:

- ❌ **NumPy** - NO floating point arrays, NO approximate arithmetic
- ❌ **Floats** - NO `float`, NO `np.float64`, NO decimal approximations
- ❌ **Numerical tolerance comparisons** - NO `abs(a - b) < eps`
- ❌ **Approximate methods** - NO gradient descent, NO numerical optimization
- ❌ **Probabilistic/statistical methods** - NO random sampling for structure discovery

### REQUIRED:

- ✅ **Exact rational arithmetic** - `fractions.Fraction` for all non-integer values
- ✅ **Exact integer arithmetic** - `int` for all integer operations
- ✅ **Half-integer quantization** - For coordinates in ℤ ∪ ½ℤ, use `Fraction(n, 2)`
- ✅ **Symbolic computation** - If needed, use `sympy` with exact types
- ✅ **Algebraic methods** - First-principles construction from group theory
- ✅ **Exact equality** - Use `==` for exact comparison, NO tolerances

## Why This Matters

**Atlas is a mathematical structure with EXACT properties:**
- 96 resonance classes (exact count)
- Cartan matrices with exact integer/half-integer entries
- Root systems with exact norm ratios (e.g., √2/√1 for F₄)
- Weyl group orders (exact: 12 for G₂, 1152 for F₄)

**Floating point arithmetic DESTROYS structure:**
- 0.99999... ≠ 1 breaks unity class constraint
- Accumulated rounding errors invalidate algebraic identities
- Cannot verify exact root relationships
- Cannot distinguish true symmetries from numerical artifacts

## Implementation Approach

**Build from first principles:**
1. Start with exact algebraic definitions
2. Construct structures combinatorially
3. Verify properties by exact computation
4. Let structures EMERGE from categorical framework
5. Do NOT impose external Lie theory assumptions

**Do NOT:**
1. Approximate and then "clean up"
2. Optimize numerically and hope for integer results
3. Use tolerance-based convergence
4. Treat Atlas as a numerical optimization problem

## Workspace Organization

**CRITICAL: This workspace is intentional and organized. Do NOT create random content.**

### Directory Structure

All code in `/workspaces/Hologram/working/` follows a strict organization:

```
working/
├── action_framework/      # Action framework (exact arithmetic)
├── exceptional_groups/    # Atlas embedding (first principles)
├── tier_a_embedding/      # E₈ foundation
├── docs/                  # ALL documentation (organized by topic)
├── context/               # Supporting theory papers
└── README.md              # Main documentation
```

### PROHIBITED - Do NOT Create:

- ❌ **Random markdown files** in root directory
- ❌ **Temporary status reports** (use existing docs/ structure)
- ❌ **Duplicate documentation** (organize in docs/)
- ❌ **Ad-hoc scripts** outside proper modules
- ❌ **Test files** in random locations (use proper test/ directories)
- ❌ **Certificates** outside `docs/exceptional_groups/certificates/`
- ❌ **Analysis results** outside proper analysis/ directories
- ❌ **"scratch" or "temp" directories**
- ❌ **Reports, analyses, or documentation** - NEVER create unless explicitly requested by user

### REQUIRED - File Creation Rules:

- ✅ **Implementation code** goes in proper module directories only
- ✅ **Documentation** goes in `docs/` with clear organization
- ✅ **Tests** go in module-specific `tests/` directories
- ✅ **Certificates** go in `docs/exceptional_groups/certificates/`
- ✅ **New modules** require planning and proper structure
- ✅ **Every file** has a clear purpose and location

### Valid Module Locations:

**Implementation**:
- `action_framework/` - Exact arithmetic, action functionals
- `exceptional_groups/` - First-principles Atlas embedding
- `tier_a_embedding/` - E₈ embedding foundation

**Documentation** (ALL docs go here):
- `docs/action_framework/` - Action framework reports
- `docs/exceptional_groups/` - Atlas embedding reports
- `docs/reports/` - Status and progress reports

### Before Creating ANY File:

1. **Ask**: Does this belong in an existing directory?
2. **Check**: Is there already similar content?
3. **Organize**: Use proper docs/ structure for documentation
4. **Clean**: Remove when obsolete (don't leave clutter)

### Workspace Hygiene:

- **No `__pycache__`** - Covered by `.gitignore`
- **No `.pyc` files** - Covered by `.gitignore`
- **No duplicate files** - Consolidate or delete
- **No temporary content** - Use proper locations or delete when done
- **Regular cleanup** - Keep workspace intentional and minimal
