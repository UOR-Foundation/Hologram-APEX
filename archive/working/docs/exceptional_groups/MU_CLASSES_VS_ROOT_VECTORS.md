# μ-Classes vs Root Vectors: Understanding the 32:16 vs 24:24 Distinction

## Executive Summary

There are **two valid ways to count "roots" in the F₄ structure from Atlas**:
1. **μ-classes** (quotient equivalence classes): 32 degree-5, 16 degree-6
2. **Root vectors** (F₄ Lie algebra): 24 short (norm²=1), 24 long (norm²=2)

Both counts are correct - they measure different aspects of the same structure.

---

## The Two Counting Methods

### Method 1: μ-Classes (Graph-Theoretic)

**What**: Equivalence classes under mirror involution μ on Atlas

**Process**:
1. Start with 96 Atlas vertices
2. Apply mirror involution μ (fixed-point-free involution)
3. Get 48 equivalence classes (quotient)
4. Each class has a **degree** in the quotient graph

**Count**:
- 32 classes with degree-5 in quotient graph
- 16 classes with degree-6 in quotient graph
- Total: 48 μ-classes

**Source**: `f4/sign_class_analysis.py`
```python
Sign class structure:
  Total classes: 48
  Degree-5 vertices: 32    # μ-classes
  Degree-6 vertices: 16    # μ-classes
  Ratio: 32:16 = 2:1
```

### Method 2: Root Vectors (Lie-Algebraic)

**What**: Actual root system vectors in ℝ⁴ for F₄ Lie algebra

**Process**:
1. Generate standard F₄ root system in 4D
2. Assign coordinates to each of the 48 μ-classes
3. Compute **norm² = ⟨root, root⟩** for each vector

**Count**:
- 24 vectors with norm² = 1 (SHORT roots)
- 24 vectors with norm² = 2 (LONG roots)
- Total: 48 root vectors

**Source**: `f4/page_correspondence.py`
```python
# Type 1: ±eᵢ (8 SHORT roots, norm² = 1)
# Type 2: ±eᵢ ± eⱼ (24 LONG roots, norm² = 2)
# Type 3: ½(±1,±1,±1,±1) (16 SHORT roots, norm² = 1)
# Total: 8 + 24 + 16 = 48
# Short: 8 + 16 = 24
# Long: 24
```

---

## Why They Differ

### The 1:1 Correspondence

Each **μ-class** corresponds to exactly one **root vector**:

| μ-class index | Root vector coordinates | Norm² | μ-class degree | Root length |
|---------------|------------------------|-------|----------------|-------------|
| 0 | (1, 0, 0, 0) | 1 | 5 | SHORT |
| 8 | (1, 1, 0, 0) | 2 | 6 | LONG |
| 32 | (1/2, 1/2, 1/2, 1/2) | 1 | 5 | SHORT |
| ... | ... | ... | ... | ... |

But the **degree of a μ-class** ≠ **norm² of its root vector**.

### Why Degrees Don't Match Norms

**μ-class degree** = number of adjacent classes in quotient graph (topology)

**Root norm²** = ⟨root, root⟩ in Euclidean space (geometry)

These are measuring **different properties**:
- Degree measures quotient graph connectivity
- Norm measures Euclidean length

### The Statistical Mismatch

**μ-class degrees**:
- 32 classes have degree 5
- 16 classes have degree 6
- Ratio: 32:16 = 2:1

**Root norms**:
- 24 vectors have norm² = 1 (short)
- 24 vectors have norm² = 2 (long)
- Ratio: 24:24 = 1:1

**These ratios differ because graph degree ≠ vector norm.**

---

## Concrete Example

Let's trace a specific root through both systems:

### Example 1: Root index 0

**As μ-class**:
```
Index: 0
Degree in quotient: 5
Adjacent to: [2, 3, 5, 11, 15] (5 neighbors)
```

**As root vector**:
```
Coordinates: (1, 0, 0, 0) = e₁
Norm²: 1² + 0² + 0² + 0² = 1
Classification: SHORT root
```

**Observation**: Degree 5 ≠ Norm² 1. They're measuring different things.

### Example 2: Root index 8

**As μ-class**:
```
Index: 8
Degree in quotient: 6
Adjacent to: [1, 7, 12, 19, 28, 38] (6 neighbors)
```

**As root vector**:
```
Coordinates: (1, 1, 0, 0) = e₁ + e₂
Norm²: 1² + 1² + 0² + 0² = 2
Classification: LONG root
```

**Observation**: Degree 6, Norm² 2. Coincidentally both "6" and "2", but not related mathematically.

---

## Why Both Are Correct

### μ-Classes Are Correct Because:

1. **They emerge naturally from Atlas**: 96 vertices → mirror quotient → 48 classes
2. **They have well-defined graph structure**: Adjacency inherited from quotient
3. **Degree counts are exact**: 32 degree-5 + 16 degree-6 verified computationally
4. **This is the quotient topology**: 32:16 pattern reflects mirror structure

### Root Vectors Are Correct Because:

1. **They match standard F₄**: 24 short + 24 long is the canonical F₄ root system
2. **Norms are exact**: Computed via ⟨α, α⟩ using fractions.Fraction
3. **They generate correct Cartan matrix**: Asymmetric F₄ matrix verified
4. **This is the Lie algebra structure**: 24:24 is F₄ as abstract Lie algebra

---

## Which Count to Use When

### Use μ-class counts (32:16) when discussing:

- **Quotient graph structure**: "The quotient has 32 degree-5 vertices"
- **Mirror involution**: "μ creates 48 equivalence classes with 32:16 degree split"
- **Atlas topology**: "The 96→48 quotient has non-uniform degree distribution"
- **Page structure**: "48 pages with varying connectivity"

### Use root vector counts (24:24) when discussing:

- **F₄ Lie algebra**: "F₄ has 24 short roots and 24 long roots"
- **Cartan matrix**: "The Cartan matrix distinguishes 24 short from 24 long"
- **Root system**: "Standard F₄ root system in ℝ⁴"
- **Weyl group**: "Weyl(F₄) acts on the 24+24 root system"

---

## The Relationship

### How are 32:16 and 24:24 related?

They are **two views of the same 48-element structure**:

```
                   48 elements
                       |
        +--------------+--------------+
        |                             |
   As μ-classes                  As root vectors
   (graph-theoretic)             (Lie-algebraic)
        |                             |
   Degree count                  Norm² count
   32 deg-5                      24 norm²=1
   16 deg-6                      24 norm²=2
```

**The same 48 objects, counted two different ways.**

### Why doesn't degree predict norm²?

**Degree** depends on:
- Quotient graph adjacency
- Mirror structure
- Topological connectivity

**Norm²** depends on:
- Root coordinates in ℝ⁴
- Euclidean geometry
- F₄ Lie algebra structure

These are **independent properties** - knowing degree doesn't tell you norm, and vice versa.

---

## Common Misconceptions

### ❌ "Degree-5 means short root"

**Wrong**. Some degree-5 μ-classes correspond to long roots.

Example: A root with coordinates (1,1,0,0) has:
- Norm² = 2 (LONG)
- Might have degree 5 or 6 in quotient (independent)

### ❌ "We have 32 short roots"

**Ambiguous**. Clarify:
- "32 degree-5 μ-classes" ✓
- "32 short root vectors" ✗ (should be 24)

### ❌ "The 32:16 ratio is wrong"

**Wrong**. The 32:16 μ-class degree pattern is correct for the quotient. The 24:24 root vector pattern is correct for F₄. Both are valid.

### ✓ "There are 48 μ-classes with 32:16 degree split, and when we assign F₄ coordinates to them, we get 24:24 short:long split"

**Correct**. Clear distinction between quotient topology and Lie algebra geometry.

---

## Technical Details

### How we compute each

**μ-class degree**:
```python
degree_sequence = f4_structure.degree_sequence
degree_counts = {5: 32, 6: 16}  # From graph analysis
```

**Root vector norm²**:
```python
from fractions import Fraction

for idx, coords in root_coords.items():
    norm_sq = sum(Fraction(c) * Fraction(c) for c in coords)
    if norm_sq == Fraction(1):
        short_roots.add(idx)  # 24 total
    elif norm_sq == Fraction(2):
        long_roots.add(idx)   # 24 total
```

### Verification

Both counts verified in `f4/page_correspondence.py`:

```
F₄ PAGE-ROOT CORRESPONDENCE
  Pages: 48
  Mirror pairs: 24
  Long roots: 24      # From norm² = 2
  Short roots: 24     # From norm² = 1
  Page adjacencies: 128

Bijection verification:
  ✓ C1_all_pages_paired
  ✓ C2_24_long       # Checks norm²-based count
  ✓ C2_24_short      # Checks norm²-based count
```

---

## Conclusion

**The key insight**: 32:16 and 24:24 are **both correct** because they measure different aspects:

- **32:16** = graph degree distribution of μ-classes (quotient topology)
- **24:24** = Euclidean norm² distribution of root vectors (F₄ Lie algebra)

**When to use each**:
- Talk about **μ-quotient structure**: Use 32:16
- Talk about **F₄ root system**: Use 24:24

**The relationship**: 1:1 correspondence between 48 μ-classes and 48 root vectors, but degree ≠ norm².

**Both are Atlas truth**: The 32:16 pattern reveals the quotient structure. The 24:24 pattern reveals the F₄ Lie algebra. Together they show how F₄ emerges from Atlas.

---

*Technical Documentation*
*UOR Foundation*
*October 2025*
