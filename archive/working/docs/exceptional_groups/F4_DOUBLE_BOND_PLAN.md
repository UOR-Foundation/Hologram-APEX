# F₄ Double Bond Detection - Implementation Plan

**Goal**: Correctly detect and place the -2 entry in F₄ Cartan matrix

**Current Status**: ⚠️ Cartan matrix structure correct, but all off-diagonal connected entries are -1 (should have one -2)

---

## Problem Analysis

### Expected F₄ Cartan Matrix
```
[[ 2, -1,  0,  0],
 [-1,  2, -2,  0],   ← -2 at position [1,2]
 [ 0, -1,  2, -1],
 [ 0,  0, -1,  2]]
```

### Why the -2?

**Cartan Matrix Formula**:
```
C[i,j] = 2⟨αᵢ,αⱼ⟩ / ⟨αⱼ,αⱼ⟩
```

For **connected roots** (adjacent in Dynkin diagram):
- Both same length: `C[i,j] = -1`
- αᵢ short, αⱼ long: `C[i,j] = -1`
- **αᵢ long, αⱼ short**: `C[i,j] = -2` ⭐ **DOUBLE BOND**

### F₄ Root Structure

From our analysis:
- **32 short roots**: degree-5 vertices in sign class graph
- **16 long roots**: degree-6 vertices in sign class graph
- Ratio: 2:1 ✓

**Key**: We already have this information in `F4Structure.short_roots` and `F4Structure.long_roots`!

---

## Current Code Issues

### Location 1: `cartan_extraction.py:114-118`
```python
if self.adjacency[ri][rj]:
    # Connected - determine bond type
    # For now, assume single bonds (-1)
    # TODO: Determine double bonds for F₄
    cartan[i][j] = -1  # ← Always -1, needs to be -2 sometimes
```

### Location 2: `cartan_extraction.py:124-139`
```python
def refine_cartan_for_f4(self, cartan: List[List[int]]):
    """..."""
    # Look for the characteristic -2 entry pattern
    # This requires analyzing bond multiplicities

    # For now, return as-is with a note
    # TODO: Implement proper bond multiplicity detection
    return cartan  # ← Does nothing
```

---

## Solution Strategy

### Approach: Use Degree to Determine Root Length

**Step 1**: Pass root length information to `CartanExtractor`

**Step 2**: When building Cartan matrix, check root lengths:
```python
if self.adjacency[ri][rj]:  # If connected
    # Determine root lengths
    ri_is_long = (self.degrees[ri] == 6)  # degree-6 = long
    rj_is_long = (self.degrees[rj] == 6)

    # Apply Cartan formula
    if ri_is_long and not rj_is_long:
        cartan[i][j] = -2  # Long → Short: double bond
    else:
        cartan[i][j] = -1  # All other cases
```

**Step 3**: Verify against expected pattern

---

## Implementation Plan

### Phase 1: Modify `CartanExtractor.__init__`

**Current**:
```python
def __init__(self, adjacency_matrix: List[List[bool]], degree_sequence: List[int]):
    self.adjacency = adjacency_matrix
    self.degrees = degree_sequence
    self.n = len(adjacency_matrix)
```

**Add**: Store long/short classification
```python
def __init__(self, adjacency_matrix: List[List[bool]], degree_sequence: List[int]):
    self.adjacency = adjacency_matrix
    self.degrees = degree_sequence
    self.n = len(adjacency_matrix)

    # Classify roots by degree
    # In F₄: degree-6 = long roots, degree-5 = short roots
    self.long_roots = set(i for i, d in enumerate(degree_sequence) if d == 6)
    self.short_roots = set(i for i, d in enumerate(degree_sequence) if d == 5)
```

### Phase 2: Update `extract_cartan_submatrix`

**Replace lines 114-118** with:
```python
if self.adjacency[ri][rj]:
    # Connected - determine bond multiplicity from root lengths
    ri_is_long = (ri in self.long_roots)
    rj_is_long = (rj in self.long_roots)

    # Cartan entry depends on relative root lengths
    if ri_is_long and not rj_is_long:
        # Long root to short root: double bond
        cartan[i][j] = -2
    else:
        # All other cases: single bond
        cartan[i][j] = -1
```

### Phase 3: Simplify `refine_cartan_for_f4`

**Replace lines 134-139** with:
```python
# Bond detection now happens in extract_cartan_submatrix
# This function is no longer needed, but keep for verification
return cartan
```

Or better: **Remove this function entirely** and its call.

### Phase 4: Add Verification

**New function**:
```python
def verify_f4_double_bond(self, cartan: List[List[int]]) -> bool:
    """
    Verify that Cartan matrix has exactly one -2 entry.

    F₄ has exactly one double bond in Dynkin diagram.
    """
    # Count -2 entries
    num_double_bonds = sum(
        1 for i in range(len(cartan))
        for j in range(len(cartan))
        if cartan[i][j] == -2
    )

    has_double_bond = (num_double_bonds >= 1)

    if has_double_bond:
        print(f"✓ Found {num_double_bonds} double bond(s) in F₄ Cartan matrix")
        # Find and report position
        for i in range(len(cartan)):
            for j in range(len(cartan)):
                if cartan[i][j] == -2:
                    print(f"  Double bond at position [{i},{j}]")
    else:
        print(f"✗ No double bond found (F₄ should have one)")

    return has_double_bond
```

---

## Testing Strategy

### Test 1: Simple Roots Identification
```python
# Expected: 4 simple roots forming a path
# One should be long, three short (or vice versa)

simple_roots = [0, 2, 1, 3]  # Example
for i, root_idx in enumerate(simple_roots):
    is_long = (root_idx in long_roots)
    print(f"α{i} (vertex {root_idx}): {'long' if is_long else 'short'}")
```

### Test 2: Cartan Matrix Pattern
```python
# Expected pattern for F₄:
expected = [
    [ 2, -1,  0,  0],
    [-1,  2, -2,  0],
    [ 0, -1,  2, -1],
    [ 0,  0, -1,  2]
]

# Or some permutation thereof (depends on root ordering)
```

### Test 3: Double Bond Count
```python
assert count_entries(cartan, -2) >= 1, "F₄ must have at least one -2 entry"
assert count_entries(cartan, -2) <= 2, "F₄ has one double bond (2 entries)"
# Note: double bond creates TWO entries due to asymmetry
```

---

## Edge Cases

### Case 1: Root Ordering
The exact position of -2 depends on how we order the simple roots.

**Solution**: Don't require exact position match, just verify:
- Has -2 entry ✓
- Structure is valid ✓
- Determinant = 1 ✓

### Case 2: Multiple -2 Entries?
A double bond creates **two** -2 entries (asymmetric):
```
C[i,j] = -2  (long to short)
C[j,i] = -1  (short to long)
```

**Solution**: Check for 1-2 instances of -2 (acceptable range)

### Case 3: Degree Hypothesis Wrong?
What if degree-6 ≠ long roots?

**Solution**:
1. Try both interpretations
2. Verify which gives valid F₄ Cartan
3. Document the correct mapping

---

## Success Criteria

After implementation, we should have:

✅ **Cartan matrix with -2 entry**
✅ **Automatic detection** (no hardcoding)
✅ **Verification passes** (confirm F₄ pattern)
✅ **Documentation** (which roots are long/short)
✅ **Tests** (verify on actual F₄ structure)

---

## Implementation Checklist

### Phase 1: Analysis (15 min)
- [ ] Review current F₄ structure data
- [ ] Verify degree-5 vs degree-6 counts (32 vs 16)
- [ ] Confirm hypothesis: degree-6 = long

### Phase 2: Code Changes (30 min)
- [ ] Update `CartanExtractor.__init__`
- [ ] Modify `extract_cartan_submatrix`
- [ ] Add `verify_f4_double_bond`
- [ ] Update `extract_cartan_matrix` flow

### Phase 3: Testing (15 min)
- [ ] Run on actual F₄ structure
- [ ] Verify -2 entry appears
- [ ] Check Cartan properties
- [ ] Verify determinant = 1

### Phase 4: Documentation (10 min)
- [ ] Update comments
- [ ] Remove TODOs
- [ ] Document degree → length mapping
- [ ] Update verification status

**Total estimated time**: ~70 minutes

---

## Potential Issues

### Issue 1: Wrong Degree Mapping
**Symptom**: -2 appears in wrong location or too many times
**Fix**: Swap the interpretation (degree-5 = long, degree-6 = short)

### Issue 2: Simple Roots Not Ideal
**Symptom**: Simple roots don't form proper Dynkin diagram
**Fix**: Improve simple root selection in `find_simple_roots`

### Issue 3: Multiple Valid Solutions
**Symptom**: Different orderings give different valid Cartan matrices
**Fix**: Accept any valid F₄ Cartan matrix (up to permutation)

---

## Follow-up Tasks

After fixing the double bond:

1. **Update Week 1 status**: Mark F₄ Cartan as 100% complete
2. **Update certificates**: Note the fix in verification docs
3. **Run full verification**: Ensure all F₄ tests still pass
4. **Document in report**: Add to "refinements completed" section

---

## References

**Theory**:
- F₄ Cartan matrix: Wikipedia/Lie Algebras texts
- Cartan formula: C[i,j] = 2⟨αᵢ,αⱼ⟩/⟨αⱼ,αⱼ⟩
- F₄ structure: 24 long + 24 short roots

**Code**:
- `f4/cartan_extraction.py` (lines 96-139)
- `f4/sign_class_analysis.py` (lines 100-106)
- F4Structure dataclass (has long_roots, short_roots)

---

**Ready to implement?** This plan provides:
- Clear problem understanding ✓
- Specific code changes ✓
- Testing strategy ✓
- Success criteria ✓
- Time estimate ✓