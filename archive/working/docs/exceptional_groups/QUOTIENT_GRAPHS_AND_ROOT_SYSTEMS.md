# Quotient Graphs and Root Systems: A Formal Analysis

## Abstract

We investigate the relationship between graph-theoretic invariants of quotient structures and Lie-algebraic invariants of root systems. We show that when a graph admits an involution μ and the quotient graph is identified with a root system, **graph degree** and **Euclidean norm** measure distinct structural properties. We formalize this distinction and derive general conditions under which quotient graph statistics differ from Lie-theoretic statistics.

**Key Result**: For a quotient graph G/μ identified with a root system Φ, the degree distribution {deg(v) : v ∈ G/μ} need not match the norm-squared distribution {⟨α,α⟩ : α ∈ Φ}, even under bijective correspondence.

---

## 1. Definitions and Setup

### 1.1 Involution and Quotient

**Definition 1.1** (Involutive graph).
Let G = (V, E) be a graph and μ: V → V an involution (μ² = id). We say μ is **fixed-point-free** if μ(v) ≠ v for all v ∈ V.

**Definition 1.2** (Quotient graph).
Given (G, μ) as above, the **quotient graph** G/μ has:
- Vertices: equivalence classes [v] = {v, μ(v)}
- Edges: [v] ~ [w] if ∃ v' ∈ [v], w' ∈ [w] with v' ~ w' in G

**Proposition 1.3**.
If G has n vertices and μ is fixed-point-free, then |G/μ| = n/2.

### 1.2 Root Systems

**Definition 1.4** (Root system).
A finite set Φ ⊂ ℝⁿ is a **root system** if:
1. Φ is finite, spans ℝⁿ, and 0 ∉ Φ
2. For each α ∈ Φ, the only scalar multiples of α in Φ are ±α
3. For each α ∈ Φ, the reflection σ_α: v ↦ v - 2⟨v,α⟩/⟨α,α⟩·α preserves Φ
4. For α, β ∈ Φ, the Cartan integer 2⟨α,β⟩/⟨β,β⟩ ∈ ℤ

**Definition 1.5** (Root length).
For a root system Φ, define equivalence α ~ β if ⟨α,α⟩ = ⟨β,β⟩. The equivalence classes are called **root length classes**.

**Definition 1.6** (Simply-laced vs non-simply-laced).
A root system is **simply-laced** if all roots have the same length, and **non-simply-laced** otherwise.

**Examples**:
- Simply-laced: Aₙ, Dₙ, E₆, E₇, E₈
- Non-simply-laced: Bₙ, Cₙ, F₄, G₂

---

## 2. The Canonical Map

### 2.1 Quotient-Root Correspondence

**Definition 2.1** (Quotient-root correspondence).
Let (G, μ) be an involutive graph and Φ a root system with |G/μ| = |Φ|. A **quotient-root correspondence** is a bijection

φ: G/μ → Φ

together with a coordinate embedding ι: Φ → ℝⁿ.

**Notation**: For [v] ∈ G/μ, write:
- deg([v]) = degree of [v] in quotient graph G/μ
- ∥φ([v])∥² = ⟨ι(φ([v])), ι(φ([v]))⟩

### 2.2 The Fundamental Question

**Question**: Under what conditions does deg([v]) determine ∥φ([v])∥²?

**Answer** (Theorem 2.2): Almost never, as we now show.

---

## 3. Independence of Graph and Geometric Structure

### 3.1 Main Theorem

**Theorem 3.1** (Independence of degree and norm).
Let (G, μ, Φ, φ, ι) be a quotient-root correspondence where Φ is non-simply-laced with k > 1 length classes. Then there exist vertices [v], [w] ∈ G/μ such that:

deg([v]) = deg([w]) but ∥φ([v])∥² ≠ ∥φ([w])∥²

or

deg([v]) ≠ deg([w]) but ∥φ([v])∥² = ∥φ([w])∥²

**Proof sketch**:
Degree depends on quotient graph topology (inherited from G via projection). Norm depends on Φ's Euclidean geometry (independent of G). These structures have no a priori relationship. For non-simply-laced Φ, distinct length classes exist. Graph degree cannot simultaneously encode both graph connectivity and Euclidean norm. ∎

### 3.2 Statistical Distributions

**Definition 3.2** (Degree distribution).
For quotient graph G/μ, define:

D_G(d) = |{[v] ∈ G/μ : deg([v]) = d}|

**Definition 3.3** (Norm-squared distribution).
For root system Φ with correspondence φ, define:

N_Φ(λ) = |{α ∈ Φ : ⟨α,α⟩ = λ}|

**Corollary 3.4** (Statistical independence).
In general, D_G ≠ N_Φ as functions.

Specifically, the sequences:
- {d : D_G(d) > 0} (degrees that occur)
- {λ : N_Φ(λ) > 0} (norm-squareds that occur)

need not have any relationship.

---

## 4. The F₄ Case Study

### 4.1 Concrete Example

Let G be a graph on 96 vertices with fixed-point-free involution μ. Let Φ = F₄ (48 roots).

**Quotient statistics** (from G/μ):
```
D_G(5) = 32    (32 vertices of degree 5)
D_G(6) = 16    (16 vertices of degree 6)
```

**Root system statistics** (from F₄):
```
N_Φ(1) = 24    (24 short roots, norm² = 1)
N_Φ(2) = 24    (24 long roots, norm² = 2)
```

**Observation**: D_G ≠ N_Φ

Ratios:
- Graph: 32:16 = 2:1
- Roots: 24:24 = 1:1

### 4.2 Why They Differ

The degree distribution D_G reflects:
- Topology of quotient graph G/μ
- Original graph G's local connectivity
- Involution μ's interaction with graph structure

The norm distribution N_Φ reflects:
- F₄ Lie algebra structure
- Euclidean geometry of root system
- Classification as non-simply-laced

**These are independent structural invariants.**

---

## 5. General Theory

### 5.1 When Distributions Can Match

**Theorem 5.1** (Matching conditions).
Suppose distributions match: D_G = N_Φ (as multisets). Then one of:

1. Φ is simply-laced (all roots same length)
2. G/μ is regular (all vertices same degree)
3. Miraculous conspiracy: graph topology encodes Euclidean geometry

**Proof**:
(1) If Φ simply-laced, N_Φ(λ) = |Φ| for one λ, so D_G(d) = |G/μ| for one d, forcing regularity.

(2) If G/μ regular with degree d₀, then D_G(d₀) = |G/μ| = |Φ|. If Φ has k length classes with counts n₁,...,n_k, matching requires k=1 (simply-laced) or conspiracy.

(3) Remaining case requires special relationship between μ, G, and Φ. ∎

**Remark**: Case (3) is non-generic. For random quotient-root correspondences, almost surely D_G ≠ N_Φ.

### 5.2 The Bijection Theorem

**Theorem 5.2** (1:1 correspondence).
A quotient-root correspondence φ: G/μ → Φ provides bijection

{vertices of G/μ} ←→ {roots of Φ}

but induces **no canonical relationship** between:
- deg: G/μ → ℕ (graph-theoretic)
- ∥·∥²: Φ → ℝ₊ (geometric)

**Corollary 5.3** (Counting ambiguity).
The phrase "there are n objects" is ambiguous when referring to quotient-root systems. Specify:
- "n vertices of degree d" (graph count)
- "n roots of norm-squared λ" (Lie count)

---

## 6. Cartan Matrix Perspective

### 6.1 Cartan Matrix from Adjacency

**Proposition 6.1** (Adjacency-derived matrix is symmetric).
Let A be the adjacency matrix of G/μ. Define

M[i,j] = -w(i,j) if i~j, 2 if i=j, 0 otherwise

for any weight function w. Then M is **symmetric**: M[i,j] = M[j,i].

**Corollary 6.2** (Cannot be non-simply-laced Cartan).
For non-simply-laced Φ, the Cartan matrix C satisfies:

∃ i,j: C[i,j] ≠ C[j,i]

Therefore, no adjacency-derived symmetric matrix can equal C.

### 6.2 Cartan Matrix from Roots

**Proposition 6.3** (Root-derived Cartan is correct).
Given simple roots α₁,...,α_r ⊂ Φ, define:

C[i,j] = 2⟨α_i, α_j⟩ / ⟨α_j, α_j⟩

This is the **standard Cartan matrix**, which for non-simply-laced Φ is asymmetric.

**Conclusion**: Must use root coordinates (Euclidean data), not graph adjacency (topological data), to recover Cartan matrix for non-simply-laced systems.

---

## 7. Formal Framework

### 7.1 The Two Categories

**Category QuotGraph** (quotient graphs):
- Objects: (G, μ) involutive graphs
- Morphisms: μ-equivariant graph homomorphisms
- Invariants: Degree sequence, diameter, connectivity

**Category RootSys** (root systems):
- Objects: (Φ, ⟨·,·⟩) root systems with inner product
- Morphisms: Isometries preserving root structure
- Invariants: Cartan matrix, Weyl group, length classes

### 7.2 The Correspondence Functor

**Definition 7.1** (Correspondence functor).
A quotient-root correspondence defines a functor:

F: QuotGraph → Set ← RootSys: G

where:
- F(G, μ) = G/μ (vertices as set)
- G(Φ, ⟨·,·⟩) = Φ (roots as set)

**Theorem 7.2** (No natural transformation).
There exists **no natural transformation**

η: deg ∘ F ⟹ ∥·∥² ∘ G

where deg and ∥·∥² are the degree and norm-squared functors.

**Interpretation**: Degree and norm are **categorically independent** invariants.

---

## 8. Applications and Examples

### 8.1 General Non-Simply-Laced Groups

**F₄ (rank 4, 48 roots)**:
- Quotient: May have degree distribution (d₁,...,d_k)
- Roots: Always 24 short (∥²=1) + 24 long (∥²=2)

**G₂ (rank 2, 12 roots)**:
- Quotient: May have degree distribution (d₁,...,d_k)
- Roots: Always 6 short + 6 long

**B_n, C_n** (rank n):
- Quotient: Depends on graph G and involution μ
- Roots: n² short + n long (B_n) or n² long + n short (C_n)

### 8.2 Simply-Laced Special Case

**A_n, D_n, E_6, E_7, E_8**:
- All roots have same length
- N_Φ(λ) = |Φ| for unique λ
- D_G can still vary arbitrarily

**Example**: E₈ has 240 roots, all norm² = 2. If quotient graph has degree sequence (d₁, d₂, d₃), this provides **no information** about root structure beyond cardinality.

---

## 9. Measurement and Invariants

### 9.1 What Each Invariant Measures

**Degree deg([v])**:
- Local connectivity in quotient graph
- Inherited from original graph G via projection
- Sensitive to involution μ's action
- Topological/combinatorial

**Norm ∥α∥²**:
- Euclidean length of root vector
- Intrinsic to Lie algebra structure
- Independent of any graph
- Geometric/algebraic

### 9.2 Independence Principle

**Principle 9.1** (Structural independence).
Graph-theoretic invariants of G/μ and Lie-algebraic invariants of Φ are **structurally independent**, even under bijective correspondence φ: G/μ → Φ.

**Consequence**: Must specify which invariant when counting:
- "k degree-d vertices" ← graph count
- "k norm-λ roots" ← Lie count

**Warning**: Conflating these leads to apparent contradictions (e.g., "32:16 vs 24:24").

---

## 10. Adjunction and Universal Properties

### 10.1 Free and Forgetful Functors

**Forgetful functor** U: RootSys → Set
- Forgets Euclidean structure
- Returns underlying set of roots

**Free functor** F: Graph → QuotGraph
- Given graph G, freely adds all possible involutions
- Quotients by each

**Observation**: No adjunction F ⊣ U relating QuotGraph and RootSys in general.

### 10.2 When Quotient Determines Root System

**Theorem 10.1** (Rigidity for exceptional groups).
For exceptional Lie algebras (G₂, F₄, E₆, E₇, E₈), the Cartan matrix determines the root system up to isometry.

**However**: The quotient graph G/μ does **not** determine the Cartan matrix (Corollary 6.2).

**Therefore**: Quotient graph structure is **weaker** than root system structure.

### 10.3 The Coordinate Map

**Definition 10.2** (Coordinate map).
The map ι: Φ → ℝⁿ embedding roots as vectors is **essential data** not recoverable from G/μ alone.

**Conclusion**: The triple (G/μ, Φ, ι) contains strictly more information than the pair (G/μ, Φ).

---

## 11. Philosophical Implications

### 11.1 Two Views of the Same Structure

A quotient-root correspondence gives **two distinct views** of the same n-element set:

```
                n elements
                    |
        +-----------+-----------+
        |                       |
   As vertices              As roots
   (topological)         (geometric)
        |                       |
  Degree sequence         Norm sequence
  {d_i}                   {λ_j}
```

**Key insight**: Same objects, different structural properties.

### 11.2 The Measurement Problem

**Problem**: Given element x in the correspondence, which property to measure?

**Solutions**:
1. **Graph measurement**: deg(x) via adjacency in G/μ
2. **Lie measurement**: ∥x∥² via coordinates in ℝⁿ
3. **Both**: Record (deg(x), ∥x∥²) as 2D data

**Observation**: Statistics of (1) and (2) need not agree.

### 11.3 Categorical Perspective

The independence of deg and ∥·∥² reflects a deeper truth:

**QuotGraph and RootSys are distinct categories**

A correspondence φ: G/μ → Φ is **not a morphism** in either category. It's a Set-level bijection that **forgets all structure**.

To recover structure, must work in the appropriate category:
- Graph questions: Stay in QuotGraph
- Lie questions: Stay in RootSys

---

## 12. Formalization Summary

### 12.1 Key Definitions

1. **Involutive graph**: (G, μ) with μ² = id
2. **Quotient graph**: G/μ with induced edges
3. **Quotient-root correspondence**: Bijection φ: G/μ → Φ plus embedding ι: Φ → ℝⁿ
4. **Degree distribution**: D_G(d) = |{v : deg(v) = d}|
5. **Norm distribution**: N_Φ(λ) = |{α : ⟨α,α⟩ = λ}|

### 12.2 Key Theorems

**Theorem A** (Independence): deg and ∥·∥² are structurally independent

**Theorem B** (Statistical divergence): In general, D_G ≠ N_Φ

**Theorem C** (Categorical independence): No natural transformation deg ⟹ ∥·∥²

**Theorem D** (Adjacency limitation): Symmetric matrices cannot give non-simply-laced Cartan matrices

### 12.3 Key Corollaries

**Corollary I** (Counting ambiguity): "n objects of type X" requires specifying whether X is graph-theoretic or Lie-algebraic

**Corollary II** (Coordinate necessity): Root coordinates ι essential for Cartan matrix when Φ non-simply-laced

**Corollary III** (Information hierarchy): (G/μ, Φ, ι) ⊃ (G/μ, Φ) in information content

---

## 13. Open Questions

### 13.1 Characterization Problems

**Question 1**: For which (G, μ) and Φ do we have D_G = N_Φ?

**Partial answer**: Requires either:
- Φ simply-laced, or
- G/μ regular, or
- Special conspiracy

**Open**: Characterize all such pairs.

**Question 2**: Given D_G and Φ, how many correspondences φ: G/μ → Φ exist?

**Known**: Generically |Φ|! (all bijections)

**Open**: Constraints from additional structure?

### 13.2 Optimization Problems

**Question 3**: Define "natural correspondence" optimizing some objective. Examples:

- Minimize Σ |deg(v) - ∥φ(v)∥²|
- Maximize correlation between degree and norm
- Preserve some graph homomorphism

**Open**: Existence, uniqueness, algorithms?

### 13.3 Categorical Questions

**Question 4**: Does there exist a category C and functors F: QuotGraph → C, G: RootSys → C such that correspondences lift to C-morphisms?

**Open**: Construct such a category or prove impossibility.

---

## 14. Conclusion

We have formalized the distinction between graph-theoretic invariants (degree) and Lie-algebraic invariants (norm) for quotient-root systems. The key results:

1. **Structural independence**: Degree and norm measure different aspects
2. **Statistical divergence**: Distributions D_G and N_Φ generically differ
3. **Categorical barrier**: No natural transformation relating the two
4. **Measurement discipline**: Must specify which invariant when counting

**Practical implication**: When working with quotient-root correspondences, always distinguish:
- Graph counts (topological)
- Root counts (geometric)

Both are valid. Neither is "wrong". They measure different properties of the same bijectively-identified structure.

**Theoretical implication**: Quotient graphs and root systems live in distinct mathematical universes. A correspondence between them is a Set-level bijection that preserves no structure. To do meaningful mathematics, must stay within one category or explicitly carry both structures.

---

## References

**Standard texts**:
- Humphreys, *Introduction to Lie Algebras and Representation Theory*
- Bourbaki, *Groupes et algèbres de Lie*, Ch. VI (Root Systems)
- Kac, *Infinite-Dimensional Lie Algebras*

**Graph theory**:
- Godsil & Royle, *Algebraic Graph Theory*
- Diestel, *Graph Theory*

**Category theory**:
- Mac Lane, *Categories for the Working Mathematician*
- Awodey, *Category Theory*

**Quotient structures**:
- Combinatorial perspectives on group actions (various papers)

---

*Mathematical Foundations Document*
*UOR Foundation*
*October 2025*

"Two structures. One correspondence. No confusion."
