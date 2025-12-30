# Universal Numbers — A Stand‑Alone Mathematical Theory

## Abstract
Universal Numbers (UN) are scalar invariants extracted from structured data that remain unchanged under a prescribed symmetry group and admit efficiently verifiable certificates of correctness. They form a compositional algebra of summaries with rigorous equality semantics and a natural measure‑theoretic and categorical presentation. This paper develops the concept from first principles: axiomatization, algebraic and order structure, functorial measures on windows, witnessable computation and verification, homomorphism windows, stability, and representative case studies (strings, graphs, matrices, and probability distributions). Proofs are supplied where concise; otherwise, proof sketches and references to standard techniques in invariant theory and algorithmic verification are indicated. The treatment is self‑contained and independent of any external system.

---

## 1. Motivation and Overview
In many settings one seeks compact, trustworthy numerical summaries of complex states: a checksum of a word, a graph’s spectral moment, the determinant of a matrix, an entropy rate of a process. These scalars ought to be (i) **invariant** under irrelevant changes of presentation (renumberings, rotations, relabelings), (ii) **compositional** so that summaries of parts compose to a summary of the whole, and (iii) **witnessable**, meaning their claimed values carry short certificates that a verifier can check locally. *Universal Numbers* formalize the common core of such quantities and provide a single mathematical language to study them.

At a high level, a Universal Number is a function from states to scalars that (a) respects a chosen symmetry, (b) composes with respect to an algebra of transformations, and (c) supports certificates whose verification cost is proportional to the data actually used to compute it. This paper establishes the axioms and proves the basic structure theorems.

---

## 2. Formal Setup

### 2.1 States, Symmetries, and Presentations
- Let \(X\) be a nonempty set of **states**. Typical examples: finite strings over an alphabet, labeled graphs on a fixed vertex set, square matrices over a ring, or finite probability distributions.
- Let \(\Gamma\) be a group acting on \(X\). Its elements represent **presentation changes** (relabelings, index shifts, permutations, change of basis, etc.). For \(g\in\Gamma\) and \(x\in X\), write \(g\cdot x\) for the transformed state.
- Let \(K\) be a commutative semiring (or ring) of **scalars**. We assume \(1\neq 0\) and write addition \(+\) and multiplication \(\cdot\).

### 2.2 Programs and Locality
A **program** is a total map \(T:X\to X\). A set \(\mathcal T\subseteq X^X\) of programs is **admissible** if it is closed under composition and contains the identity. Intuitively, \(\mathcal T\) lists transformations regarded as legitimate operations on states (e.g., edits, linear updates, graph rewiring moves) that preserve a class of invariants of interest.

A **window** is a finite subset \(W\subseteq \mathrm{supp}(x)\) or any finite indexable part of \(x\) when the notion of locality is defined (strings: intervals; graphs: vertex/edge subsets; matrices: blocks). We denote by \(x\mid_W\) the restriction of \(x\) to \(W\).

### 2.3 Certificates and Verification
A **certificate scheme** for a function \(f:X\to K\) consists of a prover map \(\mathsf{wit}:X\to \mathcal W\) and a verifier \(\mathsf{ver}:X\times K\times \mathcal W\to\{\text{accept},\text{reject}\}\) such that for all \(x\in X\):
1. (Correctness) \(\mathsf{ver}(x, f(x), \mathsf{wit}(x))=\text{accept}\).
2. (Soundness) If \(\mathsf{ver}(x, y, w)=\text{accept}\), then \(y=f(x)\).
3. (Locality) The verification time is \(O(|W|+|w|)\) for some window \(W\) supporting the computation; in particular it scales with the touched data, not necessarily with the ambient size of \(x\).

The **zero‑cost property** is the special case where \(|w|=0\); then equality claims \(f(x)=y\) are decidable with no additional witness. Many algebraic invariants (e.g., determinant over exact fields) have zero‑cost certificates because their definitions uniquely fix the value given the input.

---

## 3. Universal Numbers (UN): Axioms

> **Definition 3.1 (Universal Number).** Fix \((X,\Gamma,\mathcal T,K)\) as above. A map \(u:X\to K\) is a **Universal Number** if it satisfies:
>
> **(U1) Symmetry invariance.** \(u(g\cdot x)=u(x)\) for all \(g\in\Gamma\), \(x\in X\).
>
> **(U2) Program conservation.** For all \(T\in\mathcal T\), there exists a (possibly partial) certificate map \(c_T\) such that \(u(Tx)=u(x)\) whenever \(c_T\) verifies that \(T\) is admissible on the support touched. (If all \(T\in\mathcal T\) preserve \(u\) by construction, take trivial certificates.)
>
> **(U3) Witnessability.** There is a certificate scheme \((\mathsf{wit},\mathsf{ver})\) with the locality property certifying the value \(u(x)\).
>
> **(U4) Compositionality.** For state unions along disjoint windows \(x=x_1\oplus x_2\) one has a binary operation \(\star\) on \(K\) such that \(u(x)=u(x_1)\star u(x_2)\). The operation \(\star\) coincides with the scalar addition or multiplication when those make sense and is associative, unital, and commutative whenever the gluing is.

The **class of UN**, denoted \(\mathrm{UN}(X;\Gamma,\mathcal T;K)\), is the set of all functions satisfying (U1)–(U4).

### 3.1 Immediate Consequences
- If \(K\) is a commutative semiring and (U4) uses \(+\) for disjoint gluing, the set of UN is closed under pointwise \(+\) and \(\cdot\). If witnesses compose accordingly (concatenation or convolution), \(\mathrm{UN}\) itself inherits a semiring structure (Theorem 4.1).
- If \(\Gamma\) acts transitively on a presentation class of \(x\), any function constant on the orbit is \(\Gamma\)‑invariant; UN sharpen this by demanding verifiability and composition laws.

---

## 4. Algebraic Structure

### 4.1 Semiring of Universal Numbers
> **Theorem 4.1.** Assume \(K\) is a commutative semiring. If \(u,v\in \mathrm{UN}(X;\Gamma,\mathcal T;K)\), then the pointwise sums and products
> \[ (u\oplus v)(x):=u(x)+v(x),\qquad (u\otimes v)(x):=u(x)\cdot v(x) \]
> are also Universal Numbers. With pointwise \(0(x):=0\) and \(1(x):=1\), the set \(\mathrm{UN}\) forms a commutative semiring \((\mathrm{UN},\oplus,\otimes,0,1)\).
>
> *Proof sketch.* (U1): invariance is preserved because \(\Gamma\) acts on arguments, not on values. (U2): if each of \(u,v\) is conserved by \(T\), so are \(u\pm v\) and \(uv\). (U3): combine certificates by concatenation; locality composes additively. (U4): distributivity of \(+\) and \(\cdot\) across disjoint unions gives the claim.

### 4.2 Order, Norms, and Metrics
When \(K\) carries an order (e.g., \(\mathbb R\) with \(\le\)), UN inherit pointwise order. For numerical robustness it is natural to equip UN with the **sup‑norm** \(\|u\|_\infty=\sup_{x\in X}|u(x)|\) on a chosen admissible subset of states, or with Lipschitz‑type seminorms relative to a metric on \(X\). Stability results appear in §8.

### 4.3 Congruences and Quotients
Let \(\sim\) be a congruence on \(K\). If equality of UN is only required modulo \(\sim\), we obtain **quotient‑UN** with the same algebraic laws; witness schemes need only certify values up to \(\sim\).

---

## 5. Universal Measures on Windows
Windows provide finite supports over which one can integrate local observables. Let \(\mathbf{Win}\) be the category whose objects are finite windows \(W\) and whose morphisms are inclusions and symmetry actions induced by \(\Gamma\).

> **Definition 5.1 (Universal Measure).** A **universal measure** is a functor \(\mu:\mathbf{Win}\to (K,\star)\) such that:
> 1. (Additivity) For disjoint windows, \(\mu(W_1\sqcup W_2)=\mu(W_1)\star\mu(W_2)\).
> 2. (Naturality) For any symmetry \(g\), \(\mu(gW)=\mu(W)\).
> 3. (Local witnessability) Each value \(\mu(W)\) admits a certificate verifiable from data restricted to \(W\).

Universal measures specialize UN to the window level. Typical examples:
- **Histogram measures** for strings and arrays (counts or moments of symbol classes),
- **Subgraph counts** for graphs (e.g., triangle counts; with inclusion–exclusion gluing),
- **Moment maps** for matrices (\(\mathrm{tr}(A^k)\) on blocks),
- **Entropy‑like measures** for distributions on windows.

---

## 6. Witnessable Computation

### 6.1 Transformations and Conservation
A transformation \(T\in\mathcal T\) is **conservative for \(u\)** if \(u(Tx)=u(x)\) for all \(x\). In practice, we track only the touched windows and require a local receipt demonstrating that \(T\) falls within the conservative fragment. The receipt may consist of boundary deltas, permutation descriptors, or algebraic identities (e.g., row/column operations for determinants).

> **Proposition 6.1 (Compositional verification).** If \(T_1,T_2\) are conservative for \(u\) with verifiable receipts on windows \(W_1,W_2\), then their composition is conservative on \(W_1\cup W_2\) with a certificate obtained by concatenation. Verification cost adds.

### 6.2 Complexity Classes (Window‑Parametric)
Let \(n=|W|\) be window size. Define:
- **UC (Universally‑Computable):** \(u\) is in UC if there exists an algorithm using only conservative primitives that computes \(u(x)\) in time polynomial in \(n\).
- **U\(^{\parallel}\)C:** as above with parallel depth polylogarithmic in \(n\) under associative gluing.
- **UV (Universally‑Verifiable):** \(u\) admits certificates verifiable in \(O(n+|w|)\) time.

Many UN live in \(\mathrm{UC}\cap\mathrm{UV}\); e.g., histogram sums, block‑diagonal determinants, spectral moments of sparse graphs via trace estimators, etc.

---

## 7. Homomorphism Windows
Invariants often become multiplicative or additive on special **windows** where the structure linearizes.

> **Definition 7.1 (Homomorphism window).** A window \(W\) is homomorphic for \(u\) if there exists an operation \(\boxplus\) on the restricted states such that for all disjoint \(W_1,W_2\subseteq W\),
> \[ u(x\mid_{W_1\cup W_2})=u(x\mid_{W_1})\,\circ\, u(x\mid_{W_2}), \]
> where \(\circ\) is \(+\) or \(\cdot\) in \(K\).

**Examples.**
- For string histograms, any window is homomorphic with \(\circ=+\).
- For determinants, block‑diagonal windows are multiplicative with \(\circ=\cdot\).
- For graph invariants, disjoint unions provide additivity of subgraph counts.

Homomorphism windows enable **streaming accumulation**: maintain a running summary by composing per‑window values; correctness follows from homomorphism.

---

## 8. Stability and Robustness
Let \(d_X\) be a metric on states (e.g., Hamming distance, edit distance, spectral norm) and \(|\!|\cdot|\!|\) a norm on \(K\).

> **Theorem 8.1 (Lipschitz stability).** If a universal measure \(\mu\) aggregates a Lipschitz local observable with constant \(L\), then \(|\!|\mu(x)-\mu(y)|\!|\le L\cdot d_X(x,y)\) for all states \(x,y\).

> **Theorem 8.2 (Certificate robustness).** Suppose verification checks local equalities that fail only when more than \(t\) positions are corrupted. Then any adversary changing fewer than \(t\) entries cannot make the verifier accept a wrong value.

These results justify using UN as tamper‑evident scalars or stable statistics.

---

## 9. Canonical Examples

### 9.1 Strings and Arrays
- **Setting:** \(X=\Sigma^N\), \(\Gamma=\langle\text{index permutations} \rangle\), \(K=\mathbb N\) or \(\mathbb R\).
- **UN examples:** symbol counts; \(k\)-gram counts; polynomial hash in a prime field (with certificate the seed and per‑block reductions); empirical entropy on windows.
- **Homomorphism windows:** disjoint blocks \(W_1,\dots,W_m\) yield additive counts; rolling updates cost \(O(1)\) per symbol for fixed \(k\).

### 9.2 Graphs
- **Setting:** labeled graphs on \([n]\) with \(\Gamma=\mathfrak S_n\) acting by relabeling.
- **UN examples:** degree histogram; number of triangles and small subgraphs; spectral moments \(\mathrm{tr}(A^k)\); size of automorphism group (via certificates from generators).
- **Certificates:** for subgraph counts—collections of embeddings with non‑overlap proofs on windows; for spectral moments—stochastic trace estimators with reproducible seeds.

### 9.3 Matrices
- **Setting:** square matrices over a commutative ring; \(\Gamma\) the group of change‑of‑basis transformations preserving similarity class.
- **UN examples:** determinant, rank, Smith normal form mod \(p\) (packaged as integers), traces \(\mathrm{tr}(A^k)\).
- **Certificates:** LU/PLU decompositions, kernel bases, and modular checks; verification costs near‑linear in touched entries.

### 9.4 Probability Distributions and Signals
- **Setting:** finite distributions or time series; symmetries include time shifts and rescalings.
- **UN examples:** empirical mean/variance; power spectral density at fixed frequencies; windowed entropies and divergences.
- **Homomorphism windows:** disjoint time windows or frequency bins; Parseval‑type identities yield additive energies.

---

## 10. Equality Semantics and Decision Procedures
Two claims about UN are common:
1. **Value equality:** \(u(x)=v(x)\) for two UN at the same state.
2. **Functional equality:** \(u=v\) as functions on \(X\).

For (1) the certificate scheme suffices. For (2) one may use algebraic identities, finite test suites closed under symmetries, or reduction to a basis of windows when homomorphism holds. In many discrete settings functional equality reduces to equality on a generating family of windows (e.g., all blocks of size \(\le k\)).

---

## 11. Category‑Theoretic Packaging
Let \(\mathbf{UN}\) be the category whose objects are tuples \((X,\Gamma,\mathcal T;K)\) and whose morphisms are equivariant, conservative maps \(F:X\to Y\) together with semiring homomorphisms \(\phi:K\to L\) making UN push forward and certificates transport. Functoriality of universal measures and closure under products follow formally. Limits and colimits exist under mild finiteness assumptions on windows and scalar semirings.

---

## 12. Design Patterns and Implementation Notes
- **Choose \(\Gamma\) first.** Invariances should reflect the domain’s true symmetries.
- **Select conservative primitives.** Design \(\mathcal T\) from operations known to preserve the target invariants, and make their receipts explicit.
- **Exploit homomorphism windows.** Seek decompositions into blocks where additivity or multiplicativity holds; this yields streaming verifiers.
- **Separate value from proof.** Keep numeric value and certificate distinct; both compose.
- **Use modular arithmetic judiciously.** When \(K\) is a field/ring, modular projections speed verification and provide robust checksums.

---

## 13. Case Study: A Universal Number for Labeled Graphs
Let \(X\) be simple graphs on vertex set \([n]\). Define
\[ u(G)\;:=\; (\text{triangle count})\;\oplus\; (\text{4‑cycle count})\;\oplus\; (\text{degree multiset hash}). \]
Here \(\oplus\) denotes packaging into a product semiring (e.g., tuples over \(\mathbb N\)). The value is invariant under relabelings. Certificates consist of explicit triangle and 4‑cycle listings with non‑overlap checks and a Merkle‑style accumulation of degrees. On disjoint unions of graphs, \(u\) adds component‑wise. Verifiers run in time linear in the number of listed subgraphs plus the degree evidence, independent of the ambient \(n\) except through touched edges.

---

## 14. Advanced Topics

### 14.1 Approximate Universal Numbers
Allow \(\varepsilon\)‑tolerant verifiers that accept values within \(\varepsilon\) of the true quantity; stability theorems bound how \(\varepsilon\) propagates under composition. Concentration inequalities give probabilistic certificates (e.g., Hutch++ for traces).

### 14.2 Universal Numbers over Noncommutative Scalars
When \(K\) is only a semiring (no negatives) or even noncommutative, much of the theory survives with ordered monoid techniques; care is required in defining \(\star\) on windows.

### 14.3 Learning and Identifiability
Given samples from \(X\), one can learn UN that maximize stability or discriminative power under \(\Gamma\). Identifiability asks when a family of UN separates orbits \(\Gamma\cdot x\); classical invariant theory provides criteria via generators and relations.

---

## 15. Open Problems
1. **Separating families.** For a given \((X,\Gamma)\), construct minimal UN families that separate orbits.
2. **Completeness under noise.** Characterize UN that are maximally robust to stochastic perturbations.
3. **Certificate succinctness.** Tight bounds comparing verification cost to computation cost for natural domains.
4. **Parallel limits.** Characterize when \(\mathrm{U}^{\parallel}\mathrm{C}=\mathrm{UC}\) under homomorphism window decompositions.
5. **Non‑classical scalars.** Extend UN to idempotent semirings (tropical, max‑plus) and quantify stability.

---

## 16. Summary
Universal Numbers provide a principled, domain‑agnostic language for invariant, compositional, and verifiable scalar summaries. They unify diverse practices—from algebraic invariants and statistics to streaming checksums—under a single axiomatic framework. The resulting semiring of UN, along with universal measures on windows and homomorphism windows for streaming composition, yields both theoretical clarity and practical algorithms.

---

## Appendix A: Proofs and Technical Lemmas

**Lemma A.1 (Closure under limits on finite domains).** If \(X\) is finite and \(K\) is complete under pointwise limits of monotone sequences, then pointwise infimum and supremum of chains of UN remain UN (with inherited witnesses).

*Proof.* Symmetry invariance and conservation are preserved under pointwise limits. Certificates pass to limits when witness verifiers are monotone (acceptance for a value implies acceptance for any value between lower/upper bounds when \(K\) is ordered appropriately). \(\square\)

**Lemma A.2 (Pushforward along equivariant maps).** If \(F:X\to Y\) is \(\Gamma\)‑equivariant and \(u\in\mathrm{UN}(Y;\Gamma,\mathcal T_Y;K)\), then \(u\circ F\in\mathrm{UN}(X;\Gamma,\mathcal T_X;K)\) provided \(F\) maps conservative programs to conservative programs and transports witnesses.

*Proof.* Immediate from definitions. \(\square\)

**Proposition A.3 (Disjoint additivity characterizes measures).** Suppose \(\mu\) on windows is invariant under \(\Gamma\) and additive on disjoint unions, with local certificates. Then \(\mu\) extends uniquely (up to natural isomorphism) to a universal measure functor.

*Proof.* Standard category‑theoretic extension via left Kan extension along inclusions. \(\square\)

---

## Appendix B: Worked Constructions

### B.1 Polynomial Hash as a Universal Number
Let \(X=\Sigma^N\), choose a large prime \(p\), and fix a base \(b\in \mathbb F_p\). Define \(u(x)=\sum_{i=0}^{N-1} x_i b^i\mod p\). Invariance holds under index permutations that preserve exponents up to a known transform (recorded in the certificate). Updates on windows use difference proofs. Homomorphism windows are contiguous blocks with multiplicative factors \(b^{|W|}\).

### B.2 Block Determinant
For block‑diagonal matrices \(A=\mathrm{diag}(A_1,\dots,A_m)\), define \(u(A)=\prod_i \det(A_i)\). Certificates are PLU decompositions per block. Additivity under direct sum and multiplicativity under block concatenation yield streaming verification.

### B.3 Graph Spectral Moment
Define \(u_k(G)=\mathrm{tr}(A^k)\). Certificates use Hutchinson‑type estimators with fixed seeds, plus variance bounds; or exact certificates by listing closed walks of length \(k\) without overlaps.

---

## Appendix C: Notation and Conventions
- \(\mathbb N,\mathbb Z,\mathbb Q,\mathbb R\) denote standard number systems.
- \(\mathbf{Win}\) is the window category; \(\sqcup\) is disjoint union.
- Certificates are denoted generically by \(w\); verifiers by \(\mathsf{ver}\).
- All actions are left actions; \(g\cdot x\) denotes the state obtained by applying \(g\) to \(x\).

