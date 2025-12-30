# Deterministic Pseudorandom Invariants (DPI): A Complete Formalization

## 0. Preliminaries & Notation
- **Working arena.** A small category \(\mathcal C\) of objects you care about (models, curves, fibers, etc.).  
- **Lawful morphisms.** A wide subcategory \(\mathcal C_{\!\mathrm{law}}\subseteq\mathcal C\) whose arrows commute with your baseline projectors (resonance \(R96\), schedule \(C_{768}\), holography \(\Phi\)) and are *budget–nonincreasing* (RL/BHIC sense).  
- **Boundary algebra.** A finite (or locally finite) \(*\)-algebra \(\mathcal A\) acting on boundary states with a designated family of **central idempotents (projectors)** \(\{\Pi_L\}_{L\in\{R96,C_{768},\Phi\}}\) that pairwise commute.  
- **The 12 288 torus.** \(\mathbb T:=\mathbb Z/48\mathbb Z \times \mathbb Z/256\mathbb Z\).  
- **Probability metrics.** For a finite set \(S\): total-variation \(\|\mu-\nu\|_{\mathrm{TV}}=\tfrac12\sum_{s\in S}|\mu(s)-\nu(s)|\); χ² and discrepancy are as standard.  
- **Fourier on \(\mathbb Z/m\).** \(\widehat{\mu}(k)=\mathbb E[e^{2\pi i k X/m}]\) for \(X\sim\mu\); nontrivial modes \(k\neq0\).

---

## 1. DPI — Core Definition
Let \(I\) be a finite index set. For each \(i\in I\) fix a finite **target** \(S_i\) (set, ring, or group) and a **binning projector** schema on the boundary: for \(s\in S_i\), \(\Pi^{(i)}_s\) projects onto the sector \(\{X: F_i(X)=s\}\).

### Definition 1.1 (DPI)
A **Deterministic Pseudorandom Invariant (DPI)** on \(\mathcal C\) with targets \(\{S_i\}_{i\in I}\) is a family of maps
\[
F=\{F_i:\mathrm{Ob}(\mathcal C)\to S_i\}_{i\in I}
\]
together with **receipts** \(\rho\) certifying:

**(L) Lawfulness & determinism.**
1. *(Isomorphism–invariance)* \(F_i\) is invariant under isomorphisms in \(\mathcal C\).
2. *(Budget–0 computability)* Each \(F_i\) is computable by algebraic/finite arithmetic with budget \(\le 0\).
3. *(Commutation)* For all \(i\) and \(s\in S_i\), \(\Pi^{(i)}_s\) is a **central idempotent** in \(\mathcal A\) and \([\Pi^{(i)}_s,\Pi_L]=0\) for \(L\in\{R96,C_{768},\Phi\}\).
4. *(Nonincreasing)* Applying \(\Pi^{(i)}_s\) does not increase budget.

**(M) Mixer property (under some lawful generator).**  
There exists a **lawful generator** \(G:\mathbb N\to\mathrm{Ob}(\mathcal C)\) (e.g., a polynomial family, lawful walk, or PolyCat pipeline) such that for each \(i\) there are constants \(c_i,\varepsilon_{ij}\ge0\) with:
- *(Equidistribution)* For samples \(X_n:=F_i(G(n))\in S_i\),
  \[
  \mathrm{disc}(X_{1..N})\ \le\ c_i\,|S_i|^{-1/2}+o_{N}(1).
  \]
- *(Weak coupling)* For \(i\neq j\), the joint law of \((F_i,F_j)\) on \(G(1..N)\) satisfies
  \[
  \mathrm{MI}(F_i;F_j)\ \le\ \varepsilon_{ij}\quad\text{(or a TV bound to the product law).}
  \]

**(C) Compression to the 12 288 torus.**  
Choose **disjoint palettes** \(I_{48},I_{256}\subseteq I\) and **fixed encoders**
\[
\mathrm{enc}_{48,i}:S_i\to \mathbb Z/48,\qquad \mathrm{enc}_{256,i}:S_i\to \mathbb Z/256
\]
together with fixed public **linear compressors**
\[
H_{48}(x)=\Big(\sum_{i\in I_{48}} w_i\,\mathrm{enc}_{48,i}(x_i)\Big)\bmod 48,\quad
H_{256}(y)=\Big(\sum_{i\in I_{256}} v_i\,\mathrm{enc}_{256,i}(y_i)\Big)\bmod 256,
\]
with \(w_i\) (resp. \(v_i\)) coprime to \(48\) (resp. \(256\)). Define the **compressed DPI**
\[
\mathbf F:\mathrm{Ob}(\mathcal C)\to\mathbb T,\quad
\mathbf F(X)=(H_{48}(F_{I_{48}}(X)),\,H_{256}(F_{I_{256}}(X))).
\]
The induced projectors \(\Pi^{(\mathbf F)}_{a,b}\) are central idempotents commuting with \(\Pi_L\), hence a **budget–nonincreasing refinement** of the boundary algebra.

---

## 2. DPI Calculus (Closure & Construction)
- **Pullback.** For any lawful functor \(U:\mathcal D\to\mathcal C\), the pullback \(U^\*F_i:=F_i\circ U\) is a DPI on \(\mathcal D\) with inherited receipts.
- **Products & linear images.** If \(F,G\) are DPIs, so are \((F,G)\) and \(H\circ(F,G)\) for any fixed linear map \(H\) into a finite target. Discrepancy/TV bounds add subadditively; MI bounds sum.
- **CRT aggregation.** For residue targets \(S_i=\mathbb Z/p_i\), any CRT fold \(\prod S_i\to \mathbb Z/M\) preserves lawfulness and multiplies Fourier magnitudes (cf. §4).
- **Markov pushforward.** If a lawful dynamic on \(\mathcal C\) pushes \(F_i\) to a **finite Markov chain** with spectral gap \(\gamma>0\), then for any start, TV distance to uniform decays \(\le (1-\gamma)^t\).
- **Idempotent refinement.** Each \(F_i\) yields commuting central idempotents \(\{\Pi^{(i)}_s\}\). Adding them preserves hub initiality on finite objects (Karoubi argument).

---

## 3. Universality Classes (by Asymptotic Receipts)

### 3.1 Square-root (Weil/character-sum) class
**Definition.** A DPI index \(i\) belongs to the square-root class if, under a lawful **polynomial-type** generator, there exists a nonconstant rational map \(f_i\) over \(\mathbb F_p\) and an encoder \(\mathrm{enc}_{m}\) such that the character sums obey
\[
\max_{\psi\neq 1}\ \Big|\sum_{n\in I}\psi\!\big(\mathrm{enc}_{m}(F_i(G(n)))\big)\Big|\ \le\ C_i\sqrt{p}
\]
for all intervals \(I\subseteq\{1,\dots,N\}\) and all nontrivial characters \(\psi\) of \(\mathbb Z/m\).  
**Receipt.** Discrepancy \(\ll C_i\,m^{1/2}p^{-1/2}\) (plus finite-sample \(\tilde O(N^{-1/2})\)).

### 3.2 Expander (spectral-gap/graph-walk) class
**Definition.** Under a lawful step operator \(T\) on \(S_i\), the associated transition matrix is irreducible, aperiodic, with spectral gap \(\gamma_i>0\).  
**Receipt.** For any start, \(\|\mu_t-\mathrm{Unif}\|_{\mathrm{TV}}\le (1-\gamma_i)^t\); serial correlations decay \(\le (1-\gamma_i)^t\).

### 3.3 Renewal-mixing class
**Definition.** There exist deterministic *renewal times* \(0=\tau_0<\tau_1<\dots\) such that blocks between \(\tau_k\) are i.i.d.-like with tail \(\mathbb P(\tau_{k+1}-\tau_k>t)\le c\,e^{-t/\tau}\) (or polynomial).  
**Receipt.** φ-/ψ-mixing coefficients with exponential (or specified polynomial) decay; CLT-type stabilization of bin frequencies.

### 3.4 Hyperbolic / Lasota–Yorke class
**Definition.** A piecewise-expanding transfer operator \(\mathcal L\) acting on BV/Hölder with spectral gap \(\delta>0\); the DPI reads out a coarse alphabet from the invariant density.  
**Receipt.** Exponential correlation decay with rate \(<1-\delta\); TV to stationary law decays at the same rate.

### 3.5 Sieve/Chebotarev class (indexing over primes/Frobenius)
**Definition.** The index set itself ranges over primes or Frobenius classes and \(F\) records a conjugacy statistic; equidistribution guaranteed by Chebotarev/sieve with square-root error.  
**Receipt.** Discrepancy over primes up to \(X\) is \(\tilde O(X^{-1/2})\) (under standard hypotheses); low MI across disjoint progressions.

> **Tagging.** Each DPI comes with a *class label* and an *asymptotic receipt tuple*, e.g. \((C_i p^{-1/2})\) or \((\gamma_i)\), guiding palette design.

---

## 4. Minimal Palettes to Hit Near-Uniform on 48 and 256

### 4.1 Fourier criterion
Let \(m\in\{48,256\}\). For each candidate index \(i\) and its encoded variable \(Y_i:=\mathrm{enc}_{m,i}(F_i)\in\mathbb Z/m\), define the **spectral flatness**
\[
\rho_i := \max_{1\le k\le m-1} \big|\widehat{\mu_i}(k)\big|,\quad \mu_i=\mathsf{Law}(Y_i).
\]
For a **linear compressor** \(H_m\) and a **palette** \(I_m\), the nontrivial Fourier modes multiply:
\[
\big|\widehat{\mu_{\sum_{i\in I_m}Y_i}}(k)\big| \le \prod_{i\in I_m}\big|\widehat{\mu_i}(k)\big|\ \le\ \prod_{i\in I_m}\rho_i.
\]
Hence the **TV bound**
\[
\Big\|\sum_{i\in I_m} Y_i - \mathrm{Unif}(\mathbb Z/m)\Big\|_{\mathrm{TV}}
\ \le\ \tfrac12\sum_{k=1}^{m-1}\prod_{i\in I_m}\rho_i
\ \le\ \tfrac12(m-1)\prod_{i\in I_m}\rho_i.
\]

### 4.2 Minimality rule (ε–target)
Given tolerance \(\varepsilon_m>0\),
\[
\boxed{\ \text{Choose the smallest }I_m\text{ such that}\quad
\prod_{i\in I_m}\rho_i\ \le\ \frac{2\varepsilon_m}{m-1}\ .\ }
\]
- **Independence caveat.** If pairwise MI between \(Y_i\) and \(Y_j\) is \(\le \eta\), replace \(\rho_i\) by \(\rho_i^\star:=\rho_i+c\sqrt{\eta}\) (Pinsker-type slack); the same rule applies with \(\rho_i^\star\).

### 4.3 Greedy palette selection (practical)
**Input.** Candidate pool \(\mathcal P_m\), tolerance \(\varepsilon_m\), MI threshold \(\eta\).  
**Loop.**
1) Start \(I_m\gets\varnothing\); current product \(P\gets1\).  
2) For each \(i\in\mathcal P_m\setminus I_m\), estimate \(\rho_i\) (and MI to the current sum). Form \(\rho_i^\star\).  
3) Add \(i^\*\) minimizing \(P\cdot \rho_{i^\*}^\star\); update \(P\gets P\cdot \rho_{i^\*}^\star\).  
4) Stop when \(P\le 2\varepsilon_m/(m-1)\).

**Default targets.** \(\varepsilon_{48}=\varepsilon_{256}=0.02\). In square-root class, *one* good index often suffices; add a second as robustness/MI insurance.

---

## 5. DPI → 12 288: Structural Theorems

### Theorem 5.1 (Commutation)
For any DPI \(F\) and compressors \(H_{48},H_{256}\) as in §1, the induced projectors \(\Pi^{(\mathbf F)}_{a,b}\) are central idempotents commuting with \(\{\Pi_{R96},\Pi_{C_{768}},\Pi_{\Phi}\}\).

*Proof sketch.* Each \(\Pi^{(i)}_s\) is central and commutes with base projectors. \(\Pi^{(\mathbf F)}_{a,b}\) is a finite Boolean combination of \(\Pi^{(i)}_s\) over \(i\in I_{48}\cup I_{256}\); centrality and commutation are preserved under sums and products of commuting idempotents. ∎

### Theorem 5.2 (Budget descent)
Applying \(\Pi^{(\mathbf F)}_{a,b}\) is budget–nonincreasing.

*Reason.* Budgets are subadditive and each \(\Pi^{(i)}_s\) is nonincreasing; compositions/sums remain nonincreasing. ∎

### Theorem 5.3 (Hub initiality preserved on finite objects)
If \(\Omega\) is an initial hub among finite coherent objects for the base projectors, then the **DPI-decorated hub** \(\Omega^{(F)}\) (with \(\Pi^{(\mathbf F)}\) installed) is initial among finite objects carrying the same refinement.

*Reason.* Adding central idempotents is a Karoubi-envelope refinement; the universal arrow \(\mathrm{Hom}(\mathcal A e,\cdot)\cong e(\cdot)\) still yields uniqueness for finite modules. ∎

### Theorem 5.4 (CRT product near-uniformity)
Let \(I_{48},I_{256}\) satisfy the ε–rule in §4.2 with tolerances \(\varepsilon_{48},\varepsilon_{256}\). If the two palettes are disjoint and cross-MI between registers is \(\le \eta\), then the **torus law**
\[
\mathbf F=(H_{48},H_{256})\in \mathbb Z/48\times\mathbb Z/256
\]
is within \(\varepsilon_{48}+\varepsilon_{256}+O(\sqrt{\eta})\) of uniform in TV.

---

## 6. Canonical DPI Instances

### 6.1 J–DPI (elliptic \(j\)-invariant residues)
- **Targets.** \(S_p=\mathbb F_p\) for good primes \(p\ge 5\).  
- **Map.** \(F_p(E)=j(E)\bmod p\) via the standard rational formula.  
- **Classes.**  
  - *Square-root class* under polynomial generators \(E_n\) with nonconstant rational \(j(n)\bmod p\).  
  - *Expander class* under \(\ell\)-isogeny walks on (super)singular \(j\)-graphs (spectral gap \(>0\)).  
- **Compression.** Use \(\mathrm{enc}_{48}(x)=x\bmod 48\), \(\mathrm{enc}_{256}(x)=x\bmod 256\) (or tabulated encodings), then linear \(H_{48},H_{256}\).

### 6.2 Λ–DPI (Legendre parameter)
- **Targets.** \(S_p=\mathbb F_p\setminus\{0,1\}\) modulo the natural \(S_3\) action (optional).  
- **Class.** Square-root under polynomial generators; good independence vs J–DPI in practice.

### 6.3 Trace–DPI (Frobenius traces)
- **Targets.** \(S_{m,r}=\mathbb Z/r\) via \(F_{m,r}(E)=a_m(E)\bmod r\).  
- **Class.** Sieve/Chebotarev under prime-index scanning; expander under suitable dynamics.

---

## 7. Receipts & Acceptance Checklist

**Static (per DPI):**
- [ ] Isomorphism invariance proof/witness.  
- [ ] Budget–0 computation witness for each \(F_i\).  
- [ ] \([\Pi^{(i)}_s,\Pi_{R96}]=[\Pi^{(i)}_s,\Pi_{C_{768}}]=[\Pi^{(i)}_s,\Pi_{\Phi}]=0\).  
- [ ] Extension–restriction: adding/removing indices \(i\) gives an adjunction \(E\dashv R\).

**Dynamic (per generator \(G\)):**
- [ ] χ²/TV per index \(i\) on one schedule (\(N=768\) or \(kN\)) below target.  
- [ ] Serial correlations (lags 1–5) ≈ 0; if expander, fit \((1-\gamma)^t\).  
- [ ] Cross-index MI ≤ \(\varepsilon\) (and cross-register MI ≤ \(\eta\)).  
- [ ] Avalanche: single lawful micro-perturbation flips ≈ 50% compressed bits.  
- [ ] Budget descent maintained during measurement.

**Compression (12 288):**
- [ ] Palettes \(I_{48},I_{256}\) satisfy \(\prod \rho_i^\star \le 2\varepsilon_m/(m-1)\).  
- [ ] \(\mathbf F\) TV ≤ \(\varepsilon_{48}+\varepsilon_{256}+O(\sqrt{\eta})\).  
- [ ] Projector \(\Pi^{(\mathbf F)}\) idempotent, central, budget–nonincreasing.

---

## 8. Minimal-Palette Existence (Guidelines)

### Proposition 8.1 (Square-root sources)
Suppose each \(Y_i\in\mathbb Z/m\) arises from a square-root class DPI with \(\rho_i\le C_i\,m^{1/2}p_i^{-1/2}\). Then any palette with
\[
\sum_{i\in I_m} \log p_i\ \ge\ 2\log\!\Big(\frac{(m-1)C}{2\varepsilon_m}\Big) \quad \text{where}\ C=\max_i C_i
\]
meets the ε–rule. In particular, **one** well–chosen prime often suffices for \(m\in\{48,256\}\); a **second** gives robust slack against MI and model drift.

### Proposition 8.2 (Expander sources)
If each candidate \(i\) is driven by an expander with gap \(\gamma_i\), then for any start
\[
\big\|\sum_{i\in I_m}Y_i^{(t)} - \mathrm{Unif}\big\|_{\mathrm{TV}} \ \le\ \tfrac12(m-1)\prod_{i\in I_m}(1-\gamma_i)^t.
\]
Thus *time* (steps \(t\)) trades off with *palette size*. For a fixed \(t\), select the few indices with largest gaps.

---

## 9. SMM / RL / Hub Packaging (Schema Outline)

- **Model: `DPI.Index`**  
  `name, target(S_i), encoder(enc_m), lawfulness_certificate, budget_certificate`.

- **Model: `DPI.Compressor`**  
  `modulus(m), palette(I_m), weights({w_i}), epsilon_target`.

- **Interface: `IApplyDPI`**  
  `(object) -> {i: value in S_i}` with receipts attached.

- **Interface: `ICompressTo12288`**  
  `(values) -> (z48,z256)` and projector installation.

- **Proof-obligations:**  
  - Centrality/commutation;  
  - Budget descent;  
  - Mixer receipts (χ²/TV/MI/serial/avalanche);  
  - ε–rule satisfied for each register;  
  - Adjunction witnesses for extension–restriction (add/remove indices).

---

## 10. Worked Pattern: J–DPI on 12 288 (One-Screen Summary)
- **Indices.** Good primes \(p\ge5\).  
- **Map.** \(F_p(E)=j(E)\bmod p\).  
- **Class tags.** Square-root (poly generator) and Expander (isogeny walk).  
- **Encoders.** Mod \(48\), mod \(256\); optional tabulation for better flatness.  
- **Palettes (tiny defaults).**  
  - \(I_{48}=\{97\}\) (often enough); add \(149\) if needed.  
  - \(I_{256}=\{257\}\); add \(263\) if needed.  
- **Receipts.** χ²/TV per register ≤ 2%; MI ≤ \(10^{-3}\); commutation & budget verified; torus TV ≤ 4% (tightens with second index or more steps).

---

## 11. Optional: Generator Abstractions

- **Polynomial generator.** \(G(n)\) outputs coefficients from fixed integer polynomials; lawfulness trivial; square-root class receipts attach via character sums.  
- **Isogeny-walk generator.** \(G(n+1)\) is a lawful \(\ell\)-step from \(G(n)\); expander receipts via spectral gap.  
- **Pipeline generator.** Any PolyCat pipeline of lawful morphisms; DPI lawfulness is functorial; mixer receipts compose subadditively.

---

## 12. What “Complete” Guarantees You Get
1. **Turnkey integration.** Any DPI satisfying (L),(M),(C) can be flipped on as a projector refinement; hubs and adjunctions remain sound.  
2. **Quantitative mixing.** Universality class tags dictate how you reach uniformity: add indices (square-root) or take steps (expander).  
3. **Minimality with proof.** Fourier ε–rule gives a certifiable stopping condition for palette size.  
4. **Auditability.** Every step has receipts: determinism/budget, commutation, mixing statistics, and ε–witness for 48 & 256.

