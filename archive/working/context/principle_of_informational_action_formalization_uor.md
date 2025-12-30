# Principle of Informational Action — Formalization (UOR)

**Version:** 1.0 (Working, Normative where flagged)  
**Object:** A finite, gauge‑invariant action principle for information on the 12,288‑cell boundary lattice, with sectorized Lagrangian, discrete Euler–Lagrange equations, Noether‑style receipts, and acceptance predicates.

---

## 0 · Overview
**Thesis.** Lawful informational processes are exactly the stationary‑action trajectories of a sectorized Lagrangian on the 12,288‑site lattice. Correctness ≙ geodesic (minimal action under constraints). Truth ≙ conservation (budget‑0 with symmetry receipts). Efficiency emerges as the same minimum.

**Scope.** Fixes the index space, configuration space, paths, total action, sector densities, stationarity, discrete Noether charges, acceptance predicate, and solver sketch. Profiles at the end specify which sectors are REQUIRED/RECOMMENDED for different deployments.

---

## 1 · Mathematical Objects & Notation

### 1.1 Index torus and configuration space
- **Index torus:** \(\mathbb T := (\mathbb Z/48\mathbb Z)\times(\mathbb Z/256\mathbb Z)\), \(|\mathbb T|=12{,}288\). Elements are \((p,b)\) with componentwise modular addition.
- **Fields / configurations:** Choose a target set \(K\) (default \(\mathbb R\) or \(\mathbb C\)). A **boundary field** is \(\psi:\mathbb T\to K\).  
  **Configuration space:** \(\mathcal C := K^{\mathbb T}.\)
- **Finite differences:**  
  \(\Delta_p\psi(p,b):=\psi(p{+}1,b)-\psi(p,b),\quad\Delta_b\psi(p,b):=\psi(p,b{+}1)-\psi(p,b).\)  
  **Discrete gradient:** \(\nabla\psi:=(\Delta_p\psi,\Delta_b\psi).\)  
  **Discrete Laplacian:** \(\Delta := \nabla^{\!\top}\nabla\).

### 1.2 Paths and time discretization
A **path** (process) is a finite sequence \(\gamma = \{\psi_t\}_{t=0}^{N}\subset\mathcal C\) with step size \(h>0\). Write the forward time difference as \(\Delta_t\psi_t := (\psi_{t+1}-\psi_t)/h\). Boundary data (problem specification): \(\psi_0 = \psi^{\mathrm{in}},\ \psi_N=\psi^{\mathrm{out}}\).

### 1.3 Symmetry groups (gauge)
- **Translations:** \(G_T\cong\mathbb Z/48\times\mathbb Z/256\) acts by index shifts.  
- **Cyclic schedule:** a rotation \(R\) of order 768 acts on a designated cyclic coordinate.  
- **Boundary automorphisms:** a finite subgroup \(\mathrm{Aut}_\partial\) of lattice automorphisms.  
- **Full gauge:** \(\mathcal G := G_T \times \langle R\rangle \times \mathrm{Aut}_\partial\).  
An observable or functional is **gauge‑invariant** if constant on \(\mathcal G\)-orbits.

### 1.4 Resonance label and budgets (interfaces)
- **Resonance label:** a pointwise map \(R:K\to \mathbb Z_{96}\) ("R96" classes).  
- **Budgets:** resource values drawn from a commutative quantale or semiring \(B\). Budget‑0 denotes conservation/truth.

---

## 2 · The Action Functional

### 2.1 Total action over a path
For weights collected in a profile \(\theta\), define
\[
S[\gamma;\theta] \;=\; \sum_{t=0}^{N-1} h\,\mathcal L\!\left(\psi_t,\,\Delta_t\psi_t,\,\nabla\psi_t\,;\,\theta\right),
\]
where the **total Lagrangian density** \(\mathcal L\) factorizes into **sectors**:
\[
\mathcal L\;=\;\mathcal L_{\text{geom}}+\mathcal L_{\text{res}}+\mathcal L_{\text{sched}}+\mathcal L_{\text{cons}}+\mathcal L_{\Phi}+\mathcal L_{\text{gauge}}+\mathcal L_{\text{ent}}+\mathcal L_{\text{wit}}+\mathcal L_{\text{spec}}.
\]
All sectors are explicit and can be toggled hard (constraint) or soft (finite weight).

### 2.2 Sector definitions (normative forms)

**(G) Geometric/Kinetic sector**  
Smoothness and geodesic‑like behavior on \(\mathbb T\):
\[
\mathcal L_{\text{geom}}(\psi)=\tfrac{\kappa_p}{2}\,(\Delta_p\psi)^2+\tfrac{\kappa_b}{2}\,(\Delta_b\psi)^2+\tfrac{\alpha}{2}\,\|\Delta_t\psi\|^2.
\]

**(R) Resonance/Identity sector**  
Adherence to the 96‑class label alphabet:
\[
\mathcal L_{\text{res}}(\psi)=\lambda_R\,\mathrm{dist}_{96}\big(R\psi,\mathcal A_{96}\big)^2,
\]
with a chosen metric \(d\) defining \(\mathrm{dist}_{96}\), applied pointwise on \(\mathbb T\).

**(S) Schedule‑invariance sector (length 768)**  
\[
\mathcal L_{\text{sched}}(\psi)=\kappa_\sigma\,\|\psi-R\psi\|^2.\quad\text{(hard: enforce }\psi=R\psi\text{)}
\]

**(C) Conservation sector (row/column)**  
Lagrange multipliers \(\{\Lambda_p\}_{p},\{\Nu_b\}_{b}\) enforce
\[
\sum_b\psi(p,b)=0\quad \forall p,\qquad \sum_p\psi(p,b)=0\quad \forall b.
\]
Contribution to \(\mathcal L\):
\[
\mathcal L_{\text{cons}}(\psi,\Lambda,\Nu)=\sum_p\Lambda_p\Big(\sum_b\psi(p,b)\Big)+\sum_b\Nu_b\Big(\sum_p\psi(p,b)\Big).
\]

**(Φ) Round‑trip sector (bulk↔boundary)**  
With an equivariant endomorphism \(\Phi:\mathcal C\to\mathcal C\):
\[
\mathcal L_{\Phi}(\psi)=\eta\,\|\psi-\Phi(\psi)\|^2.\quad\text{(hard: fixed points of }\Phi\text{)}
\]

**(Gₐ) Discrete gauge sector**  
For generators \(\{g_j\}\) of \(\mathcal G\):
\[
\mathcal L_{\text{gauge}}(\psi)=\gamma\sum_j\|\psi-g_j\!\cdot\!\psi\|^2,\quad\text{or solve on the quotient }\mathcal C/\!/\mathcal G.
\]

**(H) Entropy/Elegance sector**  
Let \(H(\psi)\) be a page/cycle‑aware informational entropy (e.g., resonance entropy over R96 labels). Then
\[
\mathcal L_{\text{ent}}(\psi)=\lambda_H\,H(\psi).\quad\text{(lower entropy → lower action)}
\]

**(W) Resource/Witness sector**  
Let \(W(\psi_t\!\to\!\psi_{t+1})\in B\) be the per‑step emitted budget. Penalize any non‑zero emission:
\[
\mathcal L_{\text{wit}}(\psi_t,\psi_{t+1})=\mu\,\mathrm{val}\big(W(\psi_t\!\to\!\psi_{t+1})\big),\quad \sum_t W=0\ \text{for fully conserved runs}.
\]

**(M) Spectral‑moment fitting (optional)**  
For a target vector of summary numbers (e.g., class histogram moments), impose quadratic fits.

---

## 3 · Stationarity and Discrete Euler–Lagrange (E–L)

**Stationarity principle.** A path \(\gamma\) is lawful iff \(\delta S[\gamma]=0\) under all compactly supported variations with fixed endpoints. In discrete form, for each interior \(t=1,\dots,N{-}1\):
\[
\boxed{\;\partial_{\psi}\mathcal L\;-
\Delta_t^{\!\top}\,\partial_{\Delta_t\psi}\mathcal L\;-
\sum_{i\in\{p,b\}}\Delta_i^{\!\top}\,\partial_{\Delta_i\psi}\mathcal L\;=\;0\;}
\]
**Sector contributions (examples):**
- Geom: \(-\alpha\,\Delta_t^{\!\top}\Delta_t\psi-\kappa_p\Delta_p^{\!*}\Delta_p\psi-\kappa_b\Delta_b^{\!*}\Delta_b\psi\).
- Res: gradient of the label‑distance penalty pulling \(R\psi\) toward \(\mathcal A_{96}\).
- Sched/Gauge: quadratic residuals yield linear normal‑equations terms.  
- Cons: multiplier equations enforce row/column sums.  
- Φ: \(2\eta(I-\mathrm D\!\Phi)^{\!\top}(\psi-\Phi\psi)\).  
- Ent/Wit: add the appropriate gradients; in hard mode impose constraints directly.

---

## 4 · Symmetry ⇒ Conservation (Discrete Noether Receipts)

Let a (possibly finite) symmetry with generator \(X\) act on configurations. If \(\mathcal L\) is invariant under the action, the **Noether charge**
\[
Q_X(t):=\big\langle\partial_{\Delta_t\psi}\mathcal L,\ X\psi_t\big\rangle
\]
is conserved along any E–L solution: \(Q_X(t{+}1)=Q_X(t)\). For finite symmetries, use difference generators \(X_g\psi= g\!\cdot\!\psi-\psi\). These charges are **receipts**: flat lines across \(t\) certify symmetry respect.

---

## 5 · Acceptance Predicate (Fail‑Closed)
A path **accepts** iff all hold:
1) **Hard constraints satisfied** at every \(t\): schedule/gauge/row‑column/Φ as configured.  
2) **Stationarity** residuals below profile tolerance.  
3) **Receipts** (Noether charges) constant across \(t\) within tolerance.  
4) **Budget‑0:** \(\sum_t W(\psi_t\!\to\!\psi_{t+1})=0\) (no net resource emission).  
Acceptance is profile‑parameterized and must be auditable.

---

## 6 · Action–Complexity and Algorithmic Reification

**Action–complexity.** For boundary data \((\psi^{\mathrm{in}},\psi^{\mathrm{out}})\), define
\[\mathsf C_S(\psi^{\mathrm{in}}\!\Rightarrow\!\psi^{\mathrm{out}}) := \inf_{\gamma} S[\gamma].\]
Any minimizer \(\gamma^*\) is the **reified algorithm**. Under gauge invariances, \(\gamma^*\) is unique modulo \(\mathcal G\).

**Speed limit (Dirichlet bound).** With \(\mathcal L_{\text{geom}}\) active,
\[ S[\gamma] \;\ge\; \tfrac{\alpha}{2Nh}\,\big\|\psi^{\mathrm{out}}-\psi^{\mathrm{in}}\big\|^2. \]

**Correctness = geodesic.** Among all constraint‑satisfying paths, the correct procedure is the minimizer. Bugs are high‑action detours that either violate constraints (ruled out) or lose to a lower‑action competitor.

**Truth ≙ conservation.** A statement/process is *true* iff there exists an accepting (budget‑0, receipts‑flat) stationary path achieving it.

---

## 7 · Concrete sector choices (implementation‑ready)

- **Norms:** use \(\ell^2\) on \(\mathcal C\); optional anisotropy by page/byte.  
- **Entropy:** page‑wise and cycle‑wise resonance entropy over R96 labels; optional MDL proxies.  
- **Resonance map:** pair‑normalized evaluation into 96 classes; expose a classifier or exact table.  
- **Schedule:** canonical fair 768‑cycle; provide rotation operator \(R\).  
- **Gauge:** boundary automorphisms of order 2048 (publish generators).  
- **Φ:** round‑trip (encode→decode) projector; target fixed points.  
- **Witness budget:** quantale/semiring accumulator; enforce zero sum for acceptance.

---

## 8 · Solver Sketch (Sector‑wise Proximal Compiler)

1) **Initialize** \(\gamma\) (linear interpolation in \(\mathcal C\); project into hard constraints).  
2) **Geom step:** conjugate‑gradient on \(\alpha\Delta_t^{\!\top}\Delta_t+\kappa_p\Delta_p^{\!*}\Delta_p+\kappa_b\Delta_b^{\!*}\Delta_b\).  
3) **Project** onto hard schedule/gauge/cons/Φ subspaces.  
4) **Prox maps** for non‑quadratic sectors:  
   — **Res:** pull toward \(\mathcal A_{96}\) via labeler \(R\).  
   — **Ent:** gradient/mirror step on \(H\).  
   — **Wit:** maintain \(\sum_t W=0\) (or Lagrange dual).  
5) **Stopping:** KKT residuals small; receipts flat; acceptance passes.

---

## 9 · Worked Mini‑Patterns (sketch)

- **Format‑preserving transform.** Hard: schedule/gauge/row‑column; Φ round‑trip. Minimize \(\mathcal L_{\text{geom}}+\mathcal L_{\text{res}}\). Output path is the reified transform with receipts.  
- **Page‑local sort.** Hard: multiset conservation per page. Entropy term favors low inversion count. Reified sort arises as the minimizer; bound steps via the Dirichlet speed limit.  
- **Tamper‑evident hashing.** Hard: Φ, schedule, and label invariants; any adversarial deviation either violates acceptance or emits budget.

---

## 10 · Profiles (which sectors are REQUIRED)

- **P‑Core:** {G, S, R} REQUIRED; C as constraints for row/column; receipts exposed; minimal Ent/Wit optional.  
- **P‑Logic:** P‑Core + Wit REQUIRED (budget‑0 truth), Ent RECOMMENDED, Φ RECOMMENDED.  
- **P‑Network:** P‑Core + hard S and Φ; Gauge REQUIRED; receipts must be serialized.  
- **P‑Full:** all sectors active; acceptance as in §5.

---

## 11 · Generalizations
- **Graphs instead of tori:** replace \(\Delta_p,\Delta_b\) by graph differences and Laplacian.  
- **Continuum limit:** letting lattice spacings \(\to 0\) yields PDE‑like E–L equations.  
- **Ensemble/quantum flavors:** replace hard minimization by Gibbs weighting \(\propto e^{-S}\) or phase sums \(e^{\tfrac{i}{\hbar}S}\); hard constraints still null out forbidden paths.

---

## 12 · Glossary (selected)
- **Action–complexity:** minimal action achieving boundary conditions.  
- **Acceptance:** pass/fail predicate in §5.  
- **Budget (0):** witness/resource measure; zero means conserved/truth.  
- **Receipts:** invariants/Noether charges constant across time for accepting paths.  
- **Reified algorithm:** a minimizing, accepting trajectory.

---

**End of formalization.**

