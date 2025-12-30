---
Title: Budgeted Holographic Information Category (BHIC)
Version: 0.1 (Public Draft)
Date: 2025‑08‑23
Sponsor: UOR Foundation (uor.foundation)
Status: Draft for community review
License: CC BY 4.0
---

# Abstract
Budgeted Holographic Information Category (BHIC) is a formal framework for reasoning about information flow in systems governed by the UOR/PrimeOS invariants—R96 resonance classes, page/cycle closures, and Φ bulk↔boundary equivalence—while explicitly accounting for **budgets** (proof/transport cost) carried by Resonance Logic (RL). BHIC unifies nine primitives—resonance entropy \(H_R\), budgeted mutual information \(I_\beta\), Φ-mutual information, conservation channel capacity \(C_{\rm cons}\), Klein-window codes, a budget-safe data processing inequality (DPI), holographic rate–distortion \(R(D_\Phi)\), the triple‑cycle spectral identity (C768), and resonance Kolmogorov complexity \(K_R\)—into a categorical object‑morphism language with testable acceptance procedures. The result is a drop‑in, auditable information theory for PrimeOS modules and adjacent systems.

# 1. Motivation and scope
Modern pipelines must preserve structure, not just bits. PrimeOS specifies a fixed 96‑ary resonance alphabet (R96) and closure schedules (pages of 48; triple cycles of 768) with a Φ isomorphism between boundary and bulk. BHIC treats these invariants as the substrate and adds **budgets** as first‑class citizens. Budgets quantify the cost of assertions, lifts, and transports—enabling verifiable guarantees like “true iff budget = 0.”

BHIC is intended for:
- Protocol designers building conserving channels, codecs, and archival formats.
- Verification engineers auditing pipelines for conservation and tamper evidence.
- Researchers exploring new measures that respect UOR/PrimeOS invariants.

# 2. Preliminaries and notation
- **Alphabet.** R96: a 96‑symbol resonance class set \(\mathcal{A}\), with per‑sample labels \(r \in \{0,\dots,95\}\).
- **Windows and cycles.** Page window size 48; fair triple‑cycle schedule of 768 steps; rotation invariance is required.
- **Φ equivalence.** A boundary↔bulk isomorphism with a canonical NF‑Lift; round‑trip acceptance is equality under fixed gauge.
- **Budgets.** Elements of a commutative monoid \((\mathcal{B}, +, 0)\) attached to morphisms; 0 is “true/conserved.” Budgets compose sub‑additively under morphism composition and accumulate along pipelines.
- **Crush.** Conservative collapse from RL to Boolean: budget‑0 theorems survive.

# 3. The BHIC abstraction
## 3.1 Objects
A BHIC object \(X\) is a boundary process with structure:
\[
X = (\Sigma,\,\sigma,\,R,\,p,\,\mathcal{C},\,\Phi_X)
\]
where:
- \(\Sigma\) is the sample space; \(\sigma\) is the schedule (48/768 with rotation invariance);
- \(R: \Sigma \to \mathcal{A}\) yields R96 labels;
- \(p\) is the class histogram over windows/cycles;
- \(\mathcal{C}\) are conservation receipts (R96 checksums, page/cycle closures, C768 statistics);
- \(\Phi_X\) provides NF‑Lift (boundary→bulk) and projection (bulk→boundary) with round‑trip witness.

## 3.2 Morphisms
A morphism \(f: X \to Y\) is a **gauge‑invariant, budget‑carrying channel** with annotation \(\beta(f) \in \mathcal{B}\). Composition satisfies:
- **Budget sub‑additivity:** \(\beta(g\circ f) \le \beta(g) + \beta(f)\).
- **Conservativity:** If \(\beta(f)=0\) and inputs satisfy receipts \(\mathcal{C}\), outputs preserve labels and closures.
- **Boolean crush:** If a statement holds with budget 0 in RL, it holds after crush.

## 3.3 Monoidal structure
- **Parallel (tensor) product:** \(X \otimes Y\) for independent processes; histograms multiply; receipts combine component‑wise.
- **Sequential composition:** Morphism composition with budget summation and DPI (see §6).

## 3.4 Distinguished functors and subobjects
- **Resonance functor** \(R\) (selector→class): fixes the 96‑ary alphabet.
- **Φ functor:** boundary↔bulk equivalence; preserves information at budget 0.
- **Crush functor:** RL→Bool, conservative at budget 0.
- **Klein subobject \(V_4\):** rare linear island \(\{0,1,48,49\}\) used for parity probes inside the non‑linear space.

# 4. The nine BHIC primitives
## 4.1 Resonance entropy \(H_R\)
**Definition.** \(H_R[p] = -\sum_{r=0}^{95} p(r)\,\log_2 p(r)\). Alphabet bound: \(H_R \le \log_2 96 \approx 6.58\) bits/symbol; equality at uniform class mix.
**Role.** Object measure of intrinsic uncertainty consistent with R96.
**Practice.** Compute per‑page (48) and per‑cycle (768) to detect drift; track rotation invariance.

## 4.2 Budgeted mutual information \(I_\beta(X;Y)\)
**Definition.** Classical MI annotated by minimal witnessing budget between processes (post‑crush witnesses must be budget 0). Write \(I_\beta\) as \(\langle I, \beta_* \rangle\), where \(\beta_*\) is the minimal budget certifying the coupling.
**Role.** Morphism‑aware dependence that quantifies “how much” and “at what cost.”
**Practice.** Log \(\beta_*\) across each stage of a pipeline; require 0 for conservation claims.

## 4.3 Φ‑mutual information
**Definition.** For a Φ‑equivalence with fixed gauge, \(I_\Phi(B;\partial B)=H(\partial B)=H(B)\). At budget 0, \(I_\Phi\) aligns with \(H_R\) when boundary labels are R96.
**Role.** Certifies holographic equality (bulk=boundary information) under acceptance.

## 4.4 Conservation channel capacity \(C_{\rm cons}\)
**Definition.** Supremum rate (bits/symbol) achievable by channels that preserve R96 labels and page/cycle receipts. Bound: \(C_{\rm cons} \le \log_2 96\).
**Attainment.** Equality iff: (i) R96 checksum holds, (ii) C768 equality/rotation holds, (iii) Φ round‑trip at budget 0.

## 4.5 Klein‑window codes
**Definition.** Coding operations restricted to the unity island \(V_4=\{0,1,48,49\}\) where XOR ↔ multiplication is a homomorphism and \(R=1\). Property fails outside \(V_4\).
**Role.** Lightweight parity/tamper‑evidence probes that must pass under conservation.
**Practice.** Sprinkle Klein probes through streams; failures indicate non‑conservation or gauge error.

## 4.6 Budget‑safe data processing inequality (DPI)
**Statement.** For gauge‑invariant \(f\): \(I_\beta(X;Y) \ge I_\beta(f(X);Y)\) provided budgets are balanced\; any increase must be paid in \(\beta\). This recovers classical DPI when \(\beta=0\).
**Role.** Enforces monotonicity of information under legitimate post‑processing.

## 4.7 Holographic rate–distortion \(R(D_\Phi)\)
**Definition.** Distortion \(D_\Phi\) is the minimal budget to reconstruct a bulk state from boundary via NF‑Lift, subject to closure tolerances. \(R(D_\Phi)\) is the achievable rate.
**Corner cases.** \(D_\Phi=0 \iff\) Φ round‑trip passes and receipts hold ⇒ \(R(0)=\log_2 96\).

## 4.8 Triple‑cycle spectral identity (C768)
**Definition.** Equality of means/variances across the three residues in the fair 768 schedule; invariance under rotation. Acts as a spectral checksum.
**Role.** Necessary side‑condition for conservation and Φ‑equivariance.

## 4.9 Resonance Kolmogorov complexity \(K_R\)
**Definition.** Minimal description length to regenerate a resonance trace given the fixed invariants and the reference machine (NF‑Lift + gauge normalizer + group enum). Lower for mostly Klein/periodic traces; higher for diverse class usage.
**Role.** Lower‑bounds required budgets for faithful reconstruction and compressibility.

# 5. Cross‑primitive correspondences
**Alphabet ↔ Capacity.** R96 fixes \(|\mathcal{A}|=96\) ⇒ \(H_R^{\max}=\log_2 96\) ⇒ \(C_{\rm cons} \le H_R^{\max}\). Equality requires C768 and \(D_\Phi=0\).

**Budget ↔ Distortion.** \(D_\Phi\) is the operational meaning of budgets in reconstruction; the “knee” of \(R(D_\Phi)\) occurs at \(\beta=0\).

**Klein ↔ Tamper‑evidence.** Klein windows provide zero‑budget parity checks; violations flag non‑conserving transforms.

**C768 ↔ Φ‑equivariance.** Fair scheduling is both a checksum and a prerequisite for boundary‑equivariance; without it, Φ cannot be certified at budget 0.

**Complexity ↔ Rate.** Smaller \(K_R\) implies lower budgets to achieve a target distortion and typically higher attainable rate under conservation constraints.

# 6. Axioms and immediate theorems (BHIC‑0)
**A0 (Alphabet).** \(|\mathrm{Im}(R)|=96\); pair‑normalized evaluation on boundary windows.

**A1 (Conservation).** Page and triple‑cycle receipts hold and are rotation‑invariant.

**A2 (Holography).** There exists a Φ round‑trip with NF‑Lift; acceptance at budget 0 implies equality of bulk and boundary information.

**A3 (Budgets).** Budgets form a commutative monoid, compose sub‑additively, and 0 witnesses truth under crush.

**A4 (Klein).** \(V_4\) subobject is homomorphic and lossless; used for parity probes.

**T1 (Holographic equality).** Under A0–A3 and \(\beta=0\): \(I_\Phi(B;\partial B)=H_R\).

**T2 (Capacity sandwich).** For conserving \(f\): \(I_\beta(X;Y) \le C_{\rm cons}(f) \le H_R(X)\); equalities iff \(D_\Phi=0\) and Klein probes pass.

**T3 (Budget‑DPI).** For gauge‑invariant \(f\): \(I_\beta(X;Y) \ge I_\beta(f(X);Y)\). Equality iff \(f\) is information‑lossless at budget 0.

# 7. Compliance and acceptance testing
## 7.1 Receipts and checks
- **R96 digest:** histogram, perplexity, and checksum per page.
- **C768 spectral:** mean/variance equality across residues; rotation‑invariance check.
- **Φ round‑trip:** NF‑Lift→project with tolerance; emit \(\beta\) and acceptance bit.
- **Klein probes:** parity checks on \(V_4\) slots distributed across pages.

## 7.2 Conformance levels
- **BHIC‑L0 (observe):** Computes receipts and logs \(H_R\); no claims on budgets.
- **BHIC‑L1 (conserve):** All receipts pass; \(\beta=0\) for Φ round‑trip; Klein probes pass.
- **BHIC‑L2 (optimize):** Achieves \(C_{\rm cons}\) within \(\epsilon\) of \(H_R\) on certified windows; publishes \(R(D_\Phi)\) curve.

# 8. Algorithms and pseudocode
## 8.1 R96 entropy and receipts
```python
# windowed_resonance_entropy(stream, window=48)
# returns: list of (H_R, hist, receipts)

def windowed_resonance_entropy(stream, window=48):
    receipts = []
    out = []
    for i in range(0, len(stream), window):
        w = stream[i:i+window]
        hist = count_classes(w, K=96)            # R96 labels
        p = [c/sum(hist) for c in hist]
        H_R = -sum(q*log2(q) for q in p if q>0)
        r96_ok = checksum96(w)
        out.append((H_R, hist, {"r96": r96_ok}))
        receipts.append(r96_ok)
    return out
```

## 8.2 C768 spectral checksum
```python
# triple_cycle_receipt(labels, cycle=768)
# checks equality across residues and rotation invariance

def triple_cycle_receipt(labels, cycle=768):
    residues = [labels[j::3] for j in range(3)]
    stats = [(mean(r), var(r)) for r in residues]
    equal = all(near(stats[k], stats[0]) for k in range(1,3))
    rot_ok = rotate_invariant(labels, cycle)
    return {"equal": equal, "rot": rot_ok, "stats": stats}
```

## 8.3 Φ round‑trip with budget
```python
# phi_round_trip(boundary, nf_lift, project, tol)
# returns: (accepted: bool, beta: int)

def phi_round_trip(boundary, nf_lift, project, tol=0):
    bulk, beta1 = nf_lift(boundary)          # lift with budget
    recon, beta2 = project(bulk)             # project with budget
    acc = distance(recon, boundary) <= tol
    beta = beta1 + beta2
    return acc and beta==0, beta
```

## 8.4 Klein parity probes
```python
# apply_klein_probes(window)
# returns: list of booleans for each probe

def apply_klein_probes(window):
    probes = []
    for a,b in choose_pairs_in_V4(window):
        lhs = xor(a,b)
        rhs = mul(a,b)     # group op restricted to V4
        probes.append(resonance(lhs)==1 and lhs==rhs)
    return probes
```

# 9. Engineering playbooks
**Pipeline audit.** Wrap each stage as a morphism; compute \(H_R\) before/after; run receipts (R96, C768); execute Φ round‑trip; log \(\beta\); sprinkle Klein probes. Conservation is certified when: all receipts pass, \(H_R\) unchanged within tolerance, \(\beta=0\), probes pass.

**Codec design.** Aim for the 96‑ary bound; use Klein windows as free parity; publish \(R(D_\Phi)\) vs budget curves; enforce rotation‑invariant scheduling.

**Archival format.** Embed receipts and minimum proofs (budgets) with payloads; make Φ round‑trip reproducible; store entropy traces per page.

# 10. Security, threat models, and failure modes
- **Non‑conserving transforms.** Will break C768 or Klein probes, or require \(\beta>0\) at Φ round‑trip.
- **Gauge drift.** Appears as entropy shifts and rotation asymmetries; mitigate with stricter gauge locks.
- **Adversarial relabeling.** R96 checksum detects permuted class maps; Φ acceptance blocks inconsistent lifts.
- **Budget forgery.** Disallow unsigned budget claims; budgets MUST be carried as signed receipts.

# 11. Empirical methodology
- **Benchmarks.** Publish windowed \(H_R\) plateaus, C768 receipts, and Φ acceptance rates.
- **Ablations.** Remove Klein probes and observe false‑accept rise; limit \(\beta\) and map \(R(D_\Phi)\) knees.
- **Reproducibility.** Provide scripts and seeds; embed receipts in artifacts.

# 12. Applications
- **PrimeOS modules.** Use BHIC‑L1 for transport and storage layers; BHIC‑L2 for optimizing codecs under conservation.
- **Tamper‑evident logs.** Klein probes + C768 receipts as cheap online checks.
- **Model evaluation.** Use \(I_\beta\) to score feature pipelines; smaller \(K_R\) streams generalize better under conservation constraints.

# 13. Related ideas (orientation)
- Shannon information and classical DPI appear as the \(\beta=0\) special case.
- Holographic dualities motivate Φ‑mutual information and rate–distortion in bulk↔boundary terms.
- Algorithmic complexity (Kolmogorov) is adapted here to the resonance substrate (\(K_R\)).

# 14. Conformance statement template
> Implementation **X** is BHIC‑L1 conformant on dataset **D** with schedule **S**. Receipts: R96 ✓, C768 ✓ (rot ✓). Φ round‑trip: accepted ✓ with \(\beta=0\). Klein probes: pass rate 100%. Capacity: \(C_{\rm cons}\) within 0.02 bits/sym of \(H_R\).

# 15. Open problems
1. Tight converse bounds relating \(K_R\) to \(R(D_\Phi)\) under partial conservation.
2. Universal constructions for budget algebras beyond commutative monoids (e.g., ordered semirings).
3. Finite‑blocklength effects for C768 receipts and their tradeoff with Klein probe density.
4. Adaptive scheduling that preserves Φ‑equivariance while maximizing instantaneous capacity.

# 16. Summary
BHIC elevates budgets to first‑class information‑theoretic citizens in a world where conservation, closures, and holographic equivalence are invariants. The nine primitives cohere into a categorical framework with crisp, testable receipts. This aligns engineering practice (checksums, parity, round‑trips) with theory (entropy, capacity, DPI, rate–distortion, complexity) under one roof.

---
**Acknowledgments** — UOR Foundation and the PrimeOS community for the underlying invariants and test harnesses.

**Contact** — foundation@uor.foundation | https://uor.foundation

