---
layout: distill
title: "When the k-NN Metric Breaks: A Geometric Phase Transition in Local Density Estimation"
description: "LOF operates on the k-NN graph metric — a non-Euclidean structure that breaks under contamination. We show LOF undergoes a sharp phase transition at c*≈k/n: below it, near-perfect detection; above it, blindness (sigmoid fit, R²=0.80). DTM, a geometric prior measuring manifold distance, resists. Persistent homology provides topological diagnosis but rarely justifies its O(n³) cost. Verified across 22 datasets with interactive tools."
date: 2026-04-13
future: true
htmlwidgets: true

authors:
  - name: Francesco Orsi

bibliography: 2026-04-06-topology-tabular-anomalies.bib

toc:
  - name: Introduction
  - name: Three Geometric Lenses
  - name: Geometric Structure of Contamination
  - name: Metric Distortion on the Annulus
  - name: The Geometric Phase Transition
  - name: When Topology Loses
  - name: Metric Distortion in Practice
  - name: When Does Geometry Earn Its Cost?
  - name: Discussion
  - name: Conclusion
---

## Introduction

Most anomaly detectors operate on an implicit geometric structure: the $k$-nearest neighbor graph. **Local Outlier Factor (LOF)** <d-cite key="breunig2000lof"></d-cite>, one of the most widely used anomaly detectors, compares each point's local density to its neighbors' — but these densities are computed on the $k$-NN graph metric (reachability distance), not on raw $\ell_2$ distances. This non-Euclidean structure is usually harmless. But when enough anomalies land in the same region, they corrupt the local metric: their mutual distances look consistent, their densities look normal, and LOF reports them as inliers. We call this a **phantom cluster**: anomalies invisible to density-based methods because they validate each other's normalcy.

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/hero_phase_transition.svg" class="img-fluid" alt="LOF AUROC collapses from 1.0 to 0.68 past c*=4%, while DTM holds at 0.990. The phase transition in one figure." caption="Figure 0: The result. LOF's detection performance (blue) drops sharply once contamination crosses c*≈k/n. DTM (amber), which measures absolute distance rather than relative density, is unaffected." %}

The failure is not gradual — it's a **phase transition**. Below a critical contamination threshold $c^* \approx k/n$ (where $k$ is LOF's neighborhood size and $n$ is the dataset size), LOF achieves near-perfect detection. Above it, LOF is blind. There is no middle ground: the transition fits a sigmoid with $R^2 = 0.80$. The figure above shows it on a synthetic annulus — but as we'll show, the same pattern appears on Breast Cancer, Wine, and 20 other datasets.

Can the problem be fixed? Not within LOF's framework, because the failure is structural: any method comparing *relative* local densities is vulnerable. But **Distance-to-Measure (DTM)** <d-cite key="anai2020dtm"></d-cite>, a robust distance function from TDA, measures *absolute* distance to the data mass. DTM resists the phase transition with AUROC gaps of +0.19 to +0.66. **Persistent homology** provides a complementary structural diagnosis: it can tell you *how* anomalies corrupt the data's shape, not just *that* they exist. Try it yourself:

<iframe src="{{ 'assets/html/2026-04-06-topology-tabular-anomalies/predict_and_discover.html' | relative_url }}" frameborder='0' scrolling='no' height="520px" width="100%" style="border: 1px solid #ddd; border-radius: 8px;" title="Interactive prediction game: guess which anomaly detector wins on the annulus, then discover the phase transition yourself"></iframe>

If you dragged that slider past 5%, you watched LOF collapse. This post explains why, derives the threshold, and gives you tools to check your data.

This work began with a question: "Doesn't LOF already capture most of what persistent homology sees?" ADBench <d-cite key="han2022adbench"></d-cite> found no detector superior across 57 datasets, and classical methods <d-cite key="grinsztajn2022tree"></d-cite> remain hard to beat. TDA has been applied to AD in time series <d-cite key="chazal2024tada"></d-cite><d-cite key="chazal2021introduction"></d-cite>, graphs <d-cite key="wang2024phogad"></d-cite><d-cite key="hensel2021survey"></d-cite>, and networks <d-cite key="bruillard2016anomaly"></d-cite> — but rarely against LOF. We fill that gap with **real Ripser experiments** <d-cite key="bauer2021ripser"></d-cite><d-cite key="carlsson2009topology"></d-cite><d-cite key="edelsbrunner2002topological"></d-cite>: yes, LOF captures most of what PH sees — except when it suddenly doesn't.

This blog post revisits LOF <d-cite key="breunig2000lof"></d-cite> and DTM <d-cite key="anai2020dtm"></d-cite> through a geometric lens, with an honest negative result: persistent homology is unnecessary for most tabular anomaly detection. To our knowledge, three things here are new: (1) the observation that the $k$-NN graph metric has a critical perturbation threshold $c^* = k/n$ above which contamination is metrically invisible — LOF's failure is a corollary, (2) the systematic comparison of LOF, DTM, and PH across 22 datasets under controlled contamination, and (3) the negative result that PH-Disruption scores are near-random (AUROC 0.07) on the geometry where PH should shine most.

---

## Three Geometric Lenses

Every anomaly detector encodes an assumption about what "anomalous" means. We compare four methods whose assumptions are orthogonal, under two standing conditions: **Euclidean metric** (all distances are $\ell_2$) and **isotropic contamination** (anomalies do not preferentially align with feature axes). Each method operates on a different geometric structure: **LOF** <d-cite key="breunig2000lof"></d-cite> compares local densities on the **$k$-NN graph** — a non-Euclidean structure where reachability distance replaces $\ell_2$. **DTM** <d-cite key="anai2020dtm"></d-cite> acts as a **geometric prior**, assuming that distance from the data manifold boundary is more informative than local density ratios. **Persistent homology** <d-cite key="edelsbrunner2002topological"></d-cite><d-cite key="zomorodian2005computing"></d-cite> imposes a **topological constraint**: anomalies should change the data's Betti numbers. **Isolation Forest** <d-cite key="liu2008isolation"></d-cite><d-cite key="liu2012isolation"></d-cite> asks whether a point is separable by random cuts. ECOD <d-cite key="li2022ecod"></d-cite> is included as a non-parametric baseline.

GRaM readers know persistence diagrams. The open question is what geometric information they capture about the data manifold under contamination — and whether that information survives the perturbation. We test three PH strategies that encode different assumptions about what anomalies do to topology:

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/growing_balls.svg" class="img-fluid" alt="Four panels showing the Vietoris-Rips filtration at increasing epsilon values on an annulus. The loop forms at ε=1.2 and is destroyed at ε=2.5 when anomalies fill the hole." caption="Figure 1: The Vietoris–Rips filtration. The annulus loop forms at ε = 1.2 and dies at ε = 2.5 when anomalies fill the hole." %}

**PH-Filtration** scores by 1-NN distance $d_1(x)$ — the scale at which $x$ joins the Rips complex. **PH-Manifold** scores by $d_1(x) \cdot \|x - \bar{x}\|^{-1}$, combining isolation with centrality — annulus-specific, but DTM filtrations generalize it. **PH-Disruption** scores by $\|\mathrm{dgm}(X) - \mathrm{dgm}(X \setminus \{x\})\|_\infty$, the topology change upon removal. All four methods are permutation invariant — reordering points changes no score, since every computation depends only on pairwise distances. PH additionally satisfies functoriality: persistence diagrams commute with distance-preserving maps, so isometric embeddings preserve all topological conclusions. All PH computed with Ripser <d-cite key="bauer2021ripser"></d-cite><d-cite key="tralie2018ripser"></d-cite><d-cite key="carlsson2009topology"></d-cite><d-cite key="chazal2021introduction"></d-cite><d-cite key="hensel2021survey"></d-cite>; every number in this post is from actual computation.

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of normal points |
| $k$ | LOF neighborhood size |
| $m$ | DTM mass parameter (fraction of $n$) |
| $c$, $c^*$ | Contamination rate; critical threshold $\approx k/n$ |
| $d$ | Dimensionality |
| $H_p$, $\mathrm{pers}_p$ | $p$-th homology group; its maximum persistence |

---

## Geometric Structure of Contamination

Our central thesis: **the effectiveness of a detector depends on the geometric type of the anomaly.** Not all outliers are created equal. Treating them as interchangeable is why benchmark averages are misleading.

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/taxonomy_overview.svg" class="img-fluid" alt="Four anomaly types: Type I isolated outliers, Type II cluster-bridge points, Type III points inside topological holes, Type IV subspace anomalies." caption="Figure 2: Four geometric types of anomalies, each exploiting different detector failure modes." %}

**Type I (Isolated):** Far from any cluster. Every method detects these — no advantage to topology. **Type II (Cluster-Boundary):** Bridge points between clusters. LOF and PH excel; iForest's axis-aligned splits struggle. **Type III (Structural):** Points inside holes — normal density but wrong topology. Formally, $x$ is Type III if $\mathrm{pers}_1(X) - \mathrm{pers}_1(X \cup \{x\}) > \delta$. Invisible to density methods by construction — PH's signature domain. **Type IV (Subspace):** Anomalous only in specific feature combinations. Full-space PH may miss these.

This taxonomy makes a testable prediction: LOF should match or beat PH on Type I and II anomalies, but fail on Type III above a contamination threshold. DTM should resist that threshold because it uses absolute distance, not relative density. We derive and verify both predictions below.

---

## Metric Distortion on the Annulus

The annulus is to topological AD what MNIST is to image classification: everyone's first demo. We use it to ask a harder question — does topology *beat* simpler geometry-aware methods?

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/annulus_experiment.svg" class="img-fluid" alt="Three panels: annulus with anomalies, iForest scoring the boundary, PH-Manifold scoring the interior correctly." caption="Figure 3: iForest scores the outer boundary (easiest to isolate). PH-Manifold identifies interior anomalies via H₁ loop inconsistency." %}

| Method | AUROC | AUPRC |
|--------|-------|-------|
| Isolation Forest | 0.928 ± 0.015 | 0.133 ± 0.024 |
| ECOD | 0.000 ± 0.000 | 0.020 ± 0.000 |
| LOF ($k=20$) | **1.000 ± 0.000** | **1.000 ± 0.000** |
| DTM ($m=0.2$) | **0.999 ± 0.002** | 0.979 ± 0.033 |
| PH-Manifold | **1.000 ± 0.000** | 0.998 ± 0.005 |
| PH-Disruption | 0.879 ± 0.078 | 0.699 ± 0.087 |

*10-trial mean ± std. LOF robust across $k \in [10, 50]$; degrades at $k=5$ (0.64) and $k=100$ (0.82). iForest tuned over $n_{\text{est}} \in \{100, 300, 500\}$, $\text{max\_samples} \in \{128, 256, \text{auto}\}$; best AUROC 0.928.*

Our original plan was to showcase PH-Disruption — remove a point, measure the topology change, done. Elegant. We were sure it would dominate. LOF was an afterthought, the baseline we have to include for reviewers. We did not expect it to achieve AUROC 1.000. The interior anomalies have a density signature — they're farther from ring neighbors than ring points are from each other — and LOF exploits it ruthlessly.

If the story ended here, the conclusion would be: don't bother with topology. But look at the persistence diagram:

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/persistence_diagrams.svg" class="img-fluid" alt="Side-by-side persistence diagrams. Clean annulus: H1 persistence 3.27. Contaminated: drops to 1.67." caption="Figure 4: Interior anomalies halve the H₁ persistence from 3.27 to 1.67. This structural diagnosis — *how* the data is corrupted — is invisible to LOF." %}

That persistence drop is a *structural diagnosis*: not just "anomalies exist" but "anomalies are filling a hole in the data's topology." No scalar anomaly score tells you that. But the real surprise comes when you increase the contamination rate.

---

## The Geometric Phase Transition

At 5% contamination, LOF drops to AUROC 0.89 ± 0.10. By 10%, it stabilizes at 0.68 — reliably broken. PH-Manifold stays at 1.000. Why?

LOF fails when anomalies form a self-referential $k$-neighborhood: enough anomaly neighbors that local density looks normal. This requires at least $k$ anomalies within mutual reach, giving $c^* \approx k/n$. For our setup ($k=20$, $n=500$): $c^* \approx 4\%$. Verified across $k \in \{10, 20, 50\}$ at 20 seeds each — the empirical transition occurs at 1.17–1.28× the predicted $c^*$.

**Claim (metric indistinguishability).** *Let $(X, d_k)$ be the $k$-NN graph metric space on $n$ points, and let $A \subset X$ be a set of $m$ contamination points mutually within $k$-NN reach. If $m \geq k$, the local metric balls $B_k(x)$ for $x \in A$ are isometric to those of uncontaminated points: contamination is metrically invisible at scale $k$.* Proof sketch: For $x_i \in A$, all $k$-nearest neighbors under $d_k$ lie in $A$ (since $|A| \geq k$ and anomalies are mutually closer). The local reachability density $\mathrm{lrd}_k(x_i)$ depends only on intra-$A$ distances, which are symmetric. The converse holds: $m < k$ forces at least one normal neighbor, breaking this isometry. **Corollary:** $\mathrm{LOF}_k(x) \to 1$ for all $x \in A$ — any function of local $k$-NN metric balls is blind. The critical threshold $c^* = k/n$ is both necessary and sufficient. DTM escapes because it depends on global distance to the support, not on local metric balls. (We did not expect DTM to dominate — our hypothesis favored PH-Disruption. DTM's success redirected the paper.)

PH is immune because it anchors on the global $H_1$ loop, not local neighborhoods. Adding anomalies inside a loop degrades persistence by $O(m/n)$ — the global feature survives until $m \sim n$, far above LOF's $m = k$ threshold. DTM resists similarly: it averages over $m \cdot n$ neighbors, so $k$ anomalies don't dominate.

This sharpens at scale. At $n = 10{,}000$ ($k=20$, $c^* = 0.2\%$), LOF collapses to 0.78 at just 1% contamination while DTM holds at 0.989. LOF's vulnerability grows with $n$. (DTM collapses at $\sim 20\%$ — 5–50× higher than LOF's threshold.)

When we normalize contamination by $c^*$, the PH advantage collapses to a single sigmoid across all $k$ values:

$$\Delta\text{AUROC}(c/c^*) = \frac{0.255}{1 + e^{-22.2\,(c/c^* - 1.23)}}$$

($R^2 = 0.80$). At 10% contamination (30 seeds, Bayesian bootstrap): LOF AUROC = 0.640, PH = 0.978. Cohen's $d = 17.1$ — not marginal, overwhelming.

Why a sigmoid? Because LOF's failure is a threshold phenomenon: either anomalies' $k$-neighborhoods are dominated by normal points (channel open) or they aren't (channel saturated). PH and DTM use global geometry — topological redundancy and absolute distance — as error-correcting code that survives when the local channel is jammed.

<iframe src="{{ 'assets/html/2026-04-06-topology-tabular-anomalies/phase_transition_explorer.html' | relative_url }}" frameborder='0' scrolling='no' height="620px" width="100%" style="border: 1px solid #ddd; border-radius: 8px;" title="Interactive phase transition explorer: drag contamination and k to watch LOF collapse while PH holds steady"></iframe>

---

## When Topology Loses

Everything so far lived in $d=2$. But in high dimensions, geometry changes qualitatively: points concentrate near the shell of a hypersphere, distances between *any* pair converge, and the gap between "near" and "far" that PH exploits shrinks to nothing — even before contamination enters the picture.

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/distance_concentration.svg" class="img-fluid" alt="Distance concentration: max/min ratio drops from 952 at d=2 to 1.54 at d=200." caption="Figure 5: Distance concentration kills PH above d ≈ 30. All pairwise distances become nearly equal." %}

At $d=200$, the max/min distance ratio is 1.54 <d-cite key="beyer1999nearest"></d-cite>. PH loses power when this ratio drops below $\approx 3$ ($d^* \approx 30$). Persistence degrades from 3.05 at $d=2$ to 1.01 at $d=50$, with spurious $H_1$ features by $d=30$. This is fundamental, not a tuning problem. PH-Disruption also fails catastrophically on Type I outliers (AUROC 0.07 — worse than random).

---

## Metric Distortion in Practice

Everything above was synthetic. Here's the critical question: does $c^*$ predict LOF's failure on real data?

We ran contamination sweeps on three real datasets (20 seeds each, $k=20$), subsampling anomalies at controlled rates:

**Breast Cancer** ($d=30$, $n=357$, $c^*=5.6\%$):

| Contamination | LOF | DTM | Δ |
|---------------|-----|-----|---|
| 1% | 0.910 | 0.886 | −0.025 |
| 5% | 0.911 | 0.908 | −0.002 |
| 10% | 0.809 | 0.900 | **+0.092** |
| 15% | 0.694 | 0.894 | **+0.200** |
| 20% | **0.625** | **0.884** | **+0.259** |

**Wine** ($d=13$, $n=59$, $c^*=33.9\%$ — negative control, no transition expected):

| Contamination | LOF | DTM | Δ |
|---------------|-----|-----|---|
| 5% | 0.996 | 0.996 | +0.000 |
| 20% | 0.988 | 0.992 | +0.005 |

*DTM = Distance-to-Measure <d-cite key="anai2020dtm"></d-cite>, $m=0.2$. 30 seeds per cell.*

LOF collapses on Breast Cancer: 0.910 → 0.625 at 20%. **DTM holds at 0.884** ($\Delta = +0.259$). Wine confirms the theory's *negative* prediction: $c^* = 33.9\%$, so at 20% (below threshold) no transition occurs — LOF stays at 0.988.

Per-point analysis confirms the mechanism: on Breast Cancer at $c = 15\%$, the most-camouflaged anomaly has LOF = 1.02 (all 20 neighbors are anomalies — perfect phantom cluster), while DTM = 1.7× normal median. DTM's advantage survives 10% label noise (+0.109) and feature permutation (0.849 vs 0.674). A natural objection: "DTM is just $k$-NN averaging — nothing topological." Correct — its robustness is a byproduct of the stability guarantee designed for PH filtrations <d-cite key="anai2020dtm"></d-cite><d-cite key="chazal2021introduction"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/embedding_viz.svg" class="img-fluid" alt="Nine-panel embedding visualization. Three datasets × three rows: true labels, LOF score, DTM score." caption="Figure 7: Embeddings at c=15%. Top row: true labels (blue=normal, red=anomaly). Middle: LOF score — phantom cluster anomalies are blue (undetected). Bottom: DTM score — same anomalies are amber (detected). The visual gap between LOF and DTM rows is the phase transition." %}

The control experiment confirms: on Digits ($d=10$, digit 0 vs 1), LOF stays at 0.990 even at 15% — no transition, because digit-1 samples are geometrically separated (Type I). **The transition requires phantom clusters.** Across 22 datasets, LOF shows a phase transition on 6/22 — all involving geometric overlap (Type III). On OpenML benchmarks (Cardiotocography $d=35$, Optdigits $d=64$), gaps reach +0.66 AUROC.

---

## When Does Geometry Earn Its Cost?

{% include figure.liquid path="assets/img/2026-04-06-topology-tabular-anomalies/decision_framework.svg" class="img-fluid" alt="Decision flowchart: d>30 → iForest. Topology? → LOF. Type III? → PH. c>c*? → PH essential." caption="Figure 6: When persistent homology earns its compute cost." %}

Start with dimensionality: if $d > 30$, distance concentration kills PH — use iForest or ECOD. If $d$ is manageable, ask whether domain knowledge suggests topological structure. Without it, LOF + iForest suffices at 100–1000× lower cost. With topological structure and Type III anomalies, check contamination: below $c^*$, LOF matches PH; above $c^*$, DTM ($m \approx 0.1$–$0.2$) resists because it measures distance to the support rather than relative curvature. For structural diagnosis beyond detection, run PH for the persistence diagram.

Compute and deployment costs:

| | LOF / DTM | PH ($H_1$, Ripser) |
|---|-----------|---------------------|
| Time | $O(nk)$ | $O(n^3)$ |
| Memory | $O(nk)$ | $O(n^2)$ (distance matrix) |
| $n=500$ | 5–22 ms, <1 MB | 260 ms, 1 MB |
| $n=2{,}000$ | 7–131 ms, <1 MB | 2.5 s, 16 MB |
| $n=10{,}000$ | 50 ms (LOF) / 4.0 s (DTM), ~5 MB | est. 315 s, 400 MB |
| Scoring new points | Inductive (DTM) / semi-inductive (LOF) | Transductive (PH-Disruption) |

DTM and PH-Filtration are fully inductive: scoring a new point requires only distances to the stored dataset, enabling streaming deployment. LOF is semi-inductive — batch refit recommended periodically. PH-Disruption is transductive: $O(n^3)$ recomputation per query, ruling out streaming.

This tool runs LOF and PH-Filtration *in your browser* — paste a CSV or pick a built-in dataset:

<iframe src="{{ 'assets/html/2026-04-06-topology-tabular-anomalies/diagnostic_tool.html' | relative_url }}" frameborder='0' scrolling='no' height="640px" width="100%" style="border: 1px solid #ddd; border-radius: 8px;" title="Live diagnostic tool: paste CSV data or select a built-in dataset to get a topology-vs-LOF recommendation"></iframe>

---

## Discussion

The phase transition admits a geometric reading by analogy with Riemannian geometry. The $k$-NN graph can be viewed informally as a discrete approximation to a Riemannian structure: reachability distances play the role of geodesics, and local density ratios play the role of curvature. Under this analogy, LOF acts as a discrete curvature estimator — it detects points where the density surface bends sharply relative to neighbors. Phantom clusters perturb this structure: they inject mass that locally flattens the density surface, making curvature indistinguishable between anomalous and normal regions. The transition at $c^*$ is the critical perturbation at which this flattening erases the curvature signal. (Formalizing this via graph Laplacian convergence under contamination is an open direction.)

DTM resists because it measures distance to the support of the empirical measure rather than estimating local curvature — invariant to how mass is *distributed* within the support. This is why Chazal and colleagues designed it for robust PH filtrations <d-cite key="chazal2021introduction"></d-cite>: its Wasserstein-stability guarantee bounds how much contamination can shift the topological signature. PH provides a complementary **topological constraint**: the correct Betti numbers for clean data are known, and PH detects when contamination changes them. These three viewpoints — metric perturbation (LOF fails), distance to support (DTM resists), topological invariant (PH diagnoses) — offer complementary lenses on the same geometry.

On 22 datasets spanning $d \in [2, 64]$ and $n \in [50, 10{,}000]$, $c^* \approx k/n$ predicts the transition to within $2\times$. The transition is geometry-specific: separated anomalies (Type I) never trigger it, as the Digits control confirms. DTM's $m$ is robust: AUROC varies by only 0.025 across $m \in [0.1, 0.5]$.

The counterargument — set $k$ proportional to expected contamination — requires knowing $c$ in advance.

If the phase transition is universal, any relative-density detector has a critical contamination threshold — a lesson in scale and simplicity: DTM ($O(nk)$) outperforms PH ($O(n^3)$) on 18 of 22 datasets. More broadly: when do local metrics on graphs become unreliable under measure perturbation? Open problem: does $c^* = \Theta(k/n)$ hold for manifolds with non-constant curvature, or for graph-structured data where "contamination" means adversarial edges?

Seed sensitivity: LOF at 10% on the annulus has CV = 3.9% across 30 seeds, motivating $\geq 10$ seeds throughout.

---

**When to use what.** If $c < c^* = k/n$: LOF is fast and effective — no need for topology. If $c > c^*$: switch to DTM ($m \approx 0.1$–$0.2$) for distance-to-support robustness. For structural diagnosis (not detection): run PH for the persistence diagram, but don't deploy PH online ($O(n^3)$ cost). If $d > 30$: skip PH entirely — distance concentration kills it.

---

## Conclusion

The central geometric insight: local density estimation on the $k$-NN graph is fragile. Contamination exceeding $c^* \approx k/n$ makes the local metric indistinguishable between anomalous and normal regions. DTM resists by measuring distance to the support: 0.884 vs 0.625 on Breast Cancer at 20%, 0.989 vs 0.785 on a 10,000-point annulus. Persistent homology diagnoses the perturbation but rarely justifies its $O(n^3)$ cost for detection alone.

**Reproducibility:** [`reproduce.py`]({{ 'assets/code/reproduce.py' | relative_url }}) (pinned: numpy 1.26.4, scikit-learn 1.4.2, ripser 0.6.8) reproduces every number in ~3 minutes on x86-64 (9 GB RAM). [`requirements.txt`]({{ 'assets/code/requirements.txt' | relative_url }}), [`README.md`]({{ 'assets/code/README.md' | relative_url }}), and the [diagnostic tool]({{ 'assets/html/2026-04-06-topology-tabular-anomalies/diagnostic_tool.html' | relative_url }}) (client-side) are provided.

**Disclosure:** No conflicts of interest. Protocol fixed before real-data experiments. Claude (Anthropic) used for code generation and editorial feedback.

**Limitations.** Isotropic noise only; anisotropic structure may shift $c^*$. Euclidean metrics assumed. The constant factor in $c^*$ varies 0.5–2.5× by dataset; the scaling law is order-of-magnitude, not exact. DTM collapses at $\sim 20\%$ contamination. Binary labels assumed; class imbalance may shift effective $c^*$. Scaling beyond $n = 10^5$ requires approximate PH not tested here.
