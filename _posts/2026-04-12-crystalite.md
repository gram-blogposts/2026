---
layout: distill
title: "Crystalite: A Lightweight Transformer for Efficient Crystal Modeling"
description: "Crystalite is a lightweight diffusion Transformer for crystal generation and crystal structure prediction. This post covers its chemistry-aware atom encoding, geometry-aware attention mechanism, and benchmark results."
date: 2026-04-12
future: true
htmlwidgets: true
authors:
  - name: Tin Hadzi Veljkovic
bibliography: 2026-04-15-crystalite.bib
toc:
  - name: Generative modeling of crystals
  - name: Equivariant GNNs for crystal modeling
  - name: Chemistry-based atom encoding
  - name: Geometry-aware attention with GEM
  - name: Crystal structure prediction and de novo generation results
    subsections:
      - name: Crystal structure prediction
      - name: De novo generation
      - name: "External benchmarking: LeMat GenBench"
      - name: Large-scale generation
  - name: Balancing efficiency and expressivity
_styles: |-
  .note {
    margin: 1.5rem 0 2rem;
    padding: 1rem 1.15rem;
    border-left: 3px solid rgba(15, 23, 42, 0.7);
    background: rgba(15, 23, 42, 0.05);
  }
  .embed-frame {
    width: 100%;
    min-height: 620px;
    border: 0;
    background: transparent;
    display: block;
    border-radius: 18px;
    margin-bottom: 0.35rem;
  }
  .figure-narrow {
    max-width: 78%;
    margin: 1.6rem auto;
  }
  .motion-video {
    width: 100%;
    display: block;
    background: #000;
  }
  .figure-nudge-left {
    transform: translateX(-2%);
  }
  .table-shell {
    margin: 1.5rem 0 0.85rem;
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 18px;
    overflow-x: auto;
    background: rgba(15, 23, 42, 0.03);
  }
  .table-shell table {
    width: 100%;
    min-width: 860px;
    border-collapse: collapse;
    font-size: 0.96rem;
  }
  .table-shell--wide table {
    min-width: 1180px;
  }
  .table-shell--xwide table {
    min-width: 1280px;
  }
  .table-shell th,
  .table-shell td {
    padding: 0.8rem 0.95rem;
    border-bottom: 1px solid rgba(15, 23, 42, 0.1);
    vertical-align: middle;
  }
  .table-shell thead th {
    font-weight: 700;
    background: rgba(15, 23, 42, 0.06);
  }
  .table-shell th:first-child,
  .table-shell td:first-child {
    text-align: left;
    white-space: nowrap;
  }
  .table-shell th:not(:first-child),
  .table-shell td:not(:first-child) {
    text-align: right;
  }
  .table-shell tbody tr:last-child td {
    border-bottom: 0;
  }
  .table-shell .group {
    text-align: center !important;
  }
  .table-shell .section-row td {
    text-align: center !important;
    font-weight: 700;
    background: rgba(15, 23, 42, 0.06);
  }
  .table-caption,
  .embed-caption {
    margin: 0 0 2rem;
    font-size: 0.94rem;
    line-height: 1.64;
    color: rgba(15, 23, 42, 0.72);
  }
  @media (max-width: 900px) {
    .figure-narrow {
      max-width: 100%;
      transform: none;
    }
    .embed-frame {
      min-height: 560px;
    }
  }
---

<h2 id="generative-modeling-of-crystals">Generative modeling of crystals</h2>

Discovering new crystalline materials remains a difficult search problem and a central challenge in modern materials discovery <d-cite key="Merchant2023"></d-cite>. The number of possible compositions and structures is enormous, while only a small fraction of candidates are thermodynamically competitive. Traditional structure-search strategies such as AIRSS and evolutionary crystal structure prediction can explore this space systematically <d-cite key="Pickard_2011"></d-cite><d-cite key="Oganov2006"></d-cite><d-cite key="Oganov2019"></d-cite>. In principle, first-principles calculations can assess whether a proposed material is meaningful. In practice, this quickly becomes too costly if the objective is broad exploration rather than detailed validation of a small set of structures.

{% include video.liquid path="assets/img/2026-04-15-crystalite/sun-generation.mp4" class="motion-video img-fluid rounded z-depth-1" controls=true autoplay=true loop=true muted=true caption="Stable, unique, and novel crystal structures discovered by Crystalite. This visualization gives a qualitative preview of the kinds of candidates behind the quantitative S.U.N. results discussed later in the post." %}

Generative models are therefore attractive because they can learn to propose candidate crystals directly from data. However, crystal generation is not simply a matter of predicting discrete atom labels. Atoms occupy positions inside a periodic lattice, distances wrap across unit-cell boundaries, and small geometric errors can change the physical plausibility of a structure. This is the main reason why much of the recent literature has relied on geometry-aware and often equivariant graph neural networks <d-cite key="xieCrystalDiffusionVariational"></d-cite><d-cite key="jiaoCrystalStructurePrediction2024"></d-cite><d-cite key="hoellmerOpenMaterialsGeneration2025"></d-cite><d-cite key="zeniGenerativeModelInorganic2025"></d-cite><d-cite key="joshi2025allatom"></d-cite>.

Crystalite begins from a narrower question: how much of this geometric structure must be built into the backbone itself? Put differently, can a diffusion Transformer remain competitive if the inductive bias is placed more carefully? Recent diffusion-transformer results suggest that lighter backbones can indeed be competitive when the representation and geometry are handled well <d-cite key="joshi2025allatom"></d-cite><d-cite key="yiCrystalDiTDiffusionTransformer2025a"></d-cite>.

<div class="note">
  In this post, we focus on two design choices that define Crystalite. The first is a chemically structured atom representation. The second is a lightweight way of injecting periodic geometry into attention without replacing the Transformer with a fully geometry-specific backbone.
</div>

<h2 id="equivariant-gnns-for-crystal-modeling">Equivariant GNNs for crystal modeling</h2>

A crystal is naturally described by three coupled objects: atom identities $ \mathbf{A} $, fractional coordinates $ \mathbf{F} $, and lattice geometry $ \mathbf{L} $:

$$
\mathcal{C} = (\mathbf{A}, \mathbf{F}, \mathbf{L}),
\qquad
\mathbf{A} \in \{0,1\}^{N \times N_Z},
\quad
\mathbf{F} \in [0,1)^{N \times 3},
\quad
\mathbf{L} \in \mathbb{R}^{3 \times 3}.
$$

An important observation is that fractional coordinates should not be regarded as Cartesian coordinates in another form. A row of $ \mathbf{F} $ specifies the position of an atom within the unit cell relative to the lattice basis, whereas $ \mathbf{L} $ specifies the lattice basis itself in real space. As a result, the same fractional arrangement can correspond to substantially different Cartesian structures under different lattices, and real-space positions are only well defined when $ \mathbf{F} $ and $ \mathbf{L} $ are considered jointly.

<div class="figure-narrow figure-nudge-left">
  {% include figure.liquid path="assets/img/2026-04-15-crystalite/frac-coords.png" class="img-fluid" caption="Fractional coordinates live inside the unit cell, while real-space positions are obtained only after the lattice basis is applied. This is why the coordinate channel and the lattice channel are inseparable in a crystal model." %}
</div>

This representation immediately explains why the problem is specialized. The unit cell is repeated periodically in all directions. Local environments matter, but only through the lattice that defines how the crystal repeats in real space. Geometric relations must be computed under the minimum-image convention rather than inside an isolated box.

Equivariant GNNs are well matched to this setting. They are designed to process neighborhoods, distances, directions, and symmetry transformations in a controlled way, which has enabled strong results in both crystal structure prediction and crystal generation. At the same time, incorporating equivariance in this way can make the resulting models architecturally complex and relatively slow at sampling time, when the denoiser must be evaluated repeatedly. Representative examples include CDVAE, DiffCSP, FlowMM, MatterGen, and OMatG <d-cite key="xieCrystalDiffusionVariational"></d-cite><d-cite key="jiaoCrystalStructurePrediction2024"></d-cite><d-cite key="millerFlowMMGeneratingMaterials2024"></d-cite><d-cite key="hoellmerOpenMaterialsGeneration2025"></d-cite><d-cite key="zeniGenerativeModelInorganic2025"></d-cite>.

Crystalite does not argue that geometry can be ignored. Rather, it asks whether geometry can be introduced in a more economical way. The model retains a standard diffusion Transformer backbone and concentrates its crystal-specific structure in the representation and the attention mechanism.

<h2 id="chemistry-based-atom-encoding">Chemistry-based atom encoding</h2>

The first modification concerns atom identity. In many crystal generators, atom types are represented by one-hot vectors. This is convenient, but chemically it is a poor geometry: every element is orthogonal to every other element. Sodium is no closer to lithium than it is to xenon.

This matters especially when diffusion is formulated over a continuous atom-type signal rather than with an explicitly discrete diffusion process. In that setting, one-hot atom identity provides no notion of chemical proximity: nearby vectors in representation space do not correspond to chemically similar elements, and chemically plausible substitutions are not encoded in the representation.

Crystalite replaces one-hot atom identity with **Subatomic Tokenization**, a compact descriptor built from periodic and electronic structure: period, group, block, and valence-shell occupancy. This gives the atom representation a more chemically meaningful structure.

<div class="l-page">
  <iframe
    id="periodic-table-frame"
    class="embed-frame"
    src="{{ 'assets/html/2026-04-15-crystalite/periodic-table.html' | relative_url }}"
    title="Interactive periodic table for Subatomic Tokenization"
    loading="lazy"
    scrolling="no"
  ></iframe>
</div>
<div class="embed-caption">
  Subatomic Tokenization is based on the idea that nearby chemistry should remain nearby in representation space. Here, the periodic table is useful because period, group, block, and valence structure directly inform the token design.
</div>
<script>
  (() => {
    const frameId = "periodic-table-frame";
    const resizeMessageType = "periodic-table:height";

    const clampFrameHeight = (height) => Math.max(420, Math.min(1600, Math.ceil(height) + 6));

    const bindPeriodicTableFrame = () => {
      const frame = document.getElementById(frameId);
      if (!frame) {
        return;
      }

      const applyFrameHeight = (nextHeight) => {
        if (!Number.isFinite(nextHeight) || nextHeight <= 0) {
          return;
        }
        const clampedHeight = clampFrameHeight(nextHeight);
        frame.style.height = `${clampedHeight}px`;
        frame.style.minHeight = `${clampedHeight}px`;
      };

      const measureFrameDocument = () => {
        try {
          const doc = frame.contentWindow && frame.contentWindow.document;
          if (!doc) {
            return;
          }
          const root = doc.documentElement;
          const body = doc.body;
          const pageShell = doc.querySelector(".page-shell");
          const shellHeight = pageShell
            ? Math.ceil(pageShell.getBoundingClientRect().height || 0)
            : 0;
          const documentHeight = Math.max(
            root ? root.scrollHeight : 0,
            root ? root.offsetHeight : 0,
            body ? body.scrollHeight : 0,
            body ? body.offsetHeight : 0
          );
          const nextHeight = shellHeight > 0 ? shellHeight : documentHeight;
          applyFrameHeight(nextHeight);
        } catch (error) {
          // Same-origin in the GRaM preview, but keep failures harmless.
        }
      };

      frame.addEventListener("load", () => {
        measureFrameDocument();
        window.setTimeout(measureFrameDocument, 180);
        window.setTimeout(measureFrameDocument, 800);
      });

      window.addEventListener("message", (event) => {
        if (event.source !== frame.contentWindow) {
          return;
        }
        const data = event.data || {};
        if (data.type !== resizeMessageType || !Number.isFinite(data.height)) {
          return;
        }
        applyFrameHeight(data.height);
      });
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", bindPeriodicTableFrame, { once: true });
    } else {
      bindPeriodicTableFrame();
    }
  })();
</script>

In a simplified form, the token for element $k$ can be written as

$$
\mathbf{h}_k =
\big[
\mathrm{onehot}(r_k),
\mathrm{onehot}(g_k),
\mathrm{onehot}(b_k),
s_k/2,\,
p_k/6,\,
d_k/10,\,
f_k/14
\big].
$$

These descriptors are standardized, balanced across feature groups, optionally PCA-compressed, and then treated as continuous atom tokens. During sampling, a predicted token is decoded back to a valid element by nearest-token matching. This makes the denoising problem better aligned with chemical structure: nearby errors in token space correspond more naturally to chemically related species.

$$
\mathcal{C} = (\mathbf{H}, \mathbf{F}, \mathbf{y}),
\qquad
\mathbf{H} \in \mathbb{R}^{N \times d_H},
\quad
\mathbf{F} \in [0,1)^{N \times 3},
\quad
\mathbf{y} \in \mathbb{R}^{6}.
$$

These are the three continuous channels denoised jointly by the model: atom tokens $ \mathbf{H} $, fractional coordinates $ \mathbf{F} $, and a compact latent description of the lattice $ \mathbf{y} $.

<h2 id="geometry-aware-attention-with-gem">Geometry-aware attention with GEM</h2>

The second key design choice concerns geometry. Crystals are periodic objects, so geometric quantities should respect the torus structure of fractional coordinates. The subtle point is that Crystalite uses two closely related constructions here: a wrapped fractional residual for the coordinate loss, and a metric-aware periodic-image search for the geometry-aware attention module introduced below.

$$
\boldsymbol{\delta}^{\mathrm{wrap}}_{ij}
=
\mathrm{wrap}(\mathbf{f}_i - \mathbf{f}_j),
\qquad
\mathrm{wrap}(\mathbf{u})
=
\mathbf{u} - \mathrm{round}(\mathbf{u}),
\qquad
d^{\mathrm{wrap}}_{ij}
=
\left\| \boldsymbol{\delta}^{\mathrm{wrap}}_{ij} \mathbf{L} \right\|_2.
$$

This wrapped residual is the right object for the coordinate loss because the model predicts fractional coordinates directly. For the geometry-aware attention bias, the goal is slightly different: we want the shortest periodic displacement under the lattice metric, not just a componentwise wrap. In practice, the model searches over a small set of nearby periodic images and keeps the one with the smallest real-space norm:

$$
\Delta \mathbf{f}_{ij}^{\star}
=
\arg\min_{\mathbf{r} \in \Omega_R}
\left\|
\left( \mathbf{f}_i - \mathbf{f}_j + \mathbf{r} \right)\mathbf{L}
\right\|_2.
$$

For orthogonal cells these two notions coincide, but for skewed lattices they do not have to. That is why it is useful to distinguish between wrapped fractional residuals in the loss and metric-aware minimum-image geometry in the attention module. The backbone itself is still intentionally simple: one token per atom, one additional global token for the lattice, and a standard diffusion-conditioned Transformer trunk.

$$
\mathbf{t}_i^{\mathrm{atom}} = E_H(\mathbf{h}_i) + E_F(\mathbf{f}_i),
\qquad
\mathbf{t}^{\mathrm{lat}} = E_{\mathrm{lat}}(\mathbf{y}).
$$

{% include figure.liquid path="assets/img/2026-04-15-crystalite/crystalite-architecture.png" class="img-fluid" caption="Crystalite keeps the architecture compact: one token per atom, one lattice token, a diffusion-conditioned Transformer trunk, and lightweight heads for the atom, coordinate, and lattice channels." %}

This gives the model a clean decomposition between local atomic information and global cell information. The lattice is parameterized through a lower-triangular latent, which keeps the representation unconstrained while ensuring positive diagonal entries:

$$
\mathbf{L}(\mathbf{y}) =
\begin{bmatrix}
e^{y_1} & 0 & 0 \\
y_2 & e^{y_3} & 0 \\
y_4 & y_5 & e^{y_6}
\end{bmatrix}.
$$

The geometry-specific piece of this backbone is the **Geometric Enhancement Module (GEM)**. GEM computes periodic pairwise features from atom positions and the lattice, then feeds them into attention as a bias rather than replacing the Transformer with a heavier message-passing architecture.

Concretely, GEM adds a geometry-dependent bias directly to the attention logits:

$$
A_{\mathrm{geom}}
=
\frac{QK^\top}{\sqrt{d}}
+
B_{\mathrm{geom}}.
$$

<div class="figure-narrow">
  {% include figure.liquid path="assets/img/2026-04-15-crystalite/gem-module.png" class="img-fluid" caption="GEM computes minimum-image pair geometry under periodic boundary conditions and converts it into an additive attention bias. Geometry therefore enters the model directly at the level of token interactions." %}
</div>

The bias term $B_{\mathrm{geom}}$ is built from periodic pairwise geometry: metric-aware minimum-image displacements, normalized distances, Fourier or radial features, and a compact lattice descriptor. The resulting attention is still standard self-attention, but it is informed by crystal geometry at the point where token interactions are decided.

This is the central modeling idea. Crystalite does not attempt to remove geometric inductive bias. Instead, it introduces a softer and more modular form of that bias inside the attention mechanism.

<h2 id="crystal-structure-prediction-and-de-novo-generation-results">Crystal structure prediction and de novo generation results</h2>

We evaluate Crystalite in two settings. In **crystal structure prediction (CSP)**, the composition is known and the model predicts the structure. In **de novo generation**, the model generates atom types, coordinates, and lattice jointly from noise.

<h3 id="crystal-structure-prediction">Crystal structure prediction</h3>

The CSP results are particularly direct. Across the reported benchmarks, Crystalite achieves the strongest results in both match rate and RMSE. The RMSE improvements are especially notable, because they suggest better geometric refinement rather than merely better recovery of the correct structural mode.

<div class="table-shell">
  <table aria-label="Crystal structure prediction results">
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th class="group" colspan="2">MP-20</th>
        <th class="group" colspan="2">MPTS-52</th>
        <th class="group" colspan="2">Alex-MP-20</th>
      </tr>
      <tr>
        <th>MR &uarr;</th>
        <th>RMSE &darr;</th>
        <th>MR &uarr;</th>
        <th>RMSE &darr;</th>
        <th>MR &uarr;</th>
        <th>RMSE &darr;</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>CDVAE</td>
        <td>33.90</td>
        <td>0.1045</td>
        <td>5.34</td>
        <td>0.2106</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>DiffCSP</td>
        <td>51.49</td>
        <td>0.0631</td>
        <td>12.19</td>
        <td>0.1786</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>FlowMM</td>
        <td>61.39</td>
        <td>0.0566</td>
        <td>17.54</td>
        <td>0.1726</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>CrystalFlow</td>
        <td>62.02</td>
        <td>0.0710</td>
        <td>22.71</td>
        <td>0.1548</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>KLDM</td>
        <td>65.83</td>
        <td>0.0517</td>
        <td>23.93</td>
        <td>0.1276</td>
        <td>--</td>
        <td>--</td>
      </tr>
      <tr>
        <td>OMatG</td>
        <td>63.75</td>
        <td>0.0720</td>
        <td>25.15</td>
        <td>0.1931</td>
        <td>64.71</td>
        <td>0.1251</td>
      </tr>
      <tr>
        <td><strong>Crystalite</strong></td>
        <td><strong>66.05</strong></td>
        <td><strong>0.0329</strong></td>
        <td><strong>31.49</strong></td>
        <td><strong>0.0701</strong></td>
        <td><strong>67.52</strong></td>
        <td><strong>0.0335</strong></td>
      </tr>
    </tbody>
  </table>
</div>
<div class="table-caption">
  CSP results across the three reported benchmarks. The strongest gains appear in RMSE, which is consistent with GEM primarily improving geometric refinement.
</div>

This interpretation matches the reported ablations: GEM has only a modest effect on match rate, but it consistently improves structural accuracy.

<h3 id="de-novo-generation">De novo generation</h3>

The <em>de novo</em> generation setting is more nuanced, because validity, novelty, uniqueness, and stability pull in different directions. For that reason, the most informative summary target is often **S.U.N.**, which rewards structures that are simultaneously stable, unique, and novel. On this benchmark, Crystalite achieves the strongest S.U.N. result while remaining substantially faster than competing baselines.

<div class="table-shell table-shell--wide">
  <table aria-label="De novo generation results">
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th class="group" colspan="5">Quality and Diversity</th>
        <th class="group" colspan="5">Stability, Distribution, and Speed</th>
      </tr>
      <tr>
        <th>Struct. Val. &uarr;</th>
        <th>Comp. Val. &uarr;</th>
        <th>Unique &uarr;</th>
        <th>Novel &uarr;</th>
        <th>U.N. &uarr;</th>
        <th>Stable &uarr;</th>
        <th>S.U.N. &uarr;</th>
        <th>wdist-&rho; &darr;</th>
        <th>wdist N-ary &darr;</th>
        <th>Time / 1k &darr;</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>FlowMM</td>
        <td>93.03</td>
        <td>83.15</td>
        <td>97.44</td>
        <td>85.00</td>
        <td>83.99</td>
        <td>46.05</td>
        <td>31.64</td>
        <td>1.389</td>
        <td><strong>0.075</strong></td>
        <td>1560</td>
      </tr>
      <tr>
        <td>CrystalDiT</td>
        <td>77.82</td>
        <td>67.28</td>
        <td>90.88</td>
        <td>59.33</td>
        <td>56.86</td>
        <td><strong>83.41</strong></td>
        <td>41.70</td>
        <td>0.202</td>
        <td>0.171</td>
        <td>73.72</td>
      </tr>
      <tr>
        <td>DiffCSP</td>
        <td><strong>99.93</strong></td>
        <td>82.10</td>
        <td>96.90</td>
        <td>89.53</td>
        <td>87.89</td>
        <td>50.28</td>
        <td>38.60</td>
        <td>0.192</td>
        <td>0.344</td>
        <td>237</td>
      </tr>
      <tr>
        <td>MatterGen</td>
        <td>99.78</td>
        <td>83.72</td>
        <td><strong>98.10</strong></td>
        <td><strong>91.14</strong></td>
        <td><strong>90.26</strong></td>
        <td>51.70</td>
        <td>42.29</td>
        <td>0.088</td>
        <td>0.184</td>
        <td>2639</td>
      </tr>
      <tr>
        <td>ADiT</td>
        <td>99.52</td>
        <td><strong>90.15</strong></td>
        <td>90.25</td>
        <td>59.80</td>
        <td>56.91</td>
        <td>76.90</td>
        <td>36.76</td>
        <td>0.231</td>
        <td>0.089</td>
        <td>84.81</td>
      </tr>
      <tr>
        <td><strong>Crystalite</strong></td>
        <td>99.61</td>
        <td>81.94</td>
        <td>95.33</td>
        <td>79.15</td>
        <td>77.12</td>
        <td>70.97</td>
        <td><strong>48.55</strong></td>
        <td><strong>0.046</strong></td>
        <td>0.125</td>
        <td><strong>22.36 / 5.14&dagger;</strong></td>
      </tr>
    </tbody>
  </table>
</div>
<div class="table-caption">
  Main <em>de novo</em> generation results on 10,000 generated crystals, with stability-related metrics estimated using NequIP. In this evaluation, Crystalite achieves the strongest S.U.N. result while also being the fastest sampler by a large margin. The daggered timing denotes an optimized inference setting with FlashAttention and bfloat16.
</div>

<h3 id="external-benchmarking-lemat-genbench">External benchmarking: LeMat GenBench</h3>

We also evaluate Crystalite on LeMat GenBench <d-cite key="betala2025lemat"></d-cite>, an external benchmark that is separate from the main evaluation above. This provides an additional check that the main conclusions are not specific to a single benchmark. The pre-relaxed and non-pre-relaxed groups should be read within their own categories. On this benchmark, Crystalite achieves state-of-the-art results on most metrics, and particularly on SUN and MSUN.

<div class="table-shell table-shell--xwide">
  <table aria-label="LeMat GenBench results">
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th>Valid</th>
        <th>Unique</th>
        <th>Novel</th>
        <th>Stable</th>
        <th>Metastable</th>
        <th>SUN</th>
        <th>MSUN</th>
        <th>E Above Hull</th>
        <th>Relax. RMSD</th>
      </tr>
      <tr>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(%) &uarr;</th>
        <th>(eV) &darr;</th>
        <th>(&Aring;) &darr;</th>
      </tr>
    </thead>
    <tbody>
      <tr class="section-row">
        <td colspan="10">Pre-Relaxed Models</td>
      </tr>
      <tr>
        <td><strong>Crystalite</strong></td>
        <td><strong>97.20</strong></td>
        <td><strong>95.80</strong></td>
        <td>53.20</td>
        <td><strong>12.70</strong></td>
        <td>51.60</td>
        <td><strong>1.50</strong></td>
        <td><strong>22.60</strong></td>
        <td>0.0905</td>
        <td>0.1320</td>
      </tr>
      <tr>
        <td>OMatG</td>
        <td>96.40</td>
        <td>95.20</td>
        <td>51.20</td>
        <td>11.60</td>
        <td>49.80</td>
        <td>1.00</td>
        <td>18.00</td>
        <td>0.0956</td>
        <td><strong>0.0759</strong></td>
      </tr>
      <tr>
        <td>MatterGen</td>
        <td>95.70</td>
        <td>95.10</td>
        <td><strong>70.50</strong></td>
        <td>2.00</td>
        <td>33.40</td>
        <td>0.20</td>
        <td>15.00</td>
        <td>0.1834</td>
        <td>0.3878</td>
      </tr>
      <tr>
        <td>PLaID++</td>
        <td>96.00</td>
        <td>77.80</td>
        <td>24.20</td>
        <td>12.40</td>
        <td><strong>60.70</strong></td>
        <td>1.00</td>
        <td>7.60</td>
        <td><strong>0.0854</strong></td>
        <td>0.1286</td>
      </tr>
      <tr>
        <td>WyFormer-DFT</td>
        <td>95.20</td>
        <td>95.00</td>
        <td>66.40</td>
        <td>3.70</td>
        <td>24.80</td>
        <td>0.40</td>
        <td>7.80</td>
        <td>0.2708</td>
        <td>0.4173</td>
      </tr>
      <tr>
        <td>WyFormer</td>
        <td>93.40</td>
        <td>93.00</td>
        <td>66.40</td>
        <td>0.50</td>
        <td>15.70</td>
        <td>0.10</td>
        <td>1.90</td>
        <td>0.4988</td>
        <td>0.8121</td>
      </tr>
      <tr class="section-row">
        <td colspan="10">Non-Pre-Relaxed Models</td>
      </tr>
      <tr>
        <td>DiffCSP</td>
        <td><strong>95.70</strong></td>
        <td>94.80</td>
        <td><strong>66.20</strong></td>
        <td><strong>2.30</strong></td>
        <td>29.80</td>
        <td>0.10</td>
        <td><strong>8.50</strong></td>
        <td><strong>0.2747</strong></td>
        <td><strong>0.3794</strong></td>
      </tr>
      <tr>
        <td>DiffCSP++</td>
        <td>95.30</td>
        <td><strong>95.10</strong></td>
        <td>62.00</td>
        <td>1.00</td>
        <td>26.40</td>
        <td><strong>0.20</strong></td>
        <td>5.00</td>
        <td>0.4093</td>
        <td>0.6933</td>
      </tr>
      <tr>
        <td>SymmCD</td>
        <td>73.40</td>
        <td>73.00</td>
        <td>47.00</td>
        <td>1.40</td>
        <td>18.60</td>
        <td>0.10</td>
        <td>2.40</td>
        <td>0.8761</td>
        <td>0.8720</td>
      </tr>
      <tr>
        <td>CrystalFormer</td>
        <td>69.90</td>
        <td>69.40</td>
        <td>31.80</td>
        <td>1.40</td>
        <td>28.80</td>
        <td>0.00</td>
        <td>3.10</td>
        <td>0.7039</td>
        <td>0.6585</td>
      </tr>
      <tr>
        <td>ADiT</td>
        <td>90.60</td>
        <td>87.80</td>
        <td>26.00</td>
        <td>0.40</td>
        <td><strong>36.50</strong></td>
        <td>0.00</td>
        <td>1.00</td>
        <td>0.3333</td>
        <td><strong>0.3794</strong></td>
      </tr>
      <tr>
        <td>Crystal-GFN</td>
        <td>51.70</td>
        <td>51.70</td>
        <td>51.70</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>0.00</td>
        <td>2.0858</td>
        <td>1.8665</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="table-caption">
  LeMat GenBench results, shown separately for pre-relaxed and non-pre-relaxed models. Within the pre-relaxed group, Crystalite leads on validity, uniqueness, stable rate, SUN, and MSUN.
</div>

<h3 id="large-scale-generation">Large-scale generation</h3>

{% include figure.liquid path="assets/img/2026-04-15-crystalite/large-scale-generation.png" class="img-fluid" caption="Some diversity metrics are properties of the generated set rather than a single sample. As the sampling budget grows, duplicates accumulate. Crystalite preserves uniqueness and unique-and-novel rate more effectively at scale." %}

A good generative model should not only produce plausible samples, but continue to generate diverse candidates as the sampling budget grows. Evaluating that behavior at scale is only practical when generation is fast enough to make such studies feasible in the first place. This is exactly where Crystalite is useful: it is fast enough for large-scale generation, while preserving uniqueness and unique-and-novel rate more effectively as more crystals are sampled.

<h2 id="balancing-efficiency-and-expressivity">Balancing efficiency and expressivity</h2>

Crystalite shows that strong crystal generation does not require pushing all of the geometric inductive bias into a heavy backbone. Instead, chemical structure can be encoded in the atom representation, while periodic geometry can be placed where it matters most: in the loss and in attention through GEM.

In practice, that gives a useful balance: Crystalite remains easier to train, sample from, and extend than many geometry-heavy alternatives while still reaching state-of-the-art crystal structure prediction results and strong <em>de novo</em> generation performance. In the main generation comparison, it also achieves the highest S.U.N. score while sampling one to two orders of magnitude faster than leading GNN-based baselines, depending on the reference model and inference setting.

That speed matters for more than runtime alone. Fast sampling makes conditional generation, guidance, and steering much easier to use in practice, because adding search loops or external constraints no longer multiplies an already expensive denoising cost. In that sense, Crystalite is not only an efficient generator, but also a more practical foundation for controllable and scalable materials discovery workflows.
