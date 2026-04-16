---
layout: distill
title: To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking
description: Many popular ML datasets are heavily canonicalized — objects almost always appear in the same orientation. We measure this with a simple classifier test, showing theoretically that canonicalization can cause data augmentation to hurt performance. We give practitioners a flowchart for diagnosing their own datasets.
date: 2026-04-11
future: true
htmlwidgets: true
# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Elyssa Hofgard

# must be the exact same name as your blogpost
bibliography: 2026-04-14-ToAugmentOrNot.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Distributional Symmetry Breaking
  - name: Proposed Metric
  - name: Theory
  - name: Experiments
  - name: Hypotheses for Empirical Behavior
  - name: Conclusion
  - name: References

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  .callout {
    background: #f0f4ff;
    border-left: 4px solid #4a6fa5;
    padding: 12px 18px;
    margin: 1.5rem 0;
    font-style: italic;
    border-radius: 0 4px 4px 0;
  }
  .dataset-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9em;
  }
  .dataset-table th {
    background: #f0f4ff;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #4a6fa5;
  }
  .dataset-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #e0e0e0;
    vertical-align: top;
  }
  .dataset-table tr:last-child td {
    border-bottom: none;
  }
  .regime-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9em;
  }
  .regime-table th {
    background: #f0f4ff;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #4a6fa5;
  }
  .regime-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #e0e0e0;
  }
  .helpful { color: #2a7a2a; font-weight: 500; }
  .harmful { color: #b00000; font-weight: 500; }
---

## Introduction

For a group transformation $$g \in G$$ (e.g. a rotation or permutation), a model is **equivariant** if $$f(gx) = g f(x)$$ — rotating the input rotates the output — and **invariant** if $$f(gx) = f(x)$$ — the output is unchanged. Equivariant architectures enforce this by design; data augmentation encourages it by randomly applying $$g$$ to training inputs. Equivariant models have had successes in multiple domanains - materials science <d-cite key="liao2023equiformerv2"></d-cite>, robotics <d-cite key="wang2024equivariant"></d-cite>, drug discovery <d-cite key="igashov2024equivariant"></d-cite>, fluid dynamics <d-cite key="wangincorporating"></d-cite>, computer vision <d-cite key="esteves2019equivariant"></d-cite>, and beyond. 

Both approaches rely on the assumption that the ground truth function $$f$$ is equivariant. However, there is often an implicit assumption that the data distribution itself is symmetric, i.e. $$p(x) \approx p(gx)$$. We refer to violations of this assumption as **distributional symmetry breaking**. In this work, we study distributional symmetry breaking. Our main contributions are:

- **A classifier-based diagnostic:** We introduce a simple two-sample test to measure distributional symmetry breaking.

- **Theoretical analysis:** We show that data augmentation can harm performance under certain distributional conditions.

- **Empirical studies:** We demonstrate that widely used 3D datasets are strongly canonicalized. We correspondingly evaluate the impacts of equivariant methods on datasets across domains and propose hypotheses for differing behaviors.

## Distributional Symmetry Breaking

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/orientations.png" class="img-fluid" %}
<div class="caption">
    Examples of distributional symmetry breaking. Left: Baseballs appear in all orientations equally — the distribution is symmetric. Middle-left: Coffee mugs have a preferred orientation (handle to the side), illustrating a mild symmetry bias. Middle-right: Canonicalization maps arbitrary orientations to a single canonical one — the strongest form of symmetry breaking. Right: Inherent canonicalization (digits 6/9) and user-defined canonicalization (crystal lattice), where orientation is either intrinsically meaningful or fixed by convention.
</div>
Distributional symmetry breaking may lead equivariant methods or data augmentation to discard useful information. For example, classifying 6s and 9s in MNIST is easy when the digits appear in their natural orientation, but it becomes harder under rotational augmentation.

## Proposed Metric

**The goal:** define a metric $$m(p_X)$$ that quantifies how far a data distribution $$p_X$$ is from being group-symmetric — without assuming symmetry in the first place.

The key reference is the **symmetrized density** $$\bar{p}_X(x) := \int_{g \in G} p_X(gx)\, dg$$, the closest group-invariant distribution to $$p_X$$. Measuring $$m(p_X)$$ reduces to measuring how distinguishable $$p_X$$ and $$\bar{p}_X$$ are from finite samples.

**Why not MMD?** A natural approach is Maximum Mean Discrepancy with a chosen kernel. But there is no universal kernel — for geometric graph datasets like molecular structures, choosing a kernel that captures chemical information is non-trivial, and the resulting values are not directly interpretable.

**Our approach: a two-sample classifier test.** We train a small neural network to distinguish samples from $$p_X$$ (original) versus $$\bar{p}_X$$ (randomly transformed), and use **test accuracy** as the metric:

$$
m(p_X) := \mathbb{E}_{(x,c) \in D^*_{\text{test}}} \left[ \mathbf{1}\big(\text{NN}(x) = c\big) \right]
$$

**Algorithm:**
1. Split the dataset in half.
2. Apply random $$g \sim G$$ to one half — these approximate $$\bar{p}_X$$ (label 1).
3. Keep the other half unchanged as $$p_X$$ (label 0).
4. Train a binary classifier and report **test accuracy** as $$m(p_X)$$.

**Interpretation:**
- $$m(p_X) \approx 0.5$$: data is (approximately) group-invariant — the classifier can't do better than chance.
- $$m(p_X) \approx 1$$: data is strongly canonicalized — the classifier easily detects the preferred orientations.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;">

  <figure style="flex: 1; text-align: center;">
    {% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/dataset_vis.png" class="img-fluid" %}
    <figcaption>Visualizations of unrotated samples from several materials datasets, with their canonicalization visible.</figcaption>
  </figure>

  <figure style="flex: 1; text-align: center;">
    {% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/classifer_setup1.png" class="img-fluid" %}
    <figcaption>A classifier test for determining if a sample is from the original dataset, or rotated.</figcaption>
  </figure>

</div>

### Task-Dependent Metric

$$m(p_X)$$ tells us *whether* the data breaks symmetry — but not *whether that matters for the specific task*. Consider MNIST digits 6 and 9: they are canonicalized in a way that is directly predictive of the label. Augmenting away orientation destroys this useful signal.

We introduce $$t(p_{X,Y})$$, a metric of **task-useful** distributional symmetry breaking. Let $$c\colon\mathcal{X} \rightarrow G$$ be a canonicalization function (implemented as a randomly initialized, untrained equivariant network). Since data augmentation destroys information in $$c(x)$$, we ask: how much does $$c(x)$$ predict the label $$f(x)$$?

We compare:
- $$\mathcal{L}(c(x) \to f(x))$$: loss when predicting labels from orientations (canonical orientation intact)
- $$\mathcal{L}_{\text{rot}} = \mathcal{L}(c(gx) \to f(gx),\, g \sim G)$$: same, but with random rotations applied — removing orientation information

$$
t(p_{X,Y}) := \frac{\mathcal{L}_{\text{rot}}}{\mathcal{L}}
$$

- $$t \gg 1$$: orientations carry task-relevant signal — augmentation likely hurts by discarding it.
- $$t \approx 1$$: orientation is not predictive — augmentation is likely safe or beneficial.

## Theory

To understand when augmentation can backfire, we analyze a tractable setting: high-dimensional ridge regression where the true function is invariant, but the data distribution may not be. Our theory shows that **even when the ground-truth function is invariant, data augmentation and test-time symmetrization can be harmful when invariant and non-invariant features are strongly correlated.**

**Takeaway:** Symmetry enforcement can hurt by discarding signal that, while technically non-invariant, is informative about the label via its correlation with invariant features.

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/theory_fig.png" class="img-fluid" %}
<div class="caption">
    Summary of theoretical results. Rows: weak vs. strong correlation between invariant and non-invariant features. Columns: under- vs. over-parameterized regime. Augmentation can hurt in the over-parameterized + strong correlation setting, where non-invariant features act as proxies for invariant ones.
</div>

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/data_aug_risk.png" class="img-fluid" %}
<div class="caption">
    Left three panels: Excess risk, bias, and variance of vanilla (blue) vs. augmented (orange) models as feature correlation strength \(\sigma_w\) varies. At low \(\sigma_w\) (strongly correlated features, high symmetry breaking), augmentation increases excess risk — driven by higher variance. Right: Corresponding \(m(p_X)\) values, which are highest at low \(\sigma_w\), confirming the metric tracks symmetry breaking accurately.
</div>

We thus find invariance interacts with data geometry and high-dimensional statistics in subtle ways. When the data distribution breaks symmetry — even slightly — enforcing invariance can destroy useful signal. And in modern over-parameterized regimes, reducing effective dimension can itself introduce instability.

<div class="callout">
  Invariance is not just a property of the target function — it’s also a property of the data distribution. If those two don’t align, enforcing symmetry can hurt.
</div>

This raises a puzzle: if datasets like QM9 are strongly canonicalized, our theory predicts augmentation should hurt — yet empirically it helps. We investigate this discrepancy below.

## Experiments

### How canonicalized are real datasets?

We measure $$m(p_X)$$ on multiple benchmark datasets. Strikingly, many widely-used benchmarks are strongly canonicalized — particularly molecular and materials science datasets — even though equivariant methods are routinely applied to them under the assumption of distributional symmetry.

**Takeaway:** Real-world datasets often deviate substantially from the symmetry assumptions built into equivariant models.

### Does canonicalization predict whether augmentation helps?

We evaluate equivariant, group-averaged, and stochastic group-averaged models on each dataset. The results are summarized below.

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/model_table.png" class="img-fluid" %}
<div class="caption">
    Comparison of train/test augmentation, group-averaged, and equivariant models across datasets. Augmentation: TT = train+test, TF = train only, FT = test only, FF = none. MNIST uses a group-averaged model; other datasets use stochastic group-averaging. MAE is reported for QM7b/QM9; equivariant baselines from e3nn. Best overall in bold, best within augmentation underlined. CNN used for MNIST, graph transformer for point clouds.
</div>

<table class="dataset-table">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Symmetry</th>
      <th>\(m(p_X)\)</th>
      <th>Effect of Augmentation / Equivariance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MNIST</strong></td>
      <td>\(C_4\) rotations</td>
      <td>High</td>
      <td><span class="harmful">Minimal benefit; slight harm</span></td>
    </tr>
    <tr>
      <td><strong>ModelNet40</strong></td>
      <td>SO(3) rotations</td>
      <td>High (class-specific)</td>
      <td><span class="harmful">Reduces performance</span></td>
    </tr>
    <tr>
      <td><strong>QM9</strong></td>
      <td>SO(3) rotations</td>
      <td>High (CORINA preprocessing <d-cite key="qm9"></d-cite>)</td>
      <td><span class="helpful">Improves nearly all properties</span></td>
    </tr>
    <tr>
      <td><strong>QM7b</strong></td>
      <td>SO(3) rotations</td>
      <td>High</td>
      <td><span class="helpful">Beneficial, especially for non-scalar properties (e.g. dipole) <d-cite key="qm7original1"></d-cite><d-cite key="qm7byang2019quantum"></d-cite></span></td>
    </tr>
  </tbody>
</table>

While one might expect augmentation to consistently hurt on canonicalized datasets, molecular datasets (QM7b and QM9) defy this picture.

**Takeaway:** Whether symmetry methods help or hurt is dataset-dependent — canonicalization alone does not predict the outcome.

### Additional Materials Science Datasets

We extend the analysis to additional materials science datasets:

- **rMD17** <d-cite key="rMD17"></d-cite>: Molecular dynamics trajectories for small molecules. The degree of distributional symmetry breaking varies widely between molecules — reflecting differences in initial conditions and molecular geometry.
- **OC20** <d-cite key="ocp_dataset"></d-cite>: Adsorbates on periodic crystalline catalysts. Both the adsorbate alone and the adsorbate+catalyst system are highly canonicalized, likely due to the catalyst surface being aligned with the $$xy$$ plane.
- **LLM crystal dataset** <d-cite key="gruver2024finetuned"></d-cite>: Crystals serialized to text for LLM generation. Atoms must be listed in some order — the authors independently noted that permutation augmentations hurt generative performance. We trained a classifier head on a pretrained DistilBERT transformer to test this, and found $$m(p_X) = 95\%$$ — strong permutation canonicalization due to conventions in atom ordering.

**Takeaway:** Distributional symmetry breaking is widespread across data types and modalities — not just 3D point clouds.

## Hypotheses for Empirical Behavior
We present hypotheses for the differing performance of equivariant models/data augmentation across datasets.

### Task Dependent Metric

We apply $$t(p_{X,Y})$$ to each dataset to ask: does the canonical orientation actually carry task-relevant information?

- **QM7b dipole (artificial canonicalization):** Molecules are aligned so their dipole moments point along the $$z$$ axis — orientation is directly predictive of the label. A non-equivariant model can exploit this; an equivariant one cannot. As expected: no-augmentation (FF) outperforms the equivariant baseline, and $$t \gg 1$$.

- **ModelNet40:** Same story — equivariance hurts and $$t$$ is large. Object identity is correlated with how the object is canonically oriented.

- **QM9:** $$t \approx 1$$ — canonical orientation carries little task-relevant information globally. Equivariance does destroy orientation, but since that orientation wasn't predictive to begin with, nothing useful is lost. This is consistent with equivariance helping.

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/task_dependent_results.png" class="img-fluid" %}
<div class="caption">
    Task-dependent metric \(t(p_{X,Y})\): Accuracy (ModelNet) or MAE (QM7b/QM9) when predicting labels from the canonicalization vs. a random-rotation baseline. Values averaged over five seeds. Higher ratio = more task-relevant signal in the orientation.
</div>

**Takeaway:** $$t$$ is large for datasets where equivariance hurts, and small where it helps — it tells us whether augmentation discards task-relevant signal.

But $$t$$ only explains why equivariance *doesn't hurt* on QM9 — not why it actively *improves* performance. For that, we need to look at local structure.

### Locality

Equivariant models compute locally equivariant features over receptive fields <d-cite key="musaelian2023local"></d-cite> — their benefit may come from capturing symmetry in small recurring motifs rather than enforcing global symmetry <d-cite key="du2022se3"></d-cite><d-cite key="lippmann2025beyond"></d-cite>. We test this by applying $$m(p_X)$$ at both global and local scales.

- **QM9:** Local bond neighborhoods have much lower $$m(p_X)$$ than the full molecule — local structure is close to isotropic even when the global dataset is canonicalized. Equivariant models can exploit this local symmetry.
- **ModelNet40:** Local neighborhoods also show lower $$m(p_X)$$ at small sizes, but canonical alignment re-emerges as neighborhoods grow — the canonical orientation is object-level and tightly coupled to the task.

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/locality_exp.png" class="img-fluid" %}
<div class="caption">
    Left: Local QM9 neighborhoods and their \(m(p_X)\) values. Right: Local ModelNet40 results as neighborhood size grows.
</div>

<div class="callout">
  The relevant question is not simply whether a dataset is canonicalized, but at what scale canonicalization interacts with the task.
</div>

Based on these findings, we provide the following flowchart as a practical guide for deciding whether to use equivariant methods or data augmentation on a new dataset:

{% include figure.liquid path="assets/img/2026-04-14-ToAugmentOrNot/symm_breaking_flowchart.png" class="img-fluid" %}
<div class="caption">
    Advice for practitioners on using our metric for model selection.
</div>

## Conclusion

We provide interpretable metrics for diagnosing distributional symmetry breaking — no domain knowledge required. Every benchmark dataset we tested showed a high degree of symmetry-breaking, yet augmentation only hurt performance on ModelNet40.

Three implications stand out. First, non-equivariant models evaluated only on in-distribution data may appear accurate but fail under transformations — assessing whether this matters requires domain expertise (see the flowchart above). Second, data augmentation is often treated as universally beneficial for invariant tasks, but we show it can hurt. Third, if already-canonicalized molecular datasets still benefit from equivariance, equivariant models must provide some **additional**, possibly **domain-specific** benefit beyond global symmetry enforcement — a compelling open question.
