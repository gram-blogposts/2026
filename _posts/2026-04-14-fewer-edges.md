---
layout: distill
title: "Fewer Edges, Faster Protein Graph Learning"
description: "Protein graphs should not be constructed blindly based on spatial proximity: they must reflect directed, geometrically viable chemistry. We introduce Angle Rewiring, a biologically motivated edge criterion. Paired with a FiLM-based reformulation of IEConv that reduces memory usage, we explore the relationship between topological sparsity, computational efficiency, and geometric expressiveness across Enzyme Commission, Gene Ontology, and Fold3D benchmarks."
date: 2026-04-14
future: true
htmlwidgets: true

authors:
  - name: Pau Hidalgo

bibliography: 2026-04-14-fewer-edges.bib

toc:
  - name: Introduction
  - name: Related Work
  - name: The Geometry of Proteins
    subsections:
      - name: The Isotropic Distance Relations
      - name: Angle Rewiring
      - name: Filtering the Graph (The Variants)
  - name: Architectural Efficiency
    subsections:
      - name: The Edge-Memory Bottleneck
      - name: Efficient IEConv via FiLM
  - name: Experimental Results
    subsections:
      - name: 1. Enzyme Commission (EC) Number
      - name: 2. Gene Ontology (GO) Prediction
      - name: 3. Structural Embedding (Fold3D)
  - name: Global Pareto Optimality & Analysis
    subsections:
      - name: Topology vs. Learned Edge Features
      - name: Training Acceleration and Memory Optimization
  - name: Limitations and Future Work
  - name: Conclusion

_styles: >
  .results-table {
    font-size: 0.9em;
    margin: 1.5em auto;
    width: 100%;
    border-collapse: collapse;
  }
  .results-table th {
    background: #f8fafc;
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 2px solid #e2e8f0;
    text-align: left;
  }
  .results-table td {
    padding: 8px 14px;
    border-bottom: 1px solid #f1f5f9;
  }
  .results-table .best {
    font-weight: 700;
    color: #059669;
  }
  .results-table .paper-baseline {
    color: #64748b;
    font-style: italic;
  }
  .caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: center;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
    line-height: 1.5;
  }

  .table-caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: left;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.5;
    font-weight: 400;
  }
---

## Introduction

Graph Neural Networks (GNNs) have emerged as a powerful paradigm for learning over protein structures. By representing amino acid residues as nodes and their spatial relationships as edges, models like GearNet <d-cite key="zhang2023protein"></d-cite> can capture interactions and learn rich structural representations. These models have driven significant progress on tasks ranging from enzyme function prediction to fold classification.

Yet despite rapid advances in message-passing (MP) architectures that operate over protein graphs, the methods used to construct those initial graphs have remained naive. Most models rely on isotropic distance thresholding: any two $C_\alpha$ atoms within 10 Å are connected by an edge, regardless of whether their side-chains could plausibly interact. This produces dense and noisy graphs in which a substantial fraction of edges represent contacts that are not biologically grounded.

This has two important consequences. First, MP layers must allocate computation to edges that carry no meaningful biological signal. Second, highly expressive architectures like Intrinsic-Extrinsic Convolutions (IEConv) <d-cite key="hermosilla2021ieconv"></d-cite>, which generate per-edge transformation matrices, face catastrophic memory scaling as the number of edges grows (sometimes requiring 40+ GB of GPU memory).

This work addresses both problems. We make the following complementary contributions:

1. **Angle Rewiring**: a biologically motivated graph construction method that filters edges based on side-chain angular alignment, reducing edge counts by up to 40% while preserving (and often improving) functional information content.
   
2. **Efficient IEConv**: a reformulation of IEConv that replaces per-edge $D \times D$ transformation matrices with dimension-wise scale and shift operations, similar to FiLM <d-cite key="perez2018film"></d-cite>, reducing peak VRAM from ~53 GB to ~4 GB and achieving competitive or improved accuracy.


We make a systematic empirical study across Enzyme Commission (EC), Gene Ontology (GO) and Fold3D benchmarks, performing graph construction ablations and showing how graph topology, edge expressiveness and computational cost interact in protein structure learning.

Our two changes expose a consistent pattern: biologically grounded sparse graphs can match or improve performance while substantially reducing memory and training cost.

---

## Related Work

**Protein graph construction.** Prior work on molecular graphs has generally treated connectivity as a preprocessing step rather than a modeling choice. EGNN <d-cite key="satorras2021egnn"></d-cite> and SchNet <d-cite key="schutt2017schnet"></d-cite> use isotropic cutoffs in the small-molecule domain; GearNet <d-cite key="zhang2023protein"></d-cite> extends this to proteins with multi-relational radius and K-Nearest-Neighbor (KNN) edges. DimeNet <d-cite key="gasteiger2020directional"></d-cite> and SphereNet <d-cite key="liu2022spherical"></d-cite> incorporate angular information as **edge features** in order to capture directional interactions. The TopoBench Protein Lifting <d-cite key="telyatnikov2025topobench"></d-cite> also uses KNN, and adds the direction between $C_{\alpha}$ and $C_{\beta}$ as an edge feature. They all rely on distance criteria to determine which edges exist at all.

 Our approach is different: we encode angular information as topology, not just as features.

**Efficient edge convolutions.** There has been some work on scalable geometric message passing. EdgeConv <d-cite key="wang2019dynamic"></d-cite> uses shared-weight edge MLPs without per-edge matrix prediction. PAENet and related methods <d-cite key="schutt2021equivariant"></d-cite> use factored tensor products to reduce equivariant convolution cost. Our FiLM-based formulation draws most directly on work in conditional generation: AdaLN layers in diffusion transformers <d-cite key="peebles2023scalable"></d-cite> and hypernetworks <d-cite key="ha2017hypernetworks"></d-cite>, applying the same scale and shift modulation principle to per-edge geometric conditioning.

**Domain-informed protein GNNs.** SSProNet <d-cite key="mouhajir2026sspronet"></d-cite> and SCHull <d-cite key="wang2025a"></d-cite> demonstrate that secondary structure constraints embedded in graph topology improve performance in fold classification. ProNet <d-cite key="wang2022pronet"></d-cite> achieves state-of-the-art results through multi-scale representations that separately model residue, backbone, and side-chain geometry. 
Our approach is complementary: Angle Rewiring improves any backbone GNN architecture by filtering the input graph, and could in principle be combined with multi-scale or equivariant architectures.


---

## The Geometry of Proteins

### The Isotropic Distance Relations

Standard protein graph construction pipelines combine several types of edges:

- **Sequential edges** connect residues that are adjacent (or near-adjacent) in primary sequence, i.e., $i \pm 1$, $i \pm 2$, and self-loops ($i$ to itself).
- **KNN edges** link each residue to its $k$-nearest spatial neighbors in 3D space.
- **Radius edges** connect all $C_{\alpha}$ pairs within a fixed distance cutoff (typically 10 Å).


In the standard GearNet construction, these edges are directed and multi-relational. On the EC dataset, the resulting graphs contain, on average, 5990 edges between 298 nodes. Of these, ~3300 come from spatial distance (Radius edges) and ~1200 from KNN. 


### Angle Rewiring

Prior geometric GNNs often treat angular information and directionality as something to be learned after the graph is already fixed, incorporated as features. We take the opposite view: directionality should help determine the graph itself. In proteins, side-chain orientation provides a simple and biologically meaningful proxy for whether a residue pair is positioned to communicate through message passing.

To encode angle information directly into the graph topology, we define a side-chain direction vector $\hat{d}$ for each residue $i$ as the unit vector from the backbone $C_{\alpha_{i}}$ toward the lateral $C_{\beta_{i}}$ carbon. For glycine, which lacks a $C_{\beta}$, we compute a virtual $C_{\beta}$ position using standard idealized geometry. This vector serves as a proxy for the direction in which the residue's functional group is oriented.

For a candidate edge from node $i$ to node $j$, we compute the displacement unit vector $\hat{r}_{ij}$. We include this edge under the **Angle** criterion if and only if both residues are mutually oriented toward each other:

$$
\angle(\hat{d}_i, \hat{r}_{ij}) \leq \theta_{\text{cut}} \quad \text{and} \quad \angle(\hat{d}_j, \hat{r}_{ji}) \leq \theta_{\text{cut}}
$$

This bilateral condition keeps an edge only when each residue lies within the forward-facing cone defined by the other residue's side-chain direction. Figure 1 provides a visual explanation.


{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/angle_intuition.png" class="img-fluid" %}
<div class="caption">
  Figure 1. Geometric intuition for Angle Rewiring. <em>Left:</em> two residues whose side-chains face each other (small angles). These satisfy the bilateral angle condition and are connected by an angle edge. <em>Right:</em> two nearby residues whose side-chains point away from each other, and are excluded. Isotropic radius edges (dashed) would connect both pairs indiscriminately, introducing noise in the right-hand case.
</div>


**Choosing the cutoff.** We swept $\theta_{\text{cut}}$ over the range 60º–150º and evaluated EC classification performance. As shown in Figure 2, the optimal range is $\theta_{\text{cut}} \in [90º, 105º]$, which requires each side-chain to lie within the forward-facing hemisphere of the other. This strikes the right balance: strict enough to eliminate directionally incompatible contacts, lenient enough to retain contacts between flexible side-chains that deviate slightly from ideal geometry.

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/fig1_angle_cutoff_sweep.png" class="img-fluid" %}
<div class="caption">
  Figure 2. Hyperparameter sweep over the angle cutoff θ<sub>cut</sub> on the EC validation set (F1-max and AUPRC). Both metrics peak in the 90º–105º range, indicating that requiring side-chains to point toward each other rather than away is the most informative filtering criterion. Beyond 120º, performance declines as increasingly non-directional contacts are admitted.
</div>

We select 90º as the default cutoff because it sits near the empirical optimum while remaining easy to interpret: each residue must lie roughly in the other's forward-facing hemisphere. The results from this experiment also show that as more edges are introduced, the performance degrades, which further supports Angle Rewiring.




### Filtering the Graph (The Variants)

We evaluate six graph construction strategies that decompose the original and span a spectrum of sparsity and connectivity assumptions:

1. **Baseline**: Radius ($\leq 10$ Å) + KNN ($k=10$) + Sequential (the original GearNet construction).
2. **Sequential**: Sequential edges to distance 2 (5 relation types).
3. **KNN Only**: Sequential + $k=10$ nearest neighbors (minimum sequential distance 5).
4. **Radius Only**: Sequential + spatial radius $\leq 10$ Å (minimum sequential distance 5).
5. **Angle**: Sequential + Angle condition ($\theta_{\text{cut}} = 90°$).
6. **Angle + KNN**: Sequential + Angle + KNN.

Additionally, following the original GearNet graph construction, Radius, KNN, and Angle edges are only created for residues with a sequential distance of at least 5. To keep the number of edges from exploding, the Angle variant also limits the maximum number of connections to 10, prioritizing those with smaller angles.

<div class="table-caption">
  Table 1. Structural statistics of graph construction variants. Average edge counts calculated across the Enzyme Commission (EC) dataset.
</div>

|            | Sequence | Radius | KNN | Angle | Avg. # Edges (EC) |
|------------|----------|--------|-----|-------|-------------------|
| Baseline   | ✓        | ✓      | ✓   |       | 5990.5            |
| Sequential | ✓        |        |     |       | 1481.5            |
| Radius     | ✓        | ✓      |     |       | 4793.3            |
| KNN        | ✓        |        | ✓   |       | 2678.7            |
| Angle      | ✓        |        |     | ✓     | 2536.4            |
| Angle+KNN  | ✓        |        | ✓   | ✓     | 3733.6            |


The edge statistics across the EC test set tell a clear story. The **Baseline** produces ~5,990 edges per protein on average. Pure **Angle** filtering reduces this to ~2,536 edges (a 58% reduction). The **Angle + KNN** variant, which restores connectivity via nearest-neighbor safety nets for residues that would otherwise become too isolated, lands at ~3,733 edges, a **38% reduction** relative to baseline while maintaining a connected graph. This is the topology we recommend as the default: it captures directionally viable contacts while ensuring that every residue participates in sufficient message-passing.

An important implementation detail is that GearNet uses a Relational GCN architecture, which has a weight matrix for every relation type. The Baseline and Angle + KNN constructions create 7 different edge types, while Angle, Radius and KNN use 6, and Sequential uses 5. This modifies the internal number of parameters of the model.

---

## Architectural Efficiency

Sparse graphs reduce the number of edges that a model must process. But the computational cost per edge also matters, and for Edge and IEConv-based models, it is the dominant bottleneck. We propose a solution for the Intrinsic-Extrinsic Convolution variant.

### The Edge-Memory Bottleneck

Intrinsic-Extrinsic Convolutions (IEConv) <d-cite key="hermosilla2021ieconv"></d-cite> are among the most geometrically expressive protein graph convolutions available. Rather than applying a shared weight matrix to all edges, IEConv uses a kernel to generate a distinct $D \times D$ transformation matrix for each edge, conditioned on continuous geometric features (inter-residue distances, dihedral angles, and local reference frame orientations):

$$
\mathbf{m}_{ij} = K_{ij} \cdot \mathbf{h}_j, \quad K_{ij} \in \mathbb{R}^{D \times D}
$$

where $K_{ij}$ is the output of a small MLP applied to the edge's geometric features. This design allows the network to apply qualitatively different transformations depending on the geometry of each contact, which is particularly expressive for protein structure.

The problem is its memory usage. Using a batch size of 8 (as in GearNet), for a feature dimension of $D = 128$ and a graph with ~6,000 edges, storing all $K_{ij}$ matrices simultaneously requires holding roughly $6000 \times 8 \times 129 \times 128 \times 4$ bytes: $\approx$ 3 GB of edge-level activations *per layer*, for just the forward pass activations. In practice, training GearNet with the original IEConv formulation on Enzyme Commission requires **47–53 GB of GPU VRAM**, placing it firmly out of reach for most research labs and effectively ruling out batch sizes larger than 1 or 2.

### Efficient IEConv via FiLM

Our empirical results suggest that a full $D \times D$ matrix is far more expressive than the training signal requires. Most of the expressiveness appears unnecessary relative to its memory cost.
Our aim is to apply **edge-conditioned, feature-wise modulation** without materializing dense per-edge matrices.

Feature-wise Linear Modulation (FiLM) <d-cite key="perez2018film"></d-cite> offers exactly this. Originally developed for conditioning visual reasoning networks on language, FiLM operates by predicting a scale $\gamma$ and shift $\beta$ vector per feature dimension, conditioned on auxiliary context. The same principle has proven highly effective in a range of conditional generation settings: AdaLN layers in diffusion transformers <d-cite key="peebles2023scalable"></d-cite> use it to condition on timestep and class embeddings, and hypernetwork-based approaches <d-cite key="ha2017hypernetworks"></d-cite> use related ideas to generate lightweight parameter modulations from compact codes.

We apply this principle to edge convolution. Instead of predicting a $D \times D$ matrix per edge, our **Efficient IEConv** uses the same geometric feature MLP to predict two $D$-dimensional vectors per edge: a multiplicative scale $\gamma_{ij}$ and an additive shift $\beta_{ij}$.

$$
\mathbf{m}_{ij} = \gamma_{ij} \odot \mathbf{h}_j + \beta_{ij}
$$

This is a dimension-wise affine transformation of the neighbor's feature vector, modulated by the local geometry. The prediction space shrinks from $O(D^2)$ to $O(D\times2)$, a reduction proportional to the feature dimension, while the model retains full conditioning for each edge on continuous geometric attributes. The FiLM formulation can also be interpreted as a form of soft edge gating: $\gamma_{ij}$ controls how much of each feature dimension passes through, while $\beta_{ij}$ applies a geometry-dependent offset. This is closely related to the gating mechanisms used in equivariant networks <d-cite key="schutt2021equivariant"></d-cite> and dynamic graph convolutions <d-cite key="wang2019dynamic"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/memory_comparison.png" class="img-fluid" %}
<div class="caption">
  Figure 3. GPU memory comparison between original GearNet-IEConv and GearNet-Efficient IEConv across batch sizes and feature dimensions. Original IEConv requires 47–53 GB of VRAM under standard training conditions, restricting use to high-end multi-GPU servers. GearNet-Efficient IEConv reduces this to ~4 GB, enabling training on a single consumer GPU.
</div>

The memory savings are substantial: VRAM drops from ~52.8 GB to ~4.3 GB, which represents a **12× reduction**, making protein structure modeling accessible on a single consumer-grade GPU. But the benefits go beyond memory. The FiLM parameterization acts as an implicit regularizer, preventing the model from overfitting to the dense geometric features that the full rank $K_{ij}$ matrix could memorize. In practice, **GearNet-Efficient IEConv** reaches similar or improved accuracies across all benchmarks tested.

This efficiency is not just a convenience. With 12× less memory per run, one can train larger models, use larger batch sizes, or most significantly, **scale pretraining to larger protein databases** (e.g., AlphaFold DB's 200M+ predicted structures) that were previously inaccessible due to memory constraints.

---

## Experimental Results

We evaluate all combinations of graph construction and architecture on three benchmarks covering complementary aspects of protein function and structure: Enzyme Commission (EC) number prediction, Gene Ontology (GO) term prediction, and structural fold classification (Fold3D). All results report F1-max averaged over 5 random seeds unless otherwise noted. We use the hyperparameters and architectures from GearNet <d-cite key="zhang2023protein"></d-cite>. The network uses 6 layers with hidden dimension 512.

### 1. Enzyme Commission (EC) Number Prediction

The EC benchmark tests multi-label classification of enzymatic function at four levels of hierarchical specificity. It is a natural testbed for geometric graph construction: EC numbers directly reflect active-site chemistry, which depends on precise spatial and orientational relationships between catalytic residues.

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/ec_comparison.png" class="img-fluid" %}
<div class="caption">
  Figure 4. EC number prediction F1-max across graph constructions and architectures. Angle+KNN consistently achieves the highest scores for lightweight models (GearNet), while GearNet-Efficient IEConv achieves strong performance across all graph topologies. The gap between Sequential-only and richer constructions confirms that spatial edges carry task-relevant information beyond backbone connectivity.
</div>

<figure>
<div class="table-caption">
  Table 2. F1-max scores on Enzyme Commission (EC) number prediction. <b>Bold</b> indicates the best performance for a given architecture. GearNet-Efficient IEConv on the baseline graph provides the highest overall accuracy, while Angle+KNN provides the best performance-to-edge-count ratio for GearNet.
</div>
<table class="results-table">
<thead>
<tr><th>Graph Construction</th><th>GearNet</th><th>GearNet-Edge</th><th>GearNet-IEConv</th><th>Efficient IEConv</th></tr>
</thead>
<tbody>
<tr><td class="paper-baseline">Original Paper Report</td><td class="paper-baseline">0.730</td><td class="paper-baseline">0.810</td><td class="paper-baseline">0.800</td><td class="paper-baseline">—</td></tr>
<tr><td><strong>Baseline</strong></td><td>0.762 ± 0.005</td><td>0.812 ± 0.002</td><td>0.813</td><td><strong>0.828 ± 0.004</strong></td></tr>
<tr><td><strong>Sequential Only</strong></td><td>0.692 ± 0.003</td><td>0.749 ± 0.006</td><td>0.748</td><td>0.786 ± 0.002</td></tr>
<tr><td><strong>KNN Only</strong></td><td>0.743 ± 0.004</td><td>0.784 ± 0.003</td><td>0.786</td><td>0.818 ± 0.005</td></tr>
<tr><td><strong>Radius Only</strong></td><td>0.753 ± 0.004</td><td><strong>0.813 ± 0.003</strong></td><td>0.808</td><td>0.825 ± 0.002</td></tr>
<tr><td><strong>Angle Only</strong></td><td>0.759 ± 0.003</td><td>0.788 ± 0.003</td><td>0.800</td><td>0.808 ± 0.003</td></tr>
<tr><td><strong>Angle + KNN</strong></td><td><strong>0.767 ± 0.003</strong></td><td>0.798 ± 0.005</td><td><strong>0.814</strong></td><td>0.824 ± 0.005</td></tr>
</tbody>
</table>
</figure>

Several patterns emerge. First, **Angle + KNN matches or beats the Baseline** for GearNet despite using 38% fewer edges, suggesting that the pruned edges were carrying noise rather than signal. Second, GearNet-Efficient IEConv outperforms GearNet variants on the baseline graph construction (0.828 vs. 0.813), demonstrating the FiLM reformulation effectiveness.

One subtlety worth noting: GearNet results are reported without applying Dropout, following TorchDrug <d-cite key="zhu2022torchdrug"></d-cite> standard implementation. This is the reason for the slight improvement of that variant over the original paper reported values.

Comparing across model architectures, we observe an interesting pattern: for GearNet, the angle based constructions offer clear improvements, while for Edge and IEConv they only maintain or slightly decrease the results. We return to this pattern in the analysis section.


### 2. Gene Ontology (GO) Prediction

Gene Ontology prediction covers three sub-tasks of increasing semantic abstraction: Molecular Function (MF), which captures direct biochemical activities; Cellular Component (CC), which reflects subcellular localization; and Biological Process (BP), which encodes high-level systemic roles. These tasks test whether geometric topology transfers across different levels of functional specificity.

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/go_comparison.png" class="img-fluid" %}
<div class="caption">
  Figure 5. GO F1-max across graph constructions with GearNet. Angle again offers improvements in this base model variant. Unlike in EC, KNN outperforms Radius in this regime.
</div>


<div class="table-caption">
  Table 3. F1-max scores for GO dataset sub-tasks: MF, CC, and BP. <b>Bold</b> indicates the best performance for a given architecture, while <u>underlined</u> values indicate better performance than the baseline. Angle-based topologies consistently outperform isotropic baselines in the lightweight GearNet setting.
</div>
<table class="results-table">
<thead>
<tr>
  <th>Ontology</th>
  <th>Model</th>
  <th>Baseline</th>
  <th>Sequential</th>
  <th>Angle Only</th>
  <th>Angle + KNN</th>
</tr>
</thead>
<tbody>


<tr>
  <td rowspan="2"><strong>MF</strong></td>
  <td>GearNet</td>
  <td>0.528 ± 0.006</td>
  <td>0.493 ± 0.002</td>
  <td><u>0.552 ± 0.005</u></td>
  <td><strong>0.553 ± 0.002</strong></td>
</tr>
<tr>
  <td>Eff. IEConv</td>
  <td>0.582 ± 0.002</td>
  <td>0.558 ± 0.002</td>
  <td>0.574 ± 0.001</td>
  <td><strong>0.590 ± 0.006</strong></td>
</tr>
<tr>
  <td rowspan="2"><strong>CC</strong></td>
  <td>GearNet</td>
  <td>0.392 ± 0.007</td>
  <td><u>0.401 ± 0.003</u></td>
  <td><u>0.409 ± 0.004</u></td>
  <td><strong>0.410 ± 0.006</strong></td>
</tr>
<tr>
  <td>Eff. IEConv</td>
  <td>0.427 ± 0.004</td>
  <td>0.416 ± 0.010</td>
  <td><strong>0.434 ± 0.006</strong></td>
  <td>0.426 ± 0.005</td>
</tr>
<tr>
  <td rowspan="2"><strong>BP</strong></td>
  <td>GearNet</td>
  <td>0.369 ± 0.010</td>
  <td><u>0.401 ± 0.004</u></td>
  <td><strong>0.407 ± 0.003</strong></td>
  <td><u>0.401 ± 0.004</u></td>
</tr>
<tr>
  <td>Eff. IEConv</td>
  <td>0.407 ± 0.002</td>
  <td><u>0.414 ± 0.004</u></td>
  <td><strong>0.420 ± 0.004</strong></td>
  <td><u>0.419 ± 0.005</u></td>
</tr>
</tbody>
</table>

The pattern of which topology wins varies slightly by sub-task. For MF, Angle + KNN is the clear winner: direct binding activities depend on precise side-chain orientation, so filtering to geometrically compatible contacts directly benefits this task. GearNet improves from 0.528 to 0.553, a 4.7% relative gain while reducing the number of edges by ~1900. However, the AUPRC values degrade slightly from 0.495 ± 0.004 to 0.487 ± 0.006.

For BP and CC, angle-only filtering sometimes outperforms Angle + KNN (despite having fewer parameters), and in the BP case, even Sequential-only edges perform comparably to richer spatial constructions for GearNet. For BP specifically, looking at the AUPRC the Angle strategies outperform the other variants: 0.2397 (Angle+KNN) and 0.2368 (Angle) compared to a baseline of 0.2187.

GearNet-Efficient IEConv consistently remains the top model across all three ontologies, suggesting that edge geometric modulation is broadly beneficial regardless of the annotation type. Also, in the GO dataset, the KNN Only variant outperformed Radius in all of the cases, while on the previous EC dataset it was the opposite. The fact that Angle based rewirings maintains performance in both suggests the intuition generalizes to other types of graphs.

Due to the prohibitive computational cost of the original GearNet-Edge, we report the performance on a single run. Results were similar to those from EC dataset:

<div class="table-caption">
  Table 4. Comparison of f1 max, average number of edges, and training wall-clock time for the GearNet-Edge architecture. While F1 scores remain stable, Angle-based filtering nearly halves the training time.
</div>

<table class="results-table">
<thead>
<tr>
  <th>Ontology</th>
  <th>Model</th>
  <th>Metric</th>
  <th>Baseline</th>
  <th>Angle Only</th>
  <th>Angle + KNN</th>
</tr>
</thead>
<tbody>
<tr>
  <td rowspan="3"><strong>MF</strong></td>
  <td rowspan="3">GearNet-Edge</td>
  <td>Test F1-max</td>
  <td><strong>0.592</strong></td>
  <td>0.579</td>
  <td>0.585</td>
</tr>
<tr>
  <td>Avg Edges</td>
  <td>5057.9</td>
  <td><strong>2155.7</strong></td>
  <td>3179.9</td>
</tr>
<tr>
  <td>Wall-clock (h)</td>
  <td>29.1</td>
  <td><strong>15.8</strong></td>
  <td>18.8</td>
</tr>
<tr>
  <td rowspan="3"><strong>CC</strong></td>
  <td rowspan="3">GearNet-Edge</td>
  <td>Test F1-max</td>
  <td><strong>0.430</strong></td>
  <td>0.417</td>
  <td>0.423</td>
</tr>
<tr>
  <td>Avg Edges</td>
  <td>5057.9</td>
  <td><strong>2155.7</strong></td>
  <td>3179.9</td>
</tr>
<tr>
  <td>Wall-clock (h)</td>
  <td>29.0</td>
  <td><strong>15.6</strong></td>
  <td>20.2</td>
</tr>
<tr>
  <td rowspan="3"><strong>BP</strong></td>
  <td rowspan="3">GearNet-Edge</td>
  <td>Test F1-max</td>
  <td>0.421</td>
  <td><strong>0.423</strong></td>
  <td><strong>0.423</strong></td>
</tr>
<tr>
  <td>Avg Edges</td>
  <td>5057.9</td>
  <td><strong>2155.7</strong></td>
  <td>3179.9</td>
</tr>
<tr>
  <td>Wall-clock (h)</td>
  <td>29.6</td>
  <td><strong>15.8</strong></td>
  <td>19.4</td>
</tr>
</tbody>
</table>



As we saw with the previous dataset, in the GearNet-Edge model variant we see no improvements in F1-max score from using Angle based graph constructions. However, they still provide computational benefits, since the reduced number of edges greatly improves the VRAM usage and total training time by a proportional amount. Note that all branches of the GO dataset use the same graphs and differ only in their annotations.


{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/delta_heatmap.png" class="img-fluid" %}
<div class="caption">
  Figure 6. Heatmap of F1-max improvement (Angle+KNN vs. Baseline) by model and GO dataset subtasks. Warmer colors indicate larger gains from geometric filtering. GearNet benefits substantially on MF (+4.7%), where side-chain orientation directly determines binding specificity. Gains are smaller for CC and BP, reflecting that these annotations are less tightly coupled to local geometric contacts. GearNet-Efficient IEConv shows more modest topology-driven gains because its per-edge conditioning already provides a form of learned geometric filtering.
</div>



### 3. Structural Embedding (Fold3D)

Fold3D tests a different question: can a model recognize structural topology even when sequence similarity is low? The dataset is split at three levels of evolutionary distance: **Fold** (most remote), **Superfamily**, and **Family** (most similar), where correct classification at the Fold level requires learning purely geometric representations with no sequence shortcut. Superfamily and Family splits allow for increasing levels of sequence-based discrimination. This makes it an ideal benchmark for evaluating whether Angle Rewiring produces topologies that are genuinely more informative for structural discrimination, rather than just function prediction.

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/fold3d_results.png" class="img-fluid" %}
<div class="caption">
  Figure 7. Fold3D classification accuracy across architecture and graph construction. The most challenging Fold-level split shows the largest benefit from geometric filtering, consistent with the hypothesis that Angle Rewiring produces representations that are more sensitive to structural topology.
</div>

<div class="table-caption">
  Table 5. Classification accuracy (%) on the Fold3D dataset. Results in *gray* are taken from the original GearNet paper <d-cite key="zhang2023protein"></d-cite>.
</div>
<table class="results-table">
<thead>
<tr><th>Architecture</th><th>Fold</th><th>Superfamily</th><th>Family</th><th>Average</th></tr>
</thead>
<tbody>
<tr><td class="paper-baseline">GearNet</td><td class="paper-baseline">28.4</td><td class="paper-baseline">42.6</td><td class="paper-baseline">95.3</td><td class="paper-baseline">55.4</td></tr>


<tr><td class="paper-baseline">GearNet-IEConv</td><td class="paper-baseline">42.3</td><td class="paper-baseline">64.1</td><td class="paper-baseline">99.1</td><td class="paper-baseline">68.5</td></tr>

<tr><td class="paper-baseline">GearNet-Edge-IEConv</td><td class="paper-baseline">48.3</td><td class="paper-baseline">70.3</td><td class="paper-baseline">99.5</td><td class="paper-baseline">72.7</td></tr>

<tr><td colspan="5" style="background:#f1f5f9;height:4px;"></td></tr>
<tr><td><strong>GearNet (Baseline)</strong></td><td>28.4 ± 0.9</td><td>41.4 ± 0.9</td><td>95.7 ± 0.6</td><td>55.2</td></tr>
<tr><td><strong>GearNet (Angle+KNN)</strong></td><td><strong>31.1 ± 0.6</strong></td><td><strong>45.0 ± 1.1</strong></td><td><strong>96.8 ± 0.3</strong></td><td><strong>57.6</strong></td></tr>
<tr><td colspan="5" style="background:#f1f5f9;height:4px;"></td></tr>
<tr><td><strong>Eff. IEConv (Baseline)</strong></td><td>43.0 ± 0.8</td><td><strong>67.3 ± 0.5</strong></td><td>99.0 ± 0.3</td><td>69.8</td></tr>
<tr><td><strong>Eff. IEConv (Angle+KNN)</strong></td><td><strong>44.8 ± 0.9</strong></td><td>66.5 ± 0.8</td><td><strong>99.2 ± 0.1</strong></td><td><strong>70.2</strong></td></tr>
</tbody>
</table>



The Fold3D results are interesting in two ways. First, Angle + KNN improves GearNet's average accuracy from 55.2% to 57.6%, with the largest gains at the most challenging Fold level (+2.7 pp) where low-level structural geometry is the only available signal. This confirms that Angle Rewiring produces topologies that are genuinely more informative for structural discrimination, not just for function prediction. It also takes an average of 6 seconds less per epoch during training (from 47s to 41s on NVIDIA H100 64 GB GPUs), which is a significant reduction in training time when increasing the number of epochs in the training process, e.g. GearNet was trained for 300 epochs.

Second, **GearNet-Efficient IEConv maintains its performance**. At 69.8–70.2% average accuracy, it improves the original GearNet-IEConv results and almost perfectly matches those of GearNet-Edge, a much more computationally expensive model which directly encodes edge information.

Our method does not aim to outperform the most elaborate protein GNNs on raw accuracy alone. Architectures such as ProNet <d-cite key="wang2022pronet"></d-cite> gain accuracy by introducing richer multi-scale representations and additional structural priors. Angle Rewiring tackles a different question: can we make the graph itself more biologically meaningful, and in doing so make learning both cheaper and cleaner? The answer appears to be yes. By filtering out directionally weak contacts at graph construction time, we reduce computation substantially while retaining competitive performance. This makes the approach complementary to heavier architectures rather than directly opposed to them. Combined with Efficient IEConv, we can deliver a strong accuracy-efficiency tradeoff with minimal added complexity.


---

## Global Pareto Optimality & Analysis

The results above show that geometric graph construction improves accuracy. But to understand the full picture, we need to consider accuracy alongside the computational cost of achieving it.

### Topology vs. Learned Edge Features
A natural question arises from the tables: why does Angle + KNN produce large accuracy gains for standard GearNet (e.g., +4.7% on GO-MF) but much smaller gains for GearNet-Efficient IEConv? And for models explicitly designed around edges, like GearNet-Edge, why does the accuracy sometimes drop slightly? If a biologically cleaner graph is better, shouldn't it help all models equally?

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/geometry_expressiveness_tradeoff.png" class="img-fluid" %}
<div class="caption">
  Figure 8. The Expressiveness Subsumption Effect. For lightweight models (GearNet, left), geometric topology is the primary carrier of structural information: the model cannot learn it from features alone. For expressive models (GearNet-Efficient IEConv, right), the per-edge MLP already learns to down-weight geometrically incompatible contacts, effectively performing soft topological filtering internally. Angle Rewiring then provides a complementary benefit: not larger accuracy gains, but equivalent accuracy with far fewer edges.
</div>

The answer lies in how different architectures access geometric information. Standard GearNet lacks detailed edge-level features to evaluate the quality of a connection, it entirely relies on the graph construction to filter out biological noise. For these lightweight models, the topology is the only source of directional geometry. Therefore, giving them a biologically accurate graph yields representational improvements.

In contrast, models like GearNet-IEConv and GearNet-Edge are specifically designed to compute and incorporate rich geometric features, such as inter-residue distances, angles, and dihedral torsions, for every single edge during the forward pass.

When an expressive network like IEConv encounters a "bad" isotropic edge, it does not blindly pass the message. Its MLP can learn to down-weight or gate that connection. For Efficient IEConv, the MLP predicting the shift and scale parameters can learn to output near zero scale factors for those edges with unfavorable geometric features.

This has an important implication: **for highly expressive models, the primary value of Angle Rewiring is computational, not representational**. Providing a sparse graph does not appreciably change what the model learns, because the architectures already possess the internal mechanisms to identify and ignore the edges. Conversely, for lightweight models where the graph topology carries the full geometric burden, Angle Rewiring is directly representationally beneficial.

### Training Acceleration and Memory Optimization

We now quantify the computational dividends of combining both contributions: downsizing the number of edges, and the memory optimization of the Efficient IEConv architecture. We analyze this on a local scale, looking at the EC dataset and the GearNet architecture:

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/pareto_edges.png" class="img-fluid" %}
<div class="caption">
  Figure 9. Pareto frontier of F1-max vs. average edges per protein on EC number prediction with base GearNet.
</div>

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/pareto_wallclock.png" class="img-fluid" %}
<div class="caption">
  Figure 10. Pareto frontier of F1-max vs. total wall-clock training time on EC. Again, we observe the same trend and a high correlation between the average edges per Graph and the training wall-clock time.
</div>

We can also analyze these results on the global scale considering all the architectures, analyzing the influence of the average number of edges and the training wall-clock time on the F1-max score:


{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/pareto_edges_global.png" class="img-fluid" %}
<div class="caption">
    Figure 11. Pareto Efficiency (Graph Sparsity versus F1 max).
</div>

{% include figure.liquid path="assets/img/2026-04-14-fewer-edges/pareto_walltime_global.png" class="img-fluid" %}
<div class="caption">
    Figure 12. Pareto Efficiency (Total Wall-clock Time). The baseline IEConv architecture requires <strong>22.7 training hours</strong>. GearNet-Efficient IEConv drops wall-clock time to <strong>6.4 hours</strong>. Adding Angle+KNN, computation goes down to 5.3 hours, an overall 76% reduction.
</div>




The combined effect of FiLM-based convolution and topological sparsification produces a substantial reordering of the efficiency landscape:

- **Peak VRAM**: 52.8 GB &rarr; 3.2 GB (16× reduction when combining sparse graph + Efficient IEConv)
- **Training time**: ~76% reduction in total wall-clock hours
- **Edge count**: ~5,990 &rarr; ~3,733 (38% reduction for Angle+KNN)

These numbers represent more than engineering convenience. Training at 4 GB of VRAM instead of 50+ GB means that a researcher with a single RTX 4090 or A10 can run experiments that previously often required high-memory accelerators. The memory headroom enables training at far larger batch sizes, opening the door to improved contrastive pretraining methods (e.g., GearNet's multi-view contrastive approach) on the full AlphaFold Database at a scale that was not practical with IEConv-class models.

---

## Limitations and Future Work

Our results reveal a boundary on when biological inductive biases help the most. Angle Rewiring benefits lightweight models like standard GearNet, but does not improve Edge message passing variants. However, Radius Only rewiring is able to obtain similar performances for those architectures despite having less parameters. This suggests that for the variants that already encode edge geometric information, pure topology based criteria could be more beneficial, since they are already able to learn the biological properties. A promising direction would be to extend these ideas beyond pairwise angle constraints toward richer topology design for protein structure, taking into account limitations of GNN architectures such as over-squashing and applying graph rewiring such as in <d-cite key="topping2021understanding"></d-cite> <d-cite key="miquel2026effective"></d-cite>.


A second limitation concerns the angle criterion itself. Our condition captures side-chain orientation, but is still blind to higher-order geometric compatibility. Extending graph construction to incorporate dihedral compatibility or other specific properties would be natural next steps. Extending graph construction to higher-order relations such as hypergraphs or cell complexes can be benifitial to understand 3D protein structure <d-cite key="telyatnikov2025topobench"></d-cite>.

Protein function prediction has been increasingly dominated by large-scale pretraining on structural databases, a regime where memory and compute constraints create a significant barrier. By making expressive geometric convolutions feasible on consumer hardware, this work opens the door to training on a larger scale of structural data <d-cite key="varadi2022alphafold"></d-cite>.

Future directions also include testing Angle Rewiring and Efficient IEConv with fully equivariant architectures like <d-cite key="brandstetter2022geometric"></d-cite> <d-cite key="liao2023equiformer"></d-cite>, as well as additional experiments controlling for the number of parameters to provide more insights into more efficient or representative protein graph constructions.


---

## Conclusion

Protein graph construction is often treated as a fixed preprocessing choice, but our results suggest it should be treated as a modeling decision. When edges are drawn between any two residues within an arbitrary distance cutoff, a significant fraction of them connect pairs that are structurally incapable of interacting, which consumes computation and injects noise.

**Angle Rewiring** addresses this by encoding biological directionality directly into graph topology. By requiring both residues in a candidate edge to orient their side-chains toward each other, it removes a substantial fraction of edges without sacrificing accuracy. The gains are largest for lightweight models and for tasks closely tied to local binding geometry (e.g., Molecular Function in GO), exactly where topological signal matters most.

**Efficient IEConv** addresses another bottleneck: the quadratic memory cost of per-edge geometric transformation matrices. Replacing $D \times D$ matrices with FiLM-style feature wise modulations, it retains edge geometric conditioning while making the model practical on standard hardware.

Taken together, these results suggest a simple message: better protein GNNs do not come only from more expressive layers, but also from better graphs. Biologically motivated topology can improve both representation quality and computational efficiency, and it composes naturally with more powerful geometric architectures.