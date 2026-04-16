---
layout: distill
title: The Role of Directionality in Graph Neural Networks
description: We investigate how graph directionality may influence GNN performance across homophilic and heterophilic benchmarks, suggesting it could be an underexplored factor.
date: 2026-04-05
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Bertran Miquel

# must be the exact same name as your blogpost
bibliography: 2026-04-15-graph_directionality_matters.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Results
    subsections:
      - name: Bidirectionalization transformation improves GNN performance
  - name: Discussion

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
    .table-caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: left;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.5;
    font-weight: 400;
  }
  .caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: center;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
    line-height: 1.5;
  }
---

# **Introduction**

Graph Neural Networks (GNNs) have achieved strong performance on node classification benchmarks, particularly in **homophilic settings**, where neighboring nodes tend to share labels. However, a widely accepted narrative is that GNNs struggle in **heterophilic graphs**, where connected nodes do not necessarily share labels. This observation has motivated a search for improving GNN performance in heterophilic regimes through architectural modifications and alternative aggregation schemes <d-cite key="kipf2016semi"></d-cite>, <d-cite key="velivckovic2017graph"></d-cite>, <d-cite key="rossi2024edge"></d-cite>.

By definition, the key distinction between homophilic and heterophilic datasets lies in the degree of label similarity among neighboring nodes. However, an often overlooked aspect is that these two regimes also differ structurally: many heterophilic datasets are defined in **directed** form, whereas homophilic datasets are typically constructed as **undirected**.

Historically, widely used homophilic benchmarks such as Cora, CiteSeer, and PubMed were constructed as **undirected graphs**. In early work (e.g., <d-cite key="kipf2016semi"></d-cite>), citation edges were explicitly symmetrized to produce a **symmetric adjacency matrix**, enabling the use of spectral graph convolutions. This design choice enabled compatibility with the problem and ensured compatibility with models relying on symmetric normalization. As a consequence, modern libraries such as PyTorch Geometric represent these graphs as **bidirectional edge lists**, even when the underlying relationships (e.g., citations) are inherently directed.

Despite the development of models capable of handling asymmetric adjacency matrices, many benchmarks continue to rely on bidirectional versions of homophilic datasets. As shown in Table 1, most homophilic datasets exhibit a **bidirectionality ratio of 1.0**, indicating that every edge has a corresponding reverse edge. The main exception is OGBN-Arxiv, which retains its original directionality and exhibits a very low bidirectionality ratio (1.4%).

This persistent design choice can be attributed to several factors:

* Some datasets are naturally undirected (e.g., co-authorship and co-purchase networks) <d-cite key="shchur2018pitfalls"></d-cite>
* Others are commonly symmetrized in practice, despite having an underlying directed structure (e.g., citation graphs)
* Even recent works, such as <d-cite key="liang2025towards"></d-cite>, explicitly enforce bidirectionality to isolate specific phenomena (e.g., long-range dependencies).

In contrast, many heterophilic datasets (e.g., Cornell, Texas, Wisconsin, Chameleon, Squirrel) are typically used in their **original directed** or asymmetric form, as their underlying relationships (e.g., hyperlinks or interactions) are inherently directional <d-cite key="pei2020geom"></d-cite>. As reported in Table 1, these datasets exhibit significantly lower bidirectionality ratios, reflecting a high degree of asymmetry.

Importantly, several works have reported poor performance of GNNs on these heterophilic benchmarks and often attributed in part to label inconsistency, analyzing structural properties of the graphs without explicitly controlling for directionality <d-cite key="zhu2020beyond"></d-cite>, <d-cite key="ma2021homophily"></d-cite>. While recent studies such as <d-cite key="rossi2024edge"></d-cite> show that **directed GNNs (DirGNNs)** can improve performance on heterophilic datasets, a performance gap with respect to homophilic benchmarks still remains. However, these analyses do not disentangle the effects of heterophily from those of directionality, leaving open the question of which factor is the primary driver of GNN performance.

Taken together, these observations raise a fundamental question:

> Are GNNs failing because of **heterophily**, or because of **directionality**?


<div class="table-caption">
Table 1. Bidirectional properties of graph datasets. We report the dataset type (homophilic vs heterophilic), whether the graph is effectively bidirectional, and the bidirectionality ratio (fraction of edges with a reverse counterpart).
</div>

| dataset                 | type         | bidirectionality | bidirectionality_ratio |
| ----------------------- | ------------ | ---------------- | ---------------------- |
| amazon-computers        | homophilic   | True             | 1.0000                 |
| amazon-photo            | homophilic   | True             | 1.0000                 |
| citeseer_full           | homophilic   | True             | 1.0000                 |
| coauthor-cs             | homophilic   | True             | 1.0000                 |
| coauthor-phy            | homophilic   | True             | 1.0000                 |
| cora_ml                 | homophilic   | True             | 1.0000                 |
| ogbn-arxiv              | homophilic   | False            | 0.0145                 |
| pubmed                  | homophilic   | True             | 1.0000                 |
| chameleon               | heterophilic | False            | 0.2596                 |
| cornell                 | heterophilic | False            | 0.1220                 |
| directed-roman-empire   | heterophilic | False            | 0.3180                 |
| directed_amazon_ratings | heterophilic | False            | 0.3571                 |
| squirrel                | heterophilic | False            | 0.1713                 |
| texas                   | heterophilic | False            | 0.1942                 |
| wisconsin               | heterophilic | False            | 0.1964                 |


To answer this question, we perform a systematic study across widely used homophilic and heterophilic benchmarks. Specifically, we consider homophilic datasets such as Cora, CiteSeer, PubMed, Amazon, OGBN-Arxiv, and Coauthor, and heterophilic datasets including Cornell, Texas, Wisconsin, Chameleon, and Squirrel.

We explicitly control for directionality by applying bidirectionality transformation. Directed or asymmetric graphs are converted into fully bidirectional graphs by adding reverse edges:
  $[
  (u, v) \rightarrow (v, u); (v, u) \rightarrow (u, v)
  ]$

After applying the transformation, we evaluate standard GNN architectures (GCN, GAT, GraphSAGE) and their directed counterparts across multiple random seeds to ensure robustness.

---

# **Results**

We evaluate the performance of GNNs on the datasets in their original form, as well as after applying the bidirectionalization and directed graph construction transformations. This allows us to partially disentangle the effects of directionality from those of homophily.
For the experiments, we use the standard splits and the same hyperparameters for all models, as in <d-cite key="rossi2024edge"></d-cite>, to ensure a fair comparison across different settings. We report the average performance across multiple random seeds to account for variability.

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/directed_baseline_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 1.</b> Performance of GNNs on directed datasets. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements.
</div>

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/bidirected_baseline_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 2.</b> Performance of GNNs on undirected or bidirectional datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) and directed GNNs (DirGCN, DirGAT, DirSAGE) perform comparably.
</div>

In Figure 1, we observe how the performance of directed GNNs (DirGCN, DirGAT, DirSAGE) consistently outperforms standard GNNs (GCN, GAT, GraphSAGE) when datasets contain directionality. While in Figure 2, we see how directed GNNs have no significant difference in performance compared to standard GNNs on bidirectional or undirected datasets, without losing performance. This suggests that directionality may play a role in explaining the observed performance gap.

## **Bidirectionalization transformation improves GNN performance**

When applying the bidirectionalization transformation of directed graphs, we observe a significant improvement in the performance of standard GNNs, which now perform with no significant difference to DirGNNs (Figure 3). These results suggest that the performance gap may not be solely explained by heterophily, but rather because of the presence of directionality in the underlying graph structure.

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/directed_bidirected_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 3.</b> Performance of GNNs on bidirected heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) and directed GNNs (DirGCN, DirGAT, DirSAGE) perform comparably after applying the bidirectionality transformation.
</div>

We compute the homophily of the original directed graphs and the bidirectionalized versions to understand whether the performance improvement can be attributed to changes in homophily.

$$
h_i = \frac{1}{\deg(i)} \sum_{j \in \mathcal{N}(i)} \mathbf{1}\left[y_i = y_j\right]
$$

In Figure 4, we can see how the homophily changes less than 0.1 in all cases, when applying this bidirectional transformation . However, the performance of GNNs significantly improves.

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/homophily_changes.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 4.</b> Homophility delta after applying the bidirectionalization transformation.
</div>


# **Discussion**

In this work, our results suggest that directionality may influence the performance of GNNs, when directed graphs are used. We show that applying a simple bidirectionalization transformation to directed graphs can significantly improve the performance of standard GNNs,
while reducing the performance gap between GNNs and directed GNNs.

This may indicate that current GNN architectures have limitations when handling directionality. Then, two main several points for future research are worth considering:
- GNNs struggle with directionality because of the way message-passing is designed. An interesting direction for future research is to **modify vanilla GNN architectures to better handle directionality.**
- **More diverse datasets are needed** to allow for a more comprehensive evaluation of GNN performance across different regimes of homophily and directionality. This would enable us to better understand the interplay between these factors and their impact on GNN performance. There is a **lack of homophilic datasets with directionality, and heterophilic datasets without directionality**, which makes it difficult to disentangle the effects of homophily and directionality on GNN performance. Directionality is a fundamental property of many real-world graphs that should be taken into account when designing new benchmarks and evaluating GNN performance.