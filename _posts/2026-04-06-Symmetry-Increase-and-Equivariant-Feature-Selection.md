---
layout: distill
title: Symmetry Increase and Equivariant Feature Selection
description: This blog shows that symmetric inputs can induce representation degeneration due to the algebraic structure of the feature space itself, leading to loss of discriminative power, and provides practical guidance for selecting equivariant features.
date: 2026-04-06
future: true
htmlwidgets: true
hidden: false

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Anyi Li

# must be the exact same name as your blogpost
bibliography: 2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: Data Symmetry
      - name: Equivariant Maps
  - name: Degeneration of Equivariant Features
    subsections:
      - name: Degneration of Average Spherical Harmonics
      - name: Visualization Results
      - name: Collapse-to-zero Model and Symmetry Increase
  - name: Symmetry Infimum
    subsections:
      - name: Symmetry Infimum and Isovariant Maps
      - name: Kernel and Relatively Isovariant Maps
  - name: Learning Guarantees under Assumptions
    subsections:
      - name: Manifold Hypothesis on the Data
      - name: Approximation Hypothesis on the Model
  - name: Computation of Symmetry Increase
  - name: Guideline for Choice of Equivariant Features
    subsections:
      - name: For Orientation-dependent Tasks
      - name: For Orientation-independent Tasks
  - name: Conclusion

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
---

## Introduction

In the pretraining of equivariant point cloud models, the choice of feature order has a  impact on model expressivity. Different orders of representations exhibit different behaviors when handling symmetric structures, a phenomenon that is observed in practice but whose theoretical mechanism is far from intuitive. This article provides an interpretive overview of the paper <d-cite key="linreducing"></d-cite>. The work show that when an equivariant network processes symmetric inputs, its representations may degenerate: the network output can acquire additional symmetry and thus lose discriminative information. We begin with a special example of feature extraction from the background material, and gradually develop an algebraic formulation of equivariant feature extraction. The conclusion is that this phenomenon arises solely from the algebraic structure of the feature space itself, rather than from the network architecture or the optimization process. By analyzing the behavior of features of different orders under symmetric inputs, we can understand the conditions under which degeneration occurs, and on this basis obtain practical guidance for choosing feature orders during pretraining. The first half of this article focuses on theoretical explanation, while the final section combines experiments to provide guidelines.

### Data Symmetry

Symmetry is common in the real world, from crystals and molecules in 3D space to ornaments and snowflakes in 2D space. As a result, symmetric data arise in many scientific and visual problems, and handling them is an important issue in machine learning.  
  
In many scientific tasks, data admit multiple possible choices of reference frame. Let $$X$$ denote the space in which the data live. Transformations between reference frames are typically modeled by a group $$G$$ This group is called the the global symmetry group. It maps data represented in one reference frame to data represented in another, so that a single physical object corresponds to the set of all data obtained by applying every possible symmetry transformation to a given datum $$x$$ This set is called the **group orbit**,

$$  
G(x)=\{g(x)\mid g\in G\}.  
$$

The group orbits partition $$X$$ into different equivalence classes, each corresponding to a different physical object. However, for a specific datum $$x$$, there may also exist local symmetries of its own, namely transformations under which $$x$$ remains unchanged. These transformations form the **isotropy subgroup** of $$x$$,  

$$  
G_x=\{g\in G\mid g(x)=x\}.  
$$

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/stablizer.png" class="img-fluid" %}
<div class="caption">
    Fig: Global symmetry and local symmetry.
</div>

Consider the following simple example. Let $$x$$ be a 3D point cloud. Point clouds are usually viewed as having global symmetry group  

$$  
G=H\times S_n,  
$$

where $$H$$ is typically $$SO(3)$$ or $$O(3)$$, $$S_n$$ is the permutation group on the nodes, and $$n$$ is the number of nodes. The factor $$H$$ corresponds to global geometric transformations of the point cloud reference frame. When $$H=SO(3)$$, we ignore the chirality of the reference frame. The factor $$S_n$$ corresponds to relabeling the nodes.

Now consider a point cloud $$x$$ consisting of $$k+1>3$$ points, and let  

$$  
G=O(3)\times S_{k+1}.  
$$

In general, $$G_x$$ is trivial, meaning that it contains only the identity transformation. However, in some special cases the situation is different. Let $$x\in X$$ be the set of vertices of a $$k$$-fold in the $$xOy$$-plane:  

$$
x=(x_0,\dots,x_k),\qquad \text{where } x_i=(\cos(2i\pi/k),\sin(2i\pi/k),0)\ \text{for } i>0,  
$$

and $$x_0$$ is a vertex at the origin. The generators of the isotropy subgroup $$G_x$$ include:  
1. a rotation about the $$z$$-axis combined with a cyclic permutation;  
2. a reflection across the $$xOz$$-plane combined with a product of transpositions;  
3. a reflection across the $$xOy$$-plane combined with the identity.  
  
Consider the projection map  $$\pi_X((g,\sigma))=g$$  Then $$\pi_X(G_x)\cong D_{kh}$$,  where $$D_{kh}$$ is the Schoenflies symbol for the $$k$$-fold dihedral group.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/k-fold.png" class="img-fluid" %}
<div class="caption">
    Fig: k-fold structures and their symmetry group elements.
</div>

### Equivariant Maps

In geometric deep learning, we aim to extract features from data that transform consistently under changes of reference frame, so that downstream tasks can access both the intrinsic information of the data and the orientation information associated with the reference frame. The former corresponds to features of the orbit, while the latter corresponds to canonicalizing transformations. In algebraic terms, this amounts to studying equivariant maps between two $$G$$-sets $$X$$ and $$Y$$ An equivariant map respects the group action, meaning that the output transforms accordingly when the input is transformed. Formally, a map $$f:X\to Y$$ is equivariant if, for all $$g\in G$$ and $$x\in X$$,  

$$  
f(\rho_X(g)(x))=\rho_Y(g)(f(x)).  
$$

In equivariant point cloud encoding, we typically seek features that are invariant to permutation. To achieve permutation invariance, the final output is designed to be invariant with respect to $$S_n$$ As a result, the feature space of interest is a direct sum of specified irreducible representations of $$H$$ The task of equivariant encoding is therefore to learn an equivariant map $$f:X\to Y$$.

Parity-matched $$O(3)$$-equivariant features, which can be characterized by spherical harmonics of different degrees. In this sense, the spherical harmonic map provides the simplest map from coordinates to equivariant features, and serves as a basic building block for many equivariant point cloud models. In the next section, we show through a simple calculation that the average spherical harmonic map may degenerate when the input possesses symmetry. This example naturally leads to the broader question of degeneration in general equivariant maps.

## Degneration of Equivariant Features
### Degneration of Average Spherical Harmonics

For 3D point clouds, perhaps the simplest permutation-invariant and rotation-equivariant feature is the average of the point coordinates. For a $$k$$-fold, however, this average is exactly zero. As a result, the feature cannot distinguish the $$k$$-fold from any of its rotated versions. This degeneration is not a coincidence of the first-order feature, but a much more general phenomenon of average spherical harmonic features.  
  
To see this, consider a map $$f$$ into the spherical harmonic representation $$V_{l=l_0}$$ of degree $$l_0$$, whose dimension is $$2l_0+1$$ We take as our object of study the permutation-invariant averaged spherical harmonic map for point clouds, defined by 

$$  
f_m(x)=f_m(x_0,\dots,x_k)=\sum_{i=0}^k h(\lVert x_i\rVert_2^2)\,Y_{l_0m}\!\left(x_i/\lVert x_i\rVert_2^2\right),\qquad m=-l_0,\dots,l_0,  
$$

where $$h$$ is an arbitrary function encoding the radial part, and $$Y_{lm}$$ denotes the real spherical harmonics. We first consider the features constructed from complex spherical harmonics,  

$$  
f_m^{\mathbb C}(x)=\sum_{i=0}^k h(\lVert x_i\rVert_2^2)\,Y_{l_0}^{m}\!\left(x_i/\lVert x_i\rVert_2^2\right).  
$$

Since $$Y_{l_0m}$$ can be obtained by realification of $$Y_{l_0}^{m}$$, and realification is simply a linear combination operation, it does not affect the conclusions below.  
  
We now examine the encoding of a $$k$$-fold. For a $$k$$-fold, the radial part is identical across points, so the distinction lies entirely in the spherical harmonic part. Expanding the expression shows that $$f_m^{\mathbb C}$$ is proportional to the real or imaginary part of a sum over complex exponential basis functions:  

$$  
f_m^{\mathbb C}(x)\propto P_{l_0}^{m}(0)\,A_m,  
$$

where $$P_{l_0}^{m}(0)$$ is the associated Legendre function evaluated at zero, which vanishes whenever $$l_0+m$$ is odd, and $$A_m$$ is the average of a geometric progression,  

$$  
A_m=\frac{1}{k}\sum_{n=0}^{k-1}e^{2\pi i n m/k}  
=  
\begin{cases}  
1, & k\mid m,\\  
0, & \mathrm{otherwise}.  
\end{cases}  
$$

Accordingly, we define  

$$  
S(l_0,k)=\{\,m\equiv 0\ (\mathrm{mod}\ k),\ (l_0+m)\equiv 0\ (\mathrm{mod}\ 2),\ \lvert m \rvert\le l_0\,\}.  
$$

For all $$\lvert m \rvert \le l_0$$ outside $$S(l_0,k)$$, we have $$f_m^{\mathbb C}(x)=0$$ We then classify the possible values of $$S(l_0,k)$$ according to $$l_0$$ and $$k$$:  

|                                 | $$l_0<k$$                           | $$l_0<k$$                          | $$k\le l_0<2k$$                     | $$k\le l_0<2k$$                    | $$l_0\ge 2k$$                       | $$l_0\ge 2k$$                      |
| ------------------------------- | --------------------------------- | -------------------------------- | --------------------------------- | -------------------------------- | --------------------------------- | -------------------------------- |
|                                 | $$l_0\ \mathrm{is}\ \mathrm{even}$$ | $$l_0\ \mathrm{is}\ \mathrm{odd}$$ | $$l_0\ \mathrm{is}\ \mathrm{even}$$ | $$l_0\ \mathrm{is}\ \mathrm{odd}$$ | $$l_0\ \mathrm{is}\ \mathrm{even}$$ | $$l_0\ \mathrm{is}\ \mathrm{odd}$$ |
| $$k\ \mathrm{is}\ \mathrm{even}$$ | $$\{0\}$$                           | $$\varnothing$$                    | $$C_1(l_0,k)$$                      | $$\varnothing$$                    | $$C_1(l_0,k)$$                      | $$\varnothing$$                    |
| $$k\ \mathrm{is}\ \mathrm{odd}$$  | $$\{0\}$$                           | $$\varnothing$$                    | $$\{0\}$$                           | $$C_1(l_0,k)$$                     | $$C_2(l_0,k)$$                      | $$C_1(l_0,k)$$                     |

Here both $$C_1(l_0,k)$$ and $$C_2(l_0,k)$$ contain nonzero elements, and  

$$  
C_1(l_0,k)=\{\,qk,\ q\in\mathbb Z\mid |qk|\le l_0\,\},\qquad  
C_2(l_0,k)=\{\,2qk,\ q\in\mathbb Z\mid |2qk|\le l_0\,\}.  
$$

Next, consider rotating the $$k$$-fold about its symmetry axis by an angle $$2\pi\alpha/k$$ We ask when the resulting features fail to distinguish a nontrivial rotation. It suffices to consider $$0<\alpha<1$$ The change in the embedding feature before and after rotation is  

$$  
f_m^{\mathbb C}(x')-f_m^{\mathbb C}(x)=\bigl(e^{2\pi i m\alpha/k}-1\bigr)f_m^{\mathbb C}(x).  
$$

We therefore ask for which $$l_0$$ there exists some $$\alpha\in(0,1)$$ such that for every $$m$$, either $$m\notin S(l_0,k)$$ or  $$e^{2\pi i m\alpha/k}=1$$ That is, for every $$m\in S(l_0,k)$$, the corresponding feature component remains unchanged under the rotation. The possibilities are as follows:
1. $$S(l_0,k)=\varnothing$$, then $$f_m(x)$$ vanishes identically, and $$\alpha$$ can be any value in $$(0,1)$$.
2. $$S(l_0,k)=\{0\}$$, then again $$\alpha$$ can be any value in $$(0,1)$$.
3. $$S(l_0,k)=C_2(l_0,k)$$, then the nontrivial choice is $$\alpha=1/2$$.
4. $$S(l_0,k)=C_1(l_0,k)$$, then there exists no nontrivial $$\alpha$$.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/deg_of_harmornics.png" class="img-fluid" %}
<div class="caption">
    Fig: Degeneration of average spherical harmonics feature.
</div>

From the behavior of the averaged spherical harmonic features under planar rotation of a $$k$$-fold, we obtain four cases:
1. **Full degeneration:** the $$k$$-fold feature vanishes identically.
2. **Axial degeneration:** although the feature does not necessarily vanish, it cannot distinguish any rotation around the $$k$$-fold symmetry axis.
3. **Half degeneration:** rotating by half of the smallest nonzero angle between neighboring nodes does not map the $$k$$-fold to itself, but the feature remains unchanged.
4. **No degeneration:** as long as the rotated $$k$$-fold does not coincide with itself, the feature is generically non-degenerate.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/deg_diagram.png" class="img-fluid" %}
<div class="caption">
    Fig: Three types of degeneration of k-fold.
</div>

### Visualization Results

<d-cite key="joshi2023expressive"></d-cite> pointed out that equivariant features of structures such as the $$k$$-fold may exhibit degeneration. Above, we presented a classification of the degeneration behavior of the spherical harmonic map. In the following, we use experiments to show that similar degeneration also occurs for general equivariant maps.

For the $$k$$-fold, we encode it using a randomly initialized equivariant neural network. Here we choose the TFN model <d-cite key="thomas2018tensor"></d-cite>. Before encoding, we rotate the $$k$$-fold by randomly sampling a rotation axis and uniformly choosing a rotation angle. This produces the corresponding equivariant features. For features of different degrees, we then visualize the results by randomly projecting the equivariant features before and after rotation onto a 2D plane, as shown below.


{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/full_axis_deg_vis.png" class="img-fluid" %}
<div class="caption">
    Fig: Visualization of representation spaces. (a) A k-fold structure is reoriented onto multiple planes. (b) Each is further rotated about the perpendicular axis. (c) All structures are embedded and projected into 2D. Marker shapes denote rotation axes, and colors denote rotation rates. Full degeneration appears at l=0,1, and axial degeneration at l=2,4.
</div>

The above visualization makes half-degeneration difficult to observe. To better reveal this phenomenon, we double the angular resolution of the rotations. The visualization results before and after increasing the resolution are shown below.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/half_vis.png" class="img-fluid" %}
<div class="caption">
    Fig: The rotation angle θ ∈ [0,2π/k) is uniformly discretized into res candidate values, showing that half degeneration appears at l=10 At res=98 and res=49, the overall shape is identical, but the yellow data points completely cover the blue data points.
</div>

### Collapse-to-zero Model and Symmetry Increase

<d-cite key="joshi2023expressive"></d-cite> showed that low-degree equivariant features can lead to rotational collapse in 2D, whereas resolving higher-order symmetries requires sufficiently high feature degrees. However, <d-cite key="joshi2023expressive"></d-cite> did not provide a deeper theoretical explanation of this phenomenon. In contrast, <d-cite key="cen_are_2024"></d-cite> analyzed different feature dimensions separately and pointed out that one source of indistinguishability is the algebraic structure of the group representation, which may force the equivariant feature to vanish.

We consider an equivariant map into the spherical harmonic representation of degree $$l = l_{0}$$, satisfying

$$
f(\rho_{X}(g)x)=\rho_{Y}(g)f(x).
$$

Let $$G_{x}$$ denote the stabilizer subgroup of $$x$$, and assume that $$G_{x}$$ is finite. Averaging over the symmetry group of $$x$$ yields

$$
f(x)=\frac{1}{|G_{x}|}\sum_{g\in G_{x}} f(\rho_{X}(g)x)
= \frac{1}{|G_{x}|}\sum_{g\in G_{x}} \rho_{Y}(g)f(x)
= P_{l_{0}}^{G_{x}}(f(x)),
$$

where

$$
P_{l_{0}}^{G_{x}}:=\frac{1}{|G_{x}|}\sum_{g\in G_{x}}\rho_{Y}(g)
$$

is the averaging operator on $$Y$$ associated with $$G_x$$.

<d-cite key="cen_are_2024"></d-cite> attributes the vanishing of $$f(x)$$ to the vanishing of the operator $$P_{l_{0}}^{G_{x}}$$, and further shows that this happens if and only if the trace of $$P_{l_{0}}^{G_{x}}$$ is zero. For irreducible complex spherical harmonic representations, $$\rho_{Y}(g)$$ is given by the Wigner matrices, and the conclusion can be transferred to real spherical harmonics by realification. Based on the trace formula of Wigner matrices computed in <d-cite key="engel2021point"></d-cite>, <d-cite key="cen_are_2024"></d-cite> determines when spherical harmonic encodings of point clouds with partial point-group symmetry vanish.

However, the theory in <d-cite key="cen_are_2024"></d-cite> explains only the case of **full collapse**, which is just a special case of the degeneracy observed in our experiments. Since random projection is a linear map, our visualizations indicate that there are also nonzero features that remain indistinguishable after rotation. In other words, collapse to zero is not the only possible failure mode. This motivates us to examine degeneracy from the perspective of **symmetry increase**. We first note two basic facts.

**Fact 1 (trace and fixed-point subspace dimension).**  
For a subgroup $$H\le G$$, define the fixed-point subspace of $$X$$ by

$$
X^{H}:=\{x\in X \mid h\cdot x=x,\ \forall h\in H\}.
$$

A closer look at the calculation in <d-cite key="linreducing"></d-cite> shows that the trace of $$P_{l_{0}}^{G_{x}}$$ is exactly the dimension of the fixed-point subspace $$Y^{G_{x}}$$ Therefore, if

$$
\dim Y^{G_{x}}= \mathrm{tr}(P_{l_{0}}^{G_{x}}) =0,
$$

then any $$G_x$$-equivariant feature in $$Y$$ must vanish.

**Fact 2 (equivariant maps do not decrease symmetry).**  
By equivariance, the output feature must inherit at least the symmetry of the input. This is a standard consequence of equivariance and can be formalized using the theorem in <d-cite key="kaba_symmetry_2023"></d-cite>.

> **Theorem.** Let $$f: X \to Y$$ be a $$G$$-equivariant map. For $$x \in X$$, the isotropy subgroup of $$x$$ is contained in that of its image $$f(x)$$, i.e., $$G_x \subseteq G_{f(x)}$$ or equivalently, $$f(x)\in Y^{G_{x}}$$.

Taken together, these two facts suggest that degeneracy of equivariant features can be understood as a phenomenon of **symmetry increase**. The full-collapse case corresponds to the extreme situation in which $$Y^{G_x}=\{0\}$$ In this case, the output collapses to the zero vector, which is invariant under the entire group and therefore cannot distinguish any rotations. 

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/sym_inc.png" class="img-fluid" %}
<div class="caption">
    Fig: Illustration of symmetry increase of equivariant neural networks: rectangle with D2 symmetry maps to square with D4 symmetry.
</div>

More generally, even when $$Y^{G_x}$$ is nontrivial, it may happen that every element of $$Y^{G_x}$$ has strictly higher symmetry than $$G_x$$ Equivalently, there exists a larger subgroup $$H\supsetneq G_x$$ such that $$Y^{G_x}=Y^{H}$$ In this case, any equivariant feature produced from an input with symmetry $$G_x$$ must automatically lie in a subspace with symmetry at least $$H$$, so its symmetry is necessarily elevated. For $$k$$-fold symmetric inputs, this means that the output of an equivariant map may acquire additional rotational symmetries beyond the original $$k$$-fold symmetry, leading to rotational indistinguishability even when the feature does not vanish. The full-collapse case can thus be viewed as the limiting case where the symmetry is elevated all the way to the whole group $$G$$.

Transformations in the larger isotropy group can map distinct data and its transformed versions to the same output, therefore reducing expressivity of equivariant features. To understand symmetry increase and mitigate its adverse effects, we need to answer the following key questions:
- **Symmetry infimum.** For a given input, does its image under equivariant maps admit a unique minimal symmetry type? Under what conditions does this minimal symmetry coincide with that of the input?
- **Learning guarantee.** If the minimal symmetry coincides with the input symmetry, do there exist equivariant maps that achieve this minimum? Are such symmetry-preserving maps generic among all equivariant maps?
- **Computation.** How can this minimal symmetry be computed in practice?

## Symmetry Infimum

### Symmetry Infimum and Isovariant Maps
  
To characterize the symmetry of an object in a frame-independent manner, we introduce the notion of **orbit type**. For any element $$y = g_{0}x$$ in the orbit $$G(x)$$, its isotropy subgroup satisfies $$G_{g_{0}x} = g_{0}G_{x}g_{0}^{-1}$$ Hence, the isotropy subgroups of all elements in the orbit $$G(x)$$ are conjugate to each other. We may therefore label the orbit by the conjugacy class $$(G_x)$$ of $$G_x$$, which is called the **orbit type** of $$x$$ This orbit type corresponds to the symmetry of the physical object. We denote by $$\mathcal{O}_{G}(X)$$ the set of all orbit types in $$X$$ For a compact Lie group $$G$$ and a representation $$X$$, the theory in <d-cite key="field_dynamics_2007"></d-cite> shows that $$\mathcal{O}_{G}(X)$$ is finite.  
  
For a subgroup $$H \le G$$, we write  

$$  
X_{(H)} := \{x \in X \mid (G_x) = (H)\},  
$$

where $$X_{(H)}$$ denotes the set of points in $$X$$ whose orbit type is $$(H)$$ We next endow orbit types, which represent physical symmetries, with a partial order. For subgroups $$H_1$$ and $$H_2$$ of $$G$$, we define  $$(H_1) > (H_2)$$ if and only if there exists some $$g_0 \in G$$ such that $$g_0 H_2 g_0^{-1}$$ is a subgroup of $$H_1$$.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/orbit_types.png" class="img-fluid" %}
<div class="caption">
    Fig: Illustration of orbit types and their order.
</div>

The following theorem shows that, under the above order, $$Y^{G_x}$$ indeed admits a minimal symmetry, which corresponds to the minimal possible symmetry of $$f(x)$$ obtained from $$x$$ under equivariant maps.  
  
> **Theorem.** Let $$X$$ be a representation of a compact Lie group $$G$$ For any closed subgroup $$H$$ of $$G$$, the fixed-point subspace $$X^H$$ has a unique minimal orbit type, denoted by $$I_G(X,H)$$ In particular, if $$(H)$$ itself is an orbit type, then the minimal orbit type in $$X^H$$ is exactly $$(H)$$.

In the problem of symmetry increase, we are concerned with the relation between $$I(Y,G_x)$$ and $$G_{f(x)}$$ Undesired symmetry elevation occurs when  $$I(Y,G_x) > (G_{f(x)})$$ The notion of **isovariant maps** introduced below captures the kind of equivariant maps we desire. For an equivariant map $$f : X \to Y$$ between $$G$$-spaces, we say that $$f$$ is **isovariant with respect to $$Y$$** if it strictly preserves symmetry for every $$x \in X$$, namely,  $$G_x = G_{f(x)}$$ The symmetry infimum immediately yields a necessary condition for the existence of isovariant maps.

> **Theorem.** For equivariant maps from a $$G$$-space $$X$$ to a $$G$$-space $$Y$$, a necessary condition for the existence of an isovariant map is  $$\mathcal{O}_G(X) \subset \mathcal{O}_G(Y)$$, that is, for every $$(H) \in \mathcal{O}_G(X)$$, one must have $$(H) \in \mathcal{O}_G(Y)$$ When $$X$$ and $$Y$$ are representations, this is equivalent to $$I_G(Y,H) = (H)$$

We can therefore answer **the first question** as follows.

**Q:** For a given data point, does its image under an equivariant map admit a unique minimal symmetry type?  

**A:** Yes.

**Q:** Under what condition does this minimal symmetry type coincide with that of the input? 

**A:** Precisely when the symmetry infimum coincides with the orbit type of the input.

### Kernel and Relatively Isovariant Maps

The symmetry of data may increase under an equivariant map. Such an increase may be caused by artificial factors, or it may arise from more subtle properties of the feature space itself. In tasks where invariance is not part of the objective, we do not want the extracted spherical features to become fully $$G$$-invariant, and symmetry increase is therefore undesirable. However, in equivariant point-cloud encoding, we require the features to be permutation invariant, so all permutation elements are necessarily introduced into the isotropy subgroup after mapping. In this case, one must distinguish more carefully between different causes of symmetry increase.

For a $$G$$-set $$X$$, denote the kernel of the action by

$$
\ker \rho_X := \{g \in G \mid g(x)=x,\ \forall x \in X\},
$$

that is, the set of group elements that fix every point in $$X$$ When $$\ker \rho_X$$ is trivial, namely $$\ker \rho_X=\{e\}$$, the action is called **faithful**, and a representation with faithful action is called a **faithful representation**. In representation learning, a nontrivial action kernel of the input space $$X$$ usually comes from redundancy in the chosen group action, namely, the introduction of group elements that act trivially. We therefore usually assume that the action of $$G$$ on $$X$$ is faithful. By contrast, a nontrivial action kernel of $$Y$$ typically arises from the requirements of the feature extraction task. For example, in the point-cloud feature extraction setting discussed in the previous section, we require the features to be permutation-invariant, and thus $$\ker \rho_Y$$ contains $$S_n$$.

We now consider the properties of $$p_X$$:
- For any subgroup $$H \le G$$, one has $$H \subset p_X(H)$$, and $$p_X$$ is idempotent, namely $$p_X^2 = p_X$$.
- For a $$G$$-set $$X$$ and any $$x \in X$$, the isotropy subgroup $$G_x$$ is invariant under the action of $$p_X$$ Hence there exists a subgroup $$K \le G/\ker \rho_X$$ such that $$G_x = \pi_X^{-1}(K)$$.

By the theorem above, from the symmetry increase relation $$G_x \subset G_{f(x)}$$, applying $$p_Y$$ to both sides yields

$$
G_x \subset p_Y(G_x) \subset G_{f(x)}.
$$

Therefore, when $$\ker \rho_Y$$ is nontrivial, the symmetry must in any case increase at least to $$p_Y(G_x)$$ The symmetry increase caused purely by this artificial factor can raise the isotropy subgroup of $$x$$ at most to $$p_Y(G_x)$$ If, however, the symmetry of $$G_{f(x)}$$ exceeds $$p_Y(G_x)$$, then the increase is no longer expected. Thus, what we are actually concerned with is the relation between $$I_G(Y,G_x)$$ and $$p_Y(G_x)$$.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/sym_inc_with_kernel.png" class="img-fluid" %}
<div class="caption">
    Fig: Illustration of inevitable increase with non-trivial kernel.
</div>

In the presence of a nontrivial kernel, an isovariant map in the usual sense is generally impossible, so the definition of isovariance must be extended. For an equivariant map $$f$$ between $$G$$-sets $$X$$ and $$Y$$, we say that $$f$$ is **isovariant relative to $$Y$$** if, for every $$x \in X$$,

$$
p_Y(G_x)=p_Y(G_{f(x)})=G_{f(x)}
\quad \Leftrightarrow \quad
\rho_Y(G_x)=\rho_Y(G_{f(x)}).
$$

In particular, when $$\ker \rho_Y=\{e\}$$, an isovariant map relative to $$Y$$ reduces to an ordinary isovariant map.

From this definition, we immediately obtain a necessary condition for the existence of isovariant maps relative to $$Y$$.

> **Theorem.** For equivariant maps from a $$G$$-set $$X$$ to a $$G$$-set $$Y$$, a necessary condition for the existence of an isovariant map relative to $$Y$$ is that for every $$(H)\in \mathcal{O}_G(X)$$, $$(p_Y(H)) \in \mathcal{O}_G(Y)$$ When $$X$$ and $$Y$$ are representations, this is equivalent to $$I_G(Y,H) = (p_Y(H))$$.

We briefly explain this equivalence. It suffices to show that if for every $$(H)\in \mathcal{O}_G(X)$$ one has $$(p_Y(H)) \in \mathcal{O}_G(Y)$$, then $$I_G(Y,H) = (p_Y(H))$$ Since $$I_G(Y,H)$$ is the symmetry infimum and $$(p_Y(H))$$ is an orbit type, we have

$$
(H) \le I_G(Y,H) \le (p_Y(H)).
$$

Applying $$p_Y$$ to both sides, and using the idempotence of $$p_Y$$ together with the fact that isotropy subgroups in $$Y$$ remain unchanged under $$p_Y$$, we obtain

$$
(p_Y(H)) \le I_G(Y,H) \le (p_Y(H)),
$$

which implies $$I_G(Y,H) = (p_Y(H))$$.

We have thus answered the first question raised in the previous section, namely the necessary condition for isovariant maps. In the next section, we return to the second question, which concerns sufficiency. In machine learning, data are distributed only on a subset of the input space, and the map we can learn is only one element in a parameterized family of mappings. We therefore introduce a data model and a parameterized mapping model in the next section to study the existence and density of **almost isovariant maps**.

## Learning Guarantees under Assumptions

### Manifold Hypothesis on the Data

Following the definition in <d-cite key="milnor_topology_1990"></d-cite>, a $$C^r$$ manifold $$M$$ is, mathematically, a subset of a linear space $$X$$ such that for every $$x \in M$$, there exists a neighborhood $$W$$ of $$x$$ in $$X$$ for which $$W \cap M$$ is $$C^r$$-homeomorphic to an open subset of another linear space $$\mathbb{R}^d$$ Here, $$d$$ is called the dimension of $$M$$, and in this case we say that $$M$$ is a submanifold of $$X$$ Another common definition does not rely on an ambient embedding space; see <d-cite key="hirsch_differential_1976"></d-cite>. Unless otherwise specified, all manifolds mentioned below are assumed to be $$C^\infty$$ manifolds.

The manifold hypothesis is a standard assumption in machine learning. Under suitable regularity assumptions, we suppose that the data are distributed on a smooth compact submanifold $$M$$ of the input space $$X$$ By the Weierstrass theorem in analysis, compactness is equivalent to closedness and boundedness. We briefly explain the reasonableness of this assumption:
- This assumption covers some classical self-intersection phenomena at the theoretical level: a data manifold may arise as the image of a map, while the image of an immersion or submersion of a manifold may exhibit self-intersections. The image of an immersion or submersion of a compact manifold is a finite union of compact manifolds.
- The boundedness assumption on the manifold comes from the boundedness assumption on the data distribution.
- The closedness assumption covers the case where the data arise as inputs under well-posed constraints. For example, given $$n$$ independent differentiable constraint equations in the input space, the solution set is a closed submanifold of codimension $$n$$ in the input space.

The action of the symmetry group $$G$$ on the input space induces a smooth action of $$G$$ on each $$M_j$$, making $$M_j$$ a smooth $$G$$-submanifold of $$X$$ As a special kind of $$G$$-set, the fixed-point stratum $$(M_j)_{(H)}$$ is a disjoint union of smooth $$G$$-submanifolds of $$M_j$$ Although in modeling we parameterize maps from $$X$$ to $$Y$$, what we actually care about is the restriction of $$f$$ to $$M$$, since the data are supported only on $$M$$ Therefore, the real question is when an isovariant map from $$M$$ to $$Y$$ exists.

We now consider the following example, which shows that even if the inclusion relation of orbit types is satisfied, an isovariant map need not exist. In the next section, we will explain that isovariant maps can appear only when the multiplicities of irreducible representations are sufficiently large.

> **Counterexample.** For any compact Lie group $$G$$, consider an orthogonal action of $$G$$ on a representation $$X$$ Restricting the action to the unit sphere $$S$$ of $$X$$ makes $$S$$ into a $$G$$-submanifold. There exists a $$G$$-manifold $$M$$ such that $$\mathcal{O}_G(M) \subset \mathcal{O}_G(Y)$$, but there is no isovariant map in $$C_G^\infty(M,Y)$$. 
>
> In particular, for $$G=\mathbb{Z}_2$$, there exists a $$G$$-manifold $$M$$ such that $$\mathcal{O}_G(M) \subset \mathcal{O}_G(Y_0)$$, but there is no isovariant map in $$C_G^\infty(M,Y_0^r)$$ whenever $$r \le \dim M$$.

Therefore, we need to relax the definition of isovariant maps and consider an “almost” isovariant notion, namely, maps that preserve isovariance at almost every point of $$M_{(H)}$$. To do so, we need to introduce a measure on $$M$$ In linear spaces, one usually chooses the Lebesgue measure. For a finite union of compact manifolds $$M$$, one may instead choose the Hausdorff $$d$$-measure and define

$$
\mu_M = \mathcal{H}^d,
$$

where $$d$$ is the Hausdorff dimension of $$M$$; for a finite union of manifolds, this is the maximum of the dimensions $$\dim M_j$$ Under $$\mathcal{H}^d$$, a manifold of dimension greater than $$d$$ has infinite measure, a manifold of dimension less than $$d$$ has zero measure, and a compact manifold of dimension exactly $$d$$ has finite positive measure. We say that a map is **almost isovariant relative to $$Y$$** if, for each $$(H)\in\mathcal{O}_G(X)$$, all $$x\in M_{(H)}$$ except for a set of $$\mu_{M_{(H)}}$$-measure zero satisfy
$$
\rho_Y(G_x)=\rho_Y(G_{f(x)}).
$$
To analyze this further, for an equivariant map $$f$$ from $$X$$ to $$Y$$, and for $$(H)\in\mathcal{O}_G(X)$$ and $$(H')\in\mathcal{O}_G(Y)$$, we consider the set of points whose symmetry is raised from $$(H)$$ to $$(H')$$ under $$f$$:

$$
S_{(H)\to(H')}(f)
=
\{x\in X \mid (G_x)=(H),\ (G_{f(x)})=(H')\}.
$$

To prove that a map is almost isovariant, one needs to show that

$$
\sum_{\substack{(H') > (\rho_Y(H)) \\ (H') \in \mathcal{O}_G(Y)}}
\mu_{M_{(H)}}\bigl(S_{(H)\to(H')}(f)\bigr)=0.
$$

Therefore, an equivariant map from $$M_{(H)}$$ to $$Y$$ is almost isovariant to $$Y$$ if and only if $$(H)\in\mathcal{O}_G(Y)$$ and, for every $$(H')>(H)$$, the set $$S_{(H)\to(H')}(f)$$ has $$\mu_{M_{(H)}}$$-measure zero.

### Approximation Hypothesis on the Model

We next turn to assumptions on the model. For point-cloud encoding, we consider the TFN model. An important property of TFNs is that they satisfy a universal approximation theorem. In topology, this is equivalent to saying that the parameterized family $$\mathcal{F}$$ is dense in the space of equivariant maps $$C_G(X,Y)$$ with respect to the $$C^0$$ topology. In some literature, this topology is called the compact-open topology. Convergence in this topology is equivalent to the following: for any $$f \in C_G(X,Y)$$, any compact set $$K$$, and any $$\epsilon>0$$, there exists $$g \in \mathcal{F}$$ such that

$$
\max_{x\in K}\|f(x)-g(x)\|<\epsilon.
$$

For this reason, the compact-open topology is also sometimes referred to as the topology of uniform convergence on compact sets.

In fact, the TFN family $$\mathcal{F}$$ is even $$C^\infty$$-dense. That is, for any nonnegative integer $$k$$, any $$f \in C_G^\infty(V,W)$$, any compact set $$K$$, and any $$\epsilon>0$$, there exists $$g \in \mathcal{F}$$ such that

$$
\max_{x\in K}\|D^k f(x)-D^k g(x)\|<\epsilon,
$$

where $$D^k$$ denotes the $$k$$th-order differential operator. Here $$D^k f$$ is a tensor whose components are of the form

$$
\partial_1^{k_1}\cdots \partial_n^{k_n}f,
\qquad
k_1+\cdots+k_n=k,
\qquad
n=\dim V.
$$

In particular, when $$k=1$$, we have $$D^1f=Df$$, which can be represented by the Jacobian matrix.

A substantial portion of maps in a dense parameterized family typically reflects the **generic properties** of the mapping space. For equivariant maps, there is a generic property closely related to almost isovariance, namely, it reveals the dimension of the sets $$S_{(H)\to(H')}(f)$$ for a generic map. The following theorem shows that for expressive models with $$C^{\infty}$$ approximation capabilities, such as the TFN discussed, almost isovariance is a generic property, and full relative isovariance can be achieved by increasing representation multiplicity. As shown in Counterexample, this requirement is tight.

> **Theorem.** Consider maps from a $$G$$-representation space $$X$$ to $$Y$$, and let $$\mathcal{F}$$ be a smooth parameterized family that is $$C^\infty$$-dense in the space of smooth equivariant maps $$C_G^\infty(X,Y)$$, i.e., it satisfies a $$C^\infty$$ universal approximation theorem. Let $$\{M_j\}$$ be any finite collection of compact smooth $$G$$-submanifolds of $$X$$ Then for any nonnegative integer $$k$$, any $$f\in C_G^\infty(X,Y)$$, and any $$\epsilon>0$$, there exists $$g\in\mathcal{F}$$ such that
>  
> $$\max_j \max_{x\in M_j}\|D^k f(x)-D^k g(x)\|<\epsilon.$$
> 
> Furthermore, if $$Y$$ contains a representation $$\tilde{Y}^{\oplus r}$$ for an integer $$r > \max_j \{\dim M_j\}$$, where $$\tilde{Y}$$ itself satisfies $$(p_{\tilde{Y}}(H)) \in \mathcal{O}_G(\tilde{Y})$$, then $$g\vert_M$$ can be chosen to be isovariant relative to $$Y$$.

We can therefore answer **the second question** as follows.

**Q:** If the minimal symmetry matches the input symmetry, do there exist equivariant maps that preserve it?

**A:** No; but yes under almost isovariant mappings.

**Q:** Are such maps common (generic) among all equivariant mappings?

**A:** No; however, almost isovariance is generic. Moreover, under sufficient multiplicity, full isovariance is also generic.

## Computation of Symmetry Increase

Existing work related to the computation of orbit types is mainly developed in the study of bifurcation theory within dynamical systems. Such works primarily focus on orbit types associated with irreducible representations. In contrast, our interest lies in feature engineering for representation learning, where the feature spaces considered in practice usually contain irreducible representations with large multiplicities. Therefore, some of the relevant computational results in the literature need to be supplemented for our setting. We first introduce a criterion extended from Michel criterion in <d-cite key="michel_symmetry_1980"></d-cite>.

> **Criterion.** A necessary condition for a closed subgroup $$H$$ to be an inert subgroup in a $$G$$-representation $$V$$ is that, for every adjacent closed subgroup $$H \subsetneq H'$$, one has $$\dim V^{H'} < \dim V^H.$$ In particular, if $$V$$ can be written as an $$r$$-fold representation $$V_0^{\oplus r}$$ with $$r > \dim G$$, then this condition is also sufficient.

Although the above criterion is not sufficient for general representations, it is already adequate in most cases of our interest, for the following reasons.

**Reason 1 (Efficient to compute).** 
The Michel criterion is very convenient to compute, since it only involves adjacent subgroups, and the dimension of the fixed-point space can be obtained via the Weyl trace formula:

$$
\dim V^H = \frac{1}{|H|}\int_{H} \chi_{V}(h)\mathrm{d}h = \frac{1}{|H|}\int_{H} \mathrm{tr}(\rho_{V}(h))\mathrm{d}h
$$

This computation method will hereafter be referred to as chain recursion.

**Reason 2 (Suitable for finite groups).** 
For finite groups, the above criterion is sufficient. Moreover, in the feature space $$Y$$, the multiplicities of the selected irreducible representations are typically larger than $$\dim G$$ (for example, for $$G = SO(3)$$, we have $$\dim G = 3$$ ). Therefore, this criterion is usually sufficient in practice.

We can therefore answer **the third question** as follows.

**Q:** For a given data point, does its representation under an equivariant map admit a unique minimal level of symmetry?

**A:** For high-multiplicity representations, the symmetry infimum can be computed via a procedure:
- Enumerate all closed supergroups;
- Retain those whose conjugacy classes are valid orbit types by criterion;
- Return the minimal one.

In most cases that concern us, the properties of symmetry types in the feature space can be derived from the properties of the symmetry types of each $$r$$-fold irreducible representation $$V_i^{\oplus r}$$ We consider the behavior of orbit types and symmetry lower bounds under direct sums:
- $$\mathcal{O}_G(V_{1}) \cup \mathcal{O}_G(V_{2}) \subseteq \mathcal{O}_G(V_{1} \oplus V_{2})$$.
- $$I_G(V_{1} \oplus V_{2},H) \le I_{G}(V_{i},H)$$ for $$i=1,2$$.

These properties provide a direct mechanism for controlling the symmetry increase of an equivariant feature.

For irreducible representations, the symmetry lower bound of any closed subgroup in $$V$$ can be obtained either by exhaustively enumerating subgroup relations or by applying theorems on symmetry lower bounds. We do not present the detailed computation here. For $$G = SO(3), O(3)$$, the symmetry lower bounds for all closed subgroups can be found in <d-cite key="linreducing"></d-cite>. Here we only quote the results for the $$k$$-fold symmetry groups (i.e. the symmetry infimum $$I_{O(3)}(V_{l = l_{0}}^{\oplus r}, D_{kh})$$ for $$k > 2, r> 3,  l_{0}> 0$$ ), as they explain the several types of degeneracies that we initially observed.

|                                 | $$l_{0} < k$$                         | $$l_{0} < k$$                        | $$k \leq l_{0} < 2k$$                 | $$k \leq l_{0} < 2k$$                | $$l_{0} \geq 2k$$                     | $$l_{0} \geq 2k$$                    |
| ------------------------------- | ----------------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------- | ----------------------------------- | ---------------------------------- |
|                                 | $$l_{0}\ \mathrm{is}\ \mathrm{even}$$ | $$l_{0}\ \mathrm{is}\ \mathrm{odd}$$ | $$l_{0}\ \mathrm{is}\ \mathrm{even}$$ | $$l_{0}\ \mathrm{is}\ \mathrm{odd}$$ | $$l_{0}\ \mathrm{is}\ \mathrm{even}$$ | $$l_{0}\ \mathrm{is}\ \mathrm{odd}$$ |
| $$k\ \mathrm{is}\ \mathrm{even}$$ | $$(D_{\infty h})$$                    | $$(O(3))$$                           | $$(D_{kh})$$                          | $$(O(3))$$                           | $$(D_{kh})$$                          | $$(O(3))$$                           |
| $$k\ \mathrm{is}\ \mathrm{odd}$$  | $$(D_{\infty h})$$                    | $$(O(3))$$                           | $$(D_{\infty h})$$                    | $$(D_{kh})$$                         | $$(D_{2kh})$$                         | $$(D_{kh})$$                         |

Although we consider high-multiplicity representations here, <d-cite key="linreducing"></d-cite> points out that the conclusion remains the same for the case $$r = 1$$ Using these symmetry lower bounds, we finally obtain explanations for all the observed degeneracies:
1. **Full degeneration:** $$I_{O(3)}(V_{l=l_{0}}, D_{kh}) = (O(3))$$
2. **Axial degeneration:** $$I_{O(3)}(V_{l=l_{0}}, D_{kh}) = (D_{\infty h})$$
3. **Half degeneration:** $$I_{O(3)}(V_{l=l_{0}}, D_{kh}) = (D_{2kh})$$
4. **No degeneration:** $$I_{O(3)}(V_{l=l_{0}}, D_{kh}) = (D_{kh})$$

## Guideline for Choice of Equivariant Features

### For Orientation-dependent Tasks

We consider equivariant neural networks for orientation-related tasks. Specifically, we construct a task in which features extracted by an equivariant neural network are used to detect the rotational state of an input point cloud. We consider $$k$$-fold point clouds and use a TFN as the encoder. For rotation detection, we examine two types of rotations: one involves rotations only around the unique symmetry axis (2D rotations), while the other involves rotations around different axes (3D rotations). A classifier is then used to determine whether a rotation has occurred. After training, the classification accuracies are reported as follows:

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/rot_detech.png" class="img-fluid" %}
<div class="caption">
    Tab: Results of distinguishing k-fold structures rotated in 2D/3D space.
</div>

The experimental results are consistent with our analysis. An accuracy of 0.5 on 2D rotation detection indicates that the corresponding degeneracy is at least **axial degeneration**, while an accuracy of 0.5 on 3D rotation detection corresponds to **full degeneration**. 

This task shows that, for highly symmetric objects, restricting the representation to low-degree features may induce symmetry increase, which in turn leads to partial or complete loss of orientation information and limits the representational capacity of the encoder. To prevent this, one should **avoid non-trivial symmetry increase** and select feature components that preserve the input orbit type as much as possible. More precisely, for a given input symmetry $$(H)$$, after taking task-relevant invariances into account, the selected feature components should contain the orbit type $$(p_Y(H))$$.

### For Orientation-independent Tasks

We consider equivariant neural networks for property prediction. On the QM9 molecular property prediction dataset, we use a pretrained equivariant neural network as a feature encoder and perform scalar prediction based on the extracted features. After excluding rare symmetry types with fewer than three samples, we first compute all possible symmetries in QM9 and the corresponding cases of symmetry increase.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/qm9_sym_inc.png" class="img-fluid" %}
<div class="caption">
    Fig: Symmetry infimum of point group symmetry on the QM9 dataset. The meaning of the colors in the table are descripted in <d-cite key="linreducing"></d-cite>.
</div>

For computational efficiency, we adopt HEGNN <d-cite key="cen_are_2024"></d-cite> as our backbone. In the experiments, to compare the contributions of features with different degrees, we consider two settings: (1) using only features with $$l = l_{0}$$, and (2) using all features with $$l \leq l_{0}$$ In this blog, we report results only for the symmetry groups $$C_{2h}$$, $$C_{3h}$$, and $$T_{d}$$, while $$C_{1}$$ (the asymmetric case) is included as a baseline.

{% include figure.liquid path="assets/img/2026-04-15-Symmetry-Increase-and-Equivariant-Feature-Selection/qm9_res.png" class="img-fluid" %}
<div class="caption">
    Fig. Prediction MAE on the QM9 molecular dataset. Each boxplot shows the distribution of errors at a given degree, while diamond markers denote the corresponding mean MAE.
</div>

For $$C_{1}$$, the prediction performance based on features from a single degree is not strongly correlated with the degree itself. As more features are introduced, the prediction error gradually decreases, which is consistent with expectations. For $$C_{2h}$$, the features exhibit full degeneration when $$l$$ is odd; for $$C_{3h}$$, full degeneration occurs at $$l = 1$$; and for $$T_{d}$$, full degeneration occurs at $$l = 1, 2, 5$$ Accordingly, we observe a significant degradation in prediction performance on these degenerate features. In particular, for $$C_{2h}$$, $$C_{3h}$$, and $$T_{d}$$, introducing the fully degenerate feature at $$l = 1$$ increases the mean prediction error. By contrast, at $$l = 2$$, the features for $$C_{2h}$$ and $$C_{3h}$$ are not fully degenerate, and incorporating them significantly reduces the mean prediction error. However, for $$T_{d}$$, the feature at $$l = 2$$ is fully degenerate, and introducing it still leads to an increase in the mean prediction error.

The output symmetry reflects the dimensionality of the fixed-point subspace. Therefore, one should generally avoid components whose symmetry infimum indicates a severe compression of the fixed-point subspace. A particularly noteworthy case is that, for all components with $$l > 0$$, fully degenerate components should be avoided, since such features do not carry meaningful information.

## Conclusion

In this blog, we revisit the phenomenon of representational degeneration characterized in <d-cite key="linreducing"></d-cite>. We address three fundamental questions concerning the symmetry increase that determines the degree of degeneration in representation features. In the experimental part, we apply our theory to both orientation-related and orientation-invariant tasks, and provide simple theoretical guidelines for feature selection, which can serve as a basis for designing more reliable equivariant neural networks.