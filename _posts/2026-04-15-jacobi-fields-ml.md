---
layout: distill
title: Jacobi Fields in Machine Learning
description: >
  Jacobi fields are a concept from differential geometry that describe how neighboring geodesics
  on a curved manifold deviate from one another. This post provides an intuitive introduction
  to Jacobi fields and illustrates their usefulness for machine learning on Riemannian manifolds,
  including an approximation result connecting tangent-space quantities to geodesic distances.
date: 2026-04-15
future: true
htmlwidgets: true

authors:
  - name: Olga Zaghen

toc:
  - name: What are Jacobi Fields?
    subsections:
      - name: Basic Definitions in Differential Geometry
      - name: Jacobi Field Theory
  - name: Approximation Result and Applications to Machine Learning
  - name: "Example: Comparing RG-VFM and RFM Losses"
  - name: Future Applications
  - name: References

_styles: >
  .def-list dt {
    font-weight: 600;
    font-style: italic;
    padding: 0.5em 1em 0.2em;
    border-left: 3px solid #888;
    background: #f8f8f8;
    margin-top: 0.6em;
    border-radius: 0 4px 0 0;
  }
  .def-list dd {
    margin-left: 0;
    padding: 0.25em 1em 0.5em;
    border-left: 3px solid #ddd;
    background: #f8f8f8;
    font-size: 0.95em;
    line-height: 1.7;
    border-radius: 0 0 4px 0;
    margin-bottom: 0.1em;
  }
  figcaption {
    font-size: 0.85em;
    font-style: italic;
    color: #555;
    margin-top: 0.4em;
  }
  .ref-list {
    padding-left: 0;
    list-style: none;
    counter-reset: ref-counter;
  }
  .ref-list li {
    counter-increment: ref-counter;
    padding-left: 2.4em;
    position: relative;
    margin-bottom: 0.6em;
    font-size: 0.9em;
  }
  .ref-list li::before {
    content: '[' counter(ref-counter) ']';
    position: absolute;
    left: 0;
    font-weight: 600;
  }
---

The goal of this blogpost is to provide an intuitive definition of **Jacobi fields**,
a concept from differential geometry, and explain their usefulness for machine learning on curved manifolds.
In a nutshell, they are particularly relevant if you are trying to determine a relation between
the difference of two vectors $v_1$ and $v_2$ in a tangent space $T_p\mathcal{M}$ to a manifold
(or a power of it), and the geodesic distance between end-points of geodesics with those vectors
as initial velocities.

<div style="max-width: 70%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/jacobi_fields_gif.gif" class="img-fluid" %}
</div>

The content of this blogpost is based on (and expands) some findings from the paper Riemannian Variational Flow Matching for Material and Protein Design [1],
where this concept is used to relate a variational objective (geodesic distance between endpoints)
to the objective of Riemannian Flow Matching [2]. We refer to the variational model
as RG-VFM (Riemannian Gaussian - Variational Flow Matching) and to Riemannian Flow Matching as RFM.

The hope is that, by providing an accessible introduction to Jacobi fields and their applications
through this blogpost, their usefulness will find more applications in future works involving
vectors and points on curved manifolds. This post will answer the following questions:

1. [What are Jacobi fields?](#what-are-jacobi-fields)
2. [How can they be used in machine learning applications?](#approximation-result-and-applications-to-machine-learning)
3. [Example: comparing RG-VFM and RFM losses](#example-comparing-rg-vfm-and-rfm-losses)
4. [Future applications](#future-applications)

Let's dive deep into it!

## What are Jacobi Fields?

Let's start from the basics, with some intuitive definitions of objects in differential geometry.

### Basic Definitions in Differential Geometry

<dl class="def-list">
  <dt>Riemannian manifold \(\mathcal{M}\)</dt>
  <dd>A smooth mathematical space that looks flat up close but can be curved globally,
    equipped with a specific tool \(g\) called a <em>Riemannian metric</em> that allows
    you to measure distances and angles everywhere on it.</dd>

  <dt>Geodesic \(\gamma(\tau)\)</dt>
  <dd>A generalization of a straight line to a curved space, representing the locally
    shortest, unaccelerated path between two points on a manifold.</dd>

  <dt>Exponential map \(\exp_x(v)\)</dt>
  <dd>Takes a tangent vector \(v\) at a base point \(x\) on a manifold and maps it to the
    point \(y\) reached by traveling along the geodesic starting from \(x\) with initial
    velocity \(v\).</dd>

  <dt>Logarithmic map \(\log_x(y)\)</dt>
  <dd>The local inverse of the exponential map: takes a target point \(y\) on the manifold
    and returns the specific tangent vector \(v\) at base point \(x\) needed to reach \(y\)
    via a geodesic.</dd>
</dl>

In the following, we will always assume we are working with *simple manifolds*, i.e. with closed-form
geodesics (geodesics that can be parametrized through the exponential map) and such that the
geodesic distance between two points can always be expressed through the norm of the logarithmic
map between them. A geodesic can thus be parametrized through the exponential map as $\gamma(\tau) := \exp_x(\tau \cdot v)$.

<figure style="max-width: 55%; margin: 0 auto; text-align: center;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/exp-map.png" class="img-fluid" %}
<figcaption>The exponential map on a manifold: a tangent vector $v$ at $x$ is mapped to the point reached by following the geodesic with initial velocity $v$.</figcaption>
</figure>

### Jacobi Field Theory

Intuitively, the **Jacobi field** is a vector field along a geodesic $\gamma(\tau)$
on a Riemannian manifold $\mathcal{M}$ describing the variation between $\gamma(\tau)$ and other
"infinitesimally close geodesics".

<figure style="max-width: 65%; margin: 0 auto; text-align: center;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-1.png" class="img-fluid" %}
<figcaption>A Jacobi field $J(\tau)$ measuring the infinitesimal separation between neighboring geodesics.</figcaption>
</figure>

In our setting, we consider a **shooting family of geodesics**
$\{\gamma_s\}$, all starting from the same point $\gamma_s(0) := x_0 \in \mathcal{M}$,
determined by an initial velocity of the form:

$$\dot{\gamma}_s(0) = v^s := v + s\, w, \quad v, w \in T_{x_0}\mathcal{M}$$

where $sw$ represents the perturbation level. This family of geodesics can be parametrized as:

$$\alpha(s, \tau) := \gamma_s \colon \tau \mapsto \exp_{x_0}\!\bigl(\tau\,(v + sw)\bigr)$$

with $s \in [0,1]$ and $\tau \in [0,1]$. The **Jacobi field** is defined at each
timestep $\tau \in [0,1]$ as the vector field obtained by differentiating with respect to the
parameter $s$ and evaluated at $s=0$. Intuitively, this corresponds to measuring the
perturbation of geodesics in the family with respect to the "reference geodesic" at $s = 0$.

Let's go through the construction step by step:

**Step 1.** We start building the shooting family of geodesics, fully determined by the starting point and the initial velocities:

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/diagram-1.png" class="img-fluid" %}
</div>

The time parameter $\tau$ traverses the geodesics from their common starting point $x_0$ at $\tau = 0$ to their endpoints at $\tau = 1$.

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/diagram-2.png" class="img-fluid" %}
</div>

We can also visualize the effect of varying the parameter $s$, which translates to picking different geodesics in the family.

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-2.png" class="img-fluid" %}
</div>

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-3.png" class="img-fluid" %}
</div>

**Step 2.** We can now derive the Jacobi field by differentiating such a family with respect to $s$:

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-4.png" class="img-fluid" %}
</div>

One key observation (proved in [1]) is that the norm of $J(1)$ equals the geodesic distance between the endpoints
of geodesics $\gamma_0$ and $\gamma_1$:
$J(1) = \log_{\gamma_0(1)}\bigl(\gamma_1(1)\bigr)$,
hence $\|J(1)\| = g\bigl(\gamma_0(1),\, \gamma_1(1)\bigr)$.
This property will be exploited in the following, by interchangeably
considering $\|J(1)\|$ and $g\bigl(\gamma_0(1),\, \gamma_1(1)\bigr)$.

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-5.png" class="img-fluid" %}
</div>

At this point, we can also introduce in the picture the **initial time derivative** of the vector field $J(0)$ at $\tau = 0$, and we observe that, by definition and straightforward derivations, **$J'(0) = w$**.

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-6.png" class="img-fluid" %}
</div>

<div style="max-width: 80%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-7.png" class="img-fluid" %}
</div>

The relation between $J'(0)$ (the initial derivative of the Jacobi field) and $J(1)$
(its value at the endpoint) is the **central object of interest**. In flat Euclidean space
these are trivially related, because all tangent spaces coincide and geodesics are
straight lines. On a curved manifold, however, they differ — making it difficult to
establish an explicit analytical relationship between them.

## Approximation Result and Applications to Machine Learning

The central result that makes Jacobi fields interesting for machine learning is the relation
between $J'(0)$ and $J(1)$, expressed by the following proposition:

> **Proposition 1** *(Approximation result)*
>
> $J'(0)$ is a linear approximation of $J(1)$.

<figure style="max-width: 80%; margin: 0 auto; text-align: center;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-8-1.png" class="img-fluid rounded" %}
<figcaption>$J'(0)$ approximates $J(1)$ up to higher-order curvature terms.</figcaption>
</figure>

The full proof of the Proposition is in [1]; intuitively it consists of:

1. Computing the Taylor expansion of $J(\tau)$ centred at $\tau = 0$ and evaluated at $\tau = 1$.
2. Identifying $J'(0)$ as the linear term of such expansion.

A natural question you may still have in mind is: **but why**? Why would this result be useful in practice? Let's explore a practical example.

<div style="max-width: 22%; margin: 0 auto;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-9.png" class="img-fluid rounded" %}
</div>

In machine learning applications, MSE losses often involve minimizing the distance between
two vectors or two points. In Euclidean space there is no conceptual difference between
the two, because the space is flat, geodesics are straight lines, and the tangent space
is the same at every point. Denoting the Euclidean distance by $d_e$:

<figure style="max-width: 80%; margin: 0 auto; text-align: center;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-10.png" class="img-fluid" %}
<figcaption>In Euclidean space, minimizing the distance between vectors is equivalent to minimizing the distance between the corresponding endpoints. This equivalence breaks down on curved manifolds.</figcaption>
</figure>

For people familiar with flow matching-based models [3], this property is what allows one to
freely switch between predicting velocities and predicting endpoints when learning a
Flow Matching model, since the latter may offer practical advantages without any
analytical differences in flat space.

The issue arises on **curved spaces**: with non-zero curvature, the trivial
equivalence between minimizing vector differences and endpoint distances no longer holds.
This is precisely where the Approximation result comes in: although there is no equivalence, we
at least have an analytical understanding of their relation.

In practice, applying the Approximation result to a machine learning setting involves two steps:

1. **Matching:** Identify how the machine learning objects correspond to the Jacobi field elements.
2. **Derivation:** Derive an explicit analytic connection between the machine learning objects, exploiting the Approximation result from step 1.

We illustrate this procedure with an example in the next section.

## Example: Comparing RG-VFM and RFM Losses

In this section we carry out the Matching and Derivation in the specific setting of [1].
We are interested in exploring the connection between the RG-VFM objective (in the intrinsic setting) and the Riemannian Flow Matching objective [2].

The key conceptual difference between the two losses is that **RFM** minimizes
the squared distance between two tangent velocities at a point on the manifold, while
**RG-VFM** minimizes the squared geodesic distance between two points on the
manifold — the endpoints of two geodesics that have such vectors as initial velocities.
With this intuition, you may already see how the Jacobi field perspective comes into play.

Concretely, the following Matching result was proved in [1]:

> **Proposition 2** *(Matching result)*
>
> The following equalities hold for the RFM and RG-VFM losses:
>
> $$\mathcal{L}_{\mathrm{RFM}}(\theta)
>   = \mathbb{E}_{t,x_1,x}\!\left[\left\|u_{t}(x \mid x_1) - v_{t}^\theta(x)\right\|_{\mathbf{g}}^2\right]
>   = \mathbb{E}_{t,x_1,x}\!\left[\left\|J'(0)\right\|_{\mathbf{g}}^2\right]$$
>
> $$\mathcal{L}_{\mathrm{RG\text{-}VFM}}(\theta)
>   = \mathbb{E}_{t,x_1,x}\!\left[\left\|\log_{x_1}\!\bigl(\mu_t^{\theta}(x)\bigr)\right\|_{\mathbf{g}}^2\right]
>   = \mathbb{E}_{t,x_1,x}\!\left[\left\|J(1)\right\|_{\mathbf{g}}^2\right]$$

Once the connection is drawn, the last step is to exploit Proposition 1 to relate
the two losses. The difference between the loss values and the Jacobi field quantities
$J(1)$ and $J'(0)$ involves taking the squared norm, which affects the Derivation:

> **Proposition 3** *(Final Derivation)*
>
> The RFM and RG-VFM losses are related through the manifold curvature:
>
> $$\mathcal{L}_{\mathrm{RG\text{-}VFM}}(\theta) = \mathcal{L}_{\mathrm{RFM}}(\theta)
>   + \underbrace{\mathbb{E}_{t,x_1,x}\!\bigl[\mathcal{C}(R,\, J'(0),\, v)
>   + \mathcal{E}_{\mathrm{higher}}\bigr]}_{\text{curvature-dependent term}}$$

<figure style="max-width: 80%; margin: 0 auto; text-align: center;">
{% include figure.liquid path="assets/img/2026-03-24-jacobi-fields-ml/image-11.png" class="img-fluid" %}
<figcaption>Schematization of how the RG-VFM and RFM losses fall into the Jacobi fields perspective.</figcaption>
</figure>

The **curvature functional** $\mathcal{C}$ captures how the manifold's geometry affects the
loss comparison, encoding first- and second-order effects of curvature on geodesic deviation.
Thus, RG-VFM *implicitly* captures the full geometric structure through the exact Jacobi field $J(1)$,
while RFM uses only the linear approximation $J'(0)$.

In summary, RG-VFM was introduced as an alternative to RFM for learning a velocity field
on a manifold, providing a variational formulation whose objective fully captures
higher-order curvature effects, unlike RFM. This results in generally different objectives
on curved manifolds. In Euclidean space, however, the RFM objective reduces to CFM [3],
while RG-VFM reduces to VFM [4] — and these two become equivalent under appropriate
normalization.

## Future Applications

In conclusion, the Approximation result could be useful in practice to
analytically relate quantities that are trivially connected in Euclidean space but whose
relationship would otherwise be obscure on curved manifolds. This could be done by
properly adapting the Matching and Derivation procedure to different settings of interest,
and we hope that this short introduction makes these concepts accessible and inspires
further applications in machine learning.

## References

<ol class="ref-list">
  <li>Zaghen, Olga, et al. "Riemannian Variational Flow Matching for Material and Protein Design." arXiv preprint arXiv:2502.12981 (ICLR 2026).</li>
  <li>Chen, Ricky T. Q. and Lipman, Yaron. "Flow Matching on General Geometries." <em>arXiv preprint arXiv:2302.03660</em> (ICLR 2024).</li>
  <li>Lipman, Yaron, et al. "Flow Matching for Generative Modeling." <em>arXiv preprint arXiv:2210.02747</em> (ICLR 2023).</li>
  <li>Eijkelboom, Floor, et al. "Variational Flow Matching for Graph Generation." <em>Advances in Neural Information Processing Systems</em> 37 (2024): 11735–11764.</li>
</ol>
