---
layout: distill
title: 4-Dimensional Objects as a Tool to Study Symmetry Learning in Humans and Machines
description: We propose four-dimensional Shepard-Metzler shapes as a tool to study symmetry learning in humans and machines.
date: 2026-04-10
future: true
htmlwidgets: true

authors:
  - name: Raihan Gafur

# must be the exact same name as your blogpost
bibliography: 2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines.bib

# table of contents
toc:
  - name: Introduction
    subsections:
      - name: "Learning symmetries from data: An open problem in ML"
      - name: "Visual object rotation: A canonical and unresolved challenge"
      - name: "Humans seem to solve it, but how?"
      - name: The problem of prior experience
  - name: 4D Objects as a Tool
    subsections:
      - name: The experimental idea
      - name: What is a 4D object?
      - name: Projective geometry of a 4D object
      - name: Building 4D Shepard-Metzler shapes
  - name: "A Geometric Curiosity: Mirroring a 3D Shape in 4D Space"
  - name: "The Human Experiment: A Pilot Study"
  - name: Results and Discussion
  - name: "Why Do the Shapes Look So Strange? Todd's Rules and Perceptual Breakdown"
  - name: Conclusions and Future Directions
  - name: Acknowledgement

---

## Introduction

### Learning symmetries from data: An open problem in ML

Symmetries are everywhere in the physical world. A chair is still a chair whether it faces left or right, upright or inverted. A molecule retains its identity across rotations. A handwritten digit “7” is recognizably a “7” from multiple viewpoints. In mathematics, these regularities are formalized as group actions, transformations that leave an object's identity unchanged. The ability to understand symmetry lies at the heart of both human perception and modern machine learning. Transformations such as rotation, reflection, and invariance are far from mere geometric curiosities; they define how objects remain identifiable across changes in viewpoint. Despite their importance, learning such transformations directly from data remains an open problem. Recent discussions on symmetry learning suggest that current models struggle to infer structured transformations without strong inductive biases <d-cite key="perin2025ability"></d-cite><d-cite key="dinh2026latent"></d-cite><d-cite key="bronstein2021geometric"></d-cite>.

A particularly revealing example of this limitation is visual object rotation. While humans perform such tasks with apparent ease, machine learning systems often fail to generalize beyond the transformations explicitly seen during training. This gap raises a fundamental question: what enables a system to truly learn symmetry?

### Visual object rotation: A canonical and unresolved challenge

Of all the symmetries present in real-world data, the rotation of visual objects is perhaps the most studied and the most practically important. When we look at an object, the image it projects onto our retina changes dramatically depending on our viewpoint. Yet we recognize objects robustly across viewpoints; a skill so automatic that we rarely notice we are doing it.

The challenge for current AI systems is not merely academic. Ollikka et al. (2024) compared humans and state-of-the-art deep networks on a task of recognizing everyday objects seen in unusual poses. Their findings were striking: humans excelled at the task, while state-of-the-art vision networks, including EfficientNet, ViT, SWIN, BEiT, and ConvNext, were systematically brittle <d-cite key="ollikka2024comparison"></d-cite>. Large vision-language models like Claude 3.5, GPT-4, and Gemini 1.5 were also tested; most showed the same brittleness, with Gemini being a notable exception <d-cite key="ollikka2024comparison"></d-cite>.

### Humans seem to solve it, but how?

The human ability to mentally rotate objects was famously documented by Shepard & Metzler (1971) in one of the most elegant experiments in cognitive science <d-cite key="shepard1971mental"></d-cite>. Participants were shown pairs of 3D objects, abstract shapes made of connected cubes; now known as Shepard-Metzler shapes, and asked to judge whether the two were identical or mirror images. The reaction time increased linearly with the angular difference in orientation between the two objects, as if participants were physically rotating one object in their minds to align it with the other. This landmark finding revealed that the mind does not simply compare images: *it simulates spatial transformations in the mind, turning cognition itself into a form of physical geometry.*

Subsequent decades of research built on this foundation. Jolicoeur (1985) showed that naming times for sketches of everyday objects were proportional to the angular departure from their canonical upright orientation, thereby extending the mental rotation paradigm beyond abstract shapes to the full repertoire of familiar objects <d-cite key="jolicoeur1985time"></d-cite>. The implication is that human object recognition, at least for non-canonical viewpoints, involves a time-consuming mental transformation, likely recruiting recurrent processes in the brain rather than purely feedforward processing <d-cite key="khazoum2025deep"></d-cite>.

The human brain appears to have a mechanism that conventional deep networks lack: the ability to perform something like analog spatial simulation, to rotate representations in a mental workspace, and to compare them. Whether this is achieved through dedicated neural circuitry (possibly partly innate, as suggested by studies of newborn chicks that can recognize novel 3D objects from birth <d-cite key="wood2015chicken"></d-cite>) or through a lifetime of embodied experience with a 3D world remains a major open question. But whatever its origin, the human ability is real, robust, and faster than any explicit algorithmic rotation search; Perin & Deny (2025) describe it as a mechanism for learning and exploiting the non-local symmetric structure of the world <d-cite key="perin2025ability"></d-cite>.

### The problem of prior experience

There is, however, a subtle confound lurking in all of this. When humans successfully recognize a rotated chair, or mentally rotate a Shepard-Metzler shape, how much of their performance is due to a genuine, generalizable capacity for symmetry learning and how much is due to the staggering amount of prior visual experience they bring to the task?

From birth, humans live in a three-dimensional world. They see objects from multiple viewpoints, handle them, and walk around them. By the time an adult participant sits down in a mental rotation experiment, they have accumulated millions of hours of rich, multimodal 3D experience. This experience is embodied, motor-linked, and comes paired with proprioceptive feedback that reinforces the connection between object identity and viewpoint transformation. No current machine learning model comes close to this depth of grounded, multimodal training, not even the largest vision-language models, which are trained on images and text but not on the continuous, interactive, motor-linked stream of experience that shapes human perception.

This creates a genuine ambiguity when comparing human and machine performance on visual rotation tasks. Is human superiority evidence that the human brain has a principled mechanism for learning object symmetries from relatively little data? Or is it simply evidence that humans have been exposed to an unfathomably larger and richer training set than any machine? In the three-dimensional case, we cannot easily come to any conclusion; there is no such thing as a truly novel 3D object for a human, because any 3D shape, however abstract, is made of familiar local features, curvatures, junctions, and surfaces that the visual system has processed countless times before.

This is precisely the problem that motivates our proposed approach.

---

## 4D Objects as a Tool

### The experimental idea

Our central proposal is simple: *use four-dimensional objects as a tool to create **genuinely novel** visual stimuli for human participants*.

A 4D object, a shape that exists in four spatial dimensions, cannot be perceived directly by the human visual system. Its 2D projection, when rendered on a screen, does not resemble any shape that participants have encountered in their daily lives. The local image features are unfamiliar. The global structure is unfamiliar. There is no learning prior to linking a particular viewpoint to the object's identity, because the object's identity is itself defined in a space our visual systems have never navigated. Humans have shown some capacity for four-dimensional spatial reasoning in controlled settings <d-cite key="ambinder2009human"></d-cite><d-cite key="aflalo2008four"></d-cite>. Here, we propose to place participants in a more demanding position closer to that of a machine learning model confronted with out-of-distribution data, by asking them to recognize a new class of genuinely novel 4D objects from new viewpoints, with no prior experience and no guarantee that familiar 3D spatial heuristics will transfer.

This allows to ask a sharper version of the question motivating the field: **given a genuinely novel symmetric transformation (rotation in 4D space), how quickly can humans and machines learn to generalize across it? And how does performance change as participants gain experience with the objects?**

To illustrate this approach, we can adapt the classic Shepard-Metzler paradigm. In the original experiment, participants compared pairs of 3D shapes made of connected cubes and judged whether they were identical or mirror images. Here we will extend this to four dimensions: we will build 4D Shepard-Metzler shapes, abstract structures made of 12 connected 4D cubes (hypercubes), and show participants pairs of these shapes, rendered as static 2D projections. The task will be the same: same shape, or mirror image?

### What is a 4D object?

Before we describe what we built, we need to explain what a 4D object is.

In everyday life, we inhabit three spatial dimensions: width *(x)*, height *(y)*, and depth *(z)*, which we denote as *(x, y, z)*. A 3D object is defined by its extent along all three axes. The fourth spatial dimension, which we call *"w"*, is a direction perpendicular to all three of *x*, *y*, and *z* simultaneously, resulting in *(x, y, z, w)*, something our visual systems cannot directly perceive, but which can be defined mathematically without contradiction.

Just as a 3D object can cast a 2D shadow on a flat surface, a 4D object can be projected into 3D space, and then that 3D projection can be projected further onto a 2D screen. The most famous example of a 4D object is the Tesseract (term coined by Charles Howard Hinton <d-cite key="hinton_wikipedia"></d-cite>), the 4D analog of a cube <d-cite key="tesseract_wikipedia"></d-cite>. Where a cube has 8 vertices, 12 edges, and 6 square faces, a tesseract has 16 vertices, 32 edges, 24 square faces, and 8 cubic cells. When a tesseract rotates, for instance, through the z-w plane, its 3D projection appears to fold in and out of itself, producing a visual effect unlike anything in ordinary 3D experience.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/tesseracts.gif" class="img-fluid" caption="Figure 1: Tesseract rotating in its z-w plane." %}

To build our stimuli, we connected 12 tesseracts end-to-end in random configurations, with 90-degree elbow angles between consecutive segments, exactly as in the original 3D Shepard-Metzler shapes. Each such configuration and its 4D mirror image form one trial in our experiment.

### Projective geometry of a 4D object

The key mathematical tool that makes all of this possible is **perspective projection**, the same principle that makes train tracks appear to converge at the horizon and makes the far face of a cube look smaller than its front face. It mimics how our eyes and cameras work: light rays from different points on an object converge at a single focal point, projecting the 3D world onto a 2D surface in a way that preserves the intuitive sense of depth.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/perspective_projection.png" class="img-fluid" caption="Figure 2: Perspective projection." %}

It works like this; if an object is at distance *d* from the eye (or camera lens), and has a real size *x*, then if we place a screen at distance *d'* to see the object, it will appear with a projected size *x'*. Mathematically, this effect can be described by a simple ratio derived from the geometry of triangles, a relationship that goes all the way back to Thales’ theorem in Euclidean geometry <d-cite key="thales_theorem_wikipedia"></d-cite><d-cite key="allman1889greek"></d-cite>:

$$
\frac{x'}{d'} = \frac{x}{d}
$$

This projection mechanism is very important here. We are trying to transform a 4D object into a 3D representation, then project it onto a 2D screen that we can actually see. Without applying the perspective projection rule, this entire process would fall apart. The shapes would look flat, distorted, and meaningless. 

If we apply simple linear projection from 3D to 2D, which simply discards the depth axis, the front and rear faces of an object would appear the same size, even though in reality, the front face should appear larger than the rear one. Therefore, perspective projection gives our visualization a sense of depth, making it feel real.

In other words, perspective projection is what allows us to see higher dimensions with a sense of depth. It bridges the mathematical world of four-dimensional geometry with the visual world our eyes can understand. Without perspective projection, both steps would produce flat, distorted representations with no sense of depth, meaningless to the human visual system. Perspective projection is what gives our 4D renderings their uncanny sense of spatial structure, however alien that structure may feel.

### Building 4D Shepard-Metzler shapes

We build our stimuli step by step, progressing from a single transparent tesseract to a fully opaque 12-tesseract Shepard-Metzler shape.

**A single tesseract:** We define all 16 vertices of a tesseract using 4-tuples *(x, y, z, w) ∈ {±1}⁴*, apply the two-step perspective projection, and render the result with PyOpenGL using 80-90% opacity, solid enough to look like a real object, transparent enough that some inner edges remain visible. We choose a non-special viewpoint (one that does not align with any symmetry axis of the tesseract) because it reveals the full strangeness of 4D geometry and makes the tesseract effect most apparent when the object rotates.

**Scaling to 12 tesseracts:** To build a full Shepard-Metzler configuration, we develop a custom shape generator that interprets a sequence of directional tokens. Each character corresponds to a unit translation along one of the eight principal axes of 4D space:

| Character | Direction |
| :-------: | :-------: |
| R / L | ± x axis |
| U / D | ± y axis |
| F / B | ± z axis |
| O / I | ± w axis |

For example, the path `"UFFFLLDDDOO"` places 12 tesseracts consecutively in 4D space, turning in multiple directions, including the fourth dimension. The resulting structure is centered at the origin before rendering. Together, we generate 10 distinct 4D shapes using different path strings and produce a 4D mirror image of each one.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/4d_object.gif" class="img-fluid" caption="Figure 3: Rotating 4D Shepard-Metzler shape." %}

The resulting 4D Shepard-Metzler shapes, when shown as static 2D images at different rotation angles, formed our experimental stimuli, images that no human participant had ever seen anything like before.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/metzler_shape_gui1.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/metzler_shape_gui2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figures 4 and 5: GUI screenshots of the experiment setup showing pairs of 4D Shepard-Metzler shapes.
</div>

---

## A Geometric Curiosity: Mirroring a 3D Shape in 4D Space

Before describing the experiment and result, we pause for a geometric observation that is both counterintuitive and fundamental to understanding our stimuli, and which connects directly to a broader curiosity about dimensionality and symmetry.

In 3D space, a mirror image of an object cannot be obtained from the original by any rotation. A left hand cannot be made to look like a right hand simply by rotating it; the two are related by a reflection, not a rotation, and reflections require stepping outside the space. The formal reason is that an n-dimensional mirror transformation requires an (n+1)-dimensional rotation to be realized as a continuous rigid motion <d-cite key="porteous1995clifford"></d-cite>.

This idea generalizes elegantly through the dimensions. Consider a 2D shape, such as the letter "L". Within the 2D plane, there is no way to rotate "L" into its mirror image without leaving the plane.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/2d.png" class="img-fluid" caption="Figure 6: Continuous rotation of a 2D \"L\" shape within the plane. The shape changes orientation but never becomes its mirror image." %}

But if we embed the "L" in 3D space, we can lift it slightly out of the plane, rotate it around an axis in the third dimension, and lay it back down, and the result is the mirror image of the original "L". The third dimension provides the necessary freedom.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/2d_in_3d.png" class="img-fluid" caption="Figure 7: Rotation of a 2D \"L\" embedded in 3D space. The red circles indicate the initial shape and its mirror image; the blue circle highlights the critical edge-on configuration during the rotation." %}

The same logic applies one dimension up. A 3D object and its mirror image are distinct in 3D; no rigid rotation in 3D can map one to the other. But if we embed the 3D object in 4D space, a continuous 4D rotation can map it to its 3D mirror image. When this 4D rotation is projected back into 3D, the result appears, from the perspective of a 3D observer, as a reflection, even though in the full 4D geometry it was simply a smooth, rigid rotation.

{% include figure.liquid path="assets/img/2026-04-13-4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/3d_in_4d.png" class="img-fluid" caption="Figure 8: A 3D Shepard-Metzler object rotating through 4D space. The black circles mark the initial shape and its mirror image, while the blue circle highlights a critical edge-on configuration during the rotation." %}

This has a concrete consequence for our experiment. What we call a "mirrored" 4D Shepard-Metzler shape is obtained by reflecting one of the four coordinates, for instance, reversing the x-axis. This produces a shape that is genuinely distinct from the original in 4D space and cannot be superimposed on it by any 4D rotation.

However, a 4D mirror is to 4D objects what a 3D mirror is to 3D objects, a genuinely distinct transformation. And just as 3D mirrors make our same-versus-mirrored task hard for 3D objects, 4D mirrors make it even harder for 4D objects, because the participant's visual system has no experience with 4D chirality.

---

## The Human Experiment: A Pilot Study

With our stimuli ready, we conducted a pilot experiment with one participant. While the sample was too small for broad generalizations, this pilot was designed to assess feasibility, reveal qualitative patterns, and inform the design of larger-scale studies.

The experiment was structured around three types of trial sessions, each addressing a distinct condition:

**Condition 1 - Random shapes, no feedback (5 sessions × 10 trials):** On each trial, a randomly selected 4D Shepard-Metzler shape from a set of 10 distinct shapes was shown as a pair of static 2D images, rendered from two different rotation angles. The participant judged: same shape, or mirror image? No feedback was given.

**Condition 2 - Random shapes, with immediate feedback (10 sessions × 10 trials):** The structure was identical to Condition 1, but immediately after each response, the participant was shown whether they were correct or incorrect. This feedback was intended to allow the participant to gradually calibrate their judgment strategy for distinguishing same versus mirrored 4D shapes.

**Condition 3 - One familiar shape, no feedback (5 sessions × 10 trials):** A single 4D Shepard-Metzler shape (path `UFFFLLDDDOO`) was shown repeatedly across all trials. On each trial, the same object appeared at a different rotation angle, paired with either another view of itself or its 4D mirror image. The participant responded without feedback.

In all conditions, the rotation angles ranged from *0°* to *360°* in *15°* increments across the *z-w* and *x-w* planes, and trial order was fully randomized across shape identity, rotation angle, and mirror status.

---

## Results and Discussion

Across 200 total trials and 20 sessions, a clear and theoretically interpretable pattern emerged.

| Condition | Correct / Total | Acc. (%) | *p*-value | Interpretation |
| --------- | :-------------: | :------: | :-------: | -------------- |
| Random shapes, no feedback | 28 / 50 | 56 | 0.2399 | Slightly above chance |
| Random shapes, with feedback | 64 / 100 | 64 | 0.0033 | Above chance |
| One familiar shape, no feedback | 47 / 50 | 94 | $$ 1.85 \times 10^{-11} $$ | Strongly above chance |

We also report results after excluding trials with angular difference ≤ 45°, to control for easy low-angle comparisons:

| Condition | Correct / Total | Acc. (%) | *p*-value | Interpretation |
| --------- | :-------------: | :------: | :-------: | -------------- |
| Random shapes, no feedback | 20 / 39 | 51 | 0.5000 | At chance |
| Random shapes, with feedback | 40 / 70 | 57 | 0.1410 | Slightly above chance |
| One familiar shape, no feedback | 35 / 37 | 95 | $$ 5.12 \times 10^{-9} $$ | Strongly above chance |

**Condition 1 (random shapes, no feedback):** The participant answered correctly on 28 out of 50 trials (56%). Performance fluctuated around chance level across sessions (ranging from 4/10 to 7/10 correct), with no clear monotonic learning trend. A binomial test against the 50% null yielded p = 0.2399, not statistically distinguishable from chance. In other words, when both the object identity and the viewing angle varied trial-to-trial without a corrective signal, the participant could not reliably discriminate between the same and mirrored 4D shapes.

**Condition 2 (random shapes, with feedback):** Accuracy improved to 64 out of 100 trials (64%), and the binomial test yielded p = 0.0033, reliably above chance. Performance started around 7/10 per session, peaked at 9/10 in session 5, then declined toward chance in later sessions. This pattern suggests that feedback supports the discovery of useful decision heuristics, but those heuristics do not generalize strongly across the diverse set of random 4D shapes.

**Condition 3 (one familiar shape, no feedback):** The participant gave 47 correct responses out of 50 (94%), with the binomial test yielding p = 1.85 × 10⁻¹¹. Performance approached the ceiling quickly, stabilizing at 9-10/10 correct after the first session, and remained stable throughout without any corrective feedback at all.

To further probe whether performance in Conditions 1 and 2 might be driven by easy low-angle comparisons where the two images in a pair look nearly identical and can be distinguished by local image similarity rather than genuine rotational reasoning, we reran the binomial tests after excluding trials with angular differences of 45° or less. The results showed that the performance in Condition 1 (p = 0.50) and Condition 2 (p = 0.1410) became non-significant, but Condition 3 remained strongly above chance (p = 5.12 × 10⁻⁹) at 95% accuracy.

This pattern suggests that low-angle comparisons may contribute to above-chance performance for unfamiliar 4D shapes, whereas robust performance at larger angular disparities depends strongly on familiarity with a specific shaped object. Together, these findings reinforce the conclusion that object familiarity plays a critical role in supporting stable and reliable recognition of 4D object geometry across rotations and mirror transformations.

Notably, the familiar-shape result mirrors findings from the developmental literature on mental rotation in infants and young children. Studies show that even very young children can develop object-specific rotation strategies through repeated exposure to a single object before generalizing these strategies to other objects <d-cite key="frick2013development"></d-cite><d-cite key="frick2013mental"></d-cite>. In a sense, adult participants encountering 4D objects are in the same epistemic position as infants encountering 3D objects for the first time, genuinely without prior experience of the geometry, and initially dependent on object-specific familiarity before any more general rotation ability can emerge.

---

## Why Do the Shapes Look So Strange? Todd's Rules and Perceptual Breakdown

The participant in our pilot reported an unusual qualitative experience beyond mere task difficulty: the 4D shapes did not feel like solid objects at all. They appeared unstable, ambiguous, almost phantom-like, more like overlapping geometric patterns than coherent physical forms. This observation is not incidental. It points to a fundamental feature of how human vision interprets 2D images as 3D objects.

Todd (2004) examined the visual cues the human visual system uses to infer 3D shape from 2D images: shading gradients, occlusion contours, texture gradients, specular highlights, and the structured ways in which these cues combine to specify surface normals and curvature. Crucially, these cues are not neutral descriptors of the world; they are deeply tuned to the statistics of three-dimensional surfaces in a three-dimensional environment. Our visual system has learned to exploit them because they reliably indicated the presence of solid surfaces throughout our evolutionary and developmental history <d-cite key="todd2004visual"></d-cite>.

What happens when these cues are applied to the 2D projection of a 4D object? They break down. A 4D Shepard-Metzler shape, rendered with perspective projection onto a 2D screen, generates image features that are inconsistent with any solid 3D surface. Shading and occlusion patterns that would, for a genuine 3D object, indicate a convex face with a consistent light source instead reflect the geometry of a 4D structure projected through two dimensions, a geometry that was never part of the statistical regularities that shaped our visual system. The result is an object that the brain cannot interpret as solid, because the visual cues it provides violate the implicit rules of 3D solidity that our perceptual system takes for granted.

It is an informative feature of the experiment rather than a rendering failure. It tells us that human 3D perception relies not just on the global structure of an image, but on a rich set of local image statistics specifically calibrated to 3D geometry. When those statistics are violated because the object genuinely exists in a higher-dimensional space, perceptual stability falters at a very basic level, before any mental rotation can even begin. Resolving this perceptual instability may itself be part of what future participants learn as they gain familiarity with 4D objects, a process that may parallel, at an accelerated pace, the slow developmental acquisition of 3D perceptual competence in infants.

---

## Conclusions and Future Directions

We introduce 4D Shepard-Metzler shapes as a new experimental tool for probing visual symmetry learning in humans and machines. Our pilot experiment confirms what the geometry predicts: **4D objects are genuinely novel to the human visual system**. Without feedback and with varied shapes, participants perform at chance. They cannot readily perform mental rotation in 4D space without prior experience, unlike in the 3D case, where a lifetime of embodied experience primes the visual system to interpret rotated objects almost effortlessly.

But the results also contain a more optimistic signal. When feedback is provided, participants might improve above chance (though this needs further study to confirm), and when a single shape becomes familiar, performance approaches the ceiling even without feedback, even at large rotation angles. The question this raises is clear and central: **How much experience and what kind is needed before a genuinely new symmetric transformation can be learned and generalized?**

This question could be pursued in two directions.

For **humans**, one can run extended training experiments tracking whether performance on random 4D shapes improves as participants accumulate more exposure. Shape complexity could be varied; dynamic, rotating stimuli could be provided rather than static snapshots; and reaction times could be measured to test whether the linear RT-versus-angle relationship documented by Shepard & Metzler (1971) for 3D shapes emerges for 4D shapes after sufficient training.

For **machines**, one could run two studies. First, train a model on 3D Shepard-Metzler shapes and test it on 4D shapes to probe how well 3D spatial reasoning transfers to a higher-dimensional setting. Second, train a model directly on 4D Shepard-Metzler shapes and track its learning curve, comparing it directly to the human one. Both comparisons will shed light on whether the mechanisms enabling human mental rotation have analogs in current deep learning models, or whether new architectures, perhaps along the lines of the latent equivariant operators proposed by Dinh & Deny (2026), are needed to bridge the gap.

We believe that 4D objects offer a unique experimental window not available in any other domain: a visual task that is simultaneously natural in structure (it obeys the same mathematical logic as 3D rotation) and genuinely novel in content (no amount of 3D experience prepares you for it). As such, it may provide one of the cleanest possible tests of how intelligence, human or artificial, learns to generalize with respect to geometric symmetry.

The fourth dimension turns out to be not just a curiosity but an experiment.

---

## Acknowledgement

To be added upon acceptance.