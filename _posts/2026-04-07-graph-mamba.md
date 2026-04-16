---
layout: distill
title: Graph Mamba - Rethinking Graph Learning
description: Graph Mamba replaces message passing by turning local subgraphs into token sequences processed by selective state space models. We explain the idea with an interactive Cora demo and a minimal reference implementation.
date: 2026-04-07

authors:
  - name: Vladislav Kalinichenko
    affiliations:
      name: Innopolis University
  - name: Polina Korobeinikova
    affiliations:
      name: Innopolis University

bibliography: 2026-01-22-graph-mamba.bib
related_posts: false
toc:
  - name: Foundations - Understanding Graphs and Sequence Models
    subsections:
      - name: The Limitations of Message Passing
      - name: Enter Mamba - The Selective State Space Model
  - name: From Random Walks to Tokens
  - name: Encoding Subgraphs with Local Encoders
    subsections:
      - name: GCN Encoder - Three Phases
  - name: The Mamba Block - Selective State Spaces
    subsections:
      - name: From Tokens to Sequences
      - name: Token Ordering
      - name: The Core Mechanism
      - name: The Selective Mechanism
  - name: Bidirectional Sequence Modeling
    subsections:
      - name: The Bi-Mamba Architecture
  - name: Two-Level Processing in Graph Mamba
  - name: End-to-End Architecture
  - name: Results on Cora
---

<script src="//unpkg.com/3d-force-graph"></script>
<script src="//unpkg.com/d3"></script>
<script>
  // Distill shows a loading/progress bar; remove it once the DOM is ready so it doesn't cover the 3D canvases.
  document.addEventListener('DOMContentLoaded', () => {
    const progress = document.getElementById('progress');
    if (progress) progress.remove();
  });
</script>
<section class="intro-section">
<p>
<strong>Graphs</strong> are everywhere: social networks, molecular structures, knowledge bases, and recommendation systems. 
            Traditional Graph Neural Networks (GNNs) have been the go-to approach for learning from graph-structured data, 
            but they face fundamental challenges with long-range dependencies and computational efficiency on large graphs.
        </p>
<p>
            This interactive article introduces <strong>Graph Mamba Networks</strong> - a novel approach that replaces traditional 
            message passing with <strong>Selective State Space Models (SSMs)</strong>. Instead of iteratively aggregating information 
            from neighbors, we linearize graph neighborhoods into sequences and process them efficiently using Mamba's selective gating mechanism. We empirically validate this methodology through experiments on the <strong>Cora citation network</strong> to demonstrate its efficacy in node classification tasks.
        </p>
<p>
            References used in this tutorial: Graph Mamba <d-cite key="behrouz2024graphmambalearninggraphs"></d-cite> and the Cora dataset <d-cite key="cora2001"></d-cite>.
        </p>
</section>
<section style="width: 100%; margin: 0 auto;">
  <div style="position: relative; width: 140%; margin-left: -20%; margin-top: 1.5rem; margin-bottom: 1.5rem;">
    <div style="position: relative; width: 100%; height: 560px; border: 1px solid #e5e7eb; background: #ffffff;
        border-radius: 8px;
        overflow: hidden; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #1f2937;">
      <div class="viz-footer" style="
        position: absolute; 
        bottom: 0; 
        left: 0; 
        width: 100%;
        z-index: 10; 
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(2px);
        color: #4b5563; 
        padding: 6px 0; 
        font-size: 11px; 
        text-align: center;
        border-top: 1px solid rgba(229, 231, 235, 0.5);
        font-family: monospace;
        pointer-events: none;">
        Left-click: rotate • Scroll: zoom
      </div>
      <div class="interactive-figure if--600" id="cora-viz" style="
          position: absolute; /* Absolute, чтобы занять всё место в обертке */
          top: 0;
          left: 0;
          width: 100%; 
          height: 100%; 
          overflow: hidden; 
          background: #ffffff;">
      </div>
    </div>
    <div class="figure-caption" style="margin-top: 8px; text-align: left; color: #666; font-size: 0.9em;">
      <strong>Figure 1: Topological Visualization of the Cora Dataset.</strong> 
      A 3D projection of the citation network, where nodes represent scientific publications and edges denote citation links. 
      Node colors correspond to the ground-truth topic classification of each paper.
    </div>
  </div>
</section>

<section class="article-width" id="foundations" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="foundations-understanding-graphs-and-sequence-models">Foundations - Understanding Graphs and Sequence Models</h2>
<p class="section-intro">
            Before diving into Graph Mamba, let's establish the fundamentals of graph-structured data 
            and why traditional approaches struggle with it.
        </p>

<h3 id="the-limitations-of-message-passing">The Limitations of Message Passing</h3>
<p>
            A <strong>graph</strong> is a data structure with <strong>nodes</strong> (entities) and 
            <strong>edges</strong> (relationships) connecting these nodes. Graph Neural Networks (GNNs) are the standard for learning on graph-structured data. Most Graph Neural Networks operate via <strong>message passing</strong>: at each layer, every node
            updates its representation by aggregating information from its immediate neighbors. One layer reaches only one-hop neighbors, two layers reach two hops, and so on.
        </p>
<p>
            This makes deep networks necessary for long-range reasoning: to connect far-apart nodes, the model often needs many layers. But stacking many message-passing layers has well-known side effects: node representations become increasingly similar (<strong>over-smoothing</strong>), and information from many distant nodes gets compressed into a limited-size embedding (<strong>over-squashing</strong>).
        </p>
<p>
            There is also a practical cost: each layer typically requires iterating over edges to aggregate messages, so deeper models mean more passes over the graph and higher memory/compute. This is especially painful for large, dense, or high-degree graphs.
        </p>
<h3 id="enter-mamba-the-selective-state-space-model">Enter Mamba - The Selective State Space Model</h3>
  <p>
    To overcome these bottlenecks, we turn to <strong>Mamba</strong>, a recent architecture based on <strong>Selective State Space Models (SSMs)</strong>. Mamba was originally designed for sequence modeling, but its core properties are uniquely suited for graphs when we view them as sequences of structural snapshots.
  </p>
	  <p>
	    At first glance, Mamba looks like an RNN—it processes sequences step-by-step with a hidden state. But architecturally there's a key difference: Mamba is fully <strong>linear</strong> with no activation functions between recurrent steps. This small change (plus a selection mechanism that adds a data-dependent <em>keep gate</em> for filtering information) is what allows parallelization during training. So you get RNN's recurrent structure with Transformer-level quality.
	  </p>

  <ul style="margin-top: 1rem; margin-bottom: 1.5rem; line-height: 1.6; color: #374151;">
    <li style="margin-bottom: 0.75rem;">
      <strong>Traditional RNN problem</strong>: sequential computation means you can't parallelize across timesteps—one step depends on the previous, making training slow.
    </li>
    <li style="margin-bottom: 0.75rem;">
      <strong>Mamba's solution</strong>: linear recurrence (no activations between steps) can be unrolled and computed in parallel during training, while still running sequentially during inference for \(O(1)\) memory.
    </li>
    <li style="margin-bottom: 0.75rem;">
      <strong>Transformer comparison</strong>: achieves Transformer-level modeling quality with linear \(O(N)\) complexity instead of quadratic \(O(N^2)\), avoiding the massive attention matrix.
    </li>
  </ul>

  <p>
    This linear efficiency is the key unlock for graphs. It allows us to process long sequences of graph snapshots without the exploding computational cost of a Transformer or the forgetting issues of an RNN. By linearizing graph neighborhoods, we can use Mamba to propagate information globally across the entire graph structure in a single, efficient pass.
  </p>

<details>
<summary><strong>Python code:</strong> Loading the Cora dataset</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Imports used throughout this notebook/tutorial.</span>
<span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torch.nn</span> <span class="k">as</span> <span class="n">nn</span>
<span class="kn">import</span> <span class="nn">torch.nn.functional</span> <span class="k">as</span> <span class="n">F</span>
<span class="kn">from</span> <span class="nn">torch_geometric.datasets</span> <span class="kn">import</span> <span class="n">Planetoid</span>
<span class="kn">from</span> <span class="nn">torch_geometric.utils</span> <span class="kn">import</span> <span class="n">to_networkx</span>
<span class="kn">from</span> <span class="nn">torch_geometric.nn</span> <span class="kn">import</span> <span class="n">GCNConv</span>
<span class="kn">from</span> <span class="nn">torch_geometric.data</span> <span class="kn">import</span> <span class="n">Data</span>
<span class="kn">from</span> <span class="nn">torch_cluster</span> <span class="kn">import</span> <span class="n">random_walk</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="n">np</span>
<span class="kn">import</span> <span class="nn">networkx</span> <span class="k">as</span> <span class="n">nx</span>
<span class="kn">import</span> <span class="nn">plotly.graph_objects</span> <span class="k">as</span> <span class="n">go</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="kn">from</span> <span class="nn">matplotlib.patches</span> <span class="kn">import</span> <span class="n">Rectangle</span>
<span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="kn">import</span> <span class="n">interact</span><span class="p">,</span> <span class="n">IntSlider</span>

<span class="c1"># Choose where to run computations (Mac GPU = mps).</span>
<span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">device</span><span class="p">(</span><span class="s">"mps"</span><span class="p">)</span>

<span class="c1"># Load the Cora citation network (one graph with node features + labels).</span>
<span class="n">ds</span> <span class="o">=</span> <span class="n">Planetoid</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s">"./data"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"Cora"</span><span class="p">)</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">ds</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="c1"># Move the graph tensors to the chosen device.</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">data</span></code></pre></figure>

</details>
</section>
<section class="article-width" id="step1" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="from-random-walks-to-tokens">From Random Walks to Tokens</h2>
<p class="section-intro">
            The first innovation in Graph Mamba is how we sample and represent graph neighborhoods. 
            Instead of fixed-radius neighbors, we use random walks to create multi-scale <strong>snapshots</strong> of the graph.
        </p>
<p>
            Fix a graph \(G=(V,E)\) and a center node \(v\in V\). Choose a maximum walk length \(m\in\mathbb{N}\) and a number of walks \(M\in\mathbb{N}\). For each length \(\ell\in\{0,1,\dots,m\}\), we sample \(M\) random walks of length \(\ell\) starting from \(v\). Let \[ T_{\ell}(v)\subseteq V \] denote the set of all vertices visited at least once by these walks (including \(v\)). The \(\ell\)-th <strong>structural snapshot</strong> is defined as the induced subgraph \[ \mathcal{T}_{\ell}(v)\;:=\;G\big[T_{\ell}(v)\big], \] which provides a stochastic, multi-scale view of the neighborhood of \(v\). (Later, this snapshot will serve as an input token).
        </p>
<p>
            <strong>Note:</strong> The demo code below starts from walk length \(\ell=1\) for simplicity; \(\ell=0\) would just be the trivial token containing only the center node \(v\).
</p>
<div class="info-box"> 
  <p>
    <strong>Interpretation (multi-scale view):</strong>
  </p>
  <ul> 
    <li><strong>\(\ell=0\):</strong> \(\mathcal{T}_{0}(v)\) contains only the center vertex \(v\).</li> <li><strong>\(\ell=1\):</strong> \(\mathcal{T}_{1}(v)\) typically captures a subset of the 1-hop neighborhood of \(v\).</li>
    <li><strong>\(\ell=2\):</strong> \(\mathcal{T}_{2}(v)\) expands toward 2-hop structure, with coverage determined by the random-walk samples.</li> 
    <li><strong>\(\ell=m\):</strong> \(\mathcal{T}_{m}(v)\) yields a larger-scale token that approaches broader context as \(m\) increases.</li> 
  </ul> 
  <p> 
    The resulting token family \(\{\mathcal{T}_{\ell}(v)\}_{\ell=0}^{m}\) forms an ordered multi-resolution description of the local environment of \(v\). 
  </p> </div>
<details>
<summary><strong>Python code:</strong> Neighborhood sampling (random walk tokenization)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Build token neighborhoods using random walks.</span>
<span class="c1"># We keep edges on CPU here because random_walk is typically run on CPU.</span>
<span class="n">edge_index</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">.</span><span class="n">cpu</span><span class="p">()</span>
<span class="n">n</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">num_nodes</span>

<span class="n">max_walk_length</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">num_of_walks</span> <span class="o">=</span> <span class="mi">4</span>

<span class="c1"># tokens[node][walk_length] will store the set of nodes we reached from that start node.</span>
<span class="n">tokens</span> <span class="o">=</span> <span class="p">{</span><span class="n">v</span><span class="p">:</span> <span class="p">{}</span> <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">)}</span>

<span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">walk_length</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
        <span class="c1"># Merge several random walks so the token isn't too noisy from one walk.</span>
        <span class="n">induced_graph</span> <span class="o">=</span> <span class="nb">set</span><span class="p">()</span>

        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">num_of_walks</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
            <span class="c1"># One random walk starting from node.</span>
            <span class="n">walk</span> <span class="o">=</span> <span class="n">random_walk</span><span class="p">(</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">edge_index</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">([</span><span class="n">node</span><span class="p">]),</span> <span class="n">walk_length</span><span class="o">=</span><span class="n">walk_length</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
            <span class="n">induced_graph</span> <span class="o">|=</span> <span class="nb">set</span><span class="p">(</span><span class="n">walk</span><span class="p">.</span><span class="n">tolist</span><span class="p">())</span>

        <span class="c1"># Save the node-set for this token (we'll turn it into a tiny induced subgraph later).</span>
        <span class="n">tokens</span><span class="p">[</span><span class="n">node</span><span class="p">][</span><span class="n">walk_length</span><span class="p">]</span> <span class="o">=</span> <span class="n">induced_graph</span></code></pre></figure>

</details>
<section style="width: 100%; margin-top: 6rem; margin-bottom: 2rem;">
  <div style="position: relative; width: 140%; margin-left: -20%; margin-top: 1.5rem; margin-bottom: 1.5rem;">
    <div class="interactive-figure" style="
        position: relative; 
        overflow: hidden; 
        background: #ffffff; 
        border: 1px solid #e5e7eb; 
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #1f2937;">
      <div id="walk-viz" style="display: flex; flex-wrap: wrap; gap: 0; width: 100%; min-height: 550px; align-items: stretch;">
        <div id="walk-controls" style="
            flex: 0 0 240px; 
            background: #f9fafb; 
            padding: 20px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            border-right: 1px solid #e5e7eb; 
            box-sizing: border-box;">
          <div style="margin-bottom: 16px;">
            <div style="font-size: 18px; margin-bottom: 6px; font-weight: 600; color: #111827;">1. Walks per Sample (M)</div>
            <input id="walk-count" max="8" min="1" step="1" style="width: 100%; cursor: pointer;" type="range" value="4"/>
            <div style="margin-top: 4px; color: #6b7280; font-size: 16px;">
              M = <span id="walk-count-value" style="font-weight: 600; color: #16a34a;">4</span> walks
            </div>
          </div>
          <div style="margin-bottom: 16px;">
            <div style="font-size: 18px; font-weight: 600; color: #111827; margin-bottom: 4px;">2. Target Node</div>
            <div style="color: #6b7280; font-size: 16px;">Click any node on the graph to select a center.</div>
          </div>
          <div style="margin-bottom: 20px;">
            <div style="font-size: 18px; margin-bottom: 6px; font-weight: 600; color: #111827;">3. Walk Length (ℓ)</div>
            <input id="walk-length" max="3" min="0" step="1" style="width: 100%; cursor: pointer;" type="range" value="1"/>
            <div style="margin-top: 4px; color: #6b7280; font-size: 16px;">
              ℓ = <span id="walk-length-value" style="font-weight: 600; color: #16a34a;">1</span> steps
            </div>
          </div>
          <button id="sample-token-btn" style="
              width: 100%; 
              background: #16a34a; 
              border: 1px solid #0f7d37; 
              color: #ffffff; 
              padding: 8px 12px; 
              cursor: pointer; 
              border-radius: 6px; 
              font-weight: 500; 
              font-size: 16px; 
              box-shadow: 0 1px 2px rgba(0,0,0,0.1);
              transition: background 0.2s;">
            Generate Token
          </button>
        </div>
        <div id="walk-canvas" style="
            flex: 1 1 400px; 
            min-width: 300px; 
            height: 520px; 
            position: relative; 
            background: #ffffff; 
            box-sizing: border-box;">
        </div>
        <div id="token-panel" style="
            flex: 0 0 240px; 
            background: #f9fafb; 
            padding: 20px; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            border-left: 1px solid #e5e7eb; 
            box-sizing: border-box;">         
          <div style="ont-size: 18px; margin-bottom: 12px; font-weight: 600; color: #111827;">Token Encoding Output</div>         
          <div style="font-size: 16px; color: #4b5563; line-height: 1.5; margin-bottom: 16px;">
            For simplified visualization (s=1), each row represents the aggregated features of M random walks at length ℓ, projected to dimension d.
          </div>        
          <div style="background: #ffffff; padding: 10px; border: 1px solid #e5e7eb; border-radius: 6px; display: flex; justify-content: center;">
            <svg height="120" id="token-matrix" width="160"></svg>
          </div>
        </div>
        <div id="walk-sequence" style="
            flex: 1 1 100%; 
            background: #ffffff; 
            color: #4b5563; 
            padding: 12px 20px; 
            font-family: 'SF Mono', Consolas, Menlo, monospace; 
            font-size: 12px; 
            border-top: 1px solid #e5e7eb; 
            box-sizing: border-box;">
          > Ready. Select parameters and click "Generate Token".
        </div>
      </div>
    </div>
    <div class="figure-caption" style="margin-top: 8px; text-align: left; color: #666; font-size: 0.9em;">
      <strong>Figure 2: Random Walk Sampling.</strong> 
      Interactive demonstration of how structural snapshots are generated. 
      The system samples M random walks of length ℓ starting from a center node (red). 
      The induced subgraph formed by visited nodes constitutes a single token \(\mathcal{T}_{\ell}(v)\).
    </div>

  </div>
</section>
<section class="article-width" id="step2" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="encoding-subgraphs-with-local-encoders">Encoding Subgraphs with Local Encoders</h2>
<p class="section-intro">
            Now that we have tokens (subgraphs), the next objective is to convert each token into a latent vector representation. This transformation is done by the local encoder \(\varphi(\cdot)\). </p>
<p>
            Each snapshot \(\mathcal{T}_{\ell}(v)\) remains a graph structure consisting of a subset of vertices, induced edges, and associated node features. To integrate this structure into the sequence model, we must project the entire subgraph into a single embedding vector \(\mathbf{x}_{\ell}(v) \in \mathbb{R}^{d}\). 
        </p>
<p>
          The Graph Mamba Network employs a parameterised local encoder \(\varphi: \mathcal{G} \to \mathbb{R}^{d}\) for this aggregation. In practice, \(\varphi(\cdot)\) may be implemented using various graph neural network architectures: 
        </p> 
        <ul> 
          <li><strong>GCN (Graph Convolutional Network):</strong> A stack of spectral convolution layers operating on the subgraph adjacency.</li> 
          <li><strong>GraphSAGE:</strong> An inductive framework aggregating feature information from sampled neighborhoods.</li> 
          <li><strong>GAT (Graph Attention Network):</strong> A mechanism computing attention coefficients to weight neighbor contributions.</li> 
          <li><strong>Mean Pooling:</strong> A baseline approach computing the centroid of node features within the subgraph.</li> 
        </ul>
<p>
            Irrespective of the specific architecture, the encoder processes the structural snapshot \(\mathcal{T}_{\ell}(v)\) and its corresponding feature matrix \(\mathbf{X}_{\mathcal{T}_{\ell}(v)}\) to yield a \(d\)-dimensional representation: \[ \mathbf{x}_{\ell}(v) \;=\; \varphi\Big( \mathcal{T}_{\ell}(v), \, \mathbf{X}_{\mathcal{T}_{\ell}(v)} \Big). \] This vector forms the \(\ell\)-th row of the token matrix \(\mathbf{X}(v) \in \mathbb{R}^{K \times d}\), where \(K\) denotes the total number of tokens per node. Specifically, if the sampling procedure is repeated \(s\) times for each of the \(m+1\) scales, the sequence length is given by \(K = s(m+1)\). The subsequent Bidirectional Mamba block then ingests this matrix as a sequence, learning dependencies across the multi-scale snapshots. 
        </p>

<details>
<summary><strong>Python code:</strong> Local encoder (GCN) for a token subgraph</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Local encoder: take a token (a small subgraph) and turn it into one vector.</span>
<span class="c1"># in_dim = size of input node features; hidden_dim = size of the output token vector.</span>
<span class="k">class</span> <span class="nc">LocalEncoder</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">in_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_1</span> <span class="o">=</span> <span class="n">GCNConv</span><span class="p">(</span><span class="n">in_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_2</span> <span class="o">=</span> <span class="n">GCNConv</span><span class="p">(</span><span class="n">hidden_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">encode_token</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token</span><span class="p">,</span> <span class="n">data</span><span class="p">):</span>
        <span class="c1"># token is a set/list of node ids. Convert it to a tensor so we can index into data.</span>
        <span class="n">token</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">(</span><span class="nb">list</span><span class="p">(</span><span class="n">token</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="p">.</span><span class="nb">long</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
        <span class="c1"># Take the features for just the nodes inside this token.</span>
        <span class="n">neighborhood_features</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">x</span><span class="p">[</span><span class="n">token</span><span class="p">]</span>

        <span class="c1"># Keep only edges where BOTH endpoints are inside this token (so we get the induced subgraph).</span>
        <span class="n">mask</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">isin</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">token</span><span class="p">)</span> <span class="o">&amp;</span> <span class="n">torch</span><span class="p">.</span><span class="n">isin</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">token</span><span class="p">)</span>
        <span class="n">edges_in_token</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[:,</span> <span class="n">mask</span><span class="p">]</span>

        <span class="c1"># PyG expects node ids inside a subgraph to be numbered 0..k-1.</span>
        <span class="c1"># So we remap global node ids -&gt; local ids (within this token).</span>
        <span class="n">idx_map</span> <span class="o">=</span> <span class="p">{</span><span class="n">old</span><span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">old</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">token</span><span class="p">.</span><span class="n">tolist</span><span class="p">())}</span>
        <span class="c1"># Rebuild the edge list using the local ids.</span>
        <span class="n">sub_edge_index</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">(</span>
            <span class="p">[[</span><span class="n">idx_map</span><span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">u</span><span class="p">)]</span> <span class="k">for</span> <span class="n">u</span> <span class="ow">in</span> <span class="n">edges_in_token</span><span class="p">[</span><span class="mi">0</span><span class="p">]],</span>
            <span class="p">[</span><span class="n">idx_map</span><span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">v</span><span class="p">)]</span> <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">edges_in_token</span><span class="p">[</span><span class="mi">1</span><span class="p">]]],</span>
            <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="p">.</span><span class="nb">long</span><span class="p">,</span>
            <span class="n">device</span><span class="o">=</span><span class="n">device</span>
        <span class="p">)</span>

        <span class="c1"># Create a standalone mini graph for this token (features + token-internal edges).</span>
        <span class="n">induced_graph</span> <span class="o">=</span> <span class="n">Data</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">neighborhood_features</span><span class="p">,</span> <span class="n">edge_index</span><span class="o">=</span><span class="n">sub_edge_index</span><span class="p">)</span>
        <span class="c1"># Run a small GCN on the token, then average node embeddings to get ONE vector per token.</span>
        <span class="n">h</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_1</span><span class="p">(</span><span class="n">induced_graph</span><span class="p">.</span><span class="n">x</span><span class="p">,</span> <span class="n">induced_graph</span><span class="p">.</span><span class="n">edge_index</span><span class="p">).</span><span class="n">relu</span><span class="p">()</span>
        <span class="n">h</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_2</span><span class="p">(</span><span class="n">h</span><span class="p">,</span> <span class="n">induced_graph</span><span class="p">.</span><span class="n">edge_index</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">h</span><span class="p">.</span><span class="n">mean</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
    
    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">token</span><span class="p">):</span>
        <span class="k">return</span> <span class="bp">self</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">token</span><span class="p">,</span> <span class="n">data</span><span class="p">)</span>

<span class="n">local_encoder</span> <span class="o">=</span> <span class="n">LocalEncoder</span><span class="p">(</span><span class="n">in_dim</span><span class="o">=</span><span class="n">data</span><span class="p">.</span><span class="n">num_features</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">token_embeddings</span> <span class="o">=</span> <span class="p">{</span><span class="n">node</span><span class="p">:</span> <span class="p">[]</span> <span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">)}</span>

<span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">walk_length</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
        <span class="n">token_embeddings</span><span class="p">[</span><span class="n">node</span><span class="p">].</span><span class="n">append</span><span class="p">(</span><span class="n">local_encoder</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">tokens</span><span class="p">[</span><span class="n">node</span><span class="p">][</span><span class="n">walk_length</span><span class="p">],</span> <span class="n">data</span><span class="p">))</span></code></pre></figure>

</details>

<details>
<summary><strong>Python code:</strong> Token ordering (reverse-by-walk-length)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Put longer-walk tokens first, and the closest-to-node token last.</span>
<span class="c1"># That way, the final step in the sequence is the most local information.</span>
<span class="n">ordered_token_embeddings</span> <span class="o">=</span> <span class="p">{}</span>

<span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="n">embeddings</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">walk_length</span> <span class="ow">in</span> <span class="nb">reversed</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)):</span>
        <span class="n">emb</span> <span class="o">=</span> <span class="n">local_encoder</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">tokens</span><span class="p">[</span><span class="n">node</span><span class="p">][</span><span class="n">walk_length</span><span class="p">],</span> <span class="n">data</span><span class="p">)</span>
        <span class="n">embeddings</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">emb</span><span class="p">)</span>
    <span class="n">ordered_token_embeddings</span><span class="p">[</span><span class="n">node</span><span class="p">]</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">embeddings</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="nb">list</span><span class="p">(</span><span class="n">ordered_token_embeddings</span><span class="p">.</span><span class="n">items</span><span class="p">())[:</span><span class="mi">5</span><span class="p">]</span></code></pre></figure>

</details>
<h3 id="gcn-encoder-three-phases">GCN Encoder - Three Phases</h3>
<p>
            Previously, we modeled the local encoder \(\varphi(\cdot)\) as an abstract operator mapping a structural snapshot \(\mathcal{T}_{\ell}(v)\) to a latent vector \(\mathbf{x}_{\ell}(v) \in \mathbb{R}^{d}\). In this section, we explain the internal mechanism of \(\varphi(\cdot)\) by instantiating it as a two-layer Graph Convolutional Network (GCN). 
        </p> 
<p> 
            Let us briefly overview the GCN propagation rule. A standard GCN layer updates node representations by aggregating features from immediate neighbors, normalized by the degree of the involved nodes. Formally, the \(k\)-th layer update is given by \(\mathbf{H}^{(k)} = \sigma(\hat{\mathbf{D}}^{-1/2}\hat{\mathbf{A}}\hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(k-1)} \mathbf{W}^{(k-1)})\), where \(\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}\) represents the adjacency matrix with added self-loops, \(\hat{\mathbf{D}}\) is the diagonal degree matrix with entries \(\hat{\mathbf{D}}_{ii} = \sum_{j} \hat{\mathbf{A}}_{ij}\),   \(\mathbf{W}\) is a learnable weight matrix, and \(\sigma\) denotes a non-linear activation function. We apply this logic to process the induced subgraph of each token. 
        </p> 
<div class="info-box"> 
  <p> 
        <strong>Computational Phases:</strong>
      </p> 
        <ol>
          <li><strong>Structural Definition:</strong> Construct the subgraph adjacency matrix \(\mathbf{A}_{\ell} \in \{0,1\}^{N_\ell \times N_\ell}\) and the associated feature matrix \(\mathbf{H}^{(0)} \in \mathbb{R}^{N_\ell \times F}\), where \(N_\ell = |T_{\ell}(v)|\).</li> 
          <li><strong>Layer I (Message Passing):</strong> Perform a linear transformation followed by neighborhood aggregation and non-linear activation: \[ \mathbf{H}^{(1)} = \sigma\left( \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}}_{\ell} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(0)} \mathbf{W}^{(0)} \right). \] </li>
          <li><strong>Layer II & Pooling:</strong> Apply a second convolution step to obtain \(\mathbf{H}^{(2)}\), followed by a readout function (e.g., mean pooling) to compress the node set into a single token embedding \(\mathbf{x}_{\ell}(v)\).</li> 
        </ol> 
</div>
<div style="width: 140%; margin-left: -20%; margin-top: 2rem; margin-bottom: 2rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  
  <div class="interactive-figure" style="background: #ffffff; border: 1px solid #e5e7eb; display: flex; flex-wrap: wrap; overflow: hidden; 
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #1f2937;">
    <div id="gcn-local-viz" style="width: 100%; min-height: 500px;"></div>
  </div>

  <div class="figure-caption" style="margin-top: 0.75rem; color: #4b5563; font-size: 0.9rem; line-height: 1.5;">
    <strong>Figure 3: Interactive visualization of the GCN local encoder.</strong> Select a step on the right to see how information propagates through the token subgraph \(\mathcal{T}_{\ell}(v)\).
  </div>

</div>


<section class="article-width" id="step3" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="the-mamba-block-selective-state-spaces">The Mamba Block - Selective State Spaces</h2>
<p class="section-intro">
          After mapping the tokens into a sequence of latent representations the next challenge is to model the dependencies within this sequence efficiently. To this end, we employ the Mamba architecture, which leverages selective state space models (SSMs) to achieve linear computational complexity with respect to sequence length, unlike attention mechanisms with quadratic computational complexity.
        </p>
<h3 id="from-tokens-to-sequences">From Tokens to Sequences</h3>
<p>
          Following the local encoding phase, the multiscale environment of each node \(v\) is represented by a set of \(K\) embeddings, derived from the structural snapshots \(\{\mathcal{T}_{\ell}(v)\}\). These embeddings are organized into a matrix \(\mathbf{X}(v) \in \mathbb{R}^{K \times d}\), where: </p> 
          <ul> 
            <li>\(K = s(m+1)\), corresponding to the product of the number of samples \(s\) and the maximum walk length \(m\).</li> 
            <li>\(d\) denotes the dimensionality of the latent feature space.</li> 
          </ul>
          <p> The matrix \(\mathbf{X}(v)\) serves as the input sequence for the Mamba block. Each row \(\mathbf{x}_{k}(v)\) constitutes a single <strong>token</strong> that represents the structural information of a specific subgraph from a random walk.
        </p>
<h3 id="token-ordering">Token Ordering</h3>
<p>
            Unlike permutation-equivariant Transformers, the Mamba architecture is inherently sequential; the state update at step \(t\) depends strictly on the previous states \(1, \dots, t-1\). Consequently, the ordering of the token sequence \(\mathbf{X}(v)\) is structurally significant. 
        </p>
<div class="info-box">
<strong>Ordering Protocol:</strong> 
<p> <strong>Hierarchical Structure (\(m \ge 1\)):</strong> Subgraph tokens possess an inherent inclusion hierarchy. GMN adopts a <strong>reverse</strong> ordering: the sequence begins with the largest scale snapshots (\(\ell=m\)) and progresses inward to the local node features (\(\ell=0\)). This ensures that the final state update, corresponding to the node itself, is conditioned on the full multi-scale context. </p> 
<p> <strong>Stochastic Permutation (\(s \ge 2\)):</strong> Within any fixed scale \(\ell\), the \(s\) independent samples are randomly permuted to induce invariance to the sampling order. </p> 
<p> <strong>Node Tokenization (\(m=0\)):</strong> In the absence of hierarchical structure, tokens are ordered by global topological metrics (e.g., Personalized PageRank or degree centrality) to provide a canonical sequence. </p>
</div>
  <p> The resulting input sequence for a node \(v\) is constructed as follows: </p> 
  <div class="formula-box"> \[ \mathbf{X}(v) = \Big[ \underbrace{\mathbf{x}^{(1)}_{m}, \dots, \mathbf{x}^{(s)}_{m}}_{\text{Scale } m \text{ (Global)}}, \dots, \underbrace{\mathbf{x}^{(1)}_{1}, \dots, \mathbf{x}^{(s)}_{1}}_{\text{Scale } 1 \text{ (Local)}}, \underbrace{\mathbf{x}^{(1)}_{0}, \dots, \mathbf{x}^{(s)}_{0}}_{\text{Scale } 0 \text{ (Node)}} \Big] \] 
  <ul> 
    <li>Tokens are processed from global context (\(\ell=m\)) to local identity (\(\ell=0\)).</li>
    <li>Within each block of scale \(\ell\), the \(s\) samples are randomly shuffled.</li> 
  </ul> 
</div>

<h3 id="the-core-mechanism">The Core Mechanism</h3>
 <p> As the model processes the sequence of structural tokens \(\mathbf{x}_1, \dots, \mathbf{x}_K\), the Mamba block maintains a compressed representation of the context via a <strong>hidden state</strong>. Formally, this process is defined by a discretized state space equation: </p> 
 <div class="formula-box"> 
 <p>The latent state evolves according to the linear recurrence:</p> 
 \[ \mathbf{h}_t \;=\; \bar{\mathbf{A}}_t \mathbf{h}_{t-1} \,+\, \bar{\mathbf{B}}_t \mathbf{x}_t \] <p><strong>Definitions:</strong></p> 
 <ul> 
  <li>\(\mathbf{h}_t \in \mathbb{R}^{N}\): The hidden state at step \(t\), serving as the sequence memory.</li> 
  <li>\(\mathbf{x}_t \in \mathbb{R}^{d}\): The input token corresponding to the \(t\)-th structural snapshot.</li> 
  <li>\(\bar{\mathbf{A}}_t\): The state transition matrix (determining information retention).</li> 
  <li>\(\bar{\mathbf{B}}_t\): The input projection matrix (determining information update).</li> 
</ul> 
</div>
<h3 id="the-selective-mechanism">The Selective Mechanism</h3>
<p> In classical State Space Models (SSMs), the parameters \(\mathbf{A}\) and \(\mathbf{B}\) are static, independent of the input sequence. Mamba diverges from this by making the transition dynamics <strong>input-dependent</strong>. </p>
<p> Central to this mechanism is the <strong>timescale parameter</strong> \(\Delta_t\), which acts as a gating factor derived from the current input \(\mathbf{x}_t\): </p> 
<div class="formula-box"> <p> 
  \[ \Delta_t \;=\; \mathrm{Softplus}(\mathbf{W}_{\Delta} \mathbf{x}_t) \] </p> 
  <p>This parameter modulates the system parameters \((\mathbf{A}, \mathbf{B})\):</p> 
  <p> \[ \bar{\mathbf{A}}_t \;=\; \exp(\Delta_t \cdot \mathbf{A}), \qquad \bar{\mathbf{B}}_t \;=\; \Delta_t \cdot \mathbf{B} \] </p> 
  </div>
<p> This empowers the model with a <strong>selective mechanism</strong>: </p> 
<ul> 
  <li><strong>High \(\Delta_t\):</strong> Corresponds to a larger step size, allowing the current token \(\mathbf{x}_t\) to significantly update the state \(\mathbf{h}_t\) and reset the memory.</li>
  <li><strong>Low \(\Delta_t\):</strong> Corresponds to a small step size, causing the state to persist unchanged, effectively filtering out irrelevant or noisy tokens.</li> 
</ul> 

<details>
<summary><strong>Python code:</strong> Selective State Space block</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Note that this code is a Mamba inspired toy block (gated memory), not the exact SSM math written above.</span>
<span class="c1"># It reads token vectors one-by-one and keeps / overwrites an internal state using learned gates.</span>
<span class="k">class</span> <span class="nc">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">token_dim</span> <span class="o">=</span> <span class="n">token_dim</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">state_dim</span> <span class="o">=</span> <span class="n">state_dim</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">token_to_params</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">state_to_token</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">state_dim</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_sequence</span><span class="p">:</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">:</span>
        <span class="n">batch_size</span><span class="p">,</span> <span class="n">seq_len</span><span class="p">,</span> <span class="n">token_dim</span> <span class="o">=</span> <span class="n">token_sequence</span><span class="p">.</span><span class="n">shape</span>

        <span class="c1"># Turn each token into gate values (3 separate controllers per step).</span>
        <span class="n">params</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">token_to_params</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>
        <span class="n">keep_gate</span><span class="p">,</span> <span class="n">write_gate</span><span class="p">,</span> <span class="n">decay_gate</span> <span class="o">=</span> <span class="n">params</span><span class="p">.</span><span class="n">chunk</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

        <span class="c1"># Squash gates into sensible ranges.</span>
        <span class="n">keep_gate</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">keep_gate</span><span class="p">)</span>
        <span class="n">write_gate</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tanh</span><span class="p">(</span><span class="n">write_gate</span><span class="p">)</span>
        <span class="n">decay_gate</span> <span class="o">=</span> <span class="n">F</span><span class="p">.</span><span class="n">softplus</span><span class="p">(</span><span class="n">decay_gate</span><span class="p">)</span>

        <span class="c1"># Start with an empty memory (one state vector per item in the batch).</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">zeros</span><span class="p">(</span>
            <span class="n">batch_size</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">state_dim</span><span class="p">,</span>
            <span class="n">device</span><span class="o">=</span><span class="n">token_sequence</span><span class="p">.</span><span class="n">device</span><span class="p">,</span>
            <span class="n">dtype</span><span class="o">=</span><span class="n">token_sequence</span><span class="p">.</span><span class="n">dtype</span>
        <span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="p">[]</span>

        <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">seq_len</span><span class="p">):</span>
            <span class="c1"># Update memory: keep some of the previous state, mix in some of the current token.</span>
            <span class="n">state</span> <span class="o">=</span> <span class="n">decay_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">state</span> <span class="o">+</span> <span class="n">keep_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">token_sequence</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span>
            <span class="c1"># Produce an output for this step (what we write out from the state).</span>
            <span class="n">current_output</span> <span class="o">=</span> <span class="n">write_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">state</span>
            <span class="n">outputs</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">current_output</span><span class="p">)</span>

        <span class="c1"># Stack per-step outputs back into a sequence.</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
        <span class="c1"># Map the internal state back to the token vector size.</span>
        <span class="n">token_outputs</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">state_to_token</span><span class="p">(</span><span class="n">outputs</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">token_outputs</span></code></pre></figure>

</details>
<section style="width: 100%; margin-top: 6rem; margin-bottom: 2rem;">
  <div style="width: 140%; margin-left: -20%; margin-top: 1.5rem; margin-bottom: 1.5rem;">
    <div class="interactive-figure" style="
      position: relative;
      width: 100%;
      height: 500px;
      min-height: 500px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
      color: #111827;">
      <div id="mamba-viz" style="width: 100%; height: 100%; position: relative; background: #ffffff;"></div>
      <div style="
        position: absolute;
        bottom: 12px;
        right: 12px;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(4px);
        padding: 12px;
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        min-width: 240px;
        width: 240px;
        max-width: calc(100% - 24px);
        box-sizing: border-box;
        z-index: 10;
        color: #111827;">
        <div style="font-weight: 700; margin-bottom: 10px; color: #111827; font-size: 13px;">
          Legend
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
          <span style="width: 12px; height: 12px; background: #16a34a; display: inline-block; margin-right: 10px; border-radius: 3px;"></span>
          <span style="color: #374151;">Relevant (gate open)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
          <span style="width: 12px; height: 12px; background: #ef4444; display: inline-block; margin-right: 10px; border-radius: 3px;"></span>
          <span style="color: #374151;">Noise (gate closed)</span>
        </div>
        <div style="border-top: 1px solid #e5e7eb; padding-top: 10px; margin-top: 8px;">
          <button id="replay-btn" style="
            width: 100%;
            background: #16a34a;
            border: 1px solid #16a34a;
            color: #ffffff;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 12px;
            transition: background 0.15s ease, border-color 0.15s ease;
            box-sizing: border-box;
            border-radius: 6px;">
            Replay animation
          </button>
          <div id="step-info" style="
            margin-top: 10px;
            padding: 8px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            font-size: 11px;
            min-height: 44px;
            color: #4b5563;">
          </div>
        </div>
      </div>
    </div>
    <div class="figure-caption" style="margin-top: 8px; text-align: left; color: #666; font-size: 0.9em;">
      <strong>Figure 4: Selective gating in Mamba.</strong>
      The model reads tokens sequentially and decides via an input-dependent gate whether to write new information into the hidden state or keep it unchanged. Use Replay to step through how relevant tokens update the state while noisy tokens are filtered out.
    </div>

  </div>
</section>

<section class="article-width" id="step4" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="bidirectional-sequence-modeling">Bidirectional Sequence Modeling</h2> 
<p class="section-intro"> Unlike natural language sequences which follow a canonical temporal order, graph random walks are stochastic traversals lacking inherent directionality. Processing such sequences exclusively in a forward manner imposes some causal bias. Graph Mamba mitigates this possible bias by using a bidirectional architecture. </p> 
<h3 id="the-bi-mamba-architecture">The Bi-Mamba Architecture</h3> 
<p> Standard State Space Models are causal: the hidden state \(\mathbf{h}_t\) depends solely on the history \(\mathbf{x}_{1 \dots t}\). However, for structural representation learning, a token should be informed by the entire context of the neighborhood snapshot, regardless of its position in the sampled sequence. </p> 
<p> To achieve this, the <strong>Bidirectional Mamba</strong> block processes the input token sequence \(\mathbf{X}(v)\) using two independent SSM heads operating in opposite directions: </p>
<ul> 
  <li><strong>Forward Pass:</strong> Processes the sequence from \(t=1\) to \(K\), capturing context from the start of the walk.</li> 
  <li><strong>Backward Pass:</strong> Processes the sequence from \(t=K\) to \(1\), capturing context from the end of the walk.</li> 
</ul> 
<p> Mathematically, for each time step \(t\), the model computes two distinct states which are subsequently fused (e.g., via concatenation or element-wise addition) to form the final position-aware embedding: \[ \mathbf{z}_t \;=\; \text{Fuse}\big(\, \overrightarrow{\text{SSM}}(\mathbf{x}_t), \; \overleftarrow{\text{SSM}}(\mathbf{x}_t) \,\big). \] This mechanism ensures that every token \(\mathbf{x}_t\) integrates information from both its predecessors and successors in the sequence. </p>

<details>
<summary><strong>Python code:</strong> Bidirectional Mamba</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Bidirectional wrapper: run an SSM forward and backward over the same token sequence and fuse the results.</span>
<span class="c1"># This reduces causal bias from arbitrary random-walk ordering.</span>
<span class="k">class</span> <span class="nc">BidirectionalMamba</span><span class="p">(</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">norm_tokens</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">LayerNorm</span><span class="p">(</span><span class="n">token_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">forward_ssm</span> <span class="o">=</span> <span class="n">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">backward_ssm</span> <span class="o">=</span> <span class="n">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">output_proj</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_sequence</span><span class="p">:</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">:</span>
        <span class="c1"># Normalize token vectors before sending them into the two SSM directions.</span>
        <span class="n">token_sequence</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">norm_tokens</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>

        <span class="c1"># Forward direction: reads tokens from first to last.</span>
        <span class="n">forward_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">forward_ssm</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>

        <span class="c1"># Backward direction: flip the sequence, run the same kind of SSM, then flip back.</span>
        <span class="n">reversed_tokens</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">flip</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">,</span> <span class="n">dims</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
        <span class="n">backward_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">backward_ssm</span><span class="p">(</span><span class="n">reversed_tokens</span><span class="p">)</span>

        <span class="n">backward_tokens</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">flip</span><span class="p">(</span><span class="n">backward_tokens</span><span class="p">,</span> <span class="n">dims</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>

        <span class="c1"># Fuse the two views (here: simple add) and mix with a final linear layer.</span>
        <span class="n">mixed_tokens</span> <span class="o">=</span> <span class="n">forward_tokens</span> <span class="o">+</span> <span class="n">backward_tokens</span>
        <span class="n">mixed_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">output_proj</span><span class="p">(</span><span class="n">mixed_tokens</span><span class="p">)</span>
        
        <span class="k">return</span> <span class="n">mixed_tokens</span>

<span class="n">mamba_layer</span> <span class="o">=</span> <span class="n">BidirectionalMamba</span><span class="p">(</span><span class="n">token_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">state_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span></code></pre></figure>

</details>
<section style="width: 100%; margin-top: 6rem; margin-bottom: 2rem;">
  <div style="width: 140%; margin-left: -20%; margin-top: 1.5rem; margin-bottom: 1.5rem;">
    <div class="interactive-figure" style="
      position: relative;
      width: 100%;
      height: 500px;
      min-height: 500px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
      color: #111827;">
      <div id="bidirectional-mamba-viz" style="position: relative; width: 100%; height: 100%;"></div>
      <div style="
        position: absolute;
        bottom: 12px;
        right: 12px;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(4px);
        padding: 12px;
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        min-width: 220px;
        box-sizing: border-box;
        z-index: 10;">       
        <div style="font-weight: 700; margin-bottom: 10px; color: #111827; font-size: 13px;">
          Bi-Directional Flow
        </div>
        <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px;">
           <div style="display: flex; align-items: center;">
             <span style="width: 10px; height: 10px; background: #3b82f6; border-radius: 50%; margin-right: 8px;"></span>
             <span style="color: #4b5563;">Forward SSM (Blue)</span>
           </div>
           <div style="display: flex; align-items: center;">
             <span style="width: 10px; height: 10px; background: #f59e0b; border-radius: 50%; margin-right: 8px;"></span>
             <span style="color: #4b5563;">Backward SSM (Orange)</span>
           </div>
        </div>
        <div style="border-top: 1px solid #e5e7eb; padding-top: 10px;">
          <button id="bidir-replay-btn" style="
            width: 100%;
            background: #16a34a;
            border: 1px solid #16a34a;
            color: #ffffff;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 12px;
            border-radius: 6px;
            transition: opacity 0.2s;">
            Replay Animation
          </button>
        </div>
      </div>
    </div>
    <div class="figure-caption" style="margin-top: 8px; text-align: left; color: #666; font-size: 0.9em;">
      <strong>Figure 5: Bidirectional Processing.</strong> 
      Unlike causal models that only look left, Graph Mamba processes the token sequence in both directions (Forward & Backward) to capture the full context of the random walk.
    </div>
  </div>
</section>

<section class="article-width" id="step5" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="two-level-processing-in-graph-mamba">Two-Level Processing in Graph Mamba</h2>
<p>
  The Graph Mamba architecture is designed to be versatile. It builds representations hierarchically: first by understanding the local structure of each node, and then by modeling how these nodes interact globally. This two-level approach makes it suitable for both <strong>node-level tasks</strong> (like classifying papers in a citation network) and <strong>graph-level tasks</strong> (like predicting properties of entire molecules).
</p>

<div class="info-box">
  <p><strong>Level 1: Intra-Node Processing (Token Aggregation)</strong></p> 
  <p> 
    For a given node \(v\), the input is the sequence of \(K\) token vectors \(\mathbf{X}(v)\) representing multi-scale structural snapshots. A Bidirectional Mamba block operates on this sequence to fuse these snapshots into a single, comprehensive <strong>node embedding</strong> \(\mathbf{h}_v \in \mathbb{R}^{d}\).
  </p> 
  <p> 
    Formally, this stage acts as a learnable aggregation function. Instead of fixed pooling (like mean or max), Mamba learns to weigh the importance of different neighborhood scales and decides whether the local features (1-hop) or global context (m-hop) are more relevant for this specific node.
  </p> 
</div> 

<div class="info-box"> 
  <p><strong>Level 2: Inter-Node Processing (Global Propagation)</strong></p> 
  <p> 
    Once embeddings \(\{\mathbf{h}_v\}_{v \in V}\) are computed for all vertices, they can be arranged into a global sequence. A second Bidirectional Mamba layer processes this graph-level sequence to model interactions between nodes.
  </p> 
  <p> 
    This stage facilitates <strong>long-range dependency modeling</strong> across the entire graph. It creates a "virtual" channel where any node can influence any other node via the recurrent state without deep stacks of message-passing layers. For graph-level tasks, the outputs of this stage are simply pooled to form a single graph vector \(\mathbf{h}_G\).
  </p> 
</div> 

<p style="margin-top: 0.75rem; color: #4b5563;">
  <strong>Implementation note:</strong> Our runnable demo focuses on Level&nbsp;1 (token aggregation for node classification). Level&nbsp;2 is included here as the conceptual extension.
</p>

<div class="key-takeaway" style="margin-top: 3rem">
  This hierarchical design decouples local feature extraction from global reasoning. The first level learns optimal local structural descriptors, while the second level enables efficient, linear-time global information propagation across the entire graph structure.
</div> 
</section>

<section class="article-width" id="arch" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="end-to-end-architecture">End-to-End Architecture</h2>
<p class="section-intro">
    The Graph Mamba pipeline is characterized by its efficient design, requiring only a minimal set of trainable components. By substituting complex attention mechanisms and deep message-passing stacks with a concise sequence of local encoders and selective state space models, the architecture achieves structural depth without the associated computational overhead. 
</p>

<div class="interactive-figure" style="
      width: 140%; 
      margin-left: -20%; 
      margin-top: 3rem; 
      margin-bottom: 2rem;">
  <div style="
      background: #ffffff; 
      border: 1px solid #e5e7eb; 
      border-radius: 8px; 
      overflow: hidden; 
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
      position: relative;
      padding: 2rem 1rem;">
    <!-- SVG Diagram -->
    <svg id="e2e-architecture" style="width: 100%; height: auto; display: block;" viewBox="0 0 1100 350">
      <defs>
        <!-- Arrowhead marker (dark gray) -->
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#4b5563" />
        </marker>
      </defs>
      <!-- 1. Graph Input -->
      <g id="step1" transform="translate(40, 180)">
        <circle cx="50" cy="0" r="22" fill="#fff7ed" stroke="#f97316" stroke-width="3"></circle>
        <circle cx="75" cy="35" r="22" fill="#fff7ed" stroke="#f97316" stroke-width="3"></circle>
        <circle cx="25" cy="35" r="22" fill="#fff7ed" stroke="#f97316" stroke-width="3"></circle>
        <circle cx="50" cy="70" r="22" fill="#fff7ed" stroke="#f97316" stroke-width="3"></circle>
        <line x1="50" y1="0" x2="75" y2="35" stroke="#cbd5e1" stroke-width="2"></line>
        <line x1="50" y1="0" x2="25" y2="35" stroke="#cbd5e1" stroke-width="2"></line>
        <line x1="75" y1="35" x2="50" y2="70" stroke="#cbd5e1" stroke-width="2"></line>
        <line x1="25" y1="35" x2="50" y2="70" stroke="#cbd5e1" stroke-width="2"></line>
        <text x="50" y="115" font-size="16" font-weight="700" fill="#1f2937" text-anchor="middle" font-family="ui-monospace, monospace">Graph</text>
        <text x="50" y="135" font-size="13" fill="#6b7280" text-anchor="middle" font-family="sans-serif">Input</text>
      </g>
      <!-- Arrow 1 -->
      <line x1="160" y1="220" x2="220" y2="220" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"></line>
      <!-- 2. Tokeniser -->
      <g id="step2" transform="translate(240, 140)">
        <rect x="0" y="60" width="140" height="60" rx="8" fill="#f3f4f6" stroke="#d1d5db" stroke-width="2"></rect>
        <text x="70" y="95" font-size="15" font-weight="600" fill="#1f2937" text-anchor="middle" font-family="ui-monospace, monospace">Tokeniser</text>
        <text x="70" y="155" font-size="13" fill="#6b7280" text-anchor="middle" font-family="sans-serif">random walks</text>
        <text x="70" y="172" font-size="13" fill="#6b7280" text-anchor="middle" font-family="sans-serif">of 1-3 length</text>
      </g>
      <!-- Arrow 2 -->
      <line x1="390" y1="220" x2="450" y2="220" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"></line>
      <!-- 3. Mini-GCN -->
      <g id="step3" transform="translate(470, 130)">
        <!-- Top Box (Active) -->
        <rect x="0" y="55" width="150" height="45" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="2"></rect>
        <text x="75" y="83" font-size="15" font-weight="600" fill="#1e40af" text-anchor="middle" font-family="ui-monospace, monospace">2-layer GCN</text>     
        <!-- Bottom Box -->
        <rect x="0" y="110" width="150" height="35" rx="8" fill="#f3f4f6" stroke="#d1d5db" stroke-width="2"></rect>
        <text x="75" y="132" font-size="13" fill="#4b5563" text-anchor="middle" font-family="sans-serif">mean pooling</text>       
        <text x="75" y="170" font-size="12" fill="#6b7280" text-anchor="middle" font-family="sans-serif">produce</text>
        <text x="75" y="185" font-size="12" fill="#6b7280" text-anchor="middle" font-family="sans-serif">embeddings</text>
      </g>
      <!-- Arrow 3 -->
      <line x1="630" y1="220" x2="690" y2="220" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"></line>
      <!-- 4. Bi-SSM -->
      <g id="step4" transform="translate(710, 120)">
        <!-- Forward Arrow -->
        <path d="M 20,75 L 160,75" fill="none" stroke="#3b82f6" stroke-width="3" marker-end="url(#arrowhead)"></path>
        <text x="90" y="62" font-size="13" font-weight="600" fill="#2563eb" text-anchor="middle" font-family="ui-monospace, monospace">forward</text>      
        <!-- Tokens -->
        <rect x="40" y="85" width="32" height="42" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="2"></rect>
        <text x="56" y="111" font-size="13" font-weight="600" fill="#374151" text-anchor="middle" font-family="monospace">t₁</text>     
        <rect x="82" y="85" width="32" height="42" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="2"></rect>
        <text x="98" y="111" font-size="13" font-weight="600" fill="#374151" text-anchor="middle" font-family="monospace">t₂</text>    
        <rect x="124" y="85" width="32" height="42" rx="6" fill="#ffffff" stroke="#3b82f6" stroke-width="2"></rect>
        <text x="140" y="111" font-size="13" font-weight="600" fill="#374151" text-anchor="middle" font-family="monospace">t₃</text>
        <!-- Backward Arrow -->
        <path d="M 160,140 L 20,140" fill="none" stroke="#f59e0b" stroke-width="3" marker-end="url(#arrowhead)"></path>
        <text x="90" y="160" font-size="13" font-weight="600" fill="#d97706" text-anchor="middle" font-family="ui-monospace, monospace">backward</text>   
        <text x="90" y="185" font-size="12" fill="#6b7280" text-anchor="middle" font-family="sans-serif">context-rich</text>
        <text x="90" y="200" font-size="12" fill="#6b7280" text-anchor="middle" font-family="sans-serif">representation</text>
      </g>
      <!-- Arrow 4 -->
      <line x1="890" y1="220" x2="950" y2="220" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"></line>
      <!-- 5. Linear Classifier -->
      <g id="step5" transform="translate(970, 120)">
        <rect x="0" y="75" width="110" height="70" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"></rect>
        <text x="55" y="103" font-size="14" font-weight="700" fill="#15803d" text-anchor="middle" font-family="ui-monospace, monospace">Linear</text>
        <text x="55" y="121" font-size="14" font-weight="700" fill="#15803d" text-anchor="middle" font-family="ui-monospace, monospace">Classifier</text>     
        <text x="55" y="138" font-size="12" fill="#166534" text-anchor="middle" font-family="sans-serif">class logits</text>
        <text x="55" y="170" font-size="12" fill="#6b7280" text-anchor="middle" font-family="sans-serif">Final output</text>
      </g>
      <!-- Main Title inside SVG -->
      <text x="550" y="50" font-size="18" font-weight="700" fill="#111827" text-anchor="middle" font-family="ui-monospace, monospace" letter-spacing="0.5">
        Graph → Tokeniser → Mini-GCN → Bi-SSM → Linear Classifier
      </text>
    </svg>
  </div>
</div>

<div class="info-box">
<h4>Architectural Components</h4>
<ul>
<li><strong>Tokeniser:</strong> Generation of multi-scale structural snapshots \(\{\mathcal{T}_{\ell}(v)\}\) via random walk sampling. </li>
<li><strong>Local Encoding:</strong> Application of the parameterized encoder \(\varphi(\cdot)\) (specifically, a 2-layer GCN followed by mean pooling) to map each subgraph to a latent vector \(\mathbf{x}_{\ell} \in \mathbb{R}^d\).</li> 
<li><strong>Sequential Modeling:</strong> Contextualization of the token sequence \(\mathbf{X}(v)\) via a Bidirectional Mamba block, fusing forward and backward hidden states to produce the final node embedding.</li> 
<li><strong>Prediction Head:</strong> A linear projection layer mapping the aggregated representations to the target label space for classification.</li> </ul> </div>

<p> This design yields a model that is both parameter-efficient and computationally scalable. It achieves <strong>linear time complexity</strong> \(O(K)\) relative to sequence length and maintains a reduced memory footprint, enabling state-of-the-art performance on large-scale benchmarks. </p>
</section>

<section class="article-width" id="results" style="margin-top: 6rem; margin-bottom: 2rem;">
<h2 id="results-on-cora">Results on Cora</h2>
<p>Validation metrics (best checkpoint):</p>
<div class="info-box">
<div><strong>Accuracy:</strong> 0.6780</div>
<div><strong>Macro Precision:</strong> 0.6945</div>
<div><strong>Macro Recall:</strong> 0.6737</div>
<div><strong>Macro F1:</strong> 0.6839</div>
</div>
<div style="display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; align-items: flex-start; margin-bottom: 1.5rem;">
<figure style="flex: 1 1 320px; margin: 0; text-align: center;">
<img alt="Training/validation loss" src="{{ 'assets/img/2026-01-22-graph-mamba/output.png' | relative_url }}" style="width: 100%; height: auto; display: block;"/>
<figcaption class="figure-caption">Training vs validation loss</figcaption>
</figure>
<figure style="flex: 1 1 320px; margin: 0; text-align: center;">
<img alt="Training/validation accuracy" src="{{ 'assets/img/2026-01-22-graph-mamba/output_2.png' | relative_url }}" style="width: 100%; height: auto; display: block;"/>
<figcaption class="figure-caption">Training vs validation accuracy</figcaption>
</figure>

<p>
  These results demonstrate that <strong>Graph Mamba</strong> effectively captures structural information on the Cora benchmark. The close alignment between <strong>Precision (69.5%)</strong> and <strong>Recall (67.4%)</strong> indicates a balanced classification capability across different research topics, minimizing both false positives and false negatives.
</p>
<p>
  Notably, this performance is achieved with a <strong>linear-time complexity</strong> architecture, making it significantly more efficient than standard Graph Transformers while maintaining competitive accuracy for node classification tasks.
</p>


</div>

<details>
<summary><strong>Python code:</strong> Training loop (Graph Mamba on Cora)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Classifier head: take the node embedding (64 numbers) and output class scores.</span>
<span class="n">num_classes</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">.</span><span class="nb">max</span><span class="p">().</span><span class="n">item</span><span class="p">()</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">head</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

<span class="c1"># Train everything together: local encoder + sequence block + classifier.</span>
<span class="n">opt</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">optim</span><span class="p">.</span><span class="n">Adam</span><span class="p">(</span>
    <span class="nb">list</span><span class="p">(</span><span class="n">local_encoder</span><span class="p">.</span><span class="n">parameters</span><span class="p">())</span> <span class="o">+</span>
    <span class="nb">list</span><span class="p">(</span><span class="n">mamba_layer</span><span class="p">.</span><span class="n">parameters</span><span class="p">())</span> <span class="o">+</span>
    <span class="nb">list</span><span class="p">(</span><span class="n">head</span><span class="p">.</span><span class="n">parameters</span><span class="p">()),</span>
    <span class="n">lr</span><span class="o">=</span><span class="mf">5e-4</span><span class="p">,</span> <span class="n">weight_decay</span><span class="o">=</span><span class="mf">5e-4</span>
<span class="p">)</span>
<span class="n">loss_fn</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">CrossEntropyLoss</span><span class="p">()</span>
<span class="n">ckpt_path</span> <span class="o">=</span> <span class="s">"graph_mamba_best.pth"</span>
<span class="n">best_val_acc</span> <span class="o">=</span> <span class="o">-</span><span class="mf">1.0</span>
<span class="n">best_epoch</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span>
<span class="n">best_state</span> <span class="o">=</span> <span class="bp">None</span>

<span class="k">def</span> <span class="nf">build_token_batch</span><span class="p">(</span><span class="n">node_ids</span><span class="p">):</span>
    <span class="c1"># For each node id, build its token sequence by encoding several walk-length tokens with the LocalEncoder.</span>
    <span class="n">seqs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">node_ids</span><span class="p">:</span>
        <span class="n">token_vectors</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">reversed</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)):</span>
            <span class="n">token_vectors</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">local_encoder</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">tokens</span><span class="p">[</span><span class="n">v</span><span class="p">][</span><span class="n">i</span><span class="p">],</span> <span class="n">data</span><span class="p">))</span>
        <span class="n">seqs</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">token_vectors</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">seqs</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="n">EPOCHS</span> <span class="o">=</span> <span class="mi">5</span>
<span class="n">BATCH</span> <span class="o">=</span> <span class="mi">256</span>
<span class="n">train_loss_hist</span><span class="p">,</span> <span class="n">val_loss_hist</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
<span class="n">train_acc_hist</span><span class="p">,</span>  <span class="n">val_acc_hist</span>  <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">ep</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">EPOCHS</span><span class="p">):</span>
    <span class="c1"># Training pass.</span>
    <span class="n">local_encoder</span><span class="p">.</span><span class="n">train</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="n">train</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="n">train</span><span class="p">()</span>
    <span class="c1"># Shuffle nodes before batching.</span>
    <span class="n">perm</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">randperm</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">tot_loss</span> <span class="o">=</span> <span class="n">tot_cnt</span> <span class="o">=</span> <span class="n">corr</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">BATCH</span><span class="p">):</span>
        <span class="c1"># Take a chunk of node ids; then keep only those that belong to the training split.</span>
        <span class="n">bn</span> <span class="o">=</span> <span class="n">perm</span><span class="p">[</span><span class="n">s</span><span class="p">:</span><span class="n">s</span><span class="o">+</span><span class="n">BATCH</span><span class="p">]</span>
        <span class="n">bn</span> <span class="o">=</span> <span class="n">bn</span><span class="p">[</span><span class="n">data</span><span class="p">.</span><span class="n">train_mask</span><span class="p">[</span><span class="n">bn</span><span class="p">]]</span>

        <span class="c1"># Build token sequences for this mini-batch (this is the expensive step).</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">build_token_batch</span><span class="p">(</span><span class="n">bn</span><span class="p">.</span><span class="n">tolist</span><span class="p">())</span>
        <span class="c1"># Run the sequence model over tokens.</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="c1"># Use the last step as the final node embedding.</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>
        <span class="n">y</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">[</span><span class="n">bn</span><span class="p">]</span>

        <span class="n">logits</span> <span class="o">=</span> <span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">loss_fn</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

        <span class="c1"># Backprop + optimizer step.</span>
        <span class="n">opt</span><span class="p">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">loss</span><span class="p">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">opt</span><span class="p">.</span><span class="n">step</span><span class="p">()</span>

        <span class="n">tot_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="p">.</span><span class="n">item</span><span class="p">()</span> <span class="o">*</span> <span class="n">bn</span><span class="p">.</span><span class="n">numel</span><span class="p">()</span>
        <span class="n">tot_cnt</span>  <span class="o">+=</span> <span class="n">bn</span><span class="p">.</span><span class="n">numel</span><span class="p">()</span>
        <span class="n">corr</span>     <span class="o">+=</span> <span class="p">(</span><span class="n">logits</span><span class="p">.</span><span class="n">argmax</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="o">==</span> <span class="n">y</span><span class="p">).</span><span class="nb">sum</span><span class="p">().</span><span class="n">item</span><span class="p">()</span>

    <span class="n">train_loss</span> <span class="o">=</span> <span class="n">tot_loss</span> <span class="o">/</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">tot_cnt</span><span class="p">)</span>
    <span class="n">train_acc</span>  <span class="o">=</span> <span class="n">corr</span> <span class="o">/</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">tot_cnt</span><span class="p">)</span>
    <span class="n">train_loss_hist</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">train_loss</span><span class="p">)</span>
    <span class="n">train_acc_hist</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">train_acc</span><span class="p">)</span>

    <span class="c1"># Validation pass (no gradients).</span>
    <span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>
    <span class="k">with</span> <span class="n">torch</span><span class="p">.</span><span class="n">no_grad</span><span class="p">():</span>
        <span class="n">val_nodes</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">arange</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)[</span><span class="n">data</span><span class="p">.</span><span class="n">val_mask</span><span class="p">].</span><span class="n">tolist</span><span class="p">()</span>
        <span class="n">v_loss</span> <span class="o">=</span> <span class="n">v_cnt</span> <span class="o">=</span> <span class="n">v_corr</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">val_nodes</span><span class="p">),</span> <span class="n">BATCH</span><span class="p">):</span>
            <span class="n">part</span> <span class="o">=</span> <span class="n">val_nodes</span><span class="p">[</span><span class="n">s</span><span class="p">:</span><span class="n">s</span><span class="o">+</span><span class="n">BATCH</span><span class="p">]</span>
            <span class="n">x</span> <span class="o">=</span> <span class="n">build_token_batch</span><span class="p">(</span><span class="n">part</span><span class="p">)</span>
            <span class="n">x</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">x</span><span class="p">)[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>
            <span class="n">y</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">[</span><span class="n">part</span><span class="p">]</span>
            <span class="n">logits</span> <span class="o">=</span> <span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
            <span class="n">loss</span> <span class="o">=</span> <span class="n">loss_fn</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
            <span class="n">v_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="p">.</span><span class="n">item</span><span class="p">()</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">part</span><span class="p">)</span>
            <span class="n">v_cnt</span>  <span class="o">+=</span> <span class="nb">len</span><span class="p">(</span><span class="n">part</span><span class="p">)</span>
            <span class="n">v_corr</span> <span class="o">+=</span> <span class="p">(</span><span class="n">logits</span><span class="p">.</span><span class="n">argmax</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="o">==</span> <span class="n">y</span><span class="p">).</span><span class="nb">sum</span><span class="p">().</span><span class="n">item</span><span class="p">()</span>
        <span class="n">val_loss</span> <span class="o">=</span> <span class="n">v_loss</span> <span class="o">/</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">v_cnt</span><span class="p">)</span>
        <span class="n">val_acc</span>  <span class="o">=</span> <span class="n">v_corr</span> <span class="o">/</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">v_cnt</span><span class="p">)</span>
        <span class="n">val_loss_hist</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">val_loss</span><span class="p">)</span>
        <span class="n">val_acc_hist</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">val_acc</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">val_acc</span> <span class="o">&gt;</span> <span class="n">best_val_acc</span><span class="p">:</span>
        <span class="n">best_val_acc</span> <span class="o">=</span> <span class="n">val_acc</span>
        <span class="n">best_epoch</span> <span class="o">=</span> <span class="n">ep</span> <span class="o">+</span> <span class="mi">1</span>
        <span class="n">best_state</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s">"epoch"</span><span class="p">:</span> <span class="n">best_epoch</span><span class="p">,</span>
            <span class="s">"train_loss"</span><span class="p">:</span> <span class="n">train_loss</span><span class="p">,</span>
            <span class="s">"val_loss"</span><span class="p">:</span> <span class="n">val_loss</span><span class="p">,</span>
            <span class="s">"train_acc"</span><span class="p">:</span> <span class="n">train_acc</span><span class="p">,</span>
            <span class="s">"val_acc"</span><span class="p">:</span> <span class="n">val_acc</span><span class="p">,</span>
            <span class="s">"local_encoder"</span><span class="p">:</span> <span class="n">local_encoder</span><span class="p">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s">"mamba_layer"</span><span class="p">:</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="n">state_dict</span><span class="p">(),</span>
            <span class="s">"head"</span><span class="p">:</span> <span class="n">head</span><span class="p">.</span><span class="n">state_dict</span><span class="p">(),</span>
        <span class="p">}</span>

    <span class="k">print</span><span class="p">(</span><span class="sa">f</span><span class="s">"epoch </span><span class="si">{</span><span class="n">ep</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s">: loss=</span><span class="si">{</span><span class="n">train_loss</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s"> val_loss=</span><span class="si">{</span><span class="n">val_loss</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s"> | acc=</span><span class="si">{</span><span class="n">train_acc</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s"> val_acc=</span><span class="si">{</span><span class="n">val_acc</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">"</span><span class="p">)</span>

<span class="k">if</span> <span class="n">best_state</span> <span class="ow">is</span> <span class="ow">not</span> <span class="bp">None</span><span class="p">:</span>
    <span class="n">torch</span><span class="p">.</span><span class="n">save</span><span class="p">(</span><span class="n">best_state</span><span class="p">,</span> <span class="n">ckpt_path</span><span class="p">)</span>
    <span class="n">local_encoder</span><span class="p">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">best_state</span><span class="p">[</span><span class="s">"local_encoder"</span><span class="p">])</span>
    <span class="n">mamba_layer</span><span class="p">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">best_state</span><span class="p">[</span><span class="s">"mamba_layer"</span><span class="p">])</span>
    <span class="n">head</span><span class="p">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">best_state</span><span class="p">[</span><span class="s">"head"</span><span class="p">])</span>
    <span class="k">print</span><span class="p">(</span><span class="sa">f</span><span class="s">"Saved best checkpoint to </span><span class="si">{</span><span class="n">ckpt_path</span><span class="si">}</span><span class="s"> (epoch </span><span class="si">{</span><span class="n">best_epoch</span><span class="si">}</span><span class="s">, val_acc=</span><span class="si">{</span><span class="n">best_val_acc</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">)"</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="k">print</span><span class="p">(</span><span class="s">"No best checkpoint captured; check your training masks and data pipeline."</span><span class="p">)</span>

<span class="n">plt</span><span class="p">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">3</span><span class="p">))</span>
<span class="n">plt</span><span class="p">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train_loss_hist</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s">"train"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">plot</span><span class="p">(</span><span class="n">val_loss_hist</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s">"val"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="p">.</span><span class="n">title</span><span class="p">(</span><span class="s">"loss"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">show</span><span class="p">()</span>

<span class="n">plt</span><span class="p">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">3</span><span class="p">))</span>
<span class="n">plt</span><span class="p">.</span><span class="n">plot</span><span class="p">(</span><span class="n">train_acc_hist</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s">"train"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">plot</span><span class="p">(</span><span class="n">val_acc_hist</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s">"val"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="p">.</span><span class="n">title</span><span class="p">(</span><span class="s">"acc"</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">ylim</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="p">.</span><span class="n">show</span><span class="p">()</span>

<span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>
<span class="n">val_nodes</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">arange</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)[</span><span class="n">data</span><span class="p">.</span><span class="n">val_mask</span><span class="p">]</span>
<span class="n">all_logits</span><span class="p">,</span> <span class="n">all_labels</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>

<span class="k">with</span> <span class="n">torch</span><span class="p">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">val_nodes</span><span class="p">),</span> <span class="n">BATCH</span><span class="p">):</span>
        <span class="n">part</span> <span class="o">=</span> <span class="n">val_nodes</span><span class="p">[</span><span class="n">s</span><span class="p">:</span><span class="n">s</span><span class="o">+</span><span class="n">BATCH</span><span class="p">]</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">build_token_batch</span><span class="p">(</span><span class="n">part</span><span class="p">.</span><span class="n">tolist</span><span class="p">())</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">x</span><span class="p">)[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>
        <span class="n">logits</span> <span class="o">=</span> <span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">all_logits</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">logits</span><span class="p">.</span><span class="n">cpu</span><span class="p">())</span>
        <span class="n">all_labels</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">[</span><span class="n">part</span><span class="p">].</span><span class="n">cpu</span><span class="p">())</span>

<span class="k">if</span> <span class="n">all_logits</span><span class="p">:</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">cat</span><span class="p">(</span><span class="n">all_logits</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">labels</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">cat</span><span class="p">(</span><span class="n">all_labels</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">preds</span> <span class="o">=</span> <span class="n">logits</span><span class="p">.</span><span class="n">argmax</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>

    <span class="n">num_classes</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">labels</span><span class="p">.</span><span class="nb">max</span><span class="p">().</span><span class="n">item</span><span class="p">()</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="n">conf</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">zeros</span><span class="p">((</span><span class="n">num_classes</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="p">.</span><span class="nb">long</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">t</span><span class="p">,</span> <span class="n">p</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">labels</span><span class="p">,</span> <span class="n">preds</span><span class="p">):</span>
        <span class="n">conf</span><span class="p">[</span><span class="n">t</span><span class="p">,</span> <span class="n">p</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>

    <span class="n">tp</span> <span class="o">=</span> <span class="n">conf</span><span class="p">.</span><span class="n">diag</span><span class="p">().</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="nb">float</span><span class="p">)</span>
    <span class="n">fp</span> <span class="o">=</span> <span class="n">conf</span><span class="p">.</span><span class="nb">sum</span><span class="p">(</span><span class="mi">0</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="nb">float</span><span class="p">)</span> <span class="o">-</span> <span class="n">tp</span>
    <span class="n">fn</span> <span class="o">=</span> <span class="n">conf</span><span class="p">.</span><span class="nb">sum</span><span class="p">(</span><span class="mi">1</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="nb">float</span><span class="p">)</span> <span class="o">-</span> <span class="n">tp</span>

    <span class="n">macro_prec</span> <span class="o">=</span> <span class="p">(</span><span class="n">tp</span> <span class="o">/</span> <span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fp</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">)).</span><span class="n">mean</span><span class="p">().</span><span class="n">item</span><span class="p">()</span>
    <span class="n">macro_rec</span>  <span class="o">=</span> <span class="p">(</span><span class="n">tp</span> <span class="o">/</span> <span class="p">(</span><span class="n">tp</span> <span class="o">+</span> <span class="n">fn</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">)).</span><span class="n">mean</span><span class="p">().</span><span class="n">item</span><span class="p">()</span>
    <span class="n">macro_f1</span>   <span class="o">=</span> <span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="n">macro_prec</span> <span class="o">*</span> <span class="n">macro_rec</span> <span class="o">/</span> <span class="p">(</span><span class="n">macro_prec</span> <span class="o">+</span> <span class="n">macro_rec</span> <span class="o">+</span> <span class="mf">1e-8</span><span class="p">))</span>
    <span class="n">acc</span> <span class="o">=</span> <span class="p">(</span><span class="n">preds</span> <span class="o">==</span> <span class="n">labels</span><span class="p">).</span><span class="nb">float</span><span class="p">().</span><span class="n">mean</span><span class="p">().</span><span class="n">item</span><span class="p">()</span>

    <span class="k">print</span><span class="p">(</span><span class="sa">f</span><span class="s">"Validation metrics — acc: </span><span class="si">{</span><span class="n">acc</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">, macro_precision: </span><span class="si">{</span><span class="n">macro_prec</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">, macro_recall: </span><span class="si">{</span><span class="n">macro_rec</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">, macro_f1: </span><span class="si">{</span><span class="n">macro_f1</span><span class="p">:.</span><span class="mi">4</span><span class="n">f</span><span class="si">}</span><span class="s">"</span><span class="p">)</span>
<span class="k">else</span><span class="p">:</span>
    <span class="k">print</span><span class="p">(</span><span class="s">"No validation samples found to compute metrics."</span><span class="p">)</span></code></pre></figure>

</details>

<details>
<summary><strong>Python code:</strong> Build node representations after training</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Inference-time embedding build: compute token sequences for all nodes and take the last SSM step as the node embedding.</span>
<span class="c1"># Shapes: all_nodes_token_seqs [N, L, d] -> mixed_token_seqs [N, L, d] -> node_representations [N, d].</span>
<span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>
<span class="n">mamba_layer</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>

<span class="k">with</span> <span class="n">torch</span><span class="p">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="n">ordered_token_embeddings</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
        <span class="n">vectors</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">reversed</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)):</span>
            <span class="n">vectors</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">local_encoder</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">tokens</span><span class="p">[</span><span class="n">v</span><span class="p">][</span><span class="n">i</span><span class="p">],</span> <span class="n">data</span><span class="p">))</span>
        
        <span class="n">ordered_token_embeddings</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">vectors</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>

    <span class="n">all_nodes_token_seqs</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">ordered_token_embeddings</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">mixed_token_seqs</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">all_nodes_token_seqs</span><span class="p">)</span>
    <span class="n">node_representations</span> <span class="o">=</span> <span class="n">mixed_token_seqs</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>

<span class="n">node_representations</span></code></pre></figure>

</details>

<details>
<summary><strong>Python code:</strong> Export predictions for the website visualization</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="c1"># Export predicted labels to cora_visualization_pred.json for the website
# Output JSON schema matches ForceGraph3D: { nodes: [{id,label,labelIdx,val}], links: [{source,target}] }.
</span><span class="n">label_names</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s">"Case_Based"</span><span class="p">,</span> <span class="s">"Genetic_Algorithms"</span><span class="p">,</span> <span class="s">"Neural_Networks"</span><span class="p">,</span>
    <span class="s">"Probabilistic_Methods"</span><span class="p">,</span> <span class="s">"Reinforcement_Learning"</span><span class="p">,</span>
    <span class="s">"Rule_Learning"</span><span class="p">,</span> <span class="s">"Theory"</span>
<span class="p">]</span>

<span class="n">pred_json_path</span> <span class="o">=</span> <span class="s">"cora_visualization_pred.json"</span>
<span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>

<span class="n">nodes_pred</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">links_pred</span> <span class="o">=</span> <span class="p">[]</span>

<span class="c1"># build undirected edge list without duplicates for visualization
</span><span class="n">seen_edges</span> <span class="o">=</span> <span class="nb">set</span><span class="p">()</span>
<span class="n">edge_pairs</span> <span class="o">=</span> <span class="n">edge_index</span><span class="p">.</span><span class="n">t</span><span class="p">().</span><span class="n">cpu</span><span class="p">().</span><span class="n">tolist</span><span class="p">()</span>
<span class="k">for</span> <span class="n">s</span><span class="p">,</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">edge_pairs</span><span class="p">:</span>
    <span class="n">a</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="n">s</span><span class="p">),</span> <span class="nb">int</span><span class="p">(</span><span class="n">t</span><span class="p">))</span>
    <span class="k">if</span> <span class="n">a</span> <span class="o">&gt;</span> <span class="n">b</span><span class="p">:</span>
        <span class="n">a</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="n">b</span><span class="p">,</span> <span class="n">a</span>
    <span class="n">key</span> <span class="o">=</span> <span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">seen_edges</span><span class="p">:</span>
        <span class="k">continue</span>
    <span class="n">seen_edges</span><span class="p">.</span><span class="n">add</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
    <span class="n">links_pred</span><span class="p">.</span><span class="n">append</span><span class="p">({</span><span class="s">"source"</span><span class="p">:</span> <span class="n">a</span><span class="p">,</span> <span class="s">"target"</span><span class="p">:</span> <span class="n">b</span><span class="p">})</span>

<span class="k">with</span> <span class="n">torch</span><span class="p">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">start</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">BATCH</span><span class="p">):</span>
        <span class="n">ids</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="nb">min</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">start</span> <span class="o">+</span> <span class="n">BATCH</span><span class="p">)))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">build_token_batch</span><span class="p">(</span><span class="n">ids</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">x</span><span class="p">)[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>
        <span class="n">logits</span> <span class="o">=</span> <span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">pred</span> <span class="o">=</span> <span class="n">logits</span><span class="p">.</span><span class="n">argmax</span><span class="p">(</span><span class="mi">1</span><span class="p">).</span><span class="n">cpu</span><span class="p">().</span><span class="n">tolist</span><span class="p">()</span>

        <span class="k">for</span> <span class="n">node_id</span><span class="p">,</span> <span class="n">cls</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">ids</span><span class="p">,</span> <span class="n">pred</span><span class="p">):</span>
            <span class="n">label_idx</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">cls</span><span class="p">)</span>
            <span class="n">label_name</span> <span class="o">=</span> <span class="n">label_names</span><span class="p">[</span><span class="n">label_idx</span><span class="p">]</span> <span class="k">if</span> <span class="n">label_idx</span> <span class="o">&lt;</span> <span class="nb">len</span><span class="p">(</span><span class="n">label_names</span><span class="p">)</span> <span class="k">else</span> <span class="nb">str</span><span class="p">(</span><span class="n">label_idx</span><span class="p">)</span>
            <span class="n">nodes_pred</span><span class="p">.</span><span class="n">append</span><span class="p">({</span>
                <span class="s">"id"</span><span class="p">:</span> <span class="nb">int</span><span class="p">(</span><span class="n">node_id</span><span class="p">),</span>
                <span class="s">"label"</span><span class="p">:</span> <span class="n">label_name</span><span class="p">,</span>
                <span class="s">"labelIdx"</span><span class="p">:</span> <span class="n">label_idx</span><span class="p">,</span>
                <span class="s">"val"</span><span class="p">:</span> <span class="mf">1.5</span>
            <span class="p">})</span>

<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">pred_json_path</span><span class="p">,</span> <span class="s">"w"</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s">"utf-8"</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">json</span><span class="p">.</span><span class="n">dump</span><span class="p">({</span><span class="s">"nodes"</span><span class="p">:</span> <span class="n">nodes_pred</span><span class="p">,</span> <span class="s">"links"</span><span class="p">:</span> <span class="n">links_pred</span><span class="p">},</span> <span class="n">f</span><span class="p">)</span>

<span class="k">print</span><span class="p">(</span><span class="sa">f</span><span class="s">"Saved predictions to </span><span class="si">{</span><span class="n">pred_json_path</span><span class="si">}</span><span class="s">: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">nodes_pred</span><span class="p">)</span><span class="si">}</span><span class="s"> nodes, </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">links_pred</span><span class="p">)</span><span class="si">}</span><span class="s"> edges"</span><span class="p">)</span></code></pre></figure>

</details>
</section>
<section style="width: 100%; margin: 0 auto;">
  <h3 class="article-width">Classified Cora Graph using Graph Mamba</h3>
    <div style="width: 140%; margin-left: -20%; margin-top: 1.5rem; margin-bottom: 1.5rem;">
    <div style="
        position: relative; 
        width: 100%; 
        border: 1px solid #e5e7eb; 
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden; /* Важно для закругления углов */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #1f2937;">       
      <div class="viz-header" style="
          position: absolute; 
          top: 15px; 
          left: 15px; 
          z-index: 10; 
          background: rgba(255, 255, 255, 0.95); 
          color: #1f2937; 
          padding: 8px 12px; 
          border: 1px solid #e5e7eb; 
          border-radius: 6px; 
          font-size: 12px; 
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
          pointer-events: auto;">
        <strong style="display:block; margin-bottom:4px;">Prediction Status</strong>
        <div id="cora-classified-status" style="color: #4b5563;">Loading predictions...</div>
      </div>
      <div class="viz-footer" style="
          position: absolute; 
          bottom: 0; 
          left: 0; 
          width: 100%;
          z-index: 10; 
          background: rgba(255, 255, 255, 0.5);
          backdrop-filter: blur(2px);
          color: #4b5563;
          padding: 6px 0; 
          font-size: 11px; 
          text-align: center;
          border-top: 1px solid rgba(229, 231, 235, 0.5);
          font-family: monospace;
          pointer-events: none;">
        Left-click: rotate • Scroll: zoom
      </div>
      <div class="interactive-figure if--600" id="cora-classified-viz" style="
          position: relative; 
          width: 100%; 
          height: 560px; 
          min-height: 560px;
          background: #ffffff;">
      </div>
    </div>
    <div class="figure-caption" style="margin-top: 8px; text-align: left; color: #666; font-size: 0.9em;">
      <strong>Figure 2: Inference Results on Cora.</strong> 
      Visualization of node classifications generated by the Graph Mamba model. 
      Node colors represent the predicted topic categories, demonstrating the model's ability to recover community structure.
    </div>

  </div>
</section>

<script>
      (function() {
      // Render the classified Cora graph in 3D by fetching the model's prediction JSON produced by the code above.
      const container = document.getElementById('cora-classified-viz');
      const statusEl = document.getElementById('cora-classified-status');
      if (!container || !statusEl || typeof ForceGraph3D !== 'function') return;
      fetch('{{ "assets/html/2026-01-22-graph-mamba/data/cora_visualization_pred.json" | relative_url }}')
        .then(resp => {
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          return resp.json();
        })
        .then(payload => {
          const nodes = payload.nodes || [];
          const links = payload.links || [];
          // ForceGraph3D API is fluent: configure canvas size, data, and per-node/link render attributes.
          const graph = ForceGraph3D()(container)
            .width(container.clientWidth)
            .height(container.clientHeight)
            .graphData({ nodes, links })
            .showNavInfo(false)
            .nodeRelSize(4)
            .nodeVal(node => node.val || 2)
            .nodeColor(node => coraPalette[node.labelIdx % coraPalette.length])
            .nodeOpacity(1.0)
            .nodeResolution(16)
            .nodeLabel(node => `
                <div style="
                    color: #1f2937; 
                    background: rgba(255, 255, 255, 0.95); 
                    padding: 6px 10px; 
                    border: 1px solid #e5e7eb; 
                    border-radius: 6px; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
                    font-family: sans-serif; 
                    font-size: 12px;
                    pointer-events: none;">
                  <strong>Paper ${node.id}</strong><br/>
                  <span style="color: #6b7280;">Topic: ${node.label || 'Unknown'}</span>
                </div>
            `)            
            .linkWidth(0.5)
            .linkColor(() => '#242424')
            .linkOpacity(0.3)
            .backgroundColor('#ffffff')     
            .onEngineStop(() => statusEl.textContent = `Loaded ${nodes.length} nodes / ${links.length} edges`);
          // Keep the canvas responsive: recompute width/height from the container on resize.
          window.addEventListener('resize', () => {
            graph.width(container.clientWidth);
            graph.height(container.clientHeight);
          });
        })
        .catch(err => {
          statusEl.textContent = `Failed to load cora_visualization_pred.json: ${err.message}`;
        });
    })();
    </script>
<script>
// Static payload for the GCN local encoder widget below.
// This is a tiny, hand-authored toy subgraph so the visualization can show shapes/steps deterministically.
window.GCN_LOCAL_PAYLOAD = {
  // Which center node is being explained (in the toy example graph).
  "nodeId": 0,
  "graph": {
    // nodes: include fixed 2D positions (x,y) used for drawing and some metadata (distance/degree).
    "nodes": [
      {"id": 0, "x": 0.0, "y": 0.0, "distance": 0, "is_center": true, "degree": 3},
      {"id": 1, "x": -0.8, "y": 0.5, "distance": 1, "is_center": false, "degree": 3},
      {"id": 2, "x": 0.8, "y": 0.5, "distance": 1, "is_center": false, "degree": 3},
      {"id": 3, "x": -1.2, "y": -0.3, "distance": 2, "is_center": false, "degree": 2},
      {"id": 4, "x": 1.2, "y": -0.3, "distance": 2, "is_center": false, "degree": 2},
      {"id": 5, "x": 0.0, "y": -0.8, "distance": 1, "is_center": false, "degree": 1}
    ],
    // links: simple undirected-looking edges (the renderer treats these as connections).
    "links": [
      {"source": 0, "target": 1},
      {"source": 0, "target": 2},
      {"source": 0, "target": 5},
      {"source": 1, "target": 2},
      {"source": 1, "target": 3},
      {"source": 2, "target": 4},
      {"source": 3, "target": 4}
    ]
  },
  // layers: per-GCN-layer explanation frames (token membership + expected tensor shapes + step-by-step visuals).
  "layers": [
    {
      "walkLength": 2,
      "tokenNodes": [0, 1, 2, 3, 4, 5],
      // shapeInfo: k = |tokenNodes|, featureDim = F, hiddenDim = d for this visualization (toy numbers here).
      "shapeInfo": {"k": 6, "featureDim": 3, "hiddenDim": 3},
      "steps": [
        {
          "id": "adjacency",
          "title": "Adjacency Matrix 𝐀_ℓ",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["0", "1", "2", "3", "4", "5"],
            "values": [
              [0, 1, 1, 0, 0, 1],
              [1, 0, 1, 1, 0, 0],
              [1, 1, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 1, 0, 0],
              [1, 0, 0, 0, 0, 0]
            ]
          },
          "explain": "The adjacency matrix 𝐀_ℓ represents the connectivity within the token 𝓣_ℓ(v).",
          "formula": {
            "lhs": "𝐀_ℓ[i,j]",
            "rhs": "1 if i,j connected in 𝓣_ℓ, else 0"
          }
        },
        {
          "id": "input_features",
          "title": "Node Features 𝐇⁽⁰⁾",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["f0", "f1", "f2"],
            "values": [
              [0.22, -0.06, 0.29],
              [0.46, -0.07, -0.07],
              [0.47, 0.23, -0.14],
              [0.16, -0.14, -0.14],
              [0.07, -0.57, -0.52],
              [-0.17, -0.3, 0.09]
            ]
          },
          "explain": "Initial node features for the subgraph nodes, forming the matrix 𝐇⁽⁰⁾.",
          "formula": {
            "lhs": "𝐇⁽⁰⁾[i,f]",
            "rhs": "Raw feature f of node i in dimension f"
          }
        },
        {
          "id": "x1",
          "title": "Linear Transform I",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1", "h2"],
            "values": [
              [0.03, 0.25, 0.02],
              [0.22, 0.17, -0.14],
              [0.33, 0.05, -0.04],
              [0.07, 0.04, -0.13],
              [-0.03, -0.06, -0.4],
              [-0.19, 0.07, -0.06]
            ]
          },
          "explain": "Projects features into the first hidden space via weight matrix 𝐖⁽⁰⁾.",
          "formula": {
            "lhs": "Temp",
            "rhs": "𝐇⁽⁰⁾𝐖⁽⁰⁾"
          }
        },
        {
          "id": "ax1",
          "title": "Aggregation I",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1", "h2"],
            "values": [
              [0.36, 0.29, -0.24],
              [0.44, 0.34, -0.16],
              [0.22, 0.37, -0.52],
              [0.19, 0.11, -0.54],
              [0.4, 0.09, -0.17],
              [0.03, 0.25, 0.02]
            ]
          },
          "explain": "Aggregates messages from neighbors (simulates normalized multiplication Ã·Temp).",
          "formula": {
            "lhs": "Agg",
            "rhs": "𝐃⁻½𝐀_ℓ𝐃⁻½ · (𝐇⁽⁰⁾𝐖⁽⁰⁾)"
          }
        },
        {
          "id": "h1",
          "title": "Activation I (𝐇⁽¹⁾)",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1", "h2"],
            "values": [
              [0.36, 0.29, 0.0],
              [0.44, 0.34, 0.0],
              [0.22, 0.37, 0.0],
              [0.19, 0.11, 0.0],
              [0.4, 0.09, 0.0],
              [0.03, 0.25, 0.02]
            ]
          },
          "explain": "Applies ReLU activation σ(·) to obtain the first layer embeddings 𝐇⁽¹⁾.",
          "formula": {
            "lhs": "𝐇⁽¹⁾",
            "rhs": "σ(Agg)"
          }
        },
        {
          "id": "x2",
          "title": "Linear Transform II",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1"],
            "values": [
              [0.13, 0.0],
              [0.16, -0.0],
              [0.02, 0.09],
              [0.08, -0.02],
              [0.21, -0.12],
              [-0.05, 0.12]
            ]
          },
          "explain": "Projects 𝐇⁽¹⁾ via the second weight matrix 𝐖⁽¹⁾.",
          "formula": {
            "lhs": "Temp",
            "rhs": "𝐇⁽¹⁾𝐖⁽¹⁾"
          }
        },
        {
          "id": "ax2",
          "title": "Aggregation II",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1"],
            "values": [
              [0.14, 0.21],
              [0.23, 0.07],
              [0.5, -0.12],
              [0.37, -0.12],
              [0.1, 0.07],
              [0.13, 0.0]
            ]
          },
          "explain": "Second round of neighborhood aggregation using the same structure 𝐀_ℓ.",
          "formula": {
            "lhs": "Agg",
            "rhs": "𝐃⁻½𝐀_ℓ𝐃⁻½ · (𝐇⁽¹⁾𝐖⁽¹⁾)"
          }
        },
        {
          "id": "h2",
          "title": "Activation II (𝐇⁽²⁾)",
          "type": "matrix",
          "matrix": {
            "rows": ["0", "1", "2", "3", "4", "5"],
            "cols": ["h0", "h1"],
            "values": [
              [0.14, 0.21],
              [0.23, 0.07],
              [0.5, 0.0],
              [0.37, 0.0],
              [0.1, 0.07],
              [0.13, 0.0]
            ]
          },
          "explain": "Final node representations 𝐇⁽²⁾ for the snapshot.",
          "formula": {
            "lhs": "𝐇⁽²⁾",
            "rhs": "σ(Agg)"
          }
        },
        {
          "id": "z",
          "title": "Token Embedding 𝐱_ℓ(v)",
          "type": "vector",
          "vector": {
            "labels": ["h0", "h1"],
            "values": [0.247, 0.059]
          },
          "explain": "Mean pooling over all nodes in the snapshot yields the final token vector 𝐱_ℓ(v).",
          "formula": {
            "lhs": "𝐱_ℓ(v)",
            "rhs": "1/k · Σ_i 𝐇⁽²⁾[i,h] with k=6"
          }
        }
      ]
    }
  ]
};




(function() {
  const container = document.getElementById('walk-canvas');
  if (!container) return;

  function init() {
    const rect = container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    if (width < 50 || height < 50) {
      requestAnimationFrame(init);
      return;
    }

    d3.select(container).selectAll("svg").remove();

    const svg = d3.select(container).append("svg")
      .attr("width", width)
      .attr("height", height);

  // Toy graph
  const nodes = d3.range(8).map(i => ({ id: i }));
  const links = [
    {source: 0, target: 1}, {source: 0, target: 2},
    {source: 1, target: 3}, {source: 1, target: 4},
    {source: 2, target: 5}, {source: 2, target: 3},
    {source: 3, target: 6}, {source: 4, target: 7},
    {source: 5, target: 6}, {source: 6, target: 7}
  ];

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(70))
    .force("charge", d3.forceManyBody().strength(-400))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg.append("g")
      .attr("stroke", "#e5e7eb") // Light gray
      .attr("stroke-width", 2)
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("class", "graph-link");

  const nodeGroup = svg.append("g")
    .selectAll("g")
    .data(nodes)
    .enter().append("g")
    .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended)
    );

  nodeGroup.append("circle")
      .attr("r", 18)
      .attr("fill", "#ffffff")
      .attr("stroke", "#9ca3af") // Gray-400
      .attr("stroke-width", 2)
      .attr("class", "graph-node")
      .attr("id", d => "node-" + d.id);

  nodeGroup.append("text")
      .text(d => d.id)
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .attr("fill", "#374151") // Gray-700
      .style("font-family", "-apple-system, BlinkMacSystemFont, sans-serif")
      .style("font-weight", "600")
      .style("pointer-events", "none");

  const nodeById = {};
  nodeGroup.each(function(d) { nodeById[d.id] = d3.select(this); });

  // Walker
  const walker = svg.append("circle")
      .attr("r", 7)
      .attr("fill", "#f59e0b") // Amber-500
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 2)
      .attr("opacity", 0);

  let centerNode = null;

  const lenSlider   = document.getElementById("walk-length");
  const lenValue    = document.getElementById("walk-length-value");
  const countSlider = document.getElementById("walk-count");
  const countValue  = document.getElementById("walk-count-value");
  const sampleBtn   = document.getElementById("sample-token-btn");
  const seqDiv      = document.getElementById("walk-sequence");

  const maxL = parseInt(lenSlider.max, 10);

  // Embedding dimension D
  const D = 6;

  // s = 1: one vector per length ℓ
  let tokenMatrix = new Array(maxL + 1).fill(null);

  // Color scale for token matrix
  const valueScale = d3.scaleLinear()
      .domain([0, 1])
      .range(["#f3f4f6", "#22c55e"]); // Gray-100 -> Green-500

  lenSlider.addEventListener("input", () => {
    lenValue.textContent = lenSlider.value;
  });
  countSlider.addEventListener("input", () => {
    countValue.textContent = countSlider.value;
  });

  // Center node selection
  nodeGroup.on("click", function(event, d) {
    centerNode = d;

    nodeGroup.selectAll("circle")
        .interrupt()
        .attr("r", 18)
        .attr("fill", "#ffffff")
        .attr("stroke", "#9ca3af")
        .attr("stroke-width", 2);

    const c = d3.select(this).select("circle");
      c.attr("fill", "#dcfce7") // Light Green bg
       .attr("stroke", "#16a34a") // Green border
       .attr("stroke-width", 3)
       .attr("r", 20)
       .transition()
       .duration(250)
       .attr("r", 22)
       .transition()
       .duration(250)
       .attr("r", 20);

    clearTokenHighlight(false);
    seqDiv.textContent = "Center node v = " + d.id +
      ". Choose ℓ and M, then press Generate token.";

    // New center -> reset matrix (s = 1)
    tokenMatrix = new Array(maxL + 1).fill(null);
    renderTokenMatrix();
  });

  sampleBtn.addEventListener("click", () => {
    if (centerNode == null) {
      seqDiv.textContent = "Pick a center node first (click on any circle).";
      return;
    }
    const L = parseInt(lenSlider.value, 10);
    const M = parseInt(countSlider.value, 10);
    generateTokenForCenter(centerNode, L, M);
  });

  function generateTokenForCenter(center, L, M) {
    clearTokenHighlight(true);

    const unionVisited = new Set();
    const visitedPaths = [];

    for (let w = 0; w < M; w++) {
      let current = center.id;
      const path = [current];
      unionVisited.add(current);

      for (let step = 0; step < L; step++) {
        const neigh = neighborsOf(current);
        if (neigh.length === 0) break;
        const next = neigh[Math.floor(Math.random() * neigh.length)];
        path.push(next);
        unionVisited.add(next);
        current = next;
      }
      visitedPaths.push(path);
    }


    // Links highlighting
    link
        .transition().duration(200)
        .attr("stroke", "#e5e7eb")
        .attr("stroke-width", 2)
        .attr("opacity", 0.4);


    const nodesArr = Array.from(unionVisited).sort((a,b) => a-b);
    seqDiv.textContent =
      "Token for center v = " + center.id +
      " with length ℓ = " + L +
      " and M = " + M +
      ": nodes { " + nodesArr.join(", ") + " }";

    animateWalker(center.id, visitedPaths, unionVisited);

    // s = 1: update row for length ℓ
    addOrUpdateTokenRow(L);
  }

  // s = 1: one row per length ℓ
  function addOrUpdateTokenRow(lengthL) {
    const vec = d3.range(D).map(() => Math.random()); // pseudo-embedding
    tokenMatrix[lengthL] = { L: lengthL, vector: vec };
    renderTokenMatrix();
  }

  function renderTokenMatrix() {
    const svgMatrix = d3.select("#token-matrix");
    if (svgMatrix.empty()) return;

    const rowHeight = 14;
    const rowGap = 4;
    const colWidth = 16;
    const leftMargin = 28;
    const topMargin = 6;

    // Use only the ℓ, for which we already have a vector
    const rowsData = tokenMatrix
      .map((row, L) => row ? row : null)
      .filter(row => row !== null)
      .sort((a, b) => a.L - b.L);

    const height = topMargin + rowsData.length * (rowHeight + rowGap);
    svgMatrix.attr("height", Math.max(60, height));

    const rows = svgMatrix.selectAll("g.token-row")
      .data(rowsData, d => d.L);

    const rowsEnter = rows.enter().append("g")
      .attr("class", "token-row")
      .attr("transform", (d, i) =>
        `translate(0, ${topMargin + i * (rowHeight + rowGap)})`);

    // Note on the left ℓ = ...
    rowsEnter.append("text")
        .attr("x", 0)
        .attr("y", rowHeight - 3)
        .attr("fill", "#4b5563") // Darker gray
        .attr("font-size", 11)
        .attr("font-family", "monospace")
        .style("font-weight", "600")
        .text(d => "ℓ=" + d.L);

    // Rectangles for vector components
    rowsEnter.each(function(rowData) {
        const g = d3.select(this);
        g.selectAll("rect")
          .data(rowData.vector)
          .enter()
          .append("rect")
          .attr("x", (v, j) => leftMargin + j * (colWidth + 2))
          .attr("y", 0)
          .attr("width", colWidth)
          .attr("height", rowHeight)
          .attr("rx", 2)
          .attr("fill", v => valueScale(v))
          .attr("stroke", "#d1d5db") // Light border
          .attr("stroke-width", 1);
      });


    rows.merge(rowsEnter)
      .attr("transform", (d, i) =>
        `translate(0, ${topMargin + i * (rowHeight + rowGap)})`);

    rows.exit().remove();
  }

  // Walker animation along multiple paths
  function animateWalker(centerId, paths, unionVisited) {
    if (!paths.length) return;

    const cSel = nodeById[centerId];
    const cData = cSel.datum();
    walker
      .attr("cx", cData.x)
      .attr("cy", cData.y)
      .attr("opacity", 1);

    let pathIndex = 0;
    const nodesAlreadyHighlighted = new Set();

    if (centerNode) nodesAlreadyHighlighted.add(centerNode.id);

    function runNextPath() {
      if (pathIndex >= paths.length) {
        walker.transition().delay(300).duration(400)
          .attr("opacity", 0);
        return;
      }
      const path = paths[pathIndex];
      pathIndex++;

      const stepDuration = 600;
      let t = 0;

      for (let i = 0; i < path.length; i++) {
        const nodeId = path[i];
        setTimeout(() => {
          const nSel = nodeById[nodeId];
          const d = nSel.datum();

          walker.transition().duration(stepDuration - 100)
            .attr("cx", d.x)
            .attr("cy", d.y);

          if (i > 0) {
            const prevId = path[i - 1];
            highlightEdge(prevId, nodeId);
          }
          const circ = nSel.select("circle");
            
            // If it is the center node - it is always green
            // If it is a regular node - color it blue (Visited)
            const finalColor = (centerNode && nodeId === centerNode.id) 
                ? "#dcfce7" 
                : "#dbeafe"; 

            // Remember, that we already highlighted this node
            nodesAlreadyHighlighted.add(nodeId);

          // Animation of the node circle
            circ.transition().duration(150)
              .attr("fill", "#fcd34d")
              .transition().duration(250)
              .attr("fill", finalColor)
              .attr("stroke", (centerNode && nodeId === centerNode.id) ? "#16a34a" : "#2563eb");

          }, t);

        t += stepDuration;
      }

      setTimeout(runNextPath, path.length * stepDuration + 200);
    }

    runNextPath();
  }

  function highlightEdge(a, b) {
    const key1 = a + "-" + b;
    const key2 = b + "-" + a;

    link
      .filter(d => {
        const k = d.source.id + "-" + d.target.id;
        return k === key1 || k === key2;
      })
      .transition().duration(200)
      .attr("stroke", "#0ea5e9")
      .attr("stroke-width", 3)
      .attr("opacity", 0.9);
  }

  function neighborsOf(nodeId) {
    const neigh = [];
    links.forEach(l => {
      if (l.source.id === nodeId) neigh.push(l.target.id);
      else if (l.target.id === nodeId) neigh.push(l.source.id);
    });
    return neigh;
  }

  // resetCenter=false → do not reset the green center
  function clearTokenHighlight(resetCenter = true) {
    nodeGroup.selectAll("circle")
      .transition().duration(200)
      .attr("fill", d => {
        if (!resetCenter && centerNode && d.id === centerNode.id) return "#dcfce7";
        return "#ffffff";
      })
      .attr("stroke", d => {
        if (!resetCenter && centerNode && d.id === centerNode.id) return "#16a34a";
        return "#9ca3af";
      })
      .attr("stroke-width", d => {
        if (!resetCenter && centerNode && d.id === centerNode.id) return 3;
        return 2;
      });

    link
      .transition().duration(200)
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 2)
      .attr("opacity", 0.7);

    walker.attr("opacity", 0);
  }

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    nodeGroup
      .attr("transform", d => `translate(${d.x},${d.y})`);
  });

  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

    window.addEventListener('resize', () => {
      const r = container.getBoundingClientRect();
      const w = Math.max(1, Math.floor(r.width));
      const h = Math.max(1, Math.floor(r.height));
      svg.attr('width', w).attr('height', h);
      simulation.force("center", d3.forceCenter(w / 2, h / 2));
      simulation.alpha(0.3).restart();
    });
  }

  init();
})();


        (function() {
  const container = document.getElementById('bidirectional-mamba-viz');
  if (!container) return;

  let hasStarted = false;
  let animationTimer = null;

  function init() {
    const rect = container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));

    if (width < 50 || height < 50) {
      requestAnimationFrame(init);
      return;
    }

    container.innerHTML = '';

    const svg = d3.select(container).append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("background", "#ffffff");

    // CONFIG
    const numTokens = 7;
    const tokenSize = 50;
    const spacing = 70;
    const startX = (width - ((numTokens - 1) * spacing)) / 2;
    const centerY = height * 0.3; 
    
    // Create token data
    const tokens = d3.range(numTokens).map(i => ({
      id: i,
      x: startX + i * spacing,
      y: centerY
    }));

    // === VISUAL ELEMENTS ===

    // 1. Draw Tokens (Base Layer)
    const tokenGroup = svg.append('g');
    
    const tokenNodes = tokenGroup.selectAll('g')
      .data(tokens)
      .enter().append('g')
      .attr('transform', d => `translate(${d.x}, ${d.y})`);

    // Rectangles
    tokenNodes.append('rect')
      .attr('width', tokenSize)
      .attr('height', tokenSize)
      .attr('x', -tokenSize/2)
      .attr('y', -tokenSize/2)
      .attr('rx', 8)
      .attr('fill', '#ffffff')      // White
      .attr('stroke', '#e5e7eb')    // Light Gray Border
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.05))');

    // Text Labels
    tokenNodes.append('text')
      .text(d => `v${d.id}`)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('fill', '#374151')      // Dark Gray Text
      .attr('font-size', '14px')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace')
      .attr('font-weight', 'bold');

    // 2. Forward SSM Path (Blue, Top)
    const forwardGroup = svg.append('g');
    const forwardY = centerY - 80;
    
    forwardGroup.append('text')
      .text('Forward SSM')
      .attr('x', startX - 80)
      .attr('y', forwardY + 5)
      .attr('fill', '#3b82f6') // Blue
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace');

    // Forward Path Line
    const forwardPath = forwardGroup.append('path')
      .attr('d', `M ${startX - 30} ${forwardY} L ${startX + (numTokens-1)*spacing + 30} ${forwardY}`)
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 4)
      .attr('stroke-linecap', 'round')
      .attr('fill', 'none')
      .attr('opacity', 0.2); // Faint background line

    // Forward wave circle (Scanner)
    const forwardWave = forwardGroup.append('circle')
      .attr('r', 8)
      .attr('fill', '#3b82f6')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('cx', startX - 30)
      .attr('cy', forwardY)
      .attr('opacity', 0)
      .style('filter', 'drop-shadow(0 0 8px rgba(59,130,246, 0.6))');

    // 3. Backward SSM Path (Orange, Bottom)
    const backwardGroup = svg.append('g');
    const backwardY = centerY + 80;

    backwardGroup.append('text')
      .text('Backward SSM')
      .attr('x', startX + (numTokens-1)*spacing + 40)
      .attr('y', backwardY + 5)
      .attr('fill', '#f59e0b') // Amber
      .attr('font-size', '12px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace');

    // Backward Path Line
    const backwardPath = backwardGroup.append('path')
      .attr('d', `M ${startX + (numTokens-1)*spacing + 30} ${backwardY} L ${startX - 30} ${backwardY}`)
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 4)
      .attr('stroke-linecap', 'round')
      .attr('fill', 'none')
      .attr('opacity', 0.2);

    // Backward wave circle
    const backwardWave = backwardGroup.append('circle')
      .attr('r', 8)
      .attr('fill', '#f59e0b')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('cx', startX + (numTokens-1)*spacing + 30)
      .attr('cy', backwardY)
      .attr('opacity', 0)
      .style('filter', 'drop-shadow(0 0 8px rgba(245,158,11, 0.6))');


    // ANIMATION LOGIC
    function runBidirectionalAnimation() {
      const duration = 3000;
      
      // Reset token colors
      tokenNodes.selectAll('rect')
        .transition().duration(300)
        .attr('fill', '#ffffff')
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 2);

      tokenNodes.selectAll('text')
         .transition().duration(300)
         .attr('fill', '#374151');

      // 1. FORWARD PASS
      forwardWave
        .attr('opacity', 1)
        .attr('cx', startX - 30)
        .transition()
        .duration(duration)
        .ease(d3.easeLinear)
        .attr('cx', startX + (numTokens-1)*spacing + 30)
        .tween("processForward", function() {
           return function(t) {
             // Highlight tokens as wave passes
             const currentX = (startX - 30) + t * ((numTokens-1)*spacing + 60);
             tokens.forEach((token, i) => {
               // Check if wave is roughly over this token
               if (currentX > token.x - spacing/2 && currentX < token.x + spacing/2) {
                 d3.select(tokenNodes.nodes()[i]).select('rect')
                   .attr('stroke', '#3b82f6') // Blue Border
                   .attr('stroke-width', 3);
               }
             });
           };
        })
        .on('end', () => { forwardWave.attr('opacity', 0); });

      // 2. BACKWARD PASS (Simultaneous)
      backwardWave
        .attr('opacity', 1)
        .attr('cx', startX + (numTokens-1)*spacing + 30)
        .transition()
        .duration(duration)
        .ease(d3.easeLinear)
        .attr('cx', startX - 30)
        .tween("processBackward", function() {
           return function(t) {
             const currentX = (startX + (numTokens-1)*spacing + 30) - t * ((numTokens-1)*spacing + 60);
             tokens.forEach((token, i) => {
               if (currentX > token.x - spacing/2 && currentX < token.x + spacing/2) {
                 d3.select(tokenNodes.nodes()[i]).select('rect')
                   .attr('stroke', '#f59e0b') // Orange Border
                   .attr('stroke-width', 3);
               }
             });
           };
        })
        .on('end', () => { 
           backwardWave.attr('opacity', 0); 
           
           // 3. FUSION (After passes complete)
           setTimeout(() => {
             tokenNodes.selectAll('rect')
               .transition().duration(600)
               .attr('fill', '#16a34a')   // Green Fill (Success)
               .attr('stroke', '#166534') // Darker Green Border
               .attr('stroke-width', 2);
             
             tokenNodes.selectAll('text')
               .transition().duration(600)
               .attr('fill', '#ffffff'); // White Text
           }, 200);
        });
    }

    // CONTROLS
    const replayBtn = document.getElementById('bidir-replay-btn');
    if(replayBtn) {
      replayBtn.onclick = () => {
        // Stop any running transitions strictly
        forwardWave.interrupt();
        backwardWave.interrupt();
        tokenNodes.selectAll('rect').interrupt();
        tokenNodes.selectAll('text').interrupt();
        
        runBidirectionalAnimation();
      };
    }

    // SCROLL TRIGGER
    if (!hasStarted) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            hasStarted = true;
            setTimeout(runBidirectionalAnimation, 500);
            observer.disconnect();
          }
        });
      }, { threshold: 0.4 });
      observer.observe(container);
    }
  }

  // Init
  init();
  window.addEventListener('resize', () => {
    // Simple re-init logic on resize
    d3.select(container).selectAll("*").interrupt(); 
    init(); 
  });

})();


        async function loadCoraFromJSON() {
            try {
                console.log('🔄 Attempting to load Cora dataset from cora_visualization.json...');
                const response = await fetch('{{ "assets/html/2026-01-22-graph-mamba/data/cora_visualization.json" | relative_url }}');

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                console.log(`✅ Loaded REAL Cora from JSON: ${data.nodes.length} nodes, ${data.links.length} edges`);

                // Add random val if missing
                data.nodes.forEach(node => {
                    if (!node.val) {
                        node.val = 1 + Math.random() * 2;
                    }
                });

                return data;

            } catch (error) {
                console.warn('⚠️ Failed to load cora_visualization.json:', error.message);
                console.log('📦 Falling back to embedded subset (300 nodes)...');
                return getEmbeddedCoraData();
            }
        }
        
        function getEmbeddedCoraData() {
    const topics = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning',
        'Rule_Learning', 'Theory'
    ];

    const classDistribution = [
        { topic: 0, count: 42 }, { topic: 1, count: 43 },
        { topic: 2, count: 85 }, { topic: 3, count: 38 },
        { topic: 4, count: 22 }, { topic: 5, count: 35 },
        { topic: 6, count: 35 }
    ];

    const nodes = [];
    let nodeId = 0;

    classDistribution.forEach(({ topic, count }) => {
        for (let i = 0; i < count; i++) {
            nodes.push({
                id: nodeId++,
                label: topics[topic],
                labelIdx: topic,
                val: 1 + Math.random() * 2
            });
        }
    });

    const links = [];
    const avgDegree = 4;

    nodes.forEach((node, idx) => {
        const numCitations = Math.max(2, Math.min(6, 
            Math.floor(avgDegree * (0.6 + Math.random() * 0.8))
        ));

        for (let i = 0; i < numCitations; i++) {
            let targetIdx;

            if (Math.random() < 0.75) {
                const sameClassNodes = nodes
                    .map((n, i) => ({ node: n, index: i }))
                    .filter(({ node }) => node.labelIdx === nodes[idx].labelIdx && node.id !== nodes[idx].id);

                if (sameClassNodes.length > 0) {
                    targetIdx = sameClassNodes[Math.floor(Math.random() * sameClassNodes.length)].index;
                } else {
                    targetIdx = Math.floor(Math.random() * nodes.length);
                }
            } else {
                targetIdx = Math.floor(Math.random() * nodes.length);
            }

            if (targetIdx !== idx && 
                !links.some(l => 
                    (l.source === idx && l.target === targetIdx) ||
                    (l.source === targetIdx && l.target === idx)
                )) {
                links.push({ source: idx, target: targetIdx });
            }
        }
    });

    console.log(`📦 Встроенные данные Cora: ${nodes.length} нод, ${links.length} связей`);
    return { nodes, links };
}

        // Shared palette so ground-truth and predicted views match colors per labelIdx
        const coraPalette = [
            '#e11d48', // Bright Pink/Red
            '#d97706', // Amber
            '#16a34a', // Bright Green
            '#2563eb', // Bright Blue
            '#9333ea', // Bright Purple
            '#0891b2', // Cyan
            '#db2777'  // Magenta
        ];

        (async function() {
            const container = document.getElementById('cora-viz');
            if (!container) return;

            const data = await loadCoraFromJSON();

            const coraGraph = ForceGraph3D()
                (container)
                .showNavInfo(false)
                .width(container.clientWidth)
                .height(container.clientHeight)
                .graphData(data)
                .nodeColor(node => coraPalette[node.labelIdx % coraPalette.length])
                .nodeOpacity(1.0)
                .nodeResolution(16)
                .nodeLabel(node => `
                  <div style="
                      color: #1f2937; 
                      background: rgba(255, 255, 255, 0.95); 
                      padding: 6px 10px; 
                      border: 1px solid #e5e7eb; 
                      border-radius: 6px; 
                      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
                      font-family: sans-serif; 
                      font-size: 12px;
                      pointer-events: none;">
                    <strong>Paper ${node.id}</strong><br/>
                    <span style="color: #6b7280;">Topic: ${node.label || 'Unknown'}</span>
                  </div>
              `)
                .linkWidth(0.5)
                .linkColor(() => '#242424')
                .linkOpacity(0.3)
                .backgroundColor('#ffffff');

            window.resetCamera = () => {
                coraGraph.cameraPosition({ x: 0, y: 0, z: 1000 }, { x: 0, y: 0, z: 0 }, 1000);
            };

            window.addEventListener('resize', () => {
                coraGraph.width(container.clientWidth);
                coraGraph.height(container.clientHeight);
            });
        })();


        (function() {
  const container = document.getElementById('mamba-viz');
  if (!container) return;

  // Flag so that animation starts only once
  let hasStarted = false; 
  let animationTimer = null;

  function init() {
    // If container is not ready, retry
    const rect = container.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));

    if (width < 50 || height < 50) {
      requestAnimationFrame(init);
      return;
    }

    // Clear previous content
    container.innerHTML = '';

    // Explicitly set white background SVG
    const svg = d3.select(container).append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("background", "#ffffff");

    // CONFIG
    const config = {
      tokenSize: 50,
      tokenSpacing: 25,
      gateRadius: 40,
      animationDuration: 700,
      pauseDuration: 800,
      tokenSlideDistance: 80
    };

    const sequence = [
      { id: 'Paper 0', type: 'relevant', value: 4.2 },
      { id: 'Paper 4', type: 'relevant', value: 3.8 },
      { id: 'Noise 12', type: 'noise', value: 0.5 },
      { id: 'Noise 7', type: 'noise', value: 0.7 },
      { id: 'Paper 3', type: 'relevant', value: 4.5 },
      { id: 'Paper 9', type: 'relevant', value: 3.5 },
      { id: 'Noise 2', type: 'noise', value: 0.3 },
      { id: 'Paper 5', type: 'relevant', value: 4.0 }
    ];

    const centerX = width / 2;
    const tokenY = height * 0.2;
    const gateY = height * 0.48;
    const stateY = height * 0.88;

    let currentStep = 0;
    let hiddenState = 0;

    // GROUPS
    const tokenGroup = svg.append('g');
    const gateGroup = svg.append('g');
    const stateGroup = svg.append('g');
    const labelGroup = svg.append('g');

    // LABELS (Dark Colors)
    labelGroup.append('text')
      .attr('x', centerX)
      .attr('y', tokenY - 60)
      .attr('text-anchor', 'middle')
      .attr('fill', '#374151') // Dark gray
      .attr('font-size', '14px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace')
      .style('pointer-events', 'none')
      .text('Token Sequence');

    labelGroup.append('text')
      .attr('x', centerX)
      .attr('y', gateY - 65)
      .attr('text-anchor', 'middle')
      .attr('fill', '#374151') // Dark gray
      .attr('font-size', '15px')
      .attr('font-weight', '700')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace')
      .style('pointer-events', 'none')
      .text('Selective Gate Δ');

    labelGroup.append('text')
      .attr('x', centerX)
      .attr('y', stateY - 80)
      .attr('text-anchor', 'middle')
      .attr('fill', '#374151') // Dark gray
      .attr('font-size', '14px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace')
      .style('pointer-events', 'none')
      .text('Hidden State h_t');

    // GATE (Neutral State)
    gateGroup.append('circle')
      .attr('cx', centerX)
      .attr('cy', gateY)
      .attr('r', config.gateRadius)
      .attr('fill', 'none')
      .attr('stroke', '#e5e7eb') // Light gray border
      .attr('stroke-width', 2);

    const gateInner = gateGroup.append('circle')
      .attr('cx', centerX)
      .attr('cy', gateY)
      .attr('r', 8)
      .attr('fill', '#d1d5db'); // Neutral gray fill

    const gateText = gateGroup.append('text')
      .attr('x', centerX)
      .attr('y', gateY + config.gateRadius + 25)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af') // Muted text
      .attr('font-size', '12px')
      .attr('font-family', 'monospace')
      .text('Closed');

    // STATE BARS (Neutral State)
    const bars = [];
    for (let i = 0; i < sequence.length; i++) {
      const bar = stateGroup.append('rect')
        .attr('x', centerX - (sequence.length * 10 / 2) + i * 10)
        .attr('y', stateY)
        .attr('width', 8)
        .attr('height', 0)
        .attr('fill', '#e5e7eb') // Very light gray placeholder
        .attr('rx', 2);
      bars.push(bar);
    }

    const stateValueText = stateGroup.append('text')
      .attr('x', centerX)
      .attr('y', stateY + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#16a34a') // Green text
      .attr('font-size', '16px')
      .attr('font-weight', '700')
      .attr('font-family', 'monospace')
      .text('h = 0.00');

    // TOKENS (Light Mode)
    const tokenElements = sequence.map((token, i) => {
      const tokenG = tokenGroup.append('g')
        .attr('transform', `translate(${centerX + (i - currentStep) * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);

      tokenG.append('rect')
        .attr('x', -config.tokenSize/2)
        .attr('y', -config.tokenSize/2)
        .attr('width', config.tokenSize)
        .attr('height', config.tokenSize)
        .attr('rx', 8)
        .attr('fill', '#ffffff')      // White fill
        .attr('stroke', '#e5e7eb')    // Light border
        .attr('stroke-width', 2)
        .style('filter', 'drop-shadow(0 1px 2px rgba(0,0,0,0.05))');

      tokenG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('fill', '#374151')      // Dark text
        .attr('font-size', '11px')
        .attr('font-weight', '600')
        .attr('font-family', 'monospace')
        .text(token.id);

      return { g: tokenG, data: token };
    });

    // STEP INFO POINTER
    const stepInfo = document.getElementById('step-info');

    // ANIMATION FUNCTION
    function processToken(index) {
      if (index >= sequence.length) {
        if (stepInfo) {
          // Dark text style
          stepInfo.innerHTML = '<div style="color:#16a34a; font-weight:600; font-family:monospace;">✓ Sequence complete!</div>';
        }
        return;
      }

      const token = sequence[index];
      const isRelevant = token.type === 'relevant';
      
      // COLORS: Green for Relevant (#16a34a), Red for Noise (#ef4444)
      const color = isRelevant ? '#16a34a' : '#ef4444';
      const gateSize = isRelevant ? config.gateRadius * 0.8 : 10;

      // 1. Slide Tokens
      tokenElements.forEach((te, i) => {
        te.g.transition()
          .duration(config.animationDuration)
          .ease(d3.easeCubicInOut)
          .attr('transform', `translate(${centerX + (i - index) * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);
      });

      // 2. Highlight Active Token
      tokenElements[index].g.select('rect')
        .transition().duration(200)
        .attr('stroke', color)
        .attr('stroke-width', 3);
      
      // Make text bold/colored slightly
      tokenElements[index].g.select('text')
        .transition().duration(200)
        .attr('fill', color);

      // 3. Animate Gate
      gateInner
        .transition()
        .duration(config.animationDuration)
        .attr('r', gateSize)
        .attr('fill', color)
        // Lighter shadow for light mode
        .style('filter', `drop-shadow(0 0 ${gateSize}px ${isRelevant ? 'rgba(22,163,74,0.4)' : 'rgba(239,68,68,0.4)'})`);

      gateText
        .transition()
        .duration(config.animationDuration)
        .attr('fill', color)
        .text(isRelevant ? 'OPEN' : 'Closed');

      // 4. Update Logic
      if (isRelevant) {
        hiddenState += token.value;
      }

      // 5. Update Bars (accumulated info)
      if (index < bars.length) {
        bars[index]
          .transition()
          .duration(config.animationDuration)
          .attr('y', stateY - Math.min(hiddenState * 3, 80))
          .attr('height', Math.min(hiddenState * 3, 80))
          .attr('fill', isRelevant ? '#16a34a' : '#d1d5db'); // Green if added, Gray if skipped/history
      }

      // 6. Update H text
      stateValueText
        .transition()
        .duration(config.animationDuration)
        .tween('text', function() {
          // Interpolate numbers
          const currentVal = parseFloat(this.textContent.split('=')[1] || 0);
          const i = d3.interpolate(currentVal, hiddenState);
          return function(t) {
            this.textContent = `h = ${i(t).toFixed(2)}`;
          };
        });

      // 7. Update HTML Info (Light Mode Styles)
      if (stepInfo) {
        const formula = isRelevant 
          ? `h<sub>${index+1}</sub> = h<sub>${index}</sub> + ${token.value.toFixed(1)}`
          : `h<sub>${index+1}</sub> = h<sub>${index}</sub> (gate closed)`;
        
        // Colors compatible with white background
        stepInfo.innerHTML = `
          <div style="font-weight:600; color:${color}; margin-bottom:4px; font-family:monospace;">
            Step ${index + 1}: ${token.id}
          </div>
          <div style="font-size:11px; font-family:monospace; color:#4b5563;">
            ${formula} &nbsp;➝&nbsp; <strong>${hiddenState.toFixed(2)}</strong>
          </div>
        `;
      }

      // Next Step
      currentStep = index + 1;
      animationTimer = setTimeout(
        () => processToken(currentStep),
        config.animationDuration + config.pauseDuration
      );
    }

    // REPLAY BUTTON
    const replayBtn = document.getElementById('replay-btn');
    if (replayBtn) {
      replayBtn.onclick = function() {
        if (animationTimer) clearTimeout(animationTimer);
        currentStep = 0;
        hiddenState = 0;

        // Reset Visuals
        bars.forEach(bar => {
          bar.transition().duration(500).attr('height', 0).attr('y', stateY).attr('fill', '#e5e7eb');
        });
        
        gateInner.transition().duration(500).attr('r', 8).attr('fill', '#d1d5db');
        gateText.transition().duration(500).text('Closed').attr('fill', '#9ca3af');
        stateValueText.text('h = 0.00');
        
        tokenElements.forEach((te, i) => {
          te.g.transition().duration(500).attr('transform', `translate(${centerX + i * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);
          te.g.select('rect').attr('stroke', '#e5e7eb').attr('stroke-width', 2);
          te.g.select('text').attr('fill', '#374151');
        });

        if (stepInfo) stepInfo.textContent = 'Restarting...';
        setTimeout(() => processToken(0), 800);
      };
    }

    // SCROLL TRIGGER (Intersection Observer)
    if (!hasStarted) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            hasStarted = true;
            // Small delay to make sure user sees the start
            setTimeout(() => processToken(0), 500);
            observer.disconnect();
          }
        });
      }, { threshold: 0.4 }); // Trigger when 40% visible
      
      observer.observe(container);
    }
  }

  // Run Init
  init();
  
  // Handle Resize
  window.addEventListener('resize', () => {
    if(animationTimer) clearTimeout(animationTimer);
    init();
    if(hasStarted) {
    }
  });

})();


(function() {
  const mount = document.getElementById('gcn-local-viz');
  if (!mount) return;

  const payload = window.GCN_LOCAL_PAYLOAD;
  if (!payload || !payload.layers || !payload.layers.length) return;

  const loadD3 = () => new Promise((resolve, reject) => {
    if (window.d3) { resolve(window.d3); return; }
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js';
    s.async = true;
    s.onload = () => resolve(window.d3);
    s.onerror = () => reject(new Error('Failed to load d3'));
    document.head.appendChild(s);
  });

  const root = document.createElement('div');
  root.className = 'gmv-root';
  root.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  mount.appendChild(root);

  loadD3().then(d3 => {
    const wrapper = d3.select(root).append('div').attr('class', 'gmv-wrapper')
      .style('display', 'flex')
      .style('flex-wrap', 'wrap')
      .style('gap', '0')
      .style('align-items', 'stretch')
      .style('height', '100%')
      .style('background', '#ffffff');

    // LEFT: Force-directed Graph
    const graphPanel = wrapper.append('div').attr('class', 'gmv-panel')
      .style('flex', '1 1 420px')
      .style('min-width', '320px')
      .style('background', '#f9fafb')
      .style('border-right', '1px solid #e5e7eb')
      .style('padding', '20px')
      .style('box-sizing', 'border-box');
    graphPanel.append('div').attr('class', 'gmv-title')
      .style('color', '#1f2937')
      .style('font-weight', '700')
      .style('margin-bottom', '12px')
      .text('Token subgraph structure');

    const graphContainer = graphPanel.append('div')
      .style('position', 'relative')
      .style('width', '100%')
      .style('height', '400px')
      .style('background', '#ffffff')
      .style('border', '1px solid #e5e7eb')
      .style('border-radius', '6px')
      .style('box-sizing', 'border-box');

    const svg = graphContainer.append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', '0 0 420 420');

    // Take graph from payload
    const gcnGraph = payload.graph;
    const nodes = gcnGraph.nodes.map(n => ({ 
      id: n.id,
      originalData: n
    }));
    const links = gcnGraph.links.map(l => ({
      source: l.source,
      target: l.target
    }));

    // Force simulation 
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(210, 210))
      .force("collide", d3.forceCollide(25));

    // Edges (Default: Light Gray)
    const gLinks = svg.append('g').attr('class', 'links');
    const linkLines = gLinks.selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('class', 'graph-link')
      .attr('stroke', '#e5e7eb') // Light Gray
      .attr('stroke-width', 2);

    // Nodes (Default: White)
    const gNodes = svg.append('g').attr('class', 'nodes');
    const nodeGroup = gNodes.selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('class', 'node-group')
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
      );

    const nodeCircles = nodeGroup.append('circle')
      .attr('class', 'graph-node')
      .attr('r', d => d.originalData.is_center ? 14 : 9)
      .attr('fill', '#ffffff') // White
      // Center gets Green border, others Gray
      .attr('stroke', d => d.originalData.is_center ? '#16a34a' : '#9ca3af') 
      .attr('stroke-width', d => d.originalData.is_center ? 3 : 2)
      .attr('id', d => 'gcn-node-' + d.id);


    // Labels (Dark Gray)
    const gLabels = svg.append('g').attr('class', 'labels');
    gLabels.selectAll('text')
      .data(nodes)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -18)
      .attr('font-size', 11)
      .attr('fill', '#374151')
      .attr('font-weight', '600')
      .text(d => d.id);

    // Update positions on tick
    simulation.on("tick", () => {
      linkLines
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      nodeGroup
        .attr("transform", d => `translate(${d.x},${d.y})`);
      
      gLabels.selectAll('text')
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    // Drag handlers
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Highlight logic
    function highlightNodes(nodeIds) {
      nodeCircles
        .transition().duration(200)
        .attr('r', d => nodeIds.includes(d.id) ? 
          (d.originalData.is_center ? 17 : 12) : 
          (d.originalData.is_center ? 14 : 9))
        .attr('fill', d => {
          if (!nodeIds.includes(d.id)) return '#ffffff';
          // Active Center -> Light Green, Active Neighbor -> Light Blue
          return d.originalData.is_center ? '#dcfce7' : '#dbeafe'; 
        })
        .attr('stroke', d => {
           if (!nodeIds.includes(d.id)) return d.originalData.is_center ? '#16a34a' : '#9ca3af';
           // Active Center -> Green, Active Neighbor -> Blue
           return d.originalData.is_center ? '#16a34a' : '#2563eb';
        })
        .attr('stroke-width', d => nodeIds.includes(d.id) ? 3 : (d.originalData.is_center ? 3 : 2));
    }

    function highlightEdge(u, v) {
      linkLines
        .transition().duration(200)
        .attr('stroke', d => {
          const sid = typeof d.source === 'object' ? d.source.id : d.source;
          const tid = typeof d.target === 'object' ? d.target.id : d.target;
          const isMatch = (sid === u && tid === v) || (sid === v && tid === u);
          return isMatch ? '#0ea5e9' : '#e5e7eb'; // Sky Blue for active edge
        })
        .attr('stroke-width', d => {
          const sid = typeof d.source === 'object' ? d.source.id : d.source;
          const tid = typeof d.target === 'object' ? d.target.id : d.target;
          const isMatch = (sid === u && tid === v) || (sid === v && tid === u);
          return isMatch ? 3 : 2;
        });
    }

    function resetGraph() {
      nodeCircles
        .interrupt()
        .transition().duration(200)
        .attr('r', d => d.originalData.is_center ? 14 : 9)
        .attr('fill', '#ffffff') // Back to white
        .attr('stroke', d => d.originalData.is_center ? '#16a34a' : '#9ca3af') // Keep center Green
        .attr('stroke-width', d => d.originalData.is_center ? 3 : 2);

      linkLines
        .interrupt()
        .transition().duration(200)
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 2);
    }

    function highlightMultipleEdges(edgePairs) {
      linkLines
        .transition().duration(200)
        .attr('stroke', d => {
          const sid = typeof d.source === 'object' ? d.source.id : d.source;
          const tid = typeof d.target === 'object' ? d.target.id : d.target;
          const isMatch = edgePairs.some(([u, v]) => (sid === u && tid === v) || (sid === v && tid === u));
          return isMatch ? '#0ea5e9' : '#e5e7eb'; // Sky Blue
        })
        .attr('stroke-width', d => {
          const sid = typeof d.source === 'object' ? d.source.id : d.source;
          const tid = typeof d.target === 'object' ? d.target.id : d.target;
          const isMatch = edgePairs.some(([u, v]) => (sid === u && tid === v) || (sid === v && tid === u));
          return isMatch ? 3 : 2;
        });
    }


    // Legend
    const legend = graphPanel.append('div').attr('class', 'gmv-legend')
      .style('margin-top', '12px')
      .style('font-size', '12px')
      .style('color', '#6b7280')
      .style('display', 'flex')
      .style('gap', '12px');
    
    // Center Legend Item
    const l1 = legend.append('div').style('display','flex').style('align-items','center');
    l1.append('span').style('width','8px').style('height','8px')
      .style('background','#fff').style('border','2px solid #16a34a').style('border-radius','50%').style('margin-right','4px');
    l1.append('span').style('font-size', '16px').text('Center (v)');

    // RIGHT: Steps
    const layer = payload.layers[0];
    const steps = layer.steps || [];

    const detailPanel = wrapper.append('div').attr('class', 'gmv-panel')
      .style('flex', '1 1 520px')
      .style('min-width', '320px')
      .style('background', '#ffffff')
      .style('padding', '20px')
      .style('box-sizing', 'border-box');

    detailPanel.append('div').attr('class', 'gmv-title')
      .style('color', '#1f2937')
      .style('font-weight', '700')
      .style('margin-bottom', '12px')
      .text('Calculation Steps');

    const stepButtonsWrap = detailPanel.append('div').attr('class', 'gmv-step-buttons')
      .style('display', 'flex')
      .style('flex-wrap', 'wrap')
      .style('gap', '6px')
      .style('margin-bottom', '16px');

    const stepCard = detailPanel.append('div').attr('class', 'gmv-step-card')
      .style('background', '#f9fafb')
      .style('border', '1px solid #e5e7eb')
      .style('border-radius', '6px')
      .style('padding', '16px');

    let activeStepId = steps.length ? steps[0].id : null;
    let hoverDetailDiv = null;

    // MATRIX DRAWING (GREEN THEME)
    function drawMatrix(containerSel, step) {
      const matrix = step.matrix;
      containerSel.selectAll('*').remove();
      if (!matrix || !matrix.values || !matrix.values.length) {
        containerSel.append('div').style('color', '#9ca3af').text('no data');
        return;
      }

      const numRows = matrix.values.length;
      const numCols = Array.isArray(matrix.values[0]) ? matrix.values[0].length : 0;
      const rowLabels = matrix.rows || Array.from({length: numRows}, (_, i) => String(i));
      const colLabels = matrix.cols || Array.from({length: numCols}, (_, j) => String(j));

      const size = 50;
      const svgWidth = numCols * size;
      const svgHeight = numRows * size;
      const svgM = containerSel.append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth).attr('height', svgHeight);

      const g = svgM.append('g');
      const flat = matrix.values.flat();
      const minVal = flat.length ? d3.min(flat) : 0;
      const maxVal = flat.length ? d3.max(flat) : 1;

      // GREEN SCALE: White (#ffffff) -> Green (#22c55e)
      const colorScale = d3.scaleLinear()
        .domain([minVal, maxVal])
        .range(['#ffffff', '#22c55e']) 
        .clamp(true);

      matrix.values.forEach((row, i) => {
        row.forEach((val, j) => {
          const cell = g.append('g').attr('transform', `translate(${j * size},${i * size})`);
          const rect = cell.append('rect')
            .attr('width', size - 1)
            .attr('height', size - 1)
            .attr('fill', colorScale(val))
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 1)
            .style('cursor', 'pointer');

          cell.append('text')
            .attr('x', size/2).attr('y', size/2)
            .attr('dy', '.35em').attr('text-anchor', 'middle')
            .style('font-size', '10px').style('fill', '#374151')
            .style('pointer-events', 'none')
            .text(val.toFixed(2));

          rect.on('mouseenter', () => {
            const rowId = rowLabels[i];
            const colId = colLabels[j];
            handleCellHover(step, i, j, rowId, colId, val);
            rect.attr('stroke', '#16a34a').attr('stroke-width', 2); // Green highlight
          });
          rect.on('mouseleave', () => {
            rect.attr('stroke', '#e5e7eb').attr('stroke-width', 1);
            resetGraph();
            if (hoverDetailDiv) hoverDetailDiv.text('Hover over a cell to see computation details.');
          });
        });
      });
    }

	  // VECTOR DRAWING (GREEN THEME)
    function drawVector(containerSel, step) {
      const vector = step.vector;
      containerSel.selectAll('*').remove();
      if (!vector || !vector.values) return;

      const numRows = 1;
      const numCols = vector.values.length;
      const colLabels = vector.labels || Array.from({length: numCols}, (_, j) => String(j));

      const size = 50; 
      const svgWidth = numCols * size;
      const svgHeight = numRows * size;
      const svgV = containerSel.append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth).attr('height', svgHeight);

      const g = svgV.append('g');

      const flat = vector.values;
      const minVal = flat.length ? d3.min(flat) : 0;
      const maxVal = flat.length ? d3.max(flat) : 1;
      const colorScale = d3.scaleLinear()
        .domain([minVal, maxVal])
        .range(['#ffffff', '#22c55e']) 
        .clamp(true);

      vector.values.forEach((val, j) => {
          const i = 0;
          const cell = g.append('g').attr('transform', `translate(${j * size},${i * size})`);

          const rect = cell.append('rect')
            .attr('width', size - 1)
            .attr('height', size - 1)
            .attr('fill', colorScale(val))
            .attr('stroke', '#e5e7eb')
            .attr('stroke-width', 1)
            .style('cursor', 'pointer');

          cell.append('text')
            .attr('x', size/2).attr('y', size/2)
            .attr('dy', '.35em').attr('text-anchor', 'middle')
            .style('font-size', '10px').style('fill', '#374151')
            .style('pointer-events', 'none')
            .text(val.toFixed(2));

          // Hover handlers
          rect.on('mouseenter', () => {
            const label = colLabels[j];
            handleVectorHover(step, j, label, val);
            rect.attr('stroke', '#16a34a').attr('stroke-width', 2);
          });
          
          rect.on('mouseleave', () => {
            rect.attr('stroke', '#e5e7eb').attr('stroke-width', 1);
            resetGraph();
            if (hoverDetailDiv) hoverDetailDiv.text('Hover over a cell to see computation details.');
          });
      });
    }


    function handleCellHover(step, rowIndex, colIndex, rowId, colId, value) {
      resetGraph();
      const layerNodes = layer.tokenNodes || [];
      const nodeIndex = rowIndex < layerNodes.length ? layerNodes[rowIndex] : null;
      let text = '';

      const valStr = value.toFixed(3);

      if (step.id === 'adjacency') {
        const u = parseInt(rowId, 10);
        const v = parseInt(colId, 10);
        highlightNodes([u, v]);

        if (value !== 0) {
          highlightEdge(u, v);
          text = `𝐀_ℓ[${u},${v}] = 1 → Node ${u} and ${v} are connected in the structural token 𝓣_ℓ(v).`;
        } else {
          text = `𝐀_ℓ[${u},${v}] = 0 → No direct edge in the token 𝓣_ℓ(v).`;
        }
      } else if (step.id === 'input_features') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `𝐇⁽⁰⁾[${nodeIndex}, ${colId}] = ${valStr} → Initial feature of node ${nodeIndex} in the token.`;
        }
      } else if (step.id === 'x1') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `(𝐇⁽⁰⁾𝐖⁽⁰⁾)[${nodeIndex}, ${colId}] = ${valStr} → Node ${nodeIndex} after linear transformation, before aggregation.`;
        }
      } else if (step.id === 'ax1') {
        if (nodeIndex !== null) {
          const adjacencyStep = layer.steps.find(s => s.id === 'adjacency');
          const x1Step = layer.steps.find(s => s.id === 'x1');
          
          let neighbors = [nodeIndex];
          let sumTerms = [];
          let sumValues = [];
          let edgePairs = [];

          if (adjacencyStep && x1Step && adjacencyStep.matrix && x1Step.matrix) {
            const numNodes = adjacencyStep.matrix.values.length;
            const colIndex = x1Step.matrix.cols.indexOf(colId);
            
            for (let j = 0; j < numNodes; j++) {
              const aValue = adjacencyStep.matrix.values[nodeIndex][j];
              if (aValue !== 0) {
                if (j !== nodeIndex) {
                  neighbors.push(j);
                  edgePairs.push([nodeIndex, j]);
                }
                const xValue = x1Step.matrix.values[j][colIndex];
                sumTerms.push(`${aValue}·${xValue.toFixed(2)}`);
                sumValues.push(aValue * xValue);
              }
            }
          }
          
          neighbors = [...new Set(neighbors)];
          highlightNodes(neighbors);
          highlightMultipleEdges(edgePairs);
          
          const sum = sumValues.reduce((a, b) => a + b, 0);
          text = `[Ã𝐇...][${nodeIndex},${colId}] = ${sumTerms.join(' + ')} = ${valStr} → Aggregating signals from neighbors (simulating 𝐃⁻½𝐀𝐃⁻½ step).`;
        }
      } else if (step.id === 'h1') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `𝐇⁽¹⁾[${nodeIndex}, ${colId}] = ${valStr} → Output of Layer I after ReLU activation σ(·).`;
        }
      } else if (step.id === 'x2') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `(𝐇⁽¹⁾𝐖⁽¹⁾)[${nodeIndex}, ${colId}] = ${valStr} → Node ${nodeIndex} transformed by second weight matrix.`;
        }
      } else if (step.id === 'ax2') {
        if (nodeIndex !== null) {
          const adjacencyStep = layer.steps.find(s => s.id === 'adjacency');
          const x2Step = layer.steps.find(s => s.id === 'x2');
          
          let neighbors = [nodeIndex];
          let sumTerms = [];
          let sumValues = [];
          let edgePairs = [];

          if (adjacencyStep && x2Step && adjacencyStep.matrix && x2Step.matrix) {
            const numNodes = adjacencyStep.matrix.values.length;
            const colIndex = x2Step.matrix.cols.indexOf(colId);
            
            for (let j = 0; j < numNodes; j++) {
              const aValue = adjacencyStep.matrix.values[nodeIndex][j];
              if (aValue !== 0) {
                if (j !== nodeIndex){
                  neighbors.push(j);
                  edgePairs.push([nodeIndex, j]);
                } 
                const xValue = x2Step.matrix.values[j][colIndex];
                sumTerms.push(`${aValue}·${xValue.toFixed(2)}`);
                sumValues.push(aValue * xValue);
              }
            }
          }
          
          neighbors = [...new Set(neighbors)];
          highlightNodes(neighbors);
          highlightMultipleEdges(edgePairs);

          const sum = sumValues.reduce((a, b) => a + b, 0);
          text = `[Ã𝐇...][${nodeIndex},${colId}] = ${sumTerms.join(' + ')} = ${valStr} → Second round of neighborhood aggregation.`;

        }
      } else if (step.id === 'h2') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `𝐇⁽²⁾[${nodeIndex}, ${colId}] = ${valStr} → Final node representations in the snapshot.`;
        }
      }

      if (hoverDetailDiv) {
        hoverDetailDiv.text(text);
      }
    }

    function handleVectorHover(step, index, label, value) {
      resetGraph();
      let text = `𝐱_ℓ(v)[${label}] = ${value.toFixed(3)} → `;
      
      if (step.id === 'z') {
        const h2Step = layer.steps.find(s => s.id === 'h2');
        
        if (h2Step && h2Step.matrix && h2Step.matrix.values) {
          const numNodes = h2Step.matrix.values.length;
          const colIndex = h2Step.matrix.cols.indexOf(label);
          
          let allNodes = [];
          let sumTerms = [];
          let sumValues = [];
          
          for (let i = 0; i < numNodes; i++) {
            allNodes.push(i);
            const hValue = h2Step.matrix.values[i][colIndex];
            sumTerms.push(hValue.toFixed(2));
            sumValues.push(hValue);
          }
          
          highlightNodes(allNodes);
          
          const sum = sumValues.reduce((a, b) => a + b, 0);
          const mean = sum / numNodes;
          
          text = `𝐱_ℓ(v)[${label}] = (${sumTerms.join(' + ')}) / ${numNodes} = ${mean.toFixed(3)}`;
        } else {
          text += `Mean pooling of all nodes in snapshot 𝓣_ℓ(v). This vector becomes one row in 𝐗(v).`;
        }
      }
      
      if (hoverDetailDiv) {
        hoverDetailDiv.text(text);
      }
    }

    function renderStep() {
      const step = steps.find(s => s.id === activeStepId) || steps[0];
      stepCard.selectAll('*').remove();

      // Title (Dark Text)
      stepCard.append('div').attr('class', 'gmv-step-title')
        .style('color', '#1f2937') 
        .style('font-weight', '700')
        .style('margin-bottom', '6px')
        .text(step.title || activeStepId);

      // Formula Box (White with Green Text)
      if (step.formula) {
        const formulaBox = stepCard.append('div').attr('class', 'gmv-formula')
          .style('margin-bottom', '8px');
        if (step.formula.lhs && step.formula.rhs) {
          formulaBox.append('div')
            .style('color', '#16a34a') // Green formula
            .style('font-family', 'monospace')
            .style('font-size', '12px')
            .text(`${step.formula.lhs} = ${step.formula.rhs}`);
        }
      }

      // Explain Text (Dark Gray)
      stepCard.append('div').attr('class', 'gmv-explain')
        .style('color', '#4b5563')
        .style('margin-top', '8px')
        .text(step.explain || '');

      const gridHolder = stepCard.append('div').attr('class', 'gmv-grid');

      if (step.type === 'matrix') {
        drawMatrix(gridHolder, step);
      } else if (step.type === 'vector') {
        drawVector(gridHolder, step);
      } else {
        gridHolder.append('div').style('color', '#9ca3af').text('No visualization.');
      }

      // Hover Detail (Dark Gray)
      hoverDetailDiv = stepCard.append('div')
        .attr('class', 'gmv-hover-detail')
        .style('border-top', '1px solid #e5e7eb')
        .style('margin-top', '8px')
        .style('padding-top', '6px')
        .style('font-size', '11px')
        .style('color', '#4b5563')
        .style('min-height', '40px')
        .text('Hover over a cell to see calculation details.');

      resetGraph();
    }

    const stepMeta = steps.map(s => ({
      id: s.id,
      label: s.title ? s.title.split(' ')[0] : s.id.toUpperCase()
    }));

    stepButtonsWrap.selectAll('button')
      .data(stepMeta)
      .enter().append('button')
      // Buttons: Active=Green, Inactive=White
      .style('background', d => d.id === activeStepId ? '#16a34a' : '#ffffff')
      .style('border', d => d.id === activeStepId ? '1px solid #16a34a' : '1px solid #d1d5db')
      .style('color', d => d.id === activeStepId ? '#ffffff' : '#374151')
      .style('cursor', 'pointer')
      .style('padding', '6px 10px')
      .style('font-size', '12px')
      .style('font-family', 'monospace')
      .style('box-sizing', 'border-box')
      .style('border-radius', '4px')
      .text(d => d.label)
      .on('click', (event, d) => {
        activeStepId = d.id;
        stepButtonsWrap.selectAll('button')
          .style('background', x => x.id === activeStepId ? '#16a34a' : '#ffffff')
          .style('color', x => x.id === activeStepId ? '#ffffff' : '#374151')
          .style('border', x => x.id === activeStepId ? '1px solid #16a34a' : '1px solid #d1d5db');
        renderStep();
      });

    renderStep();
  }).catch(err => {
    console.error(err);
  });
})();
</script>
