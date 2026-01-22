---
layout: distill
title: "Graph Mamba: Rethinking Graph Learning"
description: "Rethinking Graph Learning with Selective State Space Models"
date: 2026-01-22

authors:
  - name: Anonymous

bibliography: 2026-01-22-graph-mamba.bib
related_posts: false
toc: false
---

<script src="//unpkg.com/3d-force-graph"></script>
<script src="//unpkg.com/d3"></script>
<script>
  document.addEventListener('DOMContentLoaded', () => {
    const progress = document.getElementById('progress');
    if (progress) progress.remove();
  });
</script>
<section class="intro-section">
<p>
<strong>Graphs</strong> are everywhere: social networks, molecular structures, knowledge bases, and recommendation systems. 
            Traditional Graph Neural Networks (GNNs) have been the go-to approach for learning from graph-structured data, 
            but they face fundamental challenges with <strong>long-range dependencies</strong> and <strong>computational efficiency</strong> on large graphs.
        </p>
<p>
            This interactive article introduces <strong>Graph Mamba Networks</strong> - a novel approach that replaces traditional 
            message passing with <strong>Selective State Space Models (SSMs)</strong>. Instead of iteratively aggregating information 
            from neighbors, we linearize graph neighborhoods into sequences and process them efficiently using Mamba's selective gating mechanism.
        </p>
</section>
<section style="width: 100%; margin: 0 auto;">
<div class="interactive-figure if--600" id="cora-viz" style="position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; background: #1a1d21; border: 1px solid #3d4147; height: 560px; min-height: 560px; color: #e8e6e3;">
<div class="viz-controls" style="position: absolute; top: 15px; left: 15px; background: rgba(26,29,33,0.9); color: #e8e6e3; padding: 8px 10px; font-size: 12px; border: 1px solid #3d4147;">
<strong style="color: #e8e6e3;">Cora Citation Network</strong><br/>
<span style="color: #e8e6e3;">Drag to rotate • Scroll to zoom</span><br/>
<button onclick="resetCamera()" style="margin-top: 6px; background: #2563eb; border: none; color: #ffffff; padding: 4px 8px; cursor: pointer; font-size: 12px; box-sizing: border-box;">Reset View</button>
<div style="margin-top: 6px; color: #e8e6e3; font-size: 12px;">Ground truth topics</div>
</div>
</div>
<div class="figure-caption">
<strong>Figure 1:</strong> Ground-truth Cora citations (papers as nodes, edges as citations). Colors = true topics from the dataset.
        </div>
</section>
<nav class="table-of-contents">
<h3>Contents</h3>
<ol>
<li><a href="#foundations">Foundations: What is a Graph?</a></li>
<li><a href="#step1">Step 1: From Walks to Tokens</a></li>
<li><a href="#step2">Step 2: Encoding Subgraphs with Local Encoders</a></li>
<li><a href="#step3">Step 3: The Mamba Block &amp; Selective State Spaces</a></li>
<li><a href="#step4">Step 4: Bidirectional Processing</a></li>
<li><a href="#results">Experimental Results on Cora</a></li>
</ol>
</nav>
<section class="article-width" id="foundations">
<h2>Foundations: Understanding Graphs</h2>
<p class="section-intro">
            Before diving into Graph Mamba, let's establish the fundamentals of graph-structured data 
            and why traditional approaches struggle with it.
        </p>
<h3>What is a Graph?</h3>
<p>
            A <strong>graph</strong> is a data structure with <strong>nodes</strong> (entities) and 
            <strong>edges</strong> (relationships). Think of it like a social network: people are nodes, 
            friendships are edges.
        </p>
<p>
            Graph Neural Networks (GNNs) are the standard for learning on graph-structured data. 
            However, traditional Message Passing Neural Networks (MPNNs) often struggle with 
            <strong>long-range dependencies</strong> and computational efficiency on large graphs.
        </p>
<div class="key-takeaway">
<strong>Key Challenge:</strong> Traditional GNNs require many layers to capture long-range dependencies, 
            leading to over-smoothing and high computational costs.
        </div>

<details>
<summary><strong>Python code:</strong> Loading the Cora dataset (from <code>tutorial/graph_mamba.ipynb</code>)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="kn">import</span> <span class="nn">torch</span>
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

<span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">device</span><span class="p">(</span><span class="s">"mps"</span><span class="p">)</span>

<span class="n">ds</span> <span class="o">=</span> <span class="n">Planetoid</span><span class="p">(</span><span class="n">root</span><span class="o">=</span><span class="s">"./data"</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s">"Cora"</span><span class="p">)</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">ds</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
<span class="n">data</span></code></pre></figure>

</details>
</section>
<section class="article-width" id="step1">
<h2>From Random Walks to Tokens</h2>
<p class="section-intro">
            The first innovation in Graph Mamba is how we sample and represent graph neighborhoods. 
            Instead of fixed-radius neighbors, we use random walks to create multi-scale "snapshots" of the graph.
        </p>
<p>
            For each center node <em>v</em> we want a list of small neighbourhood snapshots, called <strong>tokens</strong>. 
            We fix a maximum walk length <strong>m</strong>. For every length ℓ = 0, 1, …, m we launch <strong>M</strong> random walks 
            starting from v and collect all nodes that are visited at least once. The union of these visited nodes forms a subgraph 
            G[T<sub>ℓ</sub>(v)] – this subgraph is one token for length ℓ.
        </p>
<div class="info-box">
<p><strong>Intuition:</strong> Imagine you're exploring a city</p>
<ul>
<li><strong>ℓ = 0:</strong> You stay at the starting point (just the center node)</li>
<li><strong>ℓ = 1:</strong> You walk one block in random directions (immediate neighbors)</li>
<li><strong>ℓ = 2:</strong> You walk two blocks (2-hop neighbors)</li>
<li><strong>ℓ = m:</strong> You explore the entire neighborhood</li>
</ul>
<p>
                Each "snapshot" at distance ℓ becomes a <strong>token</strong>. By processing tokens from different distances, 
                the model learns which neighborhood scales matter most.
            </p>
</div>
<p>
            A local encoder ϕ(·) later turns each subgraph token into a d-dimensional vector. If we have one token per length ℓ, 
            the resulting "token matrix" for node v has shape (m+1) × d. In the full GMN recipe this sampling can be repeated 
            <strong>s</strong> times, giving K = (m+1)·s token vectors per node, which are then flattened into a K × d sequence 
            for the Mamba block.
        </p>

<details>
<summary><strong>Python code:</strong> Neighborhood sampling (random walk tokenization)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="n">edge_index</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">.</span><span class="n">cpu</span><span class="p">()</span>
<span class="n">n</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">num_nodes</span>

<span class="n">max_walk_length</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">num_of_walks</span> <span class="o">=</span> <span class="mi">4</span>

<span class="n">tokens</span> <span class="o">=</span> <span class="p">{</span><span class="n">v</span><span class="p">:</span> <span class="p">{}</span> <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">)}</span>

<span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">walk_length</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
        <span class="n">induced_graph</span> <span class="o">=</span> <span class="nb">set</span><span class="p">()</span>

        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">num_of_walks</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
            <span class="n">walk</span> <span class="o">=</span> <span class="n">random_walk</span><span class="p">(</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">edge_index</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">([</span><span class="n">node</span><span class="p">]),</span> <span class="n">walk_length</span><span class="o">=</span><span class="n">walk_length</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
            <span class="n">induced_graph</span> <span class="o">|=</span> <span class="nb">set</span><span class="p">(</span><span class="n">walk</span><span class="p">.</span><span class="n">tolist</span><span class="p">())</span>

        <span class="n">tokens</span><span class="p">[</span><span class="n">node</span><span class="p">][</span><span class="n">walk_length</span><span class="p">]</span> <span class="o">=</span> <span class="n">induced_graph</span></code></pre></figure>

</details>
<div class="interactive-figure" style="position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; background: #1a1d21; border: 1px solid #3d4147; min-height: 500px; color: #e8e6e3;">
<div id="walk-viz" style="display: flex; flex-wrap: wrap; gap: 10px; padding: 10px; box-sizing: border-box; width: 100%; min-height: 550px; align-items: stretch;">
<div id="walk-controls" style="flex: 0 0 220px; background: rgba(15,23,42,0.9); padding: 10px; font-family: monospace; font-size: 12px; border: 1px solid #334155; box-sizing: border-box;">
<div style="margin-top: 2px; margin-bottom: 4px; color: #e5e7eb;"><strong style="color: #e5e7eb;">1. Number of walks M</strong></div>
<input id="walk-count" max="8" min="1" step="1" style="width: 100%; box-sizing: border-box;" type="range" value="4"/>
<div style="margin-top: 4px; color: #e5e7eb;">
                M = <span id="walk-count-value" style="color: #e5e7eb;">4</span> walks
              </div>
<div style="margin-top: 8px; color: #e5e7eb;"><strong style="color: #e5e7eb;">2. Pick a center node</strong> (click any circle)</div>
<div style="margin-top: 10px; margin-bottom: 4px; color: #e5e7eb;"><strong style="color: #e5e7eb;">3. Walk length ℓ</strong></div>
<input id="walk-length" max="3" min="0" step="1" style="width: 100%; box-sizing: border-box;" type="range" value="1"/>
<div style="margin-top: 4px; color: #e5e7eb;">
                ℓ = <span id="walk-length-value" style="color: #e5e7eb;">1</span> steps
              </div>
<button id="sample-token-btn" style="margin-top: 10px; width: 100%; background: #2563eb; border: none; color: #ffffff; padding: 6px 8px; cursor: pointer; box-sizing: border-box;">
                Generate token
              </button>
</div>
<div id="walk-canvas" style="flex: 1 1 520px; min-width: 320px; height: 520px; position: relative; background: #0b1120; border: 1px solid #334155; box-sizing: border-box;"></div>
<div id="token-panel" style="flex: 0 0 220px; background: rgba(15,23,42,0.9); padding: 10px; font-family: monospace; font-size: 12px; border: 1px solid #334155; box-sizing: border-box;">
<div style="margin-bottom: 6px; color: #e5e7eb;"><strong style="color: #e5e7eb;">Token matrix for this node</strong></div>
<div style="font-size: 10px; color: #9ca3af; margin-bottom: 8px;">
                In this demo we use the simple case s = 1:
                each row corresponds to all M walks of one length ℓ, encoded as a d‑dimensional vector.
                In the full GMN model the sampling is repeated s times and the (ℓ, j) pairs are flattened
                into a K × d sequence for Mamba.
              </div>
<svg height="120" id="token-matrix" width="160"></svg>
</div>
<div id="walk-sequence" style="flex: 1 1 100%; background: rgba(15,23,42,0.9); color: #a5b4fc; padding: 8px 10px; font-family: monospace; font-size: 12px; border: 1px solid #334155; box-sizing: border-box;">
              Choose M, click a node and choose ℓ, then press “Generate token”.
            </div>
</div>
</div>
<div class="figure-caption">
<strong>Figure:</strong> Each blue "cloud" is still a graph: it has several nodes, edges, and node features. 
            To feed it into the sequence model, we compress this whole subgraph into a single vector.
        </div>
</section>
<section class="article-width" id="step2">
<h2>Encoding Subgraphs with Local Encoders</h2>
<p class="section-intro">
            Now that we have tokens (subgraphs), we need to convert each one into a fixed-size vector. 
            This is where the local encoder ϕ(·) comes in.
        </p>
<p>
            Each blue "cloud" is still a graph: it has several nodes, edges, and node features. 
            To feed it into the sequence model, we compress this whole subgraph into a single vector — 
            a <strong>token embedding</strong> of size d.
        </p>
<p>
            Graph Mamba Networks use a small <em>local encoder</em> ϕ(·) for this step. 
            In practice ϕ(·) can be implemented in several ways:
        </p>
<ul>
<li><strong>GCN (Graph Convolutional Network):</strong> 2-3 layers of graph convolutions</li>
<li><strong>GraphSAGE:</strong> Sample and aggregate neighborhood features</li>
<li><strong>GAT (Graph Attention):</strong> Learn attention weights between nodes</li>
<li><strong>Simple pooling:</strong> Just average all node features (baseline)</li>
</ul>
<p>
            Regardless of the chosen encoder, the output is always a d-dimensional vector 
            ϕ(G[T<sub>ℓ</sub>(v)], X<sub>T<sub>ℓ</sub>(v)</sub>), which becomes one row in the K × d token matrix 
            for node v. Later, the Bidirectional Mamba block will read this matrix as a sequence of tokens and learn 
            which subgraphs are most informative.
        </p>

<details>
<summary><strong>Python code:</strong> Local encoder (GCN) for a token subgraph</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="k">class</span> <span class="nc">LocalEncoder</span><span class="p">(</span><span class="n">torch</span><span class="p">.</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">in_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_1</span> <span class="o">=</span> <span class="n">GCNConv</span><span class="p">(</span><span class="n">in_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">gcn_layer_2</span> <span class="o">=</span> <span class="n">GCNConv</span><span class="p">(</span><span class="n">hidden_dim</span><span class="p">,</span> <span class="n">hidden_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">encode_token</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token</span><span class="p">,</span> <span class="n">data</span><span class="p">):</span>
        <span class="n">token</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">(</span><span class="nb">list</span><span class="p">(</span><span class="n">token</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="p">.</span><span class="nb">long</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
        <span class="n">neighborhood_features</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">x</span><span class="p">[</span><span class="n">token</span><span class="p">]</span>

        <span class="n">mask</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">isin</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">token</span><span class="p">)</span> <span class="o">&amp;</span> <span class="n">torch</span><span class="p">.</span><span class="n">isin</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">token</span><span class="p">)</span>
        <span class="n">edges_in_token</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">edge_index</span><span class="p">[:,</span> <span class="n">mask</span><span class="p">]</span>

        <span class="n">idx_map</span> <span class="o">=</span> <span class="p">{</span><span class="n">old</span><span class="p">:</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">old</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">token</span><span class="p">.</span><span class="n">tolist</span><span class="p">())}</span>
        <span class="n">sub_edge_index</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tensor</span><span class="p">(</span>
            <span class="p">[[</span><span class="n">idx_map</span><span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">u</span><span class="p">)]</span> <span class="k">for</span> <span class="n">u</span> <span class="ow">in</span> <span class="n">edges_in_token</span><span class="p">[</span><span class="mi">0</span><span class="p">]],</span>
            <span class="p">[</span><span class="n">idx_map</span><span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">v</span><span class="p">)]</span> <span class="k">for</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">edges_in_token</span><span class="p">[</span><span class="mi">1</span><span class="p">]]],</span>
            <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="p">.</span><span class="nb">long</span><span class="p">,</span>
            <span class="n">device</span><span class="o">=</span><span class="n">device</span>
        <span class="p">)</span>

        <span class="n">induced_graph</span> <span class="o">=</span> <span class="n">Data</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">neighborhood_features</span><span class="p">,</span> <span class="n">edge_index</span><span class="o">=</span><span class="n">sub_edge_index</span><span class="p">)</span>
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

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="n">ordered_token_embeddings</span> <span class="o">=</span> <span class="p">{}</span>

<span class="k">for</span> <span class="n">node</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="n">embeddings</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">walk_length</span> <span class="ow">in</span> <span class="nb">reversed</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_walk_length</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)):</span>
        <span class="n">emb</span> <span class="o">=</span> <span class="n">local_encoder</span><span class="p">.</span><span class="n">encode_token</span><span class="p">(</span><span class="n">tokens</span><span class="p">[</span><span class="n">node</span><span class="p">][</span><span class="n">walk_length</span><span class="p">],</span> <span class="n">data</span><span class="p">)</span>
        <span class="n">embeddings</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">emb</span><span class="p">)</span>
    <span class="n">ordered_token_embeddings</span><span class="p">[</span><span class="n">node</span><span class="p">]</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">embeddings</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="nb">list</span><span class="p">(</span><span class="n">ordered_token_embeddings</span><span class="p">.</span><span class="n">items</span><span class="p">())[:</span><span class="mi">5</span><span class="p">]</span></code></pre></figure>

</details>
<h3>GCN Encoder: Three Phases</h3>
<p>
            So far we treated the local encoder ϕ(·) as a black box that turns a subgraph token G[T<sub>ℓ</sub>(v)] into a 
            d-dimensional vector. Here we open this box for a simple 2-layer GCN encoder and look at the actual tensors that 
            flow through the network for one token.
        </p>
<div class="info-box">
<p><strong>GCN Processing Steps:</strong></p>
<ol>
<li><strong>Define the graph structure (A) and node features (X)</strong></li>
<li><strong>Apply linear transform to the features, aggregate neighbors, apply ReLU</strong></li>
<li><strong>Second layer of aggregation, then pool all nodes into one vector</strong></li>
</ol>
</div>
<div class="interactive-figure" style="min-height: 600px; background: #020617; border: 1px solid #1f2937; position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; color: #e8e6e3;">
<div id="gcn-local-viz" style="position: relative; width: 100%; height: 750px;"></div>
</div>
<div class="figure-caption">
<strong>Visualization:</strong> This shows how a 2-layer GCN local encoder ϕ(·) processes a single token subgraph. 
            On the left you see the token subgraph around a node v. On the right you can inspect, step by step, 
            the tensors A, X, X1, AX1, H1, X2, AX2, H2, and the pooled vector z that becomes the token embedding.
        </div>
</section>
<section class="article-width" id="step3">
<h2>The Mamba Block: Selective State Spaces</h2>
<p class="section-intro">
            After encoding subgraphs into tokens, we need a way to process these token sequences efficiently. 
            This is where Mamba shines — offering linear complexity instead of quadratic attention.
        </p>
<h3>From Tokens to Sequences</h3>
<p>
            After the local encoder processes each subgraph, every node <em>v</em> has <strong>K token vectors</strong> of dimension <em>d</em>. 
            These K vectors are stacked into a matrix of shape <strong>K × d</strong>, where:
        </p>
<ul>
<li>K = (m+1) × s = (walk lengths × number of samples)</li>
<li>d = embedding dimension</li>
</ul>
<p>
            This K × d matrix is the <em>sequence</em> that Mamba processes. Each row is one token, 
            representing a subgraph at a certain walk length.
        </p>
<h3>Token Ordering</h3>
<p>
            Unlike Transformers, which are permutation-equivariant (they don't care about token order), 
            Mamba is a <strong>sequential encoder</strong> — it processes tokens one by one, 
            and earlier tokens influence later ones through the hidden state. 
            This means the <em>order of tokens matters</em>.
        </p>
<div class="info-box">
<h4>How are tokens ordered?</h4>
<p>
<strong>When m ≥ 1 (subgraph tokenization):</strong> tokens have an <em>implicit hierarchical order</em>. 
                The i-th token (i-hop neighborhood) is a subgraph of all j-hop neighborhoods where j ≥ i. 
                GMN uses <strong>reverse order</strong>: from outer neighborhoods (m-hop) to inner ones (1-hop, then 0-hop = node itself).
            </p>
<p>
                This way, when the model reaches the node's own embedding (last token), it already has context 
                about the entire neighborhood structure — from global to local.
            </p>
<p>
<strong>When s ≥ 2:</strong> tokens with the same walk length ℓ are <em>randomly shuffled</em> among themselves 
                to make the model robust to their permutation.
            </p>
<p>
<strong>When m = 0 (node tokenization):</strong> there's no implicit order, so nodes are sorted by 
                structural properties like <em>Personalized PageRank</em> or <em>degree</em>. 
                Domain knowledge can also be used when available.
            </p>
</div>
<p>
    The final token sequence for a node v looks like this (processed in this exact order):
</p>
<div class="formula-box">
<strong>Step 1:</strong> x<sup>s</sup><sub>v,m</sub> — s tokens from <em>m-hop</em> neighborhoods (farthest)<br/>
<strong>Step 2:</strong> x<sup>s</sup><sub>v,m-1</sub> — s tokens from <em>(m-1)-hop</em> neighborhoods<br/>
<span>...</span><br/>
<strong>Step m:</strong> x<sup>s</sup><sub>v,1</sub> — s tokens from <em>1-hop</em> neighborhoods (direct neighbors)<br/>
<strong>Final:</strong> x<sup>s</sup><sub>v,0</sub> — s tokens from <em>0-hop</em> (node v itself)
      <ul>
<li><strong>x<sup>s</sup><sub>v,ℓ</sub></strong> = <em>s</em> token vectors for node <em>v</em> at walk length <em>ℓ</em></li>
<li>Within each group of <em>s</em> tokens at the same <em>ℓ</em>, order is <strong>randomly shuffled</strong></li>
</ul>
</div>
<h3>Why Mamba?</h3>
<p>
            Traditional Transformers use <strong>self-attention</strong> with \(O(N^2)\) complexity, 
            where every token attends to every other token. For long sequences (like graph random walks), 
            this becomes computationally expensive and memory-intensive.
        </p>
<p>
<strong>Mamba</strong> offers a different approach: a <strong>Selective State Space Model (SSM)</strong> 
            with \(O(N)\) complexity. Instead of attention, Mamba maintains a <em>hidden state</em> that evolves 
            recurrently as it processes each token.
        </p>
<div class="comparison-section">
<table>
<thead>
<tr>
<th>Property</th>
<th>Transformer</th>
<th>Mamba</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Complexity</strong></td>
<td>\(O(N^2)\)</td>
<td>\(O(N)\)</td>
</tr>
<tr>
<td><strong>Memory Usage</strong></td>
<td>High (stores attention matrix)</td>
<td>Low (only hidden state)</td>
</tr>
<tr>
<td><strong>Long Sequences</strong></td>
<td>Struggles</td>
<td>Efficient</td>
</tr>
<tr>
<td><strong>Selective Filtering</strong></td>
<td>No</td>
<td>Yes (Delta)</td>
</tr>
</tbody>
</table>
</div>
<h3>The Core Mechanism</h3>
<p>
            As you read through tokens (subgraphs at different distances), Mamba maintains a 
            <strong>running summary</strong> called the "hidden state":
        </p>
<div class="formula-box">
<p>The Mamba block updates its hidden state using this equation:</p>
<p>
                \[ h_t = \bar{A} h_{t-1} + \bar{B} x_t \]
            </p>
<p><strong>Where:</strong></p>
<ul>
<li>\(h_t\) = hidden state at step t</li>
<li>\(x_t\) = current input token</li>
<li>\(\bar{A}\) = state transition matrix (how much to remember)</li>
<li>\(\bar{B}\) = input projection matrix (how much new info to add)</li>
</ul>
</div>
<h3>The Selective Mechanism</h3>
<p>
            This is similar to an RNN, but in standard state space models, <strong>A and B are fixed parameters</strong> — 
            they don't change based on the input.
        </p>
<p>
            Here's the key innovation: <strong>Mamba makes A and B input-dependent</strong> through a gating parameter 
            <strong>\(\Delta_t\)</strong> (delta).
        </p>
<p>
            For each token \(x_t\), the model computes:
        </p>
<div class="formula-box">
<p>
                \[ \Delta_t = \text{Softplus}(W_\Delta x_t) \]
            </p>
<p>Then it modulates the fixed A and computes effective parameters:</p>
<p>
                \[ \bar{A} = \exp(\Delta_t \cdot A), \quad \bar{B} = \Delta_t \cdot B \]
            </p>
</div>
<p>
            This allows the model to:
        </p>
<ul>
<li><strong>Expand the gate (Δ large):</strong> Let important tokens influence the hidden state strongly</li>
<li><strong>Contract the gate (Δ small):</strong> Filter out noisy or irrelevant tokens</li>
</ul>
<div class="key-takeaway">
<strong>Key Innovation:</strong> The selective gate Δ decides how much to update the summary based on 
            the current token's importance. This enables Mamba to focus on relevant information while filtering noise.
        </div>

<details>
<summary><strong>Python code:</strong> Selective State Space block</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="k">class</span> <span class="nc">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">token_dim</span> <span class="o">=</span> <span class="n">token_dim</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">state_dim</span> <span class="o">=</span> <span class="n">state_dim</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">token_to_params</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="mi">3</span> <span class="o">*</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">state_to_token</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">state_dim</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_sequence</span><span class="p">:</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">:</span>
        <span class="n">batch_size</span><span class="p">,</span> <span class="n">seq_len</span><span class="p">,</span> <span class="n">token_dim</span> <span class="o">=</span> <span class="n">token_sequence</span><span class="p">.</span><span class="n">shape</span>

        <span class="n">params</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">token_to_params</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>
        <span class="n">keep_gate</span><span class="p">,</span> <span class="n">write_gate</span><span class="p">,</span> <span class="n">decay_gate</span> <span class="o">=</span> <span class="n">params</span><span class="p">.</span><span class="n">chunk</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

        <span class="n">keep_gate</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">keep_gate</span><span class="p">)</span>
        <span class="n">write_gate</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">tanh</span><span class="p">(</span><span class="n">write_gate</span><span class="p">)</span>
        <span class="n">decay_gate</span> <span class="o">=</span> <span class="n">F</span><span class="p">.</span><span class="n">softplus</span><span class="p">(</span><span class="n">decay_gate</span><span class="p">)</span>

        <span class="n">state</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">zeros</span><span class="p">(</span>
            <span class="n">batch_size</span><span class="p">,</span> <span class="bp">self</span><span class="p">.</span><span class="n">state_dim</span><span class="p">,</span>
            <span class="n">device</span><span class="o">=</span><span class="n">token_sequence</span><span class="p">.</span><span class="n">device</span><span class="p">,</span>
            <span class="n">dtype</span><span class="o">=</span><span class="n">token_sequence</span><span class="p">.</span><span class="n">dtype</span>
        <span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="p">[]</span>

        <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">seq_len</span><span class="p">):</span>
            <span class="c1"># s_t = decay_t * s_{t-1} + keep_t * x_t
</span>            <span class="n">state</span> <span class="o">=</span> <span class="n">decay_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">state</span> <span class="o">+</span> <span class="n">keep_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">token_sequence</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span>
            <span class="c1"># y_t = write_t * s_t
</span>            <span class="n">current_output</span> <span class="o">=</span> <span class="n">write_gate</span><span class="p">[:,</span> <span class="n">t</span><span class="p">]</span> <span class="o">*</span> <span class="n">state</span>
            <span class="n">outputs</span><span class="p">.</span><span class="n">append</span><span class="p">(</span><span class="n">current_output</span><span class="p">)</span>

        <span class="n">outputs</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">stack</span><span class="p">(</span><span class="n">outputs</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>          <span class="c1"># (B, L, state_dim)
</span>        <span class="n">token_outputs</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">state_to_token</span><span class="p">(</span><span class="n">outputs</span><span class="p">)</span>   <span class="c1"># (B, L, token_dim)
</span>        <span class="k">return</span> <span class="n">token_outputs</span></code></pre></figure>

</details>
<div class="interactive-figure" style="background: #020617; border: 1px solid #1f2937; position: relative; height: 500px; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; min-height: 500px; color: #e8e6e3;">
<div id="mamba-viz" style="width: 100%; height: 100%; position: relative;"></div>
<div style="position: absolute; bottom: 15px; right: 15px; background: rgba(15,23,42,0.95); padding: 15px; font-size: 12px; font-family: 'Roboto', sans-serif; border: 1px solid #334155; min-width: 240px; width: 240px; max-width: calc(100% - 30px); box-sizing: border-box; z-index: 10;">
<div style="font-weight: 600; margin-bottom: 10px; color: #e2e8f0; font-size: 13px;">Legend</div>
<div style="display: flex; align-items: center; margin-bottom: 6px; color: #cbd5e1;">
<span style="width: 12px; height: 12px; background: #4ade80; display: inline-block; margin-right: 10px;"></span>
<span style="color: #cbd5e1;">Relevant (Gate Open)</span>
</div>
<div style="display: flex; align-items: center; margin-bottom: 12px; color: #cbd5e1;">
<span style="width: 12px; height: 12px; background: #f87171; display: inline-block; margin-right: 10px;"></span>
<span style="color: #cbd5e1;">Noise (Gate Closed)</span>
</div>
<div style="border-top: 1px solid #475569; padding-top: 10px; margin-top: 8px;">
<button id="replay-btn" style="width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border: none; color: #ffffff; padding: 8px 12px; cursor: pointer; font-weight: 500; font-size: 12px; transition: all 0.2s; box-sizing: border-box;">
                        ▶ Replay Animation
                    </button>
<div id="step-info" style="margin-top: 10px; padding: 8px; background: rgba(30,41,59,0.8); font-size: 11px; min-height: 40px; color: #94a3b8;"></div>
</div>
</div>
</div>
<div class="figure-caption">
<strong>Interactive Demo:</strong> Watch how Mamba processes a sequence of tokens. The <strong>gate</strong> 
            opens (expands) for relevant tokens and closes (contracts) for noise. The <strong>hidden state</strong> 
            accumulates information only from relevant tokens.
        </div>
</section>
<section class="article-width" id="step4">
<h2>Bidirectional Processing</h2>
<p class="section-intro">
            Random walks have no inherent direction like sentences. Processing them only left-to-right would introduce bias. 
            Graph Mamba solves this with bidirectional processing at two levels.
        </p>
<h3>Why Bidirectional?</h3>
<p>
            A random walk has no inherent "direction" like a sentence. If we only process it left-to-right, we bias the model.
        </p>
<p>
            To fix this, we implement <strong>Bidirectional Mamba</strong>. Watch how two SSM blocks process the same sequence in 
            <strong>opposite directions</strong>, then combine their outputs. This ensures each token "sees" context from both sides.
        </p>

<details>
<summary><strong>Python code:</strong> Bidirectional Mamba</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="k">class</span> <span class="nc">BidirectionalMamba</span><span class="p">(</span><span class="n">nn</span><span class="p">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">:</span> <span class="nb">int</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">().</span><span class="n">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">norm_tokens</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">LayerNorm</span><span class="p">(</span><span class="n">token_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">forward_ssm</span> <span class="o">=</span> <span class="n">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">backward_ssm</span> <span class="o">=</span> <span class="n">SelectiveStateSpaceBlock</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">state_dim</span><span class="p">)</span>
        <span class="bp">self</span><span class="p">.</span><span class="n">output_proj</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">token_dim</span><span class="p">,</span> <span class="n">token_dim</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token_sequence</span><span class="p">:</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">torch</span><span class="p">.</span><span class="n">Tensor</span><span class="p">:</span>
        <span class="s">"""
        token_sequence: (batch, seq_len, token_dim)
        seq_len = max_walk_length
        """</span>
        <span class="n">token_sequence</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">norm_tokens</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>

        <span class="n">forward_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">forward_ssm</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">)</span>

        <span class="n">reversed_tokens</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">flip</span><span class="p">(</span><span class="n">token_sequence</span><span class="p">,</span> <span class="n">dims</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
        <span class="n">backward_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">backward_ssm</span><span class="p">(</span><span class="n">reversed_tokens</span><span class="p">)</span>

        <span class="n">backward_tokens</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">flip</span><span class="p">(</span><span class="n">backward_tokens</span><span class="p">,</span> <span class="n">dims</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>

        <span class="n">mixed_tokens</span> <span class="o">=</span> <span class="n">forward_tokens</span> <span class="o">+</span> <span class="n">backward_tokens</span>
        <span class="n">mixed_tokens</span> <span class="o">=</span> <span class="bp">self</span><span class="p">.</span><span class="n">output_proj</span><span class="p">(</span><span class="n">mixed_tokens</span><span class="p">)</span>
        
        <span class="k">return</span> <span class="n">mixed_tokens</span>

<span class="n">mamba_layer</span> <span class="o">=</span> <span class="n">BidirectionalMamba</span><span class="p">(</span><span class="n">token_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">state_dim</span><span class="o">=</span><span class="mi">64</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span></code></pre></figure>

</details>
<div class="interactive-figure" style="min-height: 400px; position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; background: #1a1d21; border: 1px solid #3d4147; color: #e8e6e3;">
<div id="bidirectional-mamba-viz" style="position: relative; width: 100%; min-height: 500px;"></div>
<div style="position: absolute; bottom: 15px; right: 15px; background: rgba(15,23,42,0.95); padding: 15px; font-size: 12px; font-family: Roboto, sans-serif; border: 1px solid #334155; min-width: 200px; width: 200px; max-width: calc(100% - 30px); box-sizing: border-box; z-index: 10;">
<div style="font-weight: 600; margin-bottom: 10px; color: #e2e8f0; font-size: 13px;">Bidirectional Processing</div>
<div style="border-top: 1px solid #475569; padding-top: 10px; margin-top: 8px;">
<button id="bidir-replay-btn" style="width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border: none; color: #ffffff; padding: 8px 12px; cursor: pointer; font-weight: 500; font-size: 12px; transition: all 0.2s; box-sizing: border-box;">
              ▶ Replay Animation
            </button>
</div>
</div>
</div>
<div class="figure-caption">
<strong>Figure:</strong> Bidirectional Mamba processes the same token sequence 
            in both directions (forward and backward), then sums the outputs.
          </div>
<div class="formula-box">
<div>
<strong>h<sub>bidir</sub></strong> = 
          <span>h<sub>→</sub></span> + 
          <span>h<sub>←</sub></span>
</div>
<div>
          Forward and backward hidden states are summed element-wise
        </div>
</div>
<h3>Two-Level Processing in Graph Mamba</h3>
<div class="info-box">
<p><strong>Level 1: Token-Level Bidirectional Mamba</strong></p>
<p>
                For each node <em>v</em>, we have K tokens (different-radius subgraphs). Bidirectional Mamba processes these K tokens 
                to produce one <em>aggregated embedding</em> for node <em>v</em>.
            </p>
<p>
                This lets the model decide which subgraphs are most important for this specific node.
            </p>
</div>
<div class="info-box">
<p><strong>Level 2: Node-Level Bidirectional Mamba</strong></p>
<p>
                After all nodes have their aggregated embeddings, we treat <em>all N nodes</em> as one long sequence and run 
                bidirectional Mamba again.
            </p>
<p>
                This enables nodes to "see" each other at long distances <strong>without</strong> quadratic attention complexity — 
                it's like multi-hop message passing, but through Mamba instead of GNN layers.
            </p>
</div>
<div class="key-takeaway">
<strong>Why Two Levels?</strong> The first level learns which neighborhood scales matter. 
            The second level enables long-range communication between distant nodes — all in linear time.
        </div>
</section>
<section class="article-width">
<h2>End-to-End Architecture</h2>
<p class="section-intro">
    The complete Graph Mamba pipeline is surprisingly simple: only 4 trainable layers 
    (2 GCN + 1 SSM + 1 Linear). No attention, no deep stacks, no complex tricks.
</p>
<div class="interactive-figure" style="min-height: 400px; position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; background: #1a1d21; border: 1px solid #3d4147; color: #e8e6e3;">
<div style="padding: 1rem 2rem; background: var(--bg-dark); height: 400px; display: flex; align-items: center; justify-content: center; --bg-dark: #1a1d21; --bg-card: #232629; --bg-card-hover: #2d3136; --text-light: #e8e6e3; --text-muted: #9ca3af; --text-bright: #d1d5db; --color-node: #c45c3e; --color-edge: #d4a754; --color-accent: #5a7fb3; --color-forward: #5a7fb3; --color-backward: #c4964a; --color-success: #4a8a5a; --color-border: #3d4147; color: #e8e6e3;">
<svg id="e2e-architecture" style="width: 95%; max-width: 1100px; height: auto;" viewbox="0 0 1100 350">
<defs>
<marker id="arrowhead" markerheight="8" markerwidth="8" orient="auto" refx="7" refy="3">
<polygon fill="var(--text-bright)" points="0 0, 8 3, 0 6"></polygon>
</marker>
</defs>
<g id="step1" transform="translate(40, 180)">
<circle cx="50" cy="0" fill="none" r="22" stroke="var(--color-node)" stroke-width="3.5"></circle>
<circle cx="75" cy="35" fill="none" r="22" stroke="var(--color-node)" stroke-width="3.5"></circle>
<circle cx="25" cy="35" fill="none" r="22" stroke="var(--color-node)" stroke-width="3.5"></circle>
<circle cx="50" cy="70" fill="none" r="22" stroke="var(--color-node)" stroke-width="3.5"></circle>
<line stroke="var(--color-edge)" stroke-width="3" x1="50" x2="75" y1="22" y2="35"></line>
<line stroke="var(--color-edge)" stroke-width="3" x1="50" x2="25" y1="22" y2="35"></line>
<line stroke="var(--color-edge)" stroke-width="3" x1="75" x2="50" y1="35" y2="70"></line>
<line stroke="var(--color-edge)" stroke-width="3" x1="25" x2="50" y1="35" y2="70"></line>
<text fill="var(--text-light)" font-size="18" font-weight="600" text-anchor="middle" x="50" y="115">Graph</text>
<text fill="var(--text-muted)" font-size="14" text-anchor="middle" x="50" y="135">Input</text>
</g>
<line marker-end="url(#arrowhead)" stroke="var(--text-bright)" stroke-width="3" x1="160" x2="220" y1="220" y2="220"></line>
<g id="step2" transform="translate(240, 140)">
<rect fill="var(--bg-card)" height="60" rx="10" stroke="var(--color-border)" stroke-width="2.5" width="140" x="0" y="60"></rect>
<text fill="var(--text-light)" font-size="16" font-weight="600" text-anchor="middle" x="70" y="95">Tokeniser</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="70" y="155">random walks</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="70" y="172">of 1-3 length</text>
</g>
<line marker-end="url(#arrowhead)" stroke="var(--text-bright)" stroke-width="3" x1="390" x2="450" y1="220" y2="220"></line>
<g id="step3" transform="translate(470, 130)">
<rect fill="var(--bg-card-hover)" height="45" rx="8" stroke="var(--color-accent)" stroke-width="2.5" width="150" x="0" y="55"></rect>
<text fill="var(--text-light)" font-size="16" font-weight="600" text-anchor="middle" x="75" y="83">2-layer GCN</text>
<rect fill="var(--bg-card)" height="35" rx="8" stroke="var(--color-border)" stroke-width="2.5" width="150" x="0" y="110"></rect>
<text fill="var(--text-muted)" font-size="14" text-anchor="middle" x="75" y="132">mean pooling</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="75" y="170">produce</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="75" y="185">embeddings</text>
</g>
<line marker-end="url(#arrowhead)" stroke="var(--text-bright)" stroke-width="3" x1="630" x2="690" y1="220" y2="220"></line>
<g id="step4" transform="translate(710, 120)">
<path d="M 20,75 L 160,75" fill="none" marker-end="url(#arrowhead)" stroke="var(--color-forward)" stroke-width="3.5"></path>
<text fill="var(--color-forward)" font-size="14" font-weight="600" text-anchor="middle" x="90" y="62">forward</text>
<rect fill="var(--bg-card)" height="42" rx="6" stroke="var(--color-accent)" stroke-width="2.5" width="32" x="40" y="85"></rect>
<text fill="var(--text-light)" font-size="14" font-weight="500" text-anchor="middle" x="56" y="111">t₁</text>
<rect fill="var(--bg-card)" height="42" rx="6" stroke="var(--color-accent)" stroke-width="2.5" width="32" x="82" y="85"></rect>
<text fill="var(--text-light)" font-size="14" font-weight="500" text-anchor="middle" x="98" y="111">t₂</text>
<rect fill="var(--bg-card)" height="42" rx="6" stroke="var(--color-accent)" stroke-width="2.5" width="32" x="124" y="85"></rect>
<text fill="var(--text-light)" font-size="14" font-weight="500" text-anchor="middle" x="140" y="111">t₃</text>
<path d="M 160,140 L 20,140" fill="none" marker-end="url(#arrowhead)" stroke="var(--color-backward)" stroke-width="3.5"></path>
<text fill="var(--color-backward)" font-size="14" font-weight="600" text-anchor="middle" x="90" y="160">backward</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="90" y="185">context-rich</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="90" y="200">representation</text>
</g>
<line marker-end="url(#arrowhead)" stroke="var(--text-bright)" stroke-width="3" x1="890" x2="950" y1="220" y2="220"></line>
<g id="step5" transform="translate(970, 120)">
<rect fill="var(--bg-card)" height="70" rx="10" stroke="var(--color-success)" stroke-width="2.5" width="110" x="0" y="75"></rect>
<text fill="var(--text-light)" font-size="15" font-weight="600" text-anchor="middle" x="55" y="103">Linear</text>
<text fill="var(--text-light)" font-size="15" font-weight="600" text-anchor="middle" x="55" y="121">Classifier</text>
<text fill="var(--color-success)" font-size="13" text-anchor="middle" x="55" y="136">class logits</text>
<text fill="var(--text-muted)" font-size="13" text-anchor="middle" x="55" y="170">Final output</text>
</g>
<text fill="var(--text-light)" font-size="20" font-weight="700" letter-spacing="0.5" text-anchor="middle" x="550" y="35">
        Graph → Tokeniser → Mini-GCN → Bi-SSM → Linear Classifier
    </text>
</svg>
</div>
</div>
<div class="info-box">
<h4>🔍 Architecture Breakdown</h4>
<ul>
<li><strong>Tokeniser:</strong> Samples random walks of length 1-3 from the graph, creating subgraph tokens</li>
<li><strong>Mini-GCN (2 layers):</strong> Encodes each token locally using graph convolutions + mean pooling</li>
<li><strong>Bidirectional SSM:</strong> Processes token sequence in both directions (forward &amp; backward) to build context-rich node representations</li>
<li><strong>Linear Classifier:</strong> Maps final embeddings to class predictions</li>
</ul>
</div>
<p>
    This elegant design achieves <strong>state-of-the-art performance</strong> on long-range and large-scale graph benchmarks 
    with <strong>linear complexity</strong> O(N) — compared to O(N²) for Graph Transformers — 
    while using significantly less memory.
</p>
</section>
<section class="article-width" id="results">
<h2>Results on Cora</h2>
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
</div>

<details>
<summary><strong>Python code:</strong> Training loop (Graph Mamba on Cora)</summary>

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="n">num_classes</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">.</span><span class="nb">max</span><span class="p">().</span><span class="n">item</span><span class="p">()</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">head</span> <span class="o">=</span> <span class="n">nn</span><span class="p">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">num_classes</span><span class="p">).</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

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
    <span class="n">local_encoder</span><span class="p">.</span><span class="n">train</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="n">train</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="n">train</span><span class="p">()</span>
    <span class="n">perm</span> <span class="o">=</span> <span class="n">torch</span><span class="p">.</span><span class="n">randperm</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">tot_loss</span> <span class="o">=</span> <span class="n">tot_cnt</span> <span class="o">=</span> <span class="n">corr</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">BATCH</span><span class="p">):</span>
        <span class="n">bn</span> <span class="o">=</span> <span class="n">perm</span><span class="p">[</span><span class="n">s</span><span class="p">:</span><span class="n">s</span><span class="o">+</span><span class="n">BATCH</span><span class="p">]</span>
        <span class="n">bn</span> <span class="o">=</span> <span class="n">bn</span><span class="p">[</span><span class="n">data</span><span class="p">.</span><span class="n">train_mask</span><span class="p">[</span><span class="n">bn</span><span class="p">]]</span>

        <span class="n">x</span> <span class="o">=</span> <span class="n">build_token_batch</span><span class="p">(</span><span class="n">bn</span><span class="p">.</span><span class="n">tolist</span><span class="p">())</span>    <span class="c1"># (B, L, 64)
</span>        <span class="n">x</span> <span class="o">=</span> <span class="n">mamba_layer</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>                    <span class="c1"># (B, L, 64)
</span>        <span class="n">x</span> <span class="o">=</span> <span class="n">x</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>                       <span class="c1"># (B, 64)
</span>        <span class="n">y</span> <span class="o">=</span> <span class="n">data</span><span class="p">.</span><span class="n">y</span><span class="p">[</span><span class="n">bn</span><span class="p">]</span>

        <span class="n">logits</span> <span class="o">=</span> <span class="n">head</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">loss_fn</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>

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

    <span class="c1"># val
</span>    <span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">mamba_layer</span><span class="p">.</span><span class="nb">eval</span><span class="p">();</span> <span class="n">head</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>
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

<figure class="highlight"><pre style="white-space: pre; overflow-x: auto;"><code class="language-python" style="white-space: pre; display: block;"><span class="n">local_encoder</span><span class="p">.</span><span class="nb">eval</span><span class="p">()</span>
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
<h3 class="article-width">Classified Cora Graph (Graph Mamba)</h3>
<div class="interactive-figure if--600" id="cora-classified-viz" style="position: relative; width: 160%; margin-left: -30%; margin-top: 1.5rem; margin-bottom: 1.5rem; overflow: hidden; background: #1a1d21; border: 1px solid #3d4147; height: 560px; min-height: 560px; color: #e8e6e3;">
<div class="viz-controls" style="position: absolute; top: 15px; left: 15px; background: rgba(26,29,33,0.9); color: #e8e6e3; padding: 8px 10px; font-size: 12px; border: 1px solid #3d4147;">
<strong style="color: #e8e6e3;">Predicted classes</strong><br/>
<span style="color: #e8e6e3;">Drag to rotate • Scroll to zoom</span>
<div id="cora-classified-status" style="margin-top: 6px; color: #e8e6e3; font-size: 12px;">Loading cora_visualization.json…</div>
</div>
</div>
<div class="figure-caption">
<strong>Figure:</strong> Model predictions on Cora after training (Graph Mamba best checkpoint). Colors = predicted topics.
        </div>
</section>
<script>
    (function() {
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

          const graph = ForceGraph3D()(container)
            .width(container.clientWidth)
            .height(container.clientHeight)
            .graphData({ nodes, links })
            .nodeRelSize(4)
            .nodeVal(node => node.val || 2)
            .nodeColor(node => coraPalette[node.labelIdx % coraPalette.length])
            .nodeLabel(node => `Paper ${node.id}<br>${node.label || ''}`)
            .linkWidth(0.5)
            .linkOpacity(0.35)
            .backgroundColor('#0b1120')
            .onEngineStop(() => statusEl.textContent = `Loaded ${nodes.length} nodes / ${links.length} edges`);

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
window.GCN_LOCAL_PAYLOAD = {
  "nodeId": 0,
  "graph": {
    "nodes": [
      {"id": 0, "x": 0.0, "y": 0.0, "distance": 0, "is_center": true, "degree": 3},
      {"id": 1, "x": -0.8, "y": 0.5, "distance": 1, "is_center": false, "degree": 3},
      {"id": 2, "x": 0.8, "y": 0.5, "distance": 1, "is_center": false, "degree": 3},
      {"id": 3, "x": -1.2, "y": -0.3, "distance": 2, "is_center": false, "degree": 2},
      {"id": 4, "x": 1.2, "y": -0.3, "distance": 2, "is_center": false, "degree": 2},
      {"id": 5, "x": 0.0, "y": -0.8, "distance": 1, "is_center": false, "degree": 1}
    ],
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
  "layers": [
    {
      "walkLength": 2,
      "tokenNodes": [0, 1, 2, 3, 4, 5],
      "shapeInfo": {"k": 6, "featureDim": 3, "hiddenDim": 3},
      "steps": [
        {
          "id": "adjacency",
          "title": "Adjacency A",
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
          "explain": "The adjacency matrix shows which nodes in the token are directly connected.",
          "formula": {
            "lhs": "A[i,j]",
            "rhs": "1 if nodes i and j share an edge, else 0"
          }
        },
        {
          "id": "input_features",
          "title": "Input features X",
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
          "explain": "Each row is a node, each column a feature. These are the raw node features before any GCN processing.",
          "formula": {
            "lhs": "X[i,f]",
            "rhs": "raw feature value of node i in dimension f"
          }
        },
        {
          "id": "x1",
          "title": "X1 = X·W1ᵀ",
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
          "explain": "The first linear layer mixes input features into hidden dimensions h0, h1, h2.",
          "formula": {
            "lhs": "X1[i,h]",
            "rhs": "Σ_f X[i,f]·W1[h,f]"
          }
        },
        {
          "id": "ax1",
          "title": "AX1 = A·X1",
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
          "explain": "Each node sums hidden messages X1[j] from its neighbors according to A.",
          "formula": {
            "lhs": "AX1[i,h]",
            "rhs": "Σ_j A[i,j]·X1[j,h]"
          }
        },
        {
          "id": "h1",
          "title": "H1 = ReLU(AX1)",
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
          "explain": "ReLU keeps positive responses and sets negative ones to zero, focusing on strongly activated patterns.",
          "formula": {
            "lhs": "H1[i,h]",
            "rhs": "max(0, AX1[i,h])"
          }
        },
        {
          "id": "x2",
          "title": "X2 = H1·W2ᵀ",
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
          "explain": "The second linear layer refines hidden states based on H1.",
          "formula": {
            "lhs": "X2[i,h]",
            "rhs": "Σ_f H1[i,f]·W2[h,f]"
          }
        },
        {
          "id": "ax2",
          "title": "AX2 = A·X2",
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
          "explain": "Second-hop aggregation spreads information from distant nodes back through the graph.",
          "formula": {
            "lhs": "AX2[i,h]",
            "rhs": "Σ_j A[i,j]·X2[j,h]"
          }
        },
        {
          "id": "h2",
          "title": "H2 = ReLU(AX2)",
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
          "explain": "The final hidden activations for all nodes in the token.",
          "formula": {
            "lhs": "H2[i,h]",
            "rhs": "max(0, AX2[i,h])"
          }
        },
        {
          "id": "z",
          "title": "z = mean(H2)",
          "type": "vector",
          "vector": {
            "labels": ["h0", "h1"],
            "values": [0.247, 0.059]
          },
          "explain": "Pooling over all k=6 nodes yields one summary vector z for this token.",
          "formula": {
            "lhs": "z[h]",
            "rhs": "1/k · Σ_i H2[i,h] with k=6"
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

  // Игрушечный граф
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
    .attr("stroke", "#374151")
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
    .attr("fill", "#020617")
    .attr("stroke", "#9ca3af")
    .attr("stroke-width", 2)
    .attr("class", "graph-node")
    .attr("id", d => "node-" + d.id);

  nodeGroup.append("text")
    .text(d => d.id)
    .attr("text-anchor", "middle")
    .attr("dy", ".35em")
    .attr("fill", "#e5e7eb")
    .style("font-family", "Roboto, sans-serif")
    .style("font-weight", "bold")
    .style("pointer-events", "none");

  const nodeById = {};
  nodeGroup.each(function(d) { nodeById[d.id] = d3.select(this); });

  // Ходок
  const walker = svg.append("circle")
    .attr("r", 7)
    .attr("fill", "#facc15")
    .attr("stroke", "#f97316")
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

  // Размерность эмбеддинга токена (для визуализации)
  const D = 6;

  // s = 1: по одному вектору на каждую длину ℓ
  let tokenMatrix = new Array(maxL + 1).fill(null);

  // Цветовая шкала по величине компоненты
  const valueScale = d3.scaleLinear()
    .domain([0, 1])
    .range(["#020617", "#22c55e"]);

  lenSlider.addEventListener("input", () => {
    lenValue.textContent = lenSlider.value;
  });
  countSlider.addEventListener("input", () => {
    countValue.textContent = countSlider.value;
  });

  // Выбор центра: зелёная подсветка + сброс матрицы
  nodeGroup.on("click", function(event, d) {
    centerNode = d;

    nodeGroup.selectAll("circle")
      .interrupt()
      .attr("r", 18)
      .attr("fill", "#020617")
      .attr("stroke", "#9ca3af")
      .attr("stroke-width", 2);

    const c = d3.select(this).select("circle");
    c.attr("fill", "#22c55e")
     .attr("stroke", "#16a34a")
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
      ". Choose ℓ and M, then press “Generate token”.";

    // новый центр → обнуляем матрицу (s = 1)
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
    clearTokenHighlight(true); // не трогаем зелёный центр

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

    // Подсвечиваем только объединение вершин (subgraph token), рёбра пока нет
    nodeGroup.selectAll("circle")
      .transition().duration(200)
      .attr("fill", d => {
        if (center && d.id === center.id) return "#22c55e";
        return unionVisited.has(d.id) ? "#1d4ed8" : "#020617";
      })
      .attr("stroke", d => {
        if (center && d.id === center.id) return "#16a34a";
        return "#9ca3af";
      })
      .attr("stroke-width", d => (center && d.id === center.id) ? 3 : 2);

    // Рёбра: фоновые, будут загораться по мере прохождения walker’а
    link
      .transition().duration(200)
      .attr("stroke", "#374151")
      .attr("stroke-width", 2)
      .attr("opacity", 0.4);

    const nodesArr = Array.from(unionVisited).sort((a,b) => a-b);
    seqDiv.textContent =
      "Token for center v = " + center.id +
      " with length ℓ = " + L +
      " and M = " + M +
      ": nodes { " + nodesArr.join(", ") + " }";

    animateWalker(center.id, visitedPaths, unionVisited);

    // s = 1: обновляем строку для длины ℓ
    addOrUpdateTokenRow(L);
  }

  // s = 1: одна строка на каждую длину ℓ
  function addOrUpdateTokenRow(lengthL) {
    const vec = d3.range(D).map(() => Math.random()); // псевдо-эмбеддинг
    tokenMatrix[lengthL] = { L: lengthL, vector: vec };
    renderTokenMatrix();
  }

  function renderTokenMatrix() {
    const svgMatrix = d3.select("#token-matrix");
    if (svgMatrix.empty()) return;

    const rowHeight = 12;
    const rowGap = 3;
    const colWidth = 14;
    const leftMargin = 22;
    const topMargin = 6;

    // Используем только те длины ℓ, для которых уже есть вектор
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

    // Подпись слева: ℓ = ...
    rowsEnter.append("text")
      .attr("x", 0)
      .attr("y", rowHeight - 2)
      .attr("fill", "#9ca3af")
      .attr("font-size", 8)
      .attr("font-family", "monospace")
      .text(d => "ℓ=" + d.L);

    // Прямоугольники компонент вектора
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
        .attr("stroke", "#4b5563")
        .attr("stroke-width", 0.5);
    });

    rows.merge(rowsEnter)
      .attr("transform", (d, i) =>
        `translate(0, ${topMargin + i * (rowHeight + rowGap)})`);

    rows.exit().remove();
  }

  // Анимация ходока и пошаговое подсвечивание рёбер
  function animateWalker(centerId, paths, unionVisited) {
    if (!paths.length) return;

    const cSel = nodeById[centerId];
    const cData = cSel.datum();
    walker
      .attr("cx", cData.x)
      .attr("cy", cData.y)
      .attr("opacity", 1);

    let pathIndex = 0;

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
          const baseFill = (centerNode && nodeId === centerNode.id)
              ? "#22c55e"
              : (unionVisited.has(nodeId) ? "#1d4ed8" : "#020617");

          circ.transition().duration(150)
            .attr("fill", "#f97316")
            .transition().duration(250)
            .attr("fill", baseFill);
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
      .attr("stroke", "#38bdf8")
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

  // resetCenter=false → не сбрасывать зелёный центр
  function clearTokenHighlight(resetCenter = true) {
    nodeGroup.selectAll("circle")
      .transition().duration(200)
      .attr("fill", d => {
        if (!resetCenter && centerNode && d.id === centerNode.id) return "#22c55e";
        return "#020617";
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
      .attr("stroke", "#374151")
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
            function init() {
              const rect = container.getBoundingClientRect();
              const width = Math.max(1, Math.floor(rect.width));
              const height = Math.max(1, Math.floor(rect.height));
              if (width < 50 || height < 50) {
                requestAnimationFrame(init);
                return;
              }

              d3.select(container).selectAll('svg').remove();

              const svg = d3.select(container).append("svg")
                .attr("width", width)
                .attr("height", height);

            // --- CONFIG ---
            const numTokens = 7;
            const tokenSize = 50;
            const spacing = 70;
            const startX = (width - (numTokens - 1) * spacing) / 2;
            const centerY = height / 2;

            // Create Token Sequence (Middle Row)
            const tokens = d3.range(numTokens).map(i => ({
                id: i,
                x: startX + i * spacing,
                y: centerY
            }));

            // --- VISUAL ELEMENTS ---

            // 1. Draw Tokens
            const tokenGroup = svg.append("g");
            
            const tokenNodes = tokenGroup.selectAll("g")
                .data(tokens)
                .enter().append("g")
                .attr("transform", d => `translate(${d.x}, ${d.y})`);

            tokenNodes.append("rect")
                .attr("width", tokenSize)
                .attr("height", tokenSize)
                .attr("x", -tokenSize/2)
                .attr("y", -tokenSize/2)
                .attr("rx", 8)
                .attr("fill", "#1e293b")
                .attr("stroke", "#475569")
                .attr("stroke-width", 2);

            tokenNodes.append("text")
                .text(d => `v${d.id}`)
                .attr("text-anchor", "middle")
                .attr("dy", ".35em")
                .attr("fill", "#94a3b8")
                .attr("font-size", "14px")
                .attr("font-family", "monospace")
                .attr("font-weight", "bold");

            // 2. Forward SSM Path (Blue, Top)
            const forwardGroup = svg.append("g");
            const forwardY = centerY - 80;
            
            forwardGroup.append("text")
                .text("Forward SSM →")
                .attr("x", startX - 80)
                .attr("y", forwardY + 5)
                .attr("fill", "#60a5fa")
                .attr("font-size", "12px")
                .attr("font-family", "monospace");

            // Forward Path Line
            const forwardPath = forwardGroup.append("path")
                .attr("d", `M ${startX - 30} ${forwardY} L ${startX + (numTokens-1) * spacing + 30} ${forwardY}`)
                .attr("stroke", "#3b82f6")
                .attr("stroke-width", 4)
                .attr("fill", "none")
                .attr("opacity", 0.3);

            // Forward "wave" circle
            const forwardWave = forwardGroup.append("circle")
                .attr("r", 12)
                .attr("fill", "#3b82f6")
                .attr("stroke", "#fff")
                .attr("stroke-width", 2)
                .attr("cx", startX - 30)
                .attr("cy", forwardY)
                .attr("opacity", 0);

            // 3. Backward SSM Path (Orange, Bottom)
            const backwardGroup = svg.append("g");
            const backwardY = centerY + 80;
            
            backwardGroup.append("text")
                .text("← Backward SSM")
                .attr("x", startX + (numTokens-1) * spacing + 40)
                .attr("y", backwardY + 5)
                .attr("fill", "#fb923c")
                .attr("font-size", "12px")
                .attr("font-family", "monospace");

            // Backward Path Line
            const backwardPath = backwardGroup.append("path")
                .attr("d", `M ${startX + (numTokens-1) * spacing + 30} ${backwardY} L ${startX - 30} ${backwardY}`)
                .attr("stroke", "#f59e0b")
                .attr("stroke-width", 4)
                .attr("fill", "none")
                .attr("opacity", 0.3);

            // Backward "wave" circle
            const backwardWave = backwardGroup.append("circle")
                .attr("r", 12)
                .attr("fill", "#f59e0b")
                .attr("stroke", "#fff")
                .attr("stroke-width", 2)
                .attr("cx", startX + (numTokens-1) * spacing + 30)
                .attr("cy", backwardY)
                .attr("opacity", 0);

            // --- ANIMATION LOGIC ---
            function runBidirectionalAnimation() {
                const duration = 2500;

                // Reset token colors
                tokenNodes.selectAll("rect")
                    .transition().duration(300)
                    .attr("fill", "#1e293b")
                    .attr("stroke", "#475569");

                // FORWARD PASS
                forwardWave.attr("opacity", 1)
                    .attr("cx", startX - 30)
                    .transition()
                    .duration(duration)
                    .ease(d3.easeLinear)
                    .attr("cx", startX + (numTokens-1) * spacing + 30)
                    .tween("processForward", function() {
                        return function(t) {
                            // Highlight tokens as wave passes
                            const currentX = startX - 30 + t * ((numTokens-1) * spacing + 60);
                            tokens.forEach((token, i) => {
                                if (currentX >= token.x - spacing/2 && currentX < token.x + spacing/2) {
                                    d3.select(tokenNodes.nodes()[i]).select("rect")
                                        .attr("fill", "#1e3a8a")
                                        .attr("stroke", "#3b82f6");
                                }
                            });
                        };
                    })
                    .on("end", () => {
                        forwardWave.attr("opacity", 0);
                    });

                // BACKWARD PASS (starts slightly delayed for clarity)
                setTimeout(() => {
                    backwardWave.attr("opacity", 1)
                        .attr("cx", startX + (numTokens-1) * spacing + 30)
                        .transition()
                        .duration(duration)
                        .ease(d3.easeLinear)
                        .attr("cx", startX - 30)
                        .tween("processBackward", function() {
                            return function(t) {
                                const currentX = (startX + (numTokens-1) * spacing + 30) - t * ((numTokens-1) * spacing + 60);
                                tokens.forEach((token, i) => {
                                    if (currentX <= token.x + spacing/2 && currentX > token.x - spacing/2) {
                                        d3.select(tokenNodes.nodes()[i]).select("rect")
                                            .attr("fill", "#78350f")
                                            .attr("stroke", "#f59e0b");
                                    }
                                });
                            };
                        })
                        .on("end", () => {
                            backwardWave.attr("opacity", 0);
                            
                            // Final Combined State (purple glow = sum of both)
                            setTimeout(() => {
                                tokenNodes.selectAll("rect")
                                    .transition().duration(800)
                                    .attr("fill", "#581c87")
                                    .attr("stroke", "#a855f7");
                                
                                // Add glow effect
                                tokenNodes.selectAll("rect")
                                    .transition().delay(800).duration(400)
                                    .attr("stroke-width", 4)
                                    .transition().duration(400)
                                    .attr("stroke-width", 2);
                            }, 300);
                        });
                }, 400); // Slight delay so they overlap nicely
            }

            // Start on load
            runBidirectionalAnimation();

            // Replay button
            const replayBtn = document.getElementById('bidir-replay-btn');
            if (replayBtn) replayBtn.onclick = runBidirectionalAnimation;

            }

            init();
            window.addEventListener('resize', () => init());
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

                // Добавляем случайные размеры для визуализации если их нет
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
        const coraPalette = d3.schemeCategory10.concat(['#a855f7', '#14b8a6', '#fb7185', '#facc15']);

        (async function() {
            const container = document.getElementById('cora-viz');
            if (!container) return;

            const data = await loadCoraFromJSON();

            const coraGraph = ForceGraph3D()
                (container)
                .width(container.clientWidth)
                .height(container.clientHeight)
                .graphData(data)
                .nodeColor(node => coraPalette[node.labelIdx % coraPalette.length])
                .nodeLabel(node => `Paper ${node.id}\n${node.label || ''}`)
                .linkWidth(0.5)
                .linkOpacity(0.5)
                .backgroundColor('#0b1120');

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
    .attr("height", height);

  // === CONFIG ===
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
  let animationTimer = null;

  // === GROUPS ===
  const tokenGroup = svg.append('g');
  const gateGroup = svg.append('g');
  const stateGroup = svg.append('g');
  const labelGroup = svg.append('g');

  // === LABELS ===
  labelGroup.append('text')
    .attr('x', centerX)
    .attr('y', tokenY - 60)
    .attr('text-anchor', 'middle')
    .attr('fill', '#94a3b8')
    .attr('font-size', '14px')
    .attr('font-weight', '600')
    .attr('font-family', 'Roboto, sans-serif')
    .text('Token Sequence');

  labelGroup.append('text')
    .attr('x', centerX)
    .attr('y', gateY - 65)
    .attr('text-anchor', 'middle')
    .attr('fill', '#fbbf24')
    .attr('font-size', '15px')
    .attr('font-weight', '700')
    .attr('font-family', 'Roboto, sans-serif')
    .text('Selective Gate Δ');

  labelGroup.append('text')
    .attr('x', centerX)
    .attr('y', stateY - 80)  // ← ПОДНЯЛ ВЫШЕ (было -50)
    .attr('text-anchor', 'middle')
    .attr('fill', '#60a5fa')
    .attr('font-size', '14px')
    .attr('font-weight', '600')
    .attr('font-family', 'Roboto, sans-serif')
    .text('Hidden State h_t');

  // === GATE ===
  gateGroup.append('circle')
    .attr('cx', centerX)
    .attr('cy', gateY)
    .attr('r', config.gateRadius)
    .attr('fill', 'none')
    .attr('stroke', '#64748b')
    .attr('stroke-width', 2)
    .attr('opacity', 0.4);

  const gateInner = gateGroup.append('circle')
    .attr('cx', centerX)
    .attr('cy', gateY)
    .attr('r', 8)
    .attr('fill', '#fbbf24')
    .style('filter', 'drop-shadow(0 0 8px rgba(251,191,36,0.6))');

  const gateText = gateGroup.append('text')
    .attr('x', centerX)
    .attr('y', gateY + config.gateRadius + 25)
    .attr('text-anchor', 'middle')
    .attr('fill', '#cbd5e1')
    .attr('font-size', '12px')
    .attr('font-family', 'monospace')
    .text('Closed');

  // === STATE BARS ===
  const bars = [];
  for (let i = 0; i < sequence.length; i++) {
    const bar = stateGroup.append('rect')
      .attr('x', centerX - (sequence.length * 10 / 2) + i * 10)
      .attr('y', stateY)
      .attr('width', 8)
      .attr('height', 0)
      .attr('fill', '#475569')
      .attr('rx', 2);
    bars.push(bar);
  }

  const stateValueText = stateGroup.append('text')
    .attr('x', centerX)
    .attr('y', stateY + 30)
    .attr('text-anchor', 'middle')
    .attr('fill', '#60a5fa')
    .attr('font-size', '16px')
    .attr('font-weight', '700')
    .attr('font-family', 'monospace')
    .text('h = 0.00');

  // === TOKENS (создаём очередь) ===
  const tokenElements = sequence.map((token, i) => {
    const tokenG = tokenGroup.append('g')
      .attr('transform', `translate(${centerX + (i - currentStep) * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);

    tokenG.append('rect')
      .attr('x', -config.tokenSize/2)
      .attr('y', -config.tokenSize/2)
      .attr('width', config.tokenSize)
      .attr('height', config.tokenSize)
      .attr('rx', 8)
      .attr('fill', '#1e293b')
      .attr('stroke', '#475569')
      .attr('stroke-width', 2);

    tokenG.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .attr('font-family', 'monospace')
      .text(token.id);

    return { g: tokenG, data: token };
  });

  // === STEP INFO ===
  const stepInfo = document.getElementById('step-info');

  // === ANIMATION ===
  function processToken(index) {
    if (index >= sequence.length) {
      if (stepInfo) stepInfo.innerHTML = '<div style="color:#4ade80; font-weight:600">✓ Sequence complete!</div>';
      return;
    }

    const token = sequence[index];
    const isRelevant = token.type === 'relevant';
    const gateSize = isRelevant ? config.gateRadius * 0.8 : 10;
    const color = isRelevant ? '#4ade80' : '#f87171';

    // Плавно сдвигаем все токены влево
    tokenElements.forEach((te, i) => {
      te.g.transition()
        .duration(config.animationDuration)
        .ease(d3.easeCubicInOut)
        .attr('transform', `translate(${centerX + (i - index) * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);
    });

    // Подсвечиваем активный токен
    tokenElements[index].g.select('rect')
      .transition().duration(200)
      .attr('stroke', color)
      .attr('stroke-width', 3);

    // Анимация gate
    gateInner
      .transition()
      .duration(config.animationDuration)
      .attr('r', gateSize)
      .attr('fill', color)
      .style('filter', `drop-shadow(0 0 ${gateSize}px ${color})`);

    gateText
      .transition()
      .duration(config.animationDuration)
      .attr('fill', color)
      .text(isRelevant ? 'OPEN' : 'Closed');

    // ← УБРАЛИ pulse линии

    // Обновляем hidden state
    if (isRelevant) {
      hiddenState += token.value;
    }

    // Обновляем бары
    if (index < bars.length) {
      bars[index]
        .transition()
        .duration(config.animationDuration)
        .attr('y', stateY - Math.min(hiddenState * 3, 80))
        .attr('height', Math.min(hiddenState * 3, 80))
        .attr('fill', isRelevant ? '#3b82f6' : '#475569');
    }

    stateValueText
      .transition()
      .duration(config.animationDuration)
      .tween('text', function() {
        const i = d3.interpolate(parseFloat(this.textContent.split(' ')[2]), hiddenState);
        return function(t) {
          this.textContent = `h = ${i(t).toFixed(2)}`;
        };
      });

    // Step info
    if (stepInfo) {
      const formula = isRelevant 
        ? `h<sub>${index+1}</sub> = 0.9·h<sub>${index}</sub> + ${token.value.toFixed(1)} = ${hiddenState.toFixed(2)}`
        : `h<sub>${index+1}</sub> = 0.9·h<sub>${index}</sub> + 0 = ${hiddenState.toFixed(2)} (filtered)`;
      
	      stepInfo.innerHTML = `
	        <div style="font-weight:600; color:${color}; margin-bottom:4px">
	          Step ${index + 1}: ${token.id}
	        </div>
	        <div style="font-size:10px; font-family:monospace; color:#cbd5e1">${formula}</div>
	      `;
	    }

    // Следующий токен
    currentStep = index + 1;
    animationTimer = setTimeout(
      () => processToken(currentStep),
      config.animationDuration + config.pauseDuration
    );
  }

  // === REPLAY ===
  const replayBtn = document.getElementById('replay-btn');
  if (replayBtn) {
    replayBtn.onclick = function() {
      if (animationTimer) clearTimeout(animationTimer);
      currentStep = 0;
      hiddenState = 0;

      // Reset
      bars.forEach(bar => {
        bar.attr('height', 0).attr('y', stateY);
      });
      gateInner.attr('r', 8).attr('fill', '#fbbf24');
      gateText.text('Closed').attr('fill', '#cbd5e1');
      stateValueText.text('h = 0.00');
      
      tokenElements.forEach((te, i) => {
        te.g.attr('transform', `translate(${centerX + i * (config.tokenSize + config.tokenSpacing)}, ${tokenY})`);
        te.g.select('rect').attr('stroke', '#475569').attr('stroke-width', 2);
      });

      if (stepInfo) stepInfo.textContent = 'Starting animation...';
      setTimeout(() => processToken(0), 500);
    };
  }

  // Auto-start
  setTimeout(() => processToken(0), 1000);
  }

  init();
  window.addEventListener('resize', () => init());
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
  mount.appendChild(root);

  loadD3().then(d3 => {
    const wrapper = d3.select(root).append('div').attr('class', 'gmv-wrapper')
      .style('display', 'flex')
      .style('flex-wrap', 'wrap')
      .style('gap', '12px')
      .style('align-items', 'flex-start')
      .style('padding', '12px')
      .style('box-sizing', 'border-box')
      .style('height', '100%');

    // ========== LEFT: Force-directed Graph (как в Random Walks) ==========
    const graphPanel = wrapper.append('div').attr('class', 'gmv-panel')
      .style('flex', '1 1 420px')
      .style('min-width', '320px')
      .style('background', 'rgba(15,23,42,0.92)')
      .style('border', '1px solid #334155')
      .style('padding', '12px')
      .style('box-sizing', 'border-box');
    graphPanel.append('div').attr('class', 'gmv-title')
      .style('color', '#e5e7eb')
      .style('font-weight', '700')
      .style('margin-bottom', '8px')
      .text('Token subgraph');

    const graphContainer = graphPanel.append('div')
      .style('position', 'relative')
      .style('width', '100%')
      .style('height', '400px')
      .style('background', '#0b1120')
      .style('border', '1px solid #334155')
      .style('box-sizing', 'border-box');

    const svg = graphContainer.append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', '0 0 420 420');

    // Берём граф из payload
    const gcnGraph = payload.graph;
    const nodes = gcnGraph.nodes.map(n => ({ 
      id: n.id,
      originalData: n  // сохраняем оригинальные данные для hover
    }));
    const links = gcnGraph.links.map(l => ({
      source: l.source,
      target: l.target
    }));

    // Force simulation - ТОЧНО КАК В RANDOM WALKS
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(210, 210))
      .force("collide", d3.forceCollide(25));

    // Рёбра
    const gLinks = svg.append('g').attr('class', 'links');
    const linkLines = gLinks.selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('class', 'graph-link')
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 1.5);

    // Ноды
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
      .attr('r', d => d.originalData.is_center ? 13 : 8)
      .attr('fill', d => d.originalData.is_center ? '#ef4444' : '#3b82f6')
      .attr('stroke', '#6b7280')
      .attr('stroke-width', d => d.originalData.is_center ? 2.5 : 1.5)
      .attr('id', d => 'gcn-node-' + d.id);

    // Подписи нод
    const gLabels = svg.append('g').attr('class', 'labels');
    gLabels.selectAll('text')
      .data(nodes)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -15)
      .attr('font-size', 11)
      .attr('fill', '#e5e7eb')
      .attr('font-weight', 'bold')
      .text(d => d.id);

    // Обновление позиций
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

    // Функции для подсветки (вызываются из правой панели)
    function highlightNodes(nodeIds) {
      nodeCircles
        .transition().duration(200)
        .attr('r', d => nodeIds.includes(d.id) ? 
          (d.originalData.is_center ? 16 : 11) : 
          (d.originalData.is_center ? 13 : 8))
        .attr('fill', d => {
          if (!nodeIds.includes(d.id)) return d.originalData.is_center ? '#ef4444' : '#3b82f6';
          return '#facc15';
        })
        .attr('stroke-width', d => nodeIds.includes(d.id) ? 3 : 
          (d.originalData.is_center ? 2.5 : 1.5));
    }

    function highlightEdge(u, v) {
        linkLines
          .transition().duration(200)
          .style('stroke', d => {
            const sid = typeof d.source === 'object' ? d.source.id : d.source;
            const tid = typeof d.target === 'object' ? d.target.id : d.target;
            const isMatch = (sid === u && tid === v) || (sid === v && tid === u);
  return isMatch ? '#facc15' : '#4b5563';
          })
          .style('stroke-width', d => {
            const sid = typeof d.source === 'object' ? d.source.id : d.source;
            const tid = typeof d.target === 'object' ? d.target.id : d.target;
            return (sid === u && tid === v) || (sid === v && tid === u) ? '3px' : '1.5px';
          });
      }

    function resetGraph() {
      nodeCircles
        .interrupt()
        .transition().duration(200)
        .attr('r', d => d.originalData.is_center ? 13 : 8)
        .attr('fill', d => d.originalData.is_center ? '#ef4444' : '#3b82f6')
        .attr('stroke-width', d => d.originalData.is_center ? 2.5 : 1.5);

      linkLines
    .interrupt()
    .transition().duration(200)
    .style('stroke', '#4b5563')  // ← STYLE вместо ATTR
    .style('stroke-width', '1.5px');  // ← STYLE вместо ATTR
    }

    function highlightMultipleEdges(edgePairs) {
  // edgePairs = [[u1, v1], [u2, v2], ...]
  console.log('Highlighting multiple edges:', edgePairs);
  
  linkLines
    .transition().duration(200)
    .style('stroke', d => {
      const sid = typeof d.source === 'object' ? d.source.id : d.source;
      const tid = typeof d.target === 'object' ? d.target.id : d.target;
      
      // Проверяем, есть ли этот edge в массиве пар
      const isMatch = edgePairs.some(([u, v]) => 
        (sid === u && tid === v) || (sid === v && tid === u)
      );
      
      return isMatch ? '#facc15' : '#4b5563';
    })
    .style('stroke-width', d => {
      const sid = typeof d.source === 'object' ? d.source.id : d.source;
      const tid = typeof d.target === 'object' ? d.target.id : d.target;
      
      const isMatch = edgePairs.some(([u, v]) => 
        (sid === u && tid === v) || (sid === v && tid === u)
      );
      
      return isMatch ? '3px' : '1.5px';
    });
}


    // Легенда
    const legend = graphPanel.append('div').attr('class', 'gmv-legend')
      .style('margin-top', '8px')
      .style('font-size', '12px')
      .style('color', '#a5b4fc');
    legend.append('span')
      .style('color', '#a5b4fc')
      .text('center is 0, others are neighbors');

    // ========== RIGHT: Steps (ВСЯ ТВОЯ ОРИГИНАЛЬНАЯ ЛОГИКА) ==========
    const layer = payload.layers[0];
    const steps = layer.steps || [];

    const detailPanel = wrapper.append('div').attr('class', 'gmv-panel')
      .style('flex', '1 1 520px')
      .style('min-width', '320px')
      .style('background', 'rgba(15,23,42,0.92)')
      .style('border', '1px solid #334155')
      .style('padding', '12px')
      .style('box-sizing', 'border-box');
    detailPanel.append('div').attr('class', 'gmv-title')
      .style('color', '#e5e7eb')
      .style('font-weight', '700')
      .style('margin-bottom', '8px')
      .text('Local GCN encoder: step by step');

    const stepButtonsWrap = detailPanel.append('div').attr('class', 'gmv-step-buttons')
      .style('display', 'flex')
      .style('flex-wrap', 'wrap')
      .style('gap', '8px')
      .style('margin-bottom', '10px');
    const shapeNote = detailPanel.append('div').attr('class', 'gmv-shape-note')
      .style('color', '#a5b4fc')
      .style('font-size', '12px')
      .style('margin-bottom', '10px')
      .text(`Token: k = ${layer.shapeInfo.k} nodes, feature dim = ${layer.shapeInfo.featureDim}, hidden dim = ${layer.shapeInfo.hiddenDim}`);
    const stepCard = detailPanel.append('div').attr('class', 'gmv-step-card')
      .style('background', '#0b1120')
      .style('border', '1px solid #334155')
      .style('padding', '12px')
      .style('box-sizing', 'border-box');

    let activeStepId = steps.length ? steps[0].id : null;
    let hoverDetailDiv = null;

    function drawMatrix(containerSel, step) {
      const matrix = step.matrix;
      containerSel.selectAll('*').remove();
      if (!matrix || !matrix.values || !matrix.values.length) {
        containerSel.append('div').style('color', '#e5e7eb').text('no data');
        return;
      }

      const numRows = matrix.values.length;
      const numCols = Array.isArray(matrix.values[0]) ? matrix.values[0].length : 0;
      if (!numCols) {
        containerSel.append('div').style('color', '#e5e7eb').text('bad matrix format');
        return;
      }

      const rowLabels = (matrix.rows && matrix.rows.length === numRows)
        ? matrix.rows
        : Array.from({length: numRows}, (_, i) => String(i));
      const colLabels = (matrix.cols && matrix.cols.length === numCols)
        ? matrix.cols
        : Array.from({length: numCols}, (_, j) => String(j));

      const size = 60;
      const svgWidth = numCols * size;
      const svgHeight = numRows * size;
      const svgM = containerSel.append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth)
        .attr('height', svgHeight);

      const g = svgM.append('g');

      const flat = matrix.values.flat();
      const minVal = flat.length ? d3.min(flat) : -1;
      const maxVal = flat.length ? d3.max(flat) : 1;

      const colorScale = d3.scaleLinear()
        .domain([minVal, 0, maxVal])
        .range(['#1e3a8a', '#3b82f6', '#93c5fd'])  // тёмно-синий -> синий -> светло-голубой
        .clamp(true);

      matrix.values.forEach((row, i) => {
        row.forEach((val, j) => {
          const cell = g.append('g').attr('transform', `translate(${j * size},${i * size})`);
          const rect = cell.append('rect')
            .attr('width', size - 2)
            .attr('height', size - 2)
            .attr('fill', colorScale(val))
            .attr('fill-opacity', 0.9)
            .attr('stroke', '#020617')
            .attr('stroke-width', 0.5)
            .style('cursor', 'pointer');

          rect.on('mouseenter', () => {
            const rowId = rowLabels[i];
            const colId = colLabels[j];
            handleCellHover(step, i, j, rowId, colId, val);
            rect.attr('stroke', '#facc15').attr('stroke-width', 1.5);
          });
          rect.on('mouseleave', () => {
            rect.attr('stroke', '#020617').attr('stroke-width', 0.5);
            resetGraph();
            if (hoverDetailDiv) {
              hoverDetailDiv.text('Hover over a cell to see computation details.');
            }
          });

          rect.append('title').text(`[${rowLabels[i]}, ${colLabels[j]}] = ${val.toFixed(3)}`);
        });
      });
    }

	    function drawVector(containerSel, step) {
	      const vector = step.vector;
	      containerSel.selectAll('*').remove();
	      if (!vector || !vector.values) {
	        containerSel.append('div').style('color', '#e5e7eb').text('no data');
	        return;
	      }
      const cellSize = 60;
      const svgWidth = Math.max(1, vector.values.length) * cellSize;
      const svgHeight = 70;
      const svgV = containerSel.append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth)
        .attr('height', svgHeight);
      
      svgV.selectAll('rect').data(vector.values).enter().append('rect')
        .attr('x', (_, idx) => idx * cellSize)
        .attr('y', 10)
        .attr('width', cellSize - 2)
        .attr('height', 50)
        .attr('fill', d => d >= 0 ? '#0ea5e9' : '#f97316')
        .style('cursor', 'pointer')
        .on('mouseenter', function(event, d) {
          const idx = vector.values.indexOf(d);
          const label = vector.labels[idx];
          handleVectorHover(step, idx, label, d);
          d3.select(this).attr('stroke', '#facc15').attr('stroke-width', 1.5);
        })
        .on('mouseleave', function() {
          d3.select(this).attr('stroke', 'none');
          resetGraph();
          if (hoverDetailDiv) {
            hoverDetailDiv.text('Hover over a column to see averaging details.');
          }
        })
        .append('title')
        .text((d, idx) => `${vector.labels[idx]} = ${d.toFixed(3)}`);
    }

    function handleCellHover(step, rowIndex, colIndex, rowId, colId, value) {
      resetGraph();
      const layerNodes = layer.tokenNodes || [];
      const nodeIndex = rowIndex < layerNodes.length ? layerNodes[rowIndex] : null;
      let text = '';

      if (step.id === 'adjacency') {
        const u = parseInt(rowId, 10);
        const v = parseInt(colId, 10);
        highlightNodes([u, v]);

        if (value !== 0) {
          highlightEdge(u, v);
          text = `A[${u},${v}] = 1 → nodes ${u} and ${v} are directly connected within the token.`;
        } else {
          text = `A[${u},${v}] = 0 → nodes ${u} and ${v} are NOT connected (but may be indirectly connected through others).`;
        }
      } else if (step.id === 'input_features') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `X[${nodeIndex},${colId}] = ${value.toFixed(3)} → this is the original feature ${colId} of node ${nodeIndex}, before any GCN processing.`;
        }
      } else if (step.id === 'x1') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `X1[${nodeIndex},${colId}] = ${value.toFixed(3)} → after the first linear layer X·W1ᵀ, node ${nodeIndex} receives a hidden activation in dimension ${colId}.`;
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
          text = `AX1[${nodeIndex},${colId}] = ${sumTerms.join(' + ')} = ${sum.toFixed(3)}`;
        }
      } else if (step.id === 'h1') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `H1[${nodeIndex},${colId}] = ${value.toFixed(3)} → after ReLU: if AX1 was negative, this will be 0, otherwise unchanged.`;
        }
      } else if (step.id === 'x2') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `X2[${nodeIndex},${colId}] = ${value.toFixed(3)} → after the second linear layer H1·W2ᵀ, a deeper hidden representation is created.`;
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
          text = `AX2[${nodeIndex},${colId}] = ${sumTerms.join(' + ')} = ${sum.toFixed(3)}`;
        }
      } else if (step.id === 'h2') {
        if (nodeIndex !== null) {
          highlightNodes([nodeIndex]);
          text = `H2[${nodeIndex},${colId}] = ${value.toFixed(3)} → final activation of node ${nodeIndex} after the second ReLU.`;
        }
      }

      if (hoverDetailDiv) {
        hoverDetailDiv.text(text);
      }
    }

    function handleVectorHover(step, index, label, value) {
      resetGraph();
      let text = `z[${label}] = ${value.toFixed(3)} → `;
      
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
          
          text = `z[${label}] = (${sumTerms.join(' + ')}) / ${numNodes} = ${mean.toFixed(3)}`;
        } else {
          text += `this is the averaged value...`;
        }
      }
      
      if (hoverDetailDiv) {
        hoverDetailDiv.text(text);
      }
    }

    function renderStep() {
      const step = steps.find(s => s.id === activeStepId) || steps[0];
      stepCard.selectAll('*').remove();

      stepCard.append('div').attr('class', 'gmv-step-title')
        .style('color', '#e5e7eb')
        .style('font-weight', '700')
        .style('margin-bottom', '6px')
        .text(step.title || activeStepId);

      const formulaBox = stepCard.append('div').attr('class', 'gmv-formula');
      if (step.formula) {
        if (step.formula.lhs && step.formula.rhs) {
          formulaBox.append('div')
            .style('color', '#cbd5e1')
            .style('font-family', 'monospace')
            .style('font-size', '12px')
            .text(`${step.formula.lhs} = ${step.formula.rhs}`);
        }
      }

      stepCard.append('div').attr('class', 'gmv-explain')
        .style('color', '#cbd5e1')
        .style('margin-top', '8px')
        .text(step.explain || '');

      const gridHolder = stepCard.append('div').attr('class', 'gmv-grid');

      if (step.type === 'matrix') {
        drawMatrix(gridHolder, step);
      } else if (step.type === 'vector') {
        drawVector(gridHolder, step);
      } else {
        gridHolder.append('div').style('color', '#e5e7eb').text('Нет визуализации.');
      }

      hoverDetailDiv = stepCard.append('div')
        .attr('class', 'gmv-hover-detail')
        .style('border-top', '1px solid rgba(148,163,184,0.5)')
        .style('margin-top', '8px')
        .style('padding-top', '6px')
        .style('font-size', '11px')
        .style('color', '#a5b4fc')
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
      .style('background', d => d.id === activeStepId ? '#2563eb' : '#1e293b')
      .style('border', '1px solid #334155')
      .style('color', '#e5e7eb')
      .style('cursor', 'pointer')
      .style('padding', '6px 10px')
      .style('font-size', '12px')
      .style('font-family', 'monospace')
      .style('box-sizing', 'border-box')
      .text(d => d.label)
      .on('click', (event, d) => {
        activeStepId = d.id;
        stepButtonsWrap.selectAll('button')
          .style('background', x => x.id === activeStepId ? '#2563eb' : '#1e293b');
        renderStep();
      });

    renderStep();
  }).catch(err => {
    console.error(err);
  });
})();

</script>
