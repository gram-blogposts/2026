// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/2026/";
    },
  },{id: "nav-about",
          title: "about",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/about/";
          },
        },{id: "nav-call-for-blogposts",
          title: "call for blogposts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/call/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/blog/";
          },
        },{id: "nav-submitting",
          title: "submitting",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/submitting/";
          },
        },{id: "dropdown-lt-strong-gt-2026-lt-strong-gt",
              title: "&lt;strong&gt;2026&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://gram-blogposts.github.io/2026/";
              },
            },{id: "dropdown-2024",
              title: "2024",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://gram-blogposts.github.io/2024/";
              },
            },{id: "post-jacobi-fields-in-machine-learning",
        
          title: "Jacobi Fields in Machine Learning",
        
        description: "Jacobi fields are a concept from differential geometry that describe how neighboring geodesics on a curved manifold deviate from one another. This post provides an intuitive introduction to Jacobi fields and illustrates their usefulness for machine learning on Riemannian manifolds, including an approximation result connecting tangent-space quantities to geodesic distances.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/jacobi-fields-ml/";
          
        },
      },{id: "post-fewer-edges-faster-protein-graph-learning",
        
          title: "Fewer Edges, Faster Protein Graph Learning",
        
        description: "Protein graphs should not be constructed blindly based on spatial proximity: they must reflect directed, geometrically viable chemistry. We introduce Angle Rewiring, a biologically motivated edge criterion. Paired with a FiLM-based reformulation of IEConv that reduces memory usage, we explore the relationship between topological sparsity, computational efficiency, and geometric expressiveness across Enzyme Commission, Gene Ontology, and Fold3D benchmarks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/fewer-edges/";
          
        },
      },{id: "post-when-the-k-nn-metric-breaks-a-geometric-phase-transition-in-local-density-estimation",
        
          title: "When the k-NN Metric Breaks: A Geometric Phase Transition in Local Density Estimation...",
        
        description: "LOF operates on the k-NN graph metric — a non-Euclidean structure that breaks under contamination. We show LOF undergoes a sharp phase transition at c*≈k/n: below it, near-perfect detection; above it, blindness (sigmoid fit, R²=0.80). DTM, a geometric prior measuring manifold distance, resists. Persistent homology provides topological diagnosis but rarely justifies its O(n³) cost. Verified across 22 datasets with interactive tools.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/topology-tabular-anomalies/";
          
        },
      },{id: "post-crystalite-a-lightweight-transformer-for-efficient-crystal-modeling",
        
          title: "Crystalite: A Lightweight Transformer for Efficient Crystal Modeling",
        
        description: "Crystalite is a lightweight diffusion Transformer for crystal generation and crystal structure prediction. This post covers its chemistry-aware atom encoding, geometry-aware attention mechanism, and benchmark results.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/crystalite/";
          
        },
      },{id: "post-to-augment-or-not-to-augment-diagnosing-distributional-symmetry-breaking",
        
          title: "To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking",
        
        description: "Many popular ML datasets are heavily canonicalized — objects almost always appear in the same orientation. We measure this with a simple classifier test, showing theoretically that canonicalization can cause data augmentation to hurt performance. We give practitioners a flowchart for diagnosing their own datasets.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/ToAugmentOrNot/";
          
        },
      },{id: "post-4-dimensional-objects-as-a-tool-to-study-symmetry-learning-in-humans-and-machines",
        
          title: "4-Dimensional Objects as a Tool to Study Symmetry Learning in Humans and Machines...",
        
        description: "We propose four-dimensional Shepard-Metzler shapes as a tool to study symmetry learning in humans and machines.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/4-Dimensional-Objects-as-a-Tool-to-Study-Symmetry-Learning-in-Humans-and-Machines/";
          
        },
      },{id: "post-blowup-and-blowdown-in-deep-learning-tracking-symmetry-breaking-with-algebraic-geometry",
        
          title: "Blowup and Blowdown in Deep Learning: Tracking Symmetry Breaking with Algebraic Geometry",
        
        description: "We propose algebraic-geometric indicators to track how deep networks simultaneously expand representation dimension (blowup) and break input symmetries (blowdown) during training. We prove an orbit-averaged orthogonality theorem valid for arbitrary nonlinear networks and verify experimentally that generalization gap scales as the square root of effective dimension over sample size with symmetry breaking following a greedy information-theoretic order.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/algebraic-geometry-deep-learning-dynamics/";
          
        },
      },{id: "post-topos-topological-optimal-transport-partitioned-operator-solver",
        
          title: "TOPOS: Topological Optimal-transport Partitioned Operator Solver",
        
        description: "Neural operators have emerged as a powerful approach in scientific machine learning, enabling resolution-invariant mappings across infinite-dimensional function spaces for applications including weather forecasting and fluid dynamics. While successful on structured grids, these architectures often require retraining for new 3D geometries, limiting zero-shot generalization. We introduce TOPOS (Topological Optimal-transport Partitioned Operational System), a unified framework that standardizes irregular physical domains into topology-aware latent workbenches using instance-dependent optimal transport mappings and genus-based routing. For a given input mesh with density mu, TOPOS computes a diffeomorphic transport T to a uniform reference nu on a sphere (g=0) or torus (g=1) workbench, applies a spectral neural operator as a solver and decodes solutions back via inverse transport. This four-stage pipeline ensures topological integrity, discretization invariance and computational efficiency through manifold reduction. TOPOS learns geometry-agnostic representations, allowing zero-shot deployment across variable topologies.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/topos/";
          
        },
      },{id: "post-graph-mamba-rethinking-graph-learning",
        
          title: "Graph Mamba - Rethinking Graph Learning",
        
        description: "Graph Mamba replaces message passing by turning local subgraphs into token sequences processed by selective state space models. We explain the idea with an interactive Cora demo and a minimal reference implementation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/graph-mamba/";
          
        },
      },{id: "post-symmetry-increase-and-equivariant-feature-selection",
        
          title: "Symmetry Increase and Equivariant Feature Selection",
        
        description: "This blog shows that symmetric inputs can induce representation degeneration due to the algebraic structure of the feature space itself, leading to loss of discriminative power, and provides practical guidance for selecting equivariant features.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/Symmetry-Increase-and-Equivariant-Feature-Selection/";
          
        },
      },{id: "post-the-role-of-directionality-in-graph-neural-networks",
        
          title: "The Role of Directionality in Graph Neural Networks",
        
        description: "We investigate how graph directionality may influence GNN performance across homophilic and heterophilic benchmarks, suggesting it could be an underexplored factor.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/graph_directionality_matters/";
          
        },
      },{id: "post-sample-blog-post",
        
          title: "Sample Blog Post",
        
        description: "Your blog post&#39;s abstract. Please add your abstract or summary here and not in the main body of your text. Do not include math/latex or hyperlinks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/distill-example/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/2026/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/2026/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/2026/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/2026/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
