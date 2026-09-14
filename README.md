# MILP-X: A Machine Learning Framework for Mixed Integer Linear Programming

A benchmark framework for training and evaluating Graph Neural Networks (GNNs) on
Mixed Integer Linear Programming (MILP) problems.

## Overview

MILP-X is a research framework designed to:
- Generate diverse MILP problem instances across 16+ problem types
- Train Graph Neural Networks to solve or approximate solutions for MILPs
- Benchmark different learning methods against commercial solvers (Gurobi, SCIP)
- Evaluate multiple ML-based optimization approaches under a common protocol

## Key Features

- **16+ MILP Problem Types**: from graph problems (Max Cut, Independent Set) to combinatorial optimization (Bin Packing, Set Cover) and scheduling tasks
- **Rich method library**: GNN-based solution-prediction and search methods — Predict-and-Search, Apollo, CoCo-MILP, DiffILO, RoME, ConPaS, Neural Diving, discrete & guided diffusion, SHSP, and learned Large-Neighborhood Search
- **Bipartite GNN encoders**: a library of constraint-variable graph encoders (half-convolution, attention, structural / positional / spectral encodings, edge-updating, and more) selected with a single `gnn_type` flag
- **Bipartite Graph Representation**: MILP instances represented as constraint-variable bipartite graphs with real edge coefficients $a_{ij}$
- **Modular Architecture**: extensible base classes for generators, trainers, and evaluators
- **Commercial Solver Integration**: benchmark against Gurobi and SCIP

## Methods

MILP-X represents each MILP instance as a **constraint-variable bipartite graph**:
variable nodes on one side, constraint nodes on the other, and an edge for every
nonzero coefficient. Edges carry the coefficient $a_{ij}$ (normalized per
constraint, with its sign; `edge_nfeats: 2`). A structure-only variant (constant
edge weight, `use_edge_coeff: false`, `edge_nfeats: 1`) is also available. The
edge builder lives in `src/utils/utils.build_edge_features`.

### GNN encoders

Selected via `gnn_type`; all follow the bipartite half-convolution paradigm
(type-specific constraint↔variable message passing) and consume the edge
coefficient.

| `gnn_type` | Description |
|---|---|
| `gcn` | Two-layer bipartite half-convolution (default Predict-and-Search backbone) |
| `gasse` | Deep bipartite half-convolution with jumping-knowledge |
| `bipartite_attention` | Per-constraint softmax attention modulated by the edge coefficient |
| `random_feature` | Half-convolution augmented with random node features |
| `tripartite` | Adds an explicit objective node with objective-coefficient edges |
| `graph_transformer` | Bipartite local attention plus a per-graph global-attention token |
| `bipartite_gin` | Bipartite GINE with injective sum aggregation |
| `rwse` | Half-convolution with random-walk structural encodings |
| `substructure` | Half-convolution with bipartite closed-walk / butterfly-count features |
| `graphgps` | Local half-convolution + global attention + random-walk encodings |
| `lap_pe` | Half-convolution with Laplacian spectral positional encodings |
| `id_gnn` | Identity-aware GNN with per-root identity tagging |
| `edge_gnn` | Edge-updating GNN that maintains and evolves a per-edge state for $a_{ij}$ |
| `coco` | Intra-constraint competitive GNN layer (CoCo-MILP) |
| `moe` | Mixture-of-experts encoder (RoME) |

```bash
python scripts/train/train_gnn.py gnn_type=gasse task=CA
python scripts/train/train_gnn.py gnn_type=graphgps task=CA
python scripts/train/train_gnn.py gnn_type=rwse task=MVC
```

Encoder hyper-parameters (`depth`, `heads`, `walk_length`, `pe_dim`, `max_roots`,
`struct_powers`, ...) are set in `configs/train/train_gnn.yaml` and can be
overridden on the command line.

### Solution-prediction methods

Each method predicts a per-binary-variable score with a bipartite GNN; a solver
then searches within a trust region around the prediction.

| Method | Train | Test | Description |
|---|---|---|---|
| **Predict-and-Search (PS)** | `train_ps.py` | `test_ps.py` | Energy-weighted marginal prediction + trust-region search |
| **EnCore** | `train_encore.py` | `test_encore.py` | Early-to-final consistency prediction from a short probing solve + fix-to-early-value search |
| **Constraint Matters** | `train_ctc.py` | `test_ctc.py` | Joint variable + critical-tight-constraint prediction; variable trust region + soft constraint tightening |
| **Apollo** | `train_apollo.py` | `test_apollo.py` | Iterative prediction–correction with progressive fixing |
| **CoCo-MILP** | `train_coco.py` | `test_coco.py` | Intra-constraint competitive GNN + inter-variable contrastive objective |
| **ConPaS** | `train_conpas.py` | `test_ps.py` | Contrastive (InfoNCE) prediction over high- vs low-quality solutions |
| **DiffILO** | `train_diffilo.py` | `test_diffilo.py` | Unsupervised, label-free differentiable objective + constraint penalty |
| **RoME** | `train_rome.py` | `test_ps.py` | Robust multi-task (group-DRO) prediction with a mixture-of-experts encoder |
| **Neural Diving** | `train_neural_diving.py` | `test_neural_diving.py` | SelectiveNet coverage gate + hard fixing of confident variables |
| **Discrete Diffusion** | `train_diffusion.py` | `test_diffusion.py` | 2-state Bernoulli diffusion denoiser; reverse-sampled marginals feed the search |
| **Constraint-Aware Diffusion** | `train_diffusion.py` | `test_diffusion.py` (`feasibility_projection: true`) | Discrete diffusion with a training-free feasibility projection during sampling |
| **SHSP** | `train_shsp.py` | `test_shsp.py` | Hierarchical conditional decoding along a variable-coupling order + mask-and-repair |
| **Guided Diffusion** | `train_guided_diffusion.py` | `test_guided_diffusion.py` | Latent diffusion over solution embeddings conditioned on the instance, with feasibility-guided sampling |

```bash
python scripts/train/train_conpas.py gnn_type=gasse task=MVC
python scripts/train/train_neural_diving.py task=MVC target_coverage=0.6
python scripts/train/train_diffusion.py task=MVC num_timesteps=200
python scripts/train/train_shsp.py task=MVC
python scripts/train/train_guided_diffusion.py task=MVC
```

### Search strategies

The search stage that turns a prediction into a solution is configurable via
`fix_strategy`:

- `pas` — trust region (local-branching ball) around the predicted assignment;
- `soft` — soft/confidence-threshold fixing;
- `dive` — coverage-based hard fixing of the most confident variables.

**Learned Large-Neighborhood Search** (`scripts/test/test_lns.py`) provides an
iterative destroy-repair loop: starting from an incumbent, it repeatedly unfixes
a subset of variables, re-optimizes the residual sub-MIP, and keeps the best
solution. The neighborhood is chosen at random (`lns_mode: random`), guided by a
trained predictor's disagreement with the incumbent (`lns_mode: prediction`), or
by a contrastively-trained destroy policy (`lns_mode: cllns`) learned from a
Local-Branching expert (collect data with `scripts/preprocess/cllns_collect.py`,
train with `scripts/train/train_cllns.py`).

```bash
python scripts/test/test_lns.py lns_mode=prediction n_iters=5 destroy_frac=0.3
python scripts/test/test_lns.py lns_mode=cllns gnn_type=bipartite_attention variable_nfeats=9
```

## Project Structure

```
MILP-X/
├── configs/              # Configuration files (YAML)
│   ├── preprocess/       # Data preprocessing configurations
│   ├── train/            # Training configurations for different methods
│   └── test/             # Testing configurations
│
├── src/                  # Source code
│   ├── generator/        # MILP instance generators for 16+ problem types
│   ├── learning/         # Neural network models and loss functions
│   ├── dataloader/       # Data loading and graph dataset processing
│   ├── trainer/          # Training loops for different methods
│   ├── evaluator/        # Evaluation and benchmarking logic
│   ├── solver/           # Solver wrappers (Gurobi, SCIP)
│   ├── preprocessing/    # MILP data preprocessing
│   └── utils/            # Utility functions
│
├── scripts/              # Executable scripts
│   ├── train/            # Training scripts for each method
│   ├── test/             # Testing/evaluation scripts
│   ├── test_single/      # Single-instance testing
│   └── preprocess/       # Data preprocessing scripts
│
└── tests/                # Unit tests for generators
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Required Dependencies

Install everything with:

```bash
pip install -r requirements.txt
```

Core packages: `torch`, `torch-geometric`, `numpy`, `pandas`, `scipy`,
`networkx`, `omegaconf`, `tqdm`, and at least one solver (`pyscipopt` for SCIP,
`gurobipy` for Gurobi). Optional: `gurobi-logtools` (parsing Gurobi logs) and
`ecole` (DiffILO data pipeline).

### Gurobi License

Gurobi requires a license. Academic licenses are available for free at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Supported Problem Types

| Problem | Code | Type | Description |
|---------|------|------|-------------|
| Minimum Vertex Cover | MVC | Graph | Find minimum vertices covering all edges |
| Maximum Independent Set | MIS | Graph | Find maximum independent vertex set |
| Max Cut | MC | Graph | Partition graph to maximize cut edges |
| Set Cover | SC | Combinatorial | Cover elements with minimum sets |
| Bin Packing | BP | Combinatorial | Pack items into minimum bins |
| Knapsack | KS | Knapsack | Multiple knapsack problem |
| Combinatorial Auctions | CA | Combinatorial | Winner determination problem |
| Facility Location | CFLP | Location | Capacitated facility location |
| Job Scheduling | JS | Scheduling | Job scheduling optimization |
| Lot Sizing | LS | Sizing | Multi-item lot sizing |
| Graph Coloring | GC | Graph | Minimum graph coloring |
| Independent Set (GISP) | GISP | Graph | Two variants: Dimacs, Erdos-Renyi |
| Network Flow | NF | Network | Capacitated network flow |
| Protein Folding | PF | Bioinformatics | Protein structure prediction |
| Max Satisfiability | SAT | Satisfiability | Maximum satisfiability problem |

## Quick Start

### 1. Generate MILP Instances

The framework provides a unified interface to generate MILP instances for different problem types through the `generate_batch` function.

**How the Generator Works:**

- **Registry Pattern**: all generators are registered with problem codes (e.g., "mvc", "sc", "ks")
- **Unified Interface**: use `generate_batch(problem_code, n_instances, output_dir, **problem_params)` for all problem types
- **Difficulty Levels**: each problem supports "easy", "medium", and "hard" difficulties
- **Output Format**: generates instances in LP or MPS format organized by difficulty

**Basic Usage:**

```python
from src.generator import generate_batch

# Generate 10 instances of Minimum Vertex Cover (MVC) problem
files = generate_batch(
    "mvc",                          # Problem code
    n_instances=10,                 # Number of instances to generate
    output_dir="./data/train",      # Output directory
    difficulty="easy",              # Difficulty level: easy/medium/hard
    min_n=500,                      # Minimum number of nodes
    max_n=500,                      # Maximum number of nodes
    graph_type="barabasi_albert",   # Graph type
    edge=4,                         # Edge parameter
    seed=42,                        # Random seed
)
```

**Testing Generators:**
You can find usage examples for all problem types in the `tests/` directory:

```bash
python tests/test_generate_mvc.py
```

### 2. Preprocess MILP Instances

Convert raw MILP instances to graph representations for GNN training.

```bash
python scripts/preprocess/ps_milp_to_graph.py
```

### 3. Train a Model

Train a GNN-based model using one of the available methods.

```bash
# Train a Predict-and-Search model (choose the encoder with gnn_type)
python scripts/train/train_gnn.py gnn_type=gasse task=MVC

# Or train other methods
python scripts/train/train_apollo.py
python scripts/train/train_coco.py
python scripts/train/train_diffilo.py
```

### 4. Evaluate Model

Evaluate the trained model on test instances.

```bash
# Test a Predict-and-Search model
python scripts/test/test_ps.py

# Compare against commercial solvers
python scripts/test/test_solver.py
```

### 5. Single Instance Testing

Test the model on individual instances for debugging or analysis.

```bash
# Test on a single instance (per-method scripts under scripts/test_single/)
python scripts/test_single/test_ps.py
```

## Evaluation Metrics

The framework tracks:
- **Solving Time**: average time to solve instances
- **Feasibility Rate**: percentage of feasible solutions found
- **Solution Quality**: objective values compared to optimal
- **Branch-and-Bound Nodes**: number of B&B nodes explored
- **Optimality Gap**: gap from known optimal solutions

## License

Released under the MIT License — see [LICENSE](LICENSE). Update the copyright
holder in `LICENSE` to your name/organization before publishing.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{,
  title={},
  author={},
  year={},
}
```
