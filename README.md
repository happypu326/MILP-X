# MILP-X: A Machine Learning Framework for Mixed Integer Linear Programming

A comprehensive benchmark framework for training and evaluating Graph Neural Networks (GNNs) on Mixed Integer Linear Programming (MILP) problems.

## Overview

X-MILP is a research framework designed to:
- Generate diverse MILP problem instances across 16+ problem types
- Train Graph Neural Networks to solve or approximate solutions for MILPs
- Benchmark different learning methods against commercial solvers (Gurobi, SCIP)
- Evaluate multiple state-of-the-art ML-based optimization approaches

## Key Features

- **16+ MILP Problem Types**: From graph problems (Max Cut, Independent Set) to combinatorial optimization (Bin Packing, Set Cover) and scheduling tasks
- **5 Learning Methods**: Apollo, COCO, DiffILO, RoME, and PS Family
- **Bipartite Graph Representation**: MILP instances represented as constraint-variable bipartite graphs
- **Modular Architecture**: Extensible base classes for generators, trainers, and evaluators
- **Commercial Solver Integration**: Benchmark against Gurobi and SCIP

## Project Structure

```
X-MILP/
├── configs/              # Configuration files (YAML)
│   ├── preprocess/       # Data preprocessing configurations
│   ├── train/           # Training configurations for different methods
│   └── test/            # Testing configurations
│
├── src/                 # Source code (~7,500 lines)
│   ├── generator/       # MILP instance generators for 16+ problem types
│   ├── learning/        # Neural network models and loss functions
│   ├── dataloader/      # Data loading and graph dataset processing
│   ├── trainer/         # Training loops for different methods
│   ├── evaluator/       # Evaluation and benchmarking logic
│   ├── solver/          # Solver wrappers (Gurobi, SCIP)
│   ├── preprocessing/   # MILP data preprocessing
│   └── utils/           # Utility functions
│
├── scripts/             # Executable scripts
│   ├── train/           # Training scripts for each method
│   ├── test/            # Testing/evaluation scripts
│   ├── test_single/     # Single-instance testing
│   └── preprocess/      # Data preprocessing scripts
│
└── tests/               # Unit tests for generators
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Required Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install gurobipy          # Gurobi solver
pip install pyscipopt          # SCIP solver
pip install ecole              # MIP solving environments
pip install omegaconf          # Configuration management
pip install numpy pandas networkx
```

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

- **Registry Pattern**: All generators are registered with problem codes (e.g., "mvc", "sc", "ks")
- **Unified Interface**: Use `generate_batch(problem_code, n_instances, output_dir, **problem_params)` for all problem types
- **Difficulty Levels**: Each problem supports "easy", "medium", and "hard" difficulties
- **Output Format**: Generates instances in LP or MPS format organized by difficulty

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
    graph_type="barabasi_albert",  # Graph type
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
python scripts/preprocess/preprocess_data.py
```

### 3. Train a Model

Train a GNN-based model using one of the available methods.

```bash
# Train Apollo model on MVC problem
python scripts/train/train_apollo.py

# Or train other methods
python scripts/train/train_coco.py
python scripts/train/train_diffilo.py
```

### 4. Evaluate Model

Evaluate the trained model on test instances.

```bash
# Test Apollo model
python scripts/test/test_apollo.py

# Compare against commercial solvers
python scripts/test/benchmark.py
```

### 5. Single Instance Testing

Test the model on individual instances for debugging or analysis.

```bash
# Test on a single instance
python scripts/test_single/test_apollo_single.py --instance_path path/to/instance.pkl
```

## Evaluation Metrics

The framework tracks:
- **Solving Time**: Average time to solve instances
- **Feasibility Rate**: Percentage of feasible solutions found
- **Solution Quality**: Objective values compared to optimal
- **Branch-and-Bound Nodes**: Number of B&B nodes explored
- **Optimality Gap**: Gap from known optimal solutions

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{,
  title={},
  author={},
  year={},
}
```
