import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "gisp_dimacs",
    n_instances=10,
    output_dir="./data/train",
    dimacs_graph_dir="./src/generator/DIMACS_1993/test",
    num_instances_per_graph=20,
    alpha=0.75,
    node_weight=100,
    edge_cost=1,
    mip_extension="mps",
    seed=42,
)

print(files)