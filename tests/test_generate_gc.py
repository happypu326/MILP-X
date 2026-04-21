import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "gc",
    n_instances=10,
    output_dir="./data/train",
    difficulty="medium",
    n_nodes=500,
    graph_type="barabasi_albert",
    edge_probability=0.25,
    affinity=4,
    max_colors=10,
    seed=42,
)

print(files)