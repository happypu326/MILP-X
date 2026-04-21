import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "mis",
    n_instances=10,
    output_dir="./data/train",
    difficulty="easy",
    number_of_nodes=1000,
    graph_type="barabasi_albert",
    affinity=4,
    edge_probability=0.25,
    seed=42,
)

print(files)