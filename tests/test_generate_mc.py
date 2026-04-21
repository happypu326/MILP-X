import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "mc",
    n_instances=10,
    output_dir="./data/train",
    difficulty="medium",
    n_nodes=150,
    graph_type="barabasi_albert",
    affinity=4,
    weight_low=0,
    weight_high=50,
    seed=42,
)

print(files)