import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "nf",
    n_instances=10,
    output_dir="./data/train",
    difficulty="easy",
    min_n_nodes=10,
    max_n_nodes=20,
    min_n_commodities=20,
    max_n_commodities=30,
    c_range=(11, 50),
    d_range=(10, 100),
    ratio=100.0,
    k_max=10,
    er_prob=0.3,
    seed=42,
)

print(files)