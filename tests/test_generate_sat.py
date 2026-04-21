import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "sat",
    n_instances=10,
    output_dir="./data/train",
    difficulty="medium",
    min_n=75,
    max_n=125,
    er_prob=0.5,
    edge_addition_prob=0.3,
    seed=42,
)

print(files)