import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "sc",
    n_instances=10,
    output_dir="./data/train",
    difficulty="easy",
    nrows=500,
    ncols=1000,
    density=0.05,
    max_coef=100,
    seed=42,
)

print(files)