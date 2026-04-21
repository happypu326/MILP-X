import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "js",
    n_instances=10,
    output_dir="./data/train",
    difficulty="medium",
    n_jobs=150,
    n_machines=20,
    n_groups=10,
    big_m=10000,
    seed=42,
)

print(files)