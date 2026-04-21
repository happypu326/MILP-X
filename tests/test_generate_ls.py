import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "ls",
    n_instances=10,
    output_dir="./data/train",
    difficulty="medium",
    num_periods=30,
    num_products=10,
    factor=1.0,
    seed=42,
)

print(files)