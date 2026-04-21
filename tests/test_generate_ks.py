import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "ks",
    n_instances=10,
    output_dir="./data/train",
    difficulty="easy",
    number_of_items=100,
    number_of_knapsacks=6,
    min_range=10,
    max_range=20,
    scheme="subset-sum",
    seed=42,
)

print(files)