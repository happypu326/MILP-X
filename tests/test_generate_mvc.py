import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "mvc",
    n_instances=10,
    output_dir="./data/train",
    difficulty="easy",
    min_n=1200,
    max_n=1200,
    graph_type="barabasi_albert",
    edge=5,
    seed=42,
)

print(files)