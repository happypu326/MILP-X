import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.generator import generate_batch

files = generate_batch(
    "gisp_erdos",
    n_instances=10,
    output_dir="./data/train",
    nodes=150,
    edge_prob=0.3,
    alpha=0.25,
    edge_cost=1,
    node_weight=100,
    mip_extension="mps",
    file_prefix="gisp_erdos_renyi",
    seed=42,
)

print(files)