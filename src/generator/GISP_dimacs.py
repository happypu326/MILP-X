from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from .base import BaseGenerator
from .gisp import GISPModelBuilder

from typing import Union
from pathlib import Path


def read_dimacs(dimacs_file: Path) -> nx.Graph:
    with open(dimacs_file, "r") as f:
        data = f.readlines()
    edges = []
    for row in data:
        if row.startswith("e"):
            i, j = row.split()[1:]
            edges.append((int(i), int(j)))
    return nx.Graph(edges)


class GISPDimacsGenerator(BaseGenerator):
    problem_code = "GISP_DIMACS"

    def __init__(
        self,
        *,
        difficulty: str = "easy",
        dimacs_graph_dir: Union[str, Path] = "./generate/DIMACS_1993/test",
        num_instances_per_graph: int = 20,
        alpha: float = 0.75,
        node_weight: float = 100,
        edge_cost: float = 1,
        mip_extension: str = "mps",
        seed: Optional[int] = 1,
    ) -> None:
        super().__init__(seed=seed)

        difficulty = difficulty.lower()
        if difficulty == "ext-hard":
            self.alpha = alpha if alpha is not None else 0.75
        else:
            self.alpha = alpha if alpha is not None else 0.75

        self.dimacs_graph_dir = Path(dimacs_graph_dir)
        self.num_instances_per_graph = num_instances_per_graph
        self.alpha = alpha
        self.mip_extension = mip_extension
        self.builder = GISPModelBuilder(
            node_weight=node_weight,
            edge_cost=edge_cost,
        )
        self.graph_files = sorted(self.dimacs_graph_dir.rglob("*.clq"))
        if not self.graph_files:
            raise FileNotFoundError(f"No .clq graph files found under {self.dimacs_graph_dir}.")
        self._graph_cache: dict[Path, nx.Graph] = {}

    def build_instance(self, idx: int, **kwargs):
        graph_path, local_idx = self._select_graph(idx)
        graph = self._load_graph(graph_path)
        partition_seed = np.random.randint(2**31)
        model = self.builder.build_from_graph(
            graph,
            alpha=self.alpha,
            seed=partition_seed,
        )

        model.data = {
            "graph_name": graph_path.stem,
            "local_idx": local_idx,
        }
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        graph_path, local_idx = self._select_graph(idx)
        graph_name = graph_path.stem
        return f"{graph_name}_mip_{local_idx}.{self.mip_extension}"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[GISP-DIMACS] Generating instance {idx+1}: {filepath}")
        return filepath

    def _select_graph(self, idx: int):
        graph_count = len(self.graph_files)
        graph_batch_idx = (idx // self.num_instances_per_graph) % graph_count
        local_idx = idx % self.num_instances_per_graph
        return self.graph_files[graph_batch_idx], local_idx

    def _load_graph(self, graph_path: Path):
        if graph_path not in self._graph_cache:
            self._graph_cache[graph_path] = read_dimacs(graph_path)
        return self._graph_cache[graph_path]
