from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from .base import BaseGenerator
from .gisp import GISPModelBuilder


class GISPErdosRenyiGenerator(BaseGenerator):
    problem_code = "GISP_erdos"

    def __init__(
        self,
        *,
        difficulty: str = "easy",
        nodes: int = 150,
        edge_prob: float = 0.3,
        alpha: float = 0.25,
        edge_cost: float = 1,
        node_weight: float = 100,
        mip_extension: str = "mps",
        file_prefix: str = "gisp_erdos_renyi",
        seed: Optional[int] = 1,
    ) -> None:
        super().__init__(seed=seed)

        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.nodes = nodes if nodes is not None else 115
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.25
        elif difficulty == "medium":
            self.nodes = nodes if nodes is not None else 150
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.75
        elif difficulty == "hard":
            self.nodes = nodes if nodes is not None else 175
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.75
        elif difficulty == "very-hard":
            self.nodes = nodes if nodes is not None else 150
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.25
        elif difficulty == "very-hard2":
            self.nodes = nodes if nodes is not None else 175
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.25
        else:
            self.nodes = nodes if nodes is not None else 115
            self.edge_prob = edge_prob if edge_prob is not None else 0.3
            self.alpha = alpha if alpha is not None else 0.25

        self.mip_extension = mip_extension
        self.file_prefix = file_prefix
        self.builder = GISPModelBuilder(
            node_weight=node_weight,
            edge_cost=edge_cost,
        )

    def build_instance(self, idx: int, **kwargs):
        graph_seed = np.random.randint(2 ** 31)
        graph = nx.erdos_renyi_graph(self.nodes, self.edge_prob, seed=graph_seed)
        partition_seed = np.random.randint(2 ** 31)
        return self.builder.build_from_graph(
            graph,
            alpha=self.alpha,
            seed=partition_seed,
        )

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"{self.file_prefix}_{idx:02d}.{self.mip_extension}"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[GISP-ER] Generating instance {idx+1}: {filepath}")
        return filepath
