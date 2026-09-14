import networkx as nx
import numpy as np
import random
from typing import Union
from pathlib import Path

from .base import BaseGenerator

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None

def gen_graph(max_n, min_n, g_type="barabasi_albert", edge=4):
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == "erdos_renyi":
        g = nx.erdos_renyi_graph(n=cur_n, p=edge * 1.0 / max(cur_n - 1, 1) * 2)
    elif g_type == "erdos_renyi_fixed":
        g = nx.gnm_random_graph(cur_n, edge * cur_n)
    elif g_type == "powerlaw":
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == "barabasi_albert":
        g = nx.barabasi_albert_graph(n=cur_n, m=edge)
    elif g_type == "watts_strogatz":
        g = nx.watts_strogatz_graph(n=cur_n, k=max(cur_n // 10, 2), p=0.1)
    else:
        raise ValueError(f"Unsupported graph type: {g_type}")

    for (u, v) in g.edges():
        g[u][v]["weight"] = random.uniform(0, 1)
    for node in g.nodes():
        g.nodes[node]["weight"] = random.uniform(0, 1)

    return g

def create_opt_vc_gurobi(graph: nx.Graph):
    """
    Minimum Vertex Cover:
        min sum(w_v * x_v)
        s.t. x_u + x_v >= 1 for all edges (u,v)
        x_v in {0,1}
    """
    if gp is None:
        raise ImportError(
            "gurobipy is not available in your environment. "
            "Install Gurobi Python package or switch back to SCIP/CIP."
        )

    model = gp.Model("MVC")
    model.Params.OutputFlag = 0

    x = {}
    for v in graph.nodes():
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"v{v}")

    for (u, v) in graph.edges():
        model.addConstr(x[u] + x[v] >= 1, name=f"cover_{u}_{v}")

    obj = gp.quicksum(graph.nodes[v]["weight"] * x[v] for v in graph.nodes())
    model.setObjective(obj, GRB.MINIMIZE)

    model.update()
    return model


class MinimumVertexCoverGenerator(BaseGenerator):
    problem_code = "MVC"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        min_n: Union[int, None] = None,
        max_n: Union[int, None] = None,
        graph_type: str = "barabasi_albert",
        edge: Union[int, None] = None,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)

        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.min_n = min_n if min_n is not None else 1200
            self.max_n = max_n if max_n is not None else 1200
            self.edge = edge if edge is not None else 5
        elif difficulty == "medium":
            self.min_n = min_n if min_n is not None else 2000
            self.max_n = max_n if max_n is not None else 2000
            self.edge = edge if edge is not None else 5
        elif difficulty == "hard":
            self.min_n = min_n if min_n is not None else 500
            self.max_n = max_n if max_n is not None else 500
            self.edge = edge if edge is not None else 70
        elif difficulty == "very-hard":
            self.min_n = min_n if min_n is not None else 1000
            self.max_n = max_n if max_n is not None else 1000
            self.edge = edge if edge is not None else 70
        elif difficulty == "very-hard2":
            self.min_n = min_n if min_n is not None else 2000
            self.max_n = max_n if max_n is not None else 2000
            self.edge = edge if edge is not None else 70
        else:
            self.min_n = min_n if min_n is not None else 1200
            self.max_n = max_n if max_n is not None else 1200
            self.edge = edge if edge is not None else 5

        self.graph_type = graph_type
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        graph = gen_graph(self.max_n, self.min_n, self.graph_type, self.edge)
        return create_opt_vc_gurobi(graph)

    def make_filename(self, idx: int, **kwargs) -> str:
        # write the .lp file
        return f"mvc_{self.graph_type}_{self.min_n}to{self.max_n}_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[MVC] Generating instance {idx+1}: {filepath}")
        return filepath