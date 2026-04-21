import numpy as np
import networkx as nx
from pyscipopt import Model


class GISPModelBuilder:
    def __init__(self, node_weight: float = 100, edge_cost: float = 1) -> None:
        self.node_weight = node_weight
        self.edge_cost = edge_cost

    @staticmethod
    def partition_edges(edges, alpha: float, seed: int = 1):
        rng = np.random.RandomState(seed)
        E1 = set()
        E2 = set()
        for edge in edges:
            if rng.rand() <= alpha:
                E2.add(edge)
            else:
                E1.add(edge)
        return E1, E2

    def _build_mip(self, nodes, E1, E2):
        model = Model("GISP")
        node_vars = {
            i: model.addVar(vtype="B", name=f"node_{i}")
            for i in nodes
        }
        edge_vars = {
            (i, j): model.addVar(vtype="B", name=f"edge_{i}_{j}")
            for (i, j) in E2
        }
        objective = (
            sum(-self.node_weight * node_vars[i] for i in nodes) +
            sum(self.edge_cost * edge_vars[(i, j)] for (i, j) in E2)
        )
        model.setObjective(objective, "minimize")
        for (i, j) in E1:
            model.addCons(node_vars[i] + node_vars[j] <= 1, name=f"E1_{i}_{j}")
        for (i, j) in E2:
            model.addCons(
                node_vars[i] + node_vars[j] - edge_vars[(i, j)] <= 1,
                name=f"E2_{i}_{j}",
            )
        return model

    def build_from_graph(self, graph: nx.Graph, alpha: float = 0.75, seed: int = 1):
        E1, E2 = self.partition_edges(graph.edges, alpha, seed)
        return self._build_mip(graph.nodes, E1, E2)

    def build_random_problem(
        self,
        *,
        nodes: int = 75,
        edge_prob: float = 0.5,
        alpha: float = 0.75,
        seed: int = 1,
    ):
        rng = np.random.RandomState(seed)
        graph_seed = rng.randint(2 ** 31)
        graph = nx.erdos_renyi_graph(nodes, edge_prob, seed=graph_seed)
        partition_seed = rng.randint(2 ** 31)
        return self.build_from_graph(graph, alpha=alpha, seed=partition_seed)

    def write_random_problem(
        self,
        filename: str,
        *,
        nodes: int = 75,
        edge_prob: float = 0.5,
        alpha: float = 0.75,
        seed: int = 1,
    ):
        problem = self.build_random_problem(
            nodes=nodes,
            edge_prob=edge_prob,
            alpha=alpha,
            seed=seed,
        )
        problem.writeProblem(filename)
