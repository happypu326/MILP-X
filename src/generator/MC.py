import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

from .base import BaseGenerator
from typing import Union

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(
                    new_node, affinity, replace=False, p=neighbor_prob
                )
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        return Graph(number_of_nodes, edges, degrees, neighbors)


class MaxCutGenerator(BaseGenerator):
    problem_code = "MC"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        n_nodes: Union[int, None] = None,
        graph_type: str = "barabasi_albert",
        edge_probability: float = 0.25,
        affinity: int = 4,
        weight_low: int = 0,
        weight_high: int = 50,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_nodes = n_nodes if n_nodes is not None else 100
        elif difficulty == "medium":
            self.n_nodes = n_nodes if n_nodes is not None else 150
        elif difficulty == "hard":
            self.n_nodes = n_nodes if n_nodes is not None else 200
        else:
            self.n_nodes = n_nodes if n_nodes is not None else 150
        
        self.graph_type = graph_type
        self.edge_probability = edge_probability
        self.affinity = affinity
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.difficulty = difficulty 

    def build_instance(self, idx: int, **kwargs):
        instance = self._generate_instance()
        return self._build_model(instance)

    def _generate_instance(self):
        graph = self._generate_graph()
        weights = {
            edge: np.random.randint(self.weight_low, self.weight_high + 1)
            for edge in graph.edges
        }
        return {"graph": graph, "weights": weights}

    def _generate_graph(self):
        if self.graph_type == "erdos_renyi":
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        if self.graph_type == "barabasi_albert":
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        raise ValueError("Unsupported graph type.")

    def _build_model(self, instance):
        graph = instance["graph"]
        weights = instance["weights"]

        model = Model("MaxCut")
        x = {
            u: model.addVar(vtype="B", lb=0.0, ub=1.0, name=f"x_{u}")
            for u in graph.nodes
        }

        y = {}
        for e in graph.edges:
            y[e] = model.addVar(vtype="B", lb=0.0, ub=1.0, name=f"y_{e[0]}_{e[1]}")
            model.addCons(y[e] <= x[e[0]] + x[e[1]], f"cut_constraint1_{e[0]}_{e[1]}")
            model.addCons(
                y[e] <= 2 - x[e[0]] - x[e[1]],
                f"cut_constraint2_{e[0]}_{e[1]}",
            )

        objective_expr = quicksum(weights[e] * y[e] for e in graph.edges)
        model.setObjective(objective_expr, "maximize")
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        if self.graph_type == "erdos_renyi":
            return f"er_{self.n_nodes}n_{self.edge_probability}p_{idx+1:04d}.lp"
        return f"ba_{self.n_nodes}n_{self.affinity}a_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[MC] Generating instance {idx+1}: {filepath}")
        return filepath
