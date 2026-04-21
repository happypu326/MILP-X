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
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

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

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


class GraphColoringGenerator(BaseGenerator):
    problem_code = "GC"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        n_nodes: Union[int, None] = None,
        graph_type: str = "barabasi_albert",
        edge_probability: float = 0.25,
        affinity: int = 4,
        max_colors: int = 10,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_nodes = n_nodes if n_nodes is not None else 300
        elif difficulty == "medium":
            self.n_nodes = n_nodes if n_nodes is not None else 500
        elif difficulty == "hard":
            self.n_nodes = n_nodes if n_nodes is not None else 700
        else:
            self.n_nodes = n_nodes if n_nodes is not None else 500
        
        self.graph_type = graph_type
        self.edge_probability = edge_probability
        self.affinity = affinity
        self.max_colors = max_colors
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        graph = self._generate_graph()
        return self._build_model(graph)

    def _generate_graph(self):
        if self.graph_type == "erdos_renyi":
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        if self.graph_type == "barabasi_albert":
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        raise ValueError("Unsupported graph type.")

    def _build_model(self, graph: Graph):
        model = Model("GraphColoring")
        max_colors = self.max_colors
        var_names = {}

        for node in graph.nodes:
            for color in range(max_colors):
                var_names[(node, color)] = model.addVar(
                    vtype="B",
                    name=f"x_{node}_{color}",
                )

        for node in graph.nodes:
            model.addCons(
                quicksum(var_names[(node, color)] for color in range(max_colors)) == 1,
                f"one_color_{node}",
            )

        for edge in graph.edges:
            node_u, node_v = edge
            if node_u < node_v:
                for color in range(max_colors):
                    model.addCons(
                        var_names[(node_u, color)] + var_names[(node_v, color)] <= 1,
                        f"adjacent_{node_u}_{node_v}_color_{color}",
                    )

        color_used = [
            model.addVar(vtype="B", name=f"color_used_{color}")
            for color in range(max_colors)
        ]
        for node in graph.nodes:
            for color in range(max_colors):
                model.addCons(
                    var_names[(node, color)] <= color_used[color],
                    f"color_usage_{node}_{color}",
                )

        objective_expr = quicksum(color_used)
        model.setObjective(objective_expr, "minimize")
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        if self.graph_type == "erdos_renyi":
            return f"GC_er_{self.n_nodes}n_{self.edge_probability}p_{idx+1:04d}.lp"
        return f"GC_ba_{self.n_nodes}n_{self.affinity}a_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[GC] Generating instance {idx+1}: {filepath}")
        return filepath