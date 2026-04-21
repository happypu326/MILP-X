from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .base import BaseGenerator


class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        return self.number_of_nodes

    def greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_indset(graph, filename):
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(len(graph)):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")


class IndependentSetGenerator(BaseGenerator):
    problem_code = "MIS"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        number_of_nodes: int | None = None,
        graph_type: str = "barabasi_albert",
        affinity: int = 4,
        edge_probability: float = 0.25,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.number_of_nodes = number_of_nodes if number_of_nodes is not None else 1000
        elif difficulty == "medium":
            self.number_of_nodes = number_of_nodes if number_of_nodes is not None else 1500
        else:
            self.number_of_nodes = number_of_nodes if number_of_nodes is not None else 1000
        
        self.graph_type = graph_type
        self.affinity = affinity
        self.edge_probability = edge_probability
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        return {"seed": np.random.randint(2**31)}

    def _write_lp(self, instance, filepath: Path) -> None:
        rng = np.random.RandomState(instance["seed"])
        if self.graph_type == "barabasi_albert":
            graph = Graph.barabasi_albert(self.number_of_nodes, self.affinity, rng)
        elif self.graph_type == "erdos_renyi":
            graph = Graph.erdos_renyi(self.number_of_nodes, self.edge_probability, rng)
        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
        generate_indset(graph, filepath)

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"mis_{self.graph_type}_{self.number_of_nodes}n_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[MIS] Generating instance {idx+1}: {filepath}")
        return filepath

