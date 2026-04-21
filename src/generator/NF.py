from __future__ import annotations

import numpy as np
import networkx as nx
from pathlib import Path

from .base import BaseGenerator


class FCMCNFGenerator(BaseGenerator):
    problem_code = "NF"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        min_n_nodes: int | None = None,
        max_n_nodes: int | None = None,
        min_n_commodities: int | None = None,
        max_n_commodities: int | None = None,
        c_range: tuple[float, float] = (11, 50),
        d_range: tuple[float, float] = (10, 100),
        ratio: float = 100.0,
        k_max: int = 10,
        er_prob: float = 0.3,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.min_n_nodes = min_n_nodes if min_n_nodes is not None else 10
            self.max_n_nodes = max_n_nodes if max_n_nodes is not None else 20
            self.min_n_commodities = min_n_commodities if min_n_commodities is not None else 20
            self.max_n_commodities = max_n_commodities if max_n_commodities is not None else 30
        elif difficulty == "medium":
            self.min_n_nodes = min_n_nodes if min_n_nodes is not None else 20
            self.max_n_nodes = max_n_nodes if max_n_nodes is not None else 30
            self.min_n_commodities = min_n_commodities if min_n_commodities is not None else 30
            self.max_n_commodities = max_n_commodities if max_n_commodities is not None else 45
        else:
            self.min_n_nodes = min_n_nodes if min_n_nodes is not None else 10
            self.max_n_nodes = max_n_nodes if max_n_nodes is not None else 20
            self.min_n_commodities = min_n_commodities if min_n_commodities is not None else 20
            self.max_n_commodities = max_n_commodities if max_n_commodities is not None else 30
        
        self.c_range = c_range
        self.d_range = d_range
        self.ratio = ratio
        self.k_max = k_max
        self.er_prob = er_prob
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        n_commodities = np.random.randint(
            self.min_n_commodities, self.max_n_commodities + 1
        )
        graph_seed = np.random.randint(2**31)

        graph, adj_mat, edge_list, incommings, outcommings = self._generate_erdos_graph(
            n_nodes, graph_seed
        )
        commodities = self._generate_commodities(graph, n_nodes, n_commodities)

        return {
            "n_nodes": n_nodes,
            "commodities": commodities,
            "adj_mat": adj_mat,
            "edge_list": edge_list,
            "incommings": incommings,
            "outcommings": outcommings,
        }

    def _generate_erdos_graph(self, n_nodes, seed):
        G = nx.erdos_renyi_graph(
            n=n_nodes,
            p=self.er_prob,
            seed=seed,
            directed=True,
        )
        adj_mat = np.zeros((n_nodes, n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(n_nodes)}
        outcommings = {i: [] for i in range(n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def _generate_commodities(self, graph, n_nodes, n_commodities):
        commodities = []
        for _ in range(n_commodities):
            while True:
                origin = np.random.randint(0, n_nodes)
                dest = np.random.randint(0, n_nodes)
                if origin != dest and nx.has_path(graph, origin, dest):
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((origin, dest, demand_k))
        return commodities

    def _write_lp(self, instance, filepath: Path):
        commodities = instance["commodities"]
        adj_mat = instance["adj_mat"]
        edge_list = instance["edge_list"]
        n_nodes = instance["n_nodes"]

        def fmt_num(v: float) -> str:
            return f"{float(v):.12g}"

        def write_wrapped_expr(f, head: str, tokens: list[str], tail: str = "", per_line: int = 12):
            if not tokens:
                tokens = ["0"]

            chunks = [tokens[i:i + per_line] for i in range(0, len(tokens), per_line)]
            f.write(head + " ".join(chunks[0]))
            for c in chunks[1:]:
                f.write("\n  " + " ".join(c))
            if tail:
                f.write(tail)
            f.write("\n")

        def build_signed_tokens(pos_terms: list[str], neg_terms: list[str]) -> list[str]:
            tokens: list[str] = []
            for t in pos_terms:
                if not tokens:
                    tokens.append(t)
                else:
                    tokens.append("+ " + t)
            for t in neg_terms:
                if not tokens:
                    tokens.append("- " + t)
                else:
                    tokens.append("- " + t)
            return tokens if tokens else ["0"]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\\ FCMCNF Problem Instance\n")
            f.write("Minimize\n")

            obj_tokens: list[str] = []
            first = True
            for (i, j) in edge_list:
                c_ij, f_ij, _u_ij = adj_mat[i, j]
                for k in range(len(commodities)):
                    d_k = commodities[k][2]
                    term = f"{fmt_num(c_ij * d_k)} x_{i+1}_{j+1}_{k+1}"
                    if first:
                        obj_tokens.append(term); first = False
                    else:
                        obj_tokens.append("+ " + term)
                term_y = f"{fmt_num(f_ij)} y_{i+1}_{j+1}"
                if first:
                    obj_tokens.append(term_y); first = False
                else:
                    obj_tokens.append("+ " + term_y)

            write_wrapped_expr(f, head="obj: ", tokens=obj_tokens, tail="")

            f.write("Subject To\n")

            cons_count = 1
            for k in range(len(commodities)):
                origin, dest, _demand = commodities[k]
                for i in range(n_nodes):
                    delta = 1 if i == origin else (-1 if i == dest else 0)

                    out_terms = [f"x_{i+1}_{j+1}_{k+1}" for j in instance["outcommings"][i]]
                    in_terms = [f"x_{j+1}_{i+1}_{k+1}" for j in instance["incommings"][i]]

                    tokens = build_signed_tokens(out_terms, in_terms)
                    write_wrapped_expr(
                        f,
                        head=f"flow_{cons_count}: ",
                        tokens=tokens,
                        tail=f" = {delta}",
                        per_line=20
                    )
                    cons_count += 1

            for (i, j) in edge_list:
                _c_ij, _f_ij, u_ij = adj_mat[i, j]
                pos_terms = [
                    f"{commodities[k][2]} x_{i+1}_{j+1}_{k+1}"
                    for k in range(len(commodities))
                ]
                neg_terms = [f"{fmt_num(u_ij)} y_{i+1}_{j+1}"]
                tokens = build_signed_tokens(pos_terms, neg_terms)

                write_wrapped_expr(
                    f,
                    head=f"cap_{i+1}_{j+1}: ",
                    tokens=tokens,
                    tail=" <= 0",
                    per_line=10
                )

            f.write("Bounds\n")
            for (i, j) in edge_list:
                for k in range(len(commodities)):
                    f.write(f"0 <= x_{i+1}_{j+1}_{k+1} <= 1\n")

            f.write("Binary\n")
            # y only (x is continuous with bounds)
            for (i, j) in edge_list:
                f.write(f"y_{i+1}_{j+1}\n")

            f.write("End\n")


    def make_filename(self, idx: int, **kwargs) -> str:
        return f"fcmcnf_instance_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[FCMCNF] Generating instance {idx+1}: {filepath}")
        return filepath

