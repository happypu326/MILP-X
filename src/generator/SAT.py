from __future__ import annotations

from pathlib import Path
import numpy as np
import networkx as nx

from .base import BaseGenerator


def gen_maxcut_graph_clauses(rng, n: int, er_prob: float, p: float = 0.3):
    divider = rng.randint(1, 6)

    G = nx.algorithms.bipartite.generators.random_graph(
        n // divider,
        n - n // divider,
        p=er_prob,
        seed=int(rng.get_state()[1][0]),
    )

    n_edges = len(G.edges)
    edges = list(G.edges)

    added_edges = 0
    while added_edges < n_edges * p:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if (i, j) not in edges and (j, i) not in edges:
            added_edges += 1
            edges.append((i, j))

    clauses = [(f"v{i},v{j}", 1) for (i, j) in edges]
    clauses += [(f"-v{i},-v{j}", 1) for (i, j) in edges]
    return clauses

def write_lp(clauses, filename: str | Path):
    var_names = {}

    with open(filename, "w", encoding="utf-8") as file:
        file.write("maximize\nOBJ:")
        file.write(
            "".join(
                [f" +{clause[1]} cl_{idx}" for idx, clause in enumerate(clauses) if clause[1] < np.inf]
            )
        )

        file.write("\n\nSubject to\n")

        for idx, clause in enumerate(clauses):
            clause_str, weight = clause
            vars_in_clause = clause_str.split(",")

            neg_varrs = []
            pos_varrs = []

            for var in vars_in_clause:
                if var != "":
                    if var[0] == "-":
                        name = var[1:]
                        if name not in var_names:
                            var_names[name] = name
                        neg_varrs.append(var_names[name])
                    else:
                        name = var
                        if name not in var_names:
                            var_names[name] = name
                        pos_varrs.append(var_names[name])

            if weight < np.inf:
                last_part = f" +cl_{idx} <= {len(neg_varrs)}\n"
            else:
                last_part = f" <= {len(neg_varrs) - 1}\n"

            file.write(
                f"clause_{idx}:"
                + "".join([f" -{yi}" for yi in pos_varrs])
                + "".join([f" +{yi}" for yi in neg_varrs])
                + last_part
            )

        file.write("\nBinaries\n")

        for idx in range(len(clauses)):
            if clauses[idx][1] < np.inf:
                file.write(f" cl_{idx}")

        for var_name in var_names.keys():
            file.write(f" {var_name}")

        file.write("\nEnd\n")


class MaxSatisfiabilityGenerator(BaseGenerator):
    problem_code = "SAT"

    def __init__(
        self,
        *,
        difficulty: str = "easy",
        min_n: int | None = None,
        max_n: int | None = None,
        er_prob: float | None = None,
        edge_addition_prob: float = 0.3,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)

        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.min_n = min_n if min_n is not None else 50
            self.max_n = max_n if max_n is not None else 100
            self.er_prob = er_prob if er_prob is not None else 0.6
        elif difficulty == "medium":
            self.min_n = min_n if min_n is not None else 75
            self.max_n = max_n if max_n is not None else 125
            self.er_prob = er_prob if er_prob is not None else 0.5
        elif difficulty == "hard":
            self.min_n = min_n if min_n is not None else 100
            self.max_n = max_n if max_n is not None else 150
            self.er_prob = er_prob if er_prob is not None else 0.4
        else:
            self.min_n = min_n if min_n is not None else 50
            self.max_n = max_n if max_n is not None else 100
            self.er_prob = er_prob if er_prob is not None else 0.6

        self.edge_addition_prob = edge_addition_prob
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        n = np.random.randint(self.min_n, self.max_n + 1)
        rng = np.random.RandomState(np.random.randint(2**31))
        clauses = gen_maxcut_graph_clauses(
            rng=rng,
            n=n,
            er_prob=self.er_prob,
            p=self.edge_addition_prob,
        )
        m = len(clauses) // 2
        return {
            "n": n,
            "m": m,
            "clauses": clauses,
        }

    def _write_lp(self, instance, filepath: Path):
        write_lp(instance["clauses"], filepath)

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"Weighted_Partial_MaxSAT_instance_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[Weighted Partial MaxSAT] Generating instance {idx+1}: {filepath}")
        return filepath