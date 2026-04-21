import random

from pyscipopt import Model, quicksum
from typing import Union
from .base import BaseGenerator


class ProteinFoldingGenerator(BaseGenerator):
    problem_code = "PF"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        n_acid: Union[int, None] = None,
        ratio: Union[float, None] = None,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_acid = n_acid if n_acid is not None else 60
            self.ratio = ratio if ratio is not None else 0.3
        elif difficulty == "medium":
            self.n_acid = n_acid if n_acid is not None else 80
            self.ratio = ratio if ratio is not None else 0.4
        elif difficulty == "hard":
            self.n_acid = n_acid if n_acid is not None else 100
            self.ratio = ratio if ratio is not None else 0.5
        else:
            self.n_acid = n_acid if n_acid is not None else 80
            self.ratio = ratio if ratio is not None else 0.4
        
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        acids = list(range(self.n_acid))
        n_h_phobic = int(self.ratio * self.n_acid)
        h_phobic = random.sample(acids, n_h_phobic)
        return self._build_model(acids, h_phobic)

    def _build_model(self, acids, h_phobic):
        list_ij = [(i, j) for i in h_phobic for j in h_phobic if j > i + 1]

        list_ik1j = []
        list_ik2j = []
        for i, j in list_ij:
            for k in range(i, j):
                if k == (i + j - 1) / 2:
                    list_ik2j.append((i, j, k))
                else:
                    list_ik1j.append((i, j, k))

        ijfold = [(i, j) for i, j, _ in list_ik2j]

        model = Model("ProteinFolding")
        match = {(i, j): model.addVar(vtype="B", name=f"match_{i}_{j}") for i, j in list_ij}
        fold = {k: model.addVar(vtype="B", name=f"fold_{k}") for k in acids}

        for i, j, k in list_ik1j:
            model.addCons(fold[k] + match[i, j] <= 1, f"limit_folding_{i}_{j}_{k}")

        for i, j, k in list_ik2j:
            model.addCons(match[i, j] <= fold[k], f"require_folding_{i}_{j}_{k}")

        objective_expr = quicksum(match[i, j] for i, j in ijfold)
        model.setObjective(objective_expr, "maximize")
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        percent = int(self.ratio * 100)
        return f"{self.n_acid}a_{percent}p_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[PF] Generating instance {idx+1}: {filepath}")
        return filepath