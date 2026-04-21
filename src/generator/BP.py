import numpy as np
from pyscipopt import Model, quicksum

from .base import BaseGenerator
from typing import Union

class BinPackingGenerator(BaseGenerator):
    problem_code = "BP"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        n_bins: Union[int, None] = None,
        n_items: Union[int, None] = None,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_bins = n_bins if n_bins is not None else 200
            self.n_items = n_items if n_items is not None else 200
        elif difficulty == "medium":
            self.n_bins = n_bins if n_bins is not None else 400
            self.n_items = n_items if n_items is not None else 400
        elif difficulty == "hard":
            self.n_bins = n_bins if n_bins is not None else 600
            self.n_items = n_items if n_items is not None else 600
        else:
            self.n_bins = n_bins if n_bins is not None else 400
            self.n_items = n_items if n_items is not None else 400
        
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        instance = self._generate_instance()
        return self._build_model(instance)

    def _generate_instance(self):
        A = np.random.randint(5, 30, size=(self.n_bins, self.n_items))
        b = np.random.randint(10 * self.n_items, 15 * self.n_items, size=self.n_bins)
        c = np.random.randint(1, 20, size=self.n_items)
        return {"A": A, "b": b, "c": c}

    def _build_model(self, instance):
        A = instance["A"]
        b = instance["b"]
        c = instance["c"]

        model = Model("BinPacking")

        x = {
            j: model.addVar(vtype="B", lb=0.0, ub=1, name=f"x_{j+1}")
            for j in range(self.n_items)
        }

        for i in range(self.n_bins):
            model.addCons(
                quicksum(A[i, j] * x[j] for j in range(self.n_items)) <= b[i],
                f"Resource_{i+1}",
            )

        objective_expr = quicksum(c[j] * x[j] for j in range(self.n_items))
        model.setObjective(objective_expr, "maximize")
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"{self.n_bins}b_{self.n_items}i_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[BP] Generating instance {idx+1}: {filepath}")
        return filepath
