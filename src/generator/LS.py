import random

from pyscipopt import Model, quicksum
from typing import Union
from .base import BaseGenerator


class MultiItemLotSizingGenerator(BaseGenerator):
    problem_code = "LS"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        num_periods: Union[int, None] = None,
        num_products: Union[int, None] = None,
        factor: float = 1.0,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.num_periods = num_periods if num_periods is not None else 20
            self.num_products = num_products if num_products is not None else 8
        elif difficulty == "medium":
            self.num_periods = num_periods if num_periods is not None else 30
            self.num_products = num_products if num_products is not None else 10
        elif difficulty == "hard":
            self.num_periods = num_periods if num_periods is not None else 40
            self.num_products = num_products if num_products is not None else 12
        else:
            self.num_periods = num_periods if num_periods is not None else 30
            self.num_products = num_products if num_products is not None else 10
        
        self.factor = factor
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        instance = self._generate_instance()
        return self._build_model(instance)

    def _generate_instance(self):
        setup_costs, setup_times = {}, {}
        variable_costs, demands = {}, {}
        holding_costs, resource_upper_bounds = {}, {}

        sumT = 0
        for t in range(1, self.num_periods + 1):
            for p in range(1, self.num_products + 1):
                setup_times[t, p] = 10 * random.randint(1, 5)
                setup_costs[t, p] = 100 * random.randint(1, 10)
                variable_costs[t, p] = 0

                demands[t, p] = 100 + random.randint(-25, 25)
                if t <= 4 and random.random() < 0.25:
                    demands[t, p] = 0
                sumT += setup_times[t, p] + demands[t, p]
                holding_costs[t, p] = random.randint(1, 5)

        for t in range(1, self.num_periods + 1):
            resource_upper_bounds[t] = int(
                float(sumT) / (float(self.num_periods) * self.factor)
            )

        return {
            "setup_costs": setup_costs,
            "setup_times": setup_times,
            "variable_costs": variable_costs,
            "demands": demands,
            "holding_costs": holding_costs,
            "resource_upper_bounds": resource_upper_bounds,
        }

    def _build_model(self, instance):
        setup_costs = instance["setup_costs"]
        setup_times = instance["setup_times"]
        variable_costs = instance["variable_costs"]
        demands = instance["demands"]
        holding_costs = instance["holding_costs"]
        resource_upper_bounds = instance["resource_upper_bounds"]

        model = Model("multi_item_lot_sizing")

        y, x, I = {}, {}, {}
        for p in range(1, self.num_products + 1):
            for t in range(1, self.num_periods + 1):
                y[t, p] = model.addVar(vtype="B", name=f"y_{t}_{p}")
                x[t, p] = model.addVar(vtype="C", name=f"x_{t}_{p}")
                I[t, p] = model.addVar(vtype="C", name=f"I_{t}_{p}")
            I[0, p] = 0

        for t in range(1, self.num_periods + 1):
            model.addCons(
                quicksum(
                    setup_times[t, p] * y[t, p] + x[t, p]
                    for p in range(1, self.num_products + 1)
                )
                <= resource_upper_bounds[t],
                f"time_capacity_{t}",
            )

        for t in range(1, self.num_periods + 1):
            for p in range(1, self.num_products + 1):
                model.addCons(
                    I[t - 1, p] + x[t, p] == I[t, p] + demands[t, p],
                    f"flow_conservation_{t}_{p}",
                )

        for t in range(1, self.num_periods + 1):
            for p in range(1, self.num_products + 1):
                model.addCons(
                    x[t, p]
                    <= (resource_upper_bounds[t] - setup_times[t, p]) * y[t, p],
                    f"capacity_connection_{t}_{p}",
                )

        for t in range(1, self.num_periods + 1):
            for p in range(1, self.num_products + 1):
                model.addCons(
                    x[t, p] <= demands[t, p] * y[t, p] + I[t, p],
                    f"tighten_{t}_{p}",
                )

        objective_expr = quicksum(
            setup_costs[t, p] * y[t, p]
            + variable_costs[t, p] * x[t, p]
            + holding_costs[t, p] * I[t, p]
            for t in range(1, self.num_periods + 1)
            for p in range(1, self.num_products + 1)
        )

        model.setObjective(objective_expr, "minimize")
        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        factor_str = str(self.factor).replace(".", "p")
        return f"{self.num_periods}t_{self.num_products}p_{factor_str}f_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[LS] Generating instance {idx+1}: {filepath}")
        return filepath
