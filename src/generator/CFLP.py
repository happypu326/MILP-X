from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseGenerator


def generate_capacited_facility_location(random, filename, n_customers, n_facilities, ratio):
    c_x = random.rand(n_customers)
    c_y = random.rand(n_customers)

    f_x = random.rand(n_facilities)
    f_y = random.rand(n_facilities)

    demands = random.randint(5, 35 + 1, size=n_customers)
    capacities = random.randint(10, 160 + 1, size=n_facilities)
    fixed_costs = random.randint(100, 110 + 1, size=n_facilities) * np.sqrt(capacities) \
            + random.randint(90 + 1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0\n")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))


class FacilityLocationGenerator(BaseGenerator):
    problem_code = "CFLP"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        n_customers: int | None = None,
        n_facilities: int | None = None,
        ratio: float = 5.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_customers = n_customers if n_customers is not None else 100
            self.n_facilities = n_facilities if n_facilities is not None else 100
        elif difficulty == "medium":
            self.n_customers = n_customers if n_customers is not None else 200
            self.n_facilities = n_facilities if n_facilities is not None else 100
        else:
            self.n_customers = n_customers if n_customers is not None else 100
            self.n_facilities = n_facilities if n_facilities is not None else 100
        
        self.ratio = ratio
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        return {"seed": np.random.randint(2**31)}

    def _write_lp(self, instance, filepath: Path) -> None:
        rng = np.random.RandomState(instance["seed"])
        generate_capacited_facility_location(
            rng,
            filepath,
            n_customers=self.n_customers,
            n_facilities=self.n_facilities,
            ratio=self.ratio,
        )

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"cflp_{self.n_customers}c_{self.n_facilities}f_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[CFLP] Generating instance {idx+1}: {filepath}")
        return filepath

