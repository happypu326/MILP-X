from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseGenerator


def generate_mknapsack(number_of_items, number_of_knapsacks, filename, random,
    min_range=10, max_range=20, scheme='weakly correlated'):
    weights = random.randint(min_range, max_range, number_of_items)

    if scheme == 'uncorrelated':
        profits = random.randint(min_range, max_range, number_of_items)

    elif scheme == 'weakly correlated':
        profits = np.apply_along_axis(
            lambda x: random.randint(x[0], x[1]),
            axis=0,
            arr=np.vstack([
                np.maximum(weights - (max_range-min_range), 1),
                weights + (max_range-min_range)]))

    elif scheme == 'strongly correlated':
        profits = weights + (max_range - min_range) / 10

    elif scheme == 'subset-sum':
        profits = weights

    else:
        raise NotImplementedError

    capacities = np.zeros(number_of_knapsacks, dtype=int)
    capacities[:-1] = random.randint(0.4 * weights.sum() // number_of_knapsacks,
                                        0.6 * weights.sum() // number_of_knapsacks,
                                        number_of_knapsacks - 1)
    capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" +{profits[item]} x{item+number_of_items*knapsack+1}")

        file.write("\n\nsubject to\n")
        for knapsack in range(number_of_knapsacks):
            variables = "".join([f" +{weights[item]} x{item+number_of_items*knapsack+1}"
                                 for item in range(number_of_items)])
            file.write(f"C{knapsack+1}:" + variables + f" <= {capacities[knapsack]}\n")

        for item in range(number_of_items):
            variables = "".join([f" +1 x{item+number_of_items*knapsack+1}"
                                 for knapsack in range(number_of_knapsacks)])
            file.write(f"C{number_of_knapsacks+item+1}:" + variables + " <= 1\n")

        file.write("\nbinary\n")
        for knapsack in range(number_of_knapsacks):
            for item in range(number_of_items):
                file.write(f" x{item+number_of_items*knapsack+1}")


class MultipleKnapsackGenerator(BaseGenerator):
    problem_code = "KS"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        number_of_items: int | None = None,
        number_of_knapsacks: int | None = None,
        min_range: int | None = None,
        max_range: int | None = None,
        scheme: str = "weakly correlated", # subset-sum
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.number_of_items = number_of_items if number_of_items is not None else 100
            self.number_of_knapsacks = number_of_knapsacks if number_of_knapsacks is not None else 6
            self.min_range = min_range if min_range is not None else 10
            self.max_range = max_range if max_range is not None else 20
        elif difficulty == "medium":
            self.number_of_items = number_of_items if number_of_items is not None else 150
            self.number_of_knapsacks = number_of_knapsacks if number_of_knapsacks is not None else 8
            self.min_range = min_range if min_range is not None else 10
            self.max_range = max_range if max_range is not None else 20
        elif difficulty == "hard":
            self.number_of_items = number_of_items if number_of_items is not None else 200
            self.number_of_knapsacks = number_of_knapsacks if number_of_knapsacks is not None else 10
            self.min_range = min_range if min_range is not None else 10
            self.max_range = max_range if max_range is not None else 30
        else:
            self.number_of_items = number_of_items if number_of_items is not None else 100
            self.number_of_knapsacks = number_of_knapsacks if number_of_knapsacks is not None else 6
            self.min_range = min_range if min_range is not None else 10
            self.max_range = max_range if max_range is not None else 20
        
        self.scheme = scheme
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        return {"seed": np.random.randint(2**31)}

    def _write_lp(self, instance, filepath: Path) -> None:
        rng = np.random.RandomState(instance["seed"])
        generate_mknapsack(
            self.number_of_items,
            self.number_of_knapsacks,
            filepath,
            rng,
            min_range=self.min_range,
            max_range=self.max_range,
            scheme=self.scheme,
        )

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"ks_{self.number_of_items}i_{self.number_of_knapsacks}k_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[KS] Generating instance {idx+1}: {filepath}")
        return filepath

