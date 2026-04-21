import random

from pyscipopt import Model
from typing import Union
from .base import BaseGenerator


class JobSchedulingGenerator(BaseGenerator):
    problem_code = "JS"
    def __init__(
        self,
        *,
        difficulty: str = "medium",
        n_jobs: Union[int, None] = None,
        n_machines: Union[int, None] = None,
        n_groups: Union[int, None] = None,
        big_m: int = 10000,
        seed: Union[int, None] = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_jobs = n_jobs if n_jobs is not None else 100
            self.n_machines = n_machines if n_machines is not None else 15
            self.n_groups = n_groups if n_groups is not None else 8
        elif difficulty == "medium":
            self.n_jobs = n_jobs if n_jobs is not None else 150
            self.n_machines = n_machines if n_machines is not None else 20
            self.n_groups = n_groups if n_groups is not None else 10
        elif difficulty == "hard":
            self.n_jobs = n_jobs if n_jobs is not None else 200
            self.n_machines = n_machines if n_machines is not None else 25
            self.n_groups = n_groups if n_groups is not None else 12
        else:
            self.n_jobs = n_jobs if n_jobs is not None else 150
            self.n_machines = n_machines if n_machines is not None else 20
            self.n_groups = n_groups if n_groups is not None else 10
        
        self.big_m = big_m
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        instance = self._generate_instance()
        return self._build_model(instance)

    def _generate_instance(self):
        processing_times = [random.randint(1, 50) for _ in range(self.n_jobs)]
        precedence_constraints = self._create_random_precedence_constraints(
            self.n_jobs, self.n_groups
        )
        machine_assignment = [
            random.randint(0, self.n_machines - 1) for _ in range(self.n_jobs)
        ]

        return {
            "processing_times": processing_times,
            "precedence_constraints": precedence_constraints,
            "machine_assignment": machine_assignment,
        }

    def _create_random_precedence_constraints(self, n_jobs, n_groups):
        group_size = max(1, n_jobs // max(1, n_groups))
        jobs = list(range(n_jobs))
        random.shuffle(jobs)
        job_groups = [
            sorted(jobs[i * group_size : (i + 1) * group_size])
            for i in range(n_groups)
        ]

        constraints = []
        for group in job_groups:
            for i in range(len(group) - 1):
                constraints.append((group[i], group[i + 1]))
        return constraints

    def _build_model(self, instance):
        processing_times = instance["processing_times"]
        precedence_constraints = instance["precedence_constraints"]
        machine_assignment = instance["machine_assignment"]

        model = Model("JobShopScheduling")

        c = {}
        s = {}
        y = {}
        end_time = model.addVar("end_time", vtype="C", lb=0)

        for i in range(self.n_jobs):
            c[i] = model.addVar(f"c_{i}", vtype="C", lb=0)
            s[i] = model.addVar(f"s_{i}", vtype="C", lb=0)

        for i in range(self.n_jobs):
            for k in range(self.n_jobs):
                if i != k:
                    y[i, k] = model.addVar(f"y_{i}_{k}", vtype="B")

        model.setObjective(end_time, "minimize")

        for i in range(self.n_jobs):
            model.addCons(c[i] >= s[i] + processing_times[i], f"completion_time_{i}")

        for idx, (job_before, job_after) in enumerate(precedence_constraints):
            model.addCons(
                s[job_after] >= c[job_before],
                f"precedence_{idx}_{job_before}_{job_after}",
            )

        for i in range(self.n_jobs):
            model.addCons(end_time >= c[i], f"makespan_{i}")

        # M = self.big_m
        M = sum(processing_times)
        for j in range(self.n_machines):
            jobs_on_machine = [
                i for i in range(self.n_jobs) if machine_assignment[i] == j
            ]
            for i_idx in range(len(jobs_on_machine)):
                for k_idx in range(i_idx + 1, len(jobs_on_machine)):
                    job_i = jobs_on_machine[i_idx]
                    job_k = jobs_on_machine[k_idx]

                    model.addCons(
                        s[job_k] >= c[job_i] - M * (1 - y[job_i, job_k]),
                        f"sequence_{job_i}_before_{job_k}_machine{j}",
                    )
                    model.addCons(
                        s[job_i] >= c[job_k] - M * y[job_i, job_k],
                        f"sequence_{job_k}_before_{job_i}_machine{j}",
                    )
                    model.addCons(
                        y[job_i, job_k] + y[job_k, job_i] == 1,
                        f"mutual_exclusive_{job_i}_{job_k}",
                    )

        return model

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"JS_{self.n_jobs}j_{self.n_machines}m_{self.n_groups}g_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[JS] Generating instance {idx+1}: {filepath}")
        return filepath
