"""
EnCore data preprocessing.

EnCore (Learning Early-to-Final Solution Consistency for MILP Acceleration)
trains a predictor of *early-to-final consistency*: for each variable, whether
its assignment in a cheap early-stage solver solution persists in the
full-budget solution. Preparing data therefore needs BOTH:

  * the full-budget solutions X* (the usual MILPPreprocessor output), and
  * one or more early-stage solutions X_ES from a short probing solve.

This preprocessor reuses the standard pipeline and additionally attaches an
``early_sols`` list (each aligned to ``var_names``) to the solution pickle.
The consistency label y_i = 1[x*_i == x_ES_i] is derived at load time in the
dataloader (EnCore branch of GraphDataset).
"""

import os
import pickle

import numpy as np

from src.preprocessing.milp_preprocessor import MILPPreprocessor
from src.utils.utils import get_a_new2


class EnCorePreprocessor(MILPPreprocessor):
    def __init__(self, solver_name='gurobi', max_time=1000, max_solutions=500,
                 threads=1, workers=16, probe_time=20, probe_top_k=3):
        super().__init__(solver_name, max_time, max_solutions, threads, workers)
        self.probe_time = probe_time
        self.probe_top_k = probe_top_k

    def collect_early(self, filepath, var_names):
        """Short probing solve -> early solutions aligned to `var_names`."""
        solver = self.solver_class()
        solver.hide_output_to_console()
        solver.load_model(filepath)
        early_sols, _x_es, _t = solver.collect_early_solution(
            time_limit=self.probe_time, threads=self.threads, top_k=self.probe_top_k)
        if not early_sols:
            return []
        probe_names = [solver.varname(v) for v in solver.get_vars()]
        # re-map each early solution onto the full-solve var_names order
        aligned = []
        for sol in early_sols:
            d = {n: float(val) for n, val in zip(probe_names, sol)}
            aligned.append(np.array([d.get(n, 0.0) for n in var_names], dtype=np.float32))
        return aligned

    def process_single_file(self, filepath, output_dirs, mode):
        solution_data = self.solve_instance(filepath, output_dirs['logs'], mode)
        solution_data['early_sols'] = self.collect_early(filepath, solution_data['var_names'])

        adjacency, var_map, var_nodes, cons_nodes, bin_vars = get_a_new2(filepath)
        bg_data = (adjacency, var_map, var_nodes, cons_nodes, bin_vars)

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(os.path.join(output_dirs['solutions'], f'{base_name}.sol'), 'wb') as f:
            pickle.dump(solution_data, f)
        with open(os.path.join(output_dirs['BG'], f'{base_name}.bg'), 'wb') as f:
            pickle.dump(bg_data, f)
