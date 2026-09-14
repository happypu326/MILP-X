"""
Constraint Matters data preprocessing: standard full-budget solutions plus
per-constraint critical-tight-constraint (CTC) labels derived from the best
solution. Labels are aligned to get_a_new2's constraint order and stored in the
solution pickle as ``ctc_labels`` (+ ``con_types``).
"""

import os
import pickle

import numpy as np

from src.preprocessing.milp_preprocessor import MILPPreprocessor
from src.utils.utils import get_a_new2
from src.utils.constraint_types import compute_ctc_labels


class CTCPreprocessor(MILPPreprocessor):
    def __init__(self, solver_name='gurobi', max_time=1000, max_solutions=500,
                 threads=1, workers=16, use_tcp=True, tight_tol=1e-6, n_min=1):
        super().__init__(solver_name, max_time, max_solutions, threads, workers)
        self.use_tcp = use_tcp
        self.tight_tol = tight_tol
        self.n_min = n_min

    def process_single_file(self, filepath, output_dirs, mode):
        solution_data = self.solve_instance(filepath, output_dirs['logs'], mode)

        var_names = solution_data['var_names']
        sols = solution_data['sols']
        if len(sols) > 0:
            x_by_name = {n: float(v) for n, v in zip(var_names, sols[0])}
            labels, types, _tight = compute_ctc_labels(
                filepath, x_by_name, tol=self.tight_tol,
                use_tcp=self.use_tcp, n_min=self.n_min)
        else:
            labels, types = np.zeros(0, dtype=np.float32), []
        solution_data['ctc_labels'] = labels
        solution_data['con_types'] = types

        adjacency, var_map, var_nodes, cons_nodes, bin_vars = get_a_new2(filepath)
        bg_data = (adjacency, var_map, var_nodes, cons_nodes, bin_vars)

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(os.path.join(output_dirs['solutions'], f'{base_name}.sol'), 'wb') as f:
            pickle.dump(solution_data, f)
        with open(os.path.join(output_dirs['BG'], f'{base_name}.bg'), 'wb') as f:
            pickle.dump(bg_data, f)
