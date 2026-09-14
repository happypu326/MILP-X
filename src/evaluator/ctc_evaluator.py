"""
Constraint Matters evaluator: variable prediction + constraint reduction.

Reuses the Predict-and-Search variable trust region and additionally *reduces*
the MILP by softly turning the top predicted critical-tight inequalities into
equalities. For a selected constraint i (activity a_i^T x, right-hand side b_i):

    sense '<=' :  a_i^T x >= b_i - M z_i        (with z_i in {0,1})
    sense '>=' :  a_i^T x <= b_i + M z_i

so z_i = 0 forces the constraint tight (a_i^T x = b_i together with the original
inequality), z_i = 1 relaxes it; a budget sum_i z_i <= Delta_c limits how many of
the tightened constraints may be relaxed. This shrinks the feasible region toward
the predicted optimal face while staying feasibility-preserving.
"""

import time
import numpy as np
import torch
from typing import Any, Dict, List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features
from src.utils.constraint_types import get_sorted_constraints
from src.solver.solver_utils import SOLVER_CLASSES


class CTCEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.kc = config.get("kc", 50)                  # #constraints to tighten
        self.ctc_delta = config.get("ctc_delta", 5)     # relaxation budget Delta_c
        self.ctc_threshold = config.get("ctc_threshold", 0.5)
        self._con_prob = None
        self._sorted_cons = None

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        cf = c_nodes.cpu()
        cf[torch.isnan(cf)] = 1
        ei = A._indices()
        ef = build_edge_features(ei, A._values(), use_edge_coeff=self.use_edge_coeff,
                                 num_cons=cf.shape[0])
        batch = torch.zeros(v_nodes.shape[0], dtype=torch.long)
        dev = self.device

        out = self.model(cf.to(dev), ei.to(dev), ef.to(dev), v_nodes.to(dev), batch.to(dev))
        var_logit, con_logit = out
        var_prob = var_logit.sigmoid().cpu().squeeze()
        con_prob = con_logit.sigmoid().cpu().squeeze()

        # stash constraint predictions (drop the objective node = last row) for the solve
        ncons_total = cf.shape[0]
        self._con_prob = con_prob[:ncons_total - 1].numpy()
        self._sorted_cons = get_sorted_constraints(ins_path)

        all_varname = list(v_map)
        binary_name = set(all_varname[i] for i in b_vars)
        scores = []
        for i in range(len(v_map)):
            typ = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], var_prob[i].item(), -1, typ])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores

    def _select_ctc(self):
        """Indices (into self._sorted_cons) of the top-kc predicted-tight
        INEQUALITY constraints above threshold."""
        if self._con_prob is None or self._sorted_cons is None:
            return []
        probs = self._con_prob
        n = min(len(probs), len(self._sorted_cons))
        cand = [i for i in range(n)
                if self._sorted_cons[i][3] in (0, 1) and probs[i] >= self.ctc_threshold]
        cand.sort(key=lambda i: probs[i], reverse=True)
        return cand[:self.kc]

    def _solve_with_trust_region(self, ins_path, scores, delta, log_path):
        solver = SOLVER_CLASSES[self.solver]()
        solver.hide_output_to_console()
        solver.load_model(ins_path)
        solver.set_aggressive()

        instance_variables = solver.get_vars()
        instance_variables.sort(key=lambda v: solver.varname(v))
        variables_map = {solver.varname(v): v for v in instance_variables}

        # ---- variable trust region (same as Predict-and-Search) ----
        alphas = []
        for i in range(len(scores)):
            tar_var = variables_map.get(scores[i][1])
            if tar_var is None:
                continue
            x_star = scores[i][3]
            if x_star < 0:
                continue
            tmp_var = solver.create_real_var(name=f'alpha_{i}')
            alphas.append(tmp_var)
            solver.add_constraint(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
            solver.add_constraint(tmp_var >= x_star - tar_var, name=f'alpha_down_{i}')
        if alphas:
            solver.add_constraint(sum(alphas) <= delta, name="sum_alpha")

        # ---- constraint reduction: soft-tighten top predicted CTCs ----
        def _bounds(v):
            lb = float(getattr(v, "LB", 0.0))
            ub = float(getattr(v, "UB", 1.0))
            if not np.isfinite(lb):
                lb = 0.0
            if not np.isfinite(ub):
                ub = 1.0                       # default to [0,1] for unbounded vars
            return lb, ub

        z_vars = []
        for rank, ci in enumerate(self._select_ctc()):
            _name, coeff, b, sense = self._sorted_cons[ci]
            terms = [(c, variables_map[k]) for k, c in coeff.items() if k in variables_map]
            if not terms:
                continue
            expr = sum(c * v for c, v in terms)
            z = solver.create_binary_var(name=f'ctc_z_{rank}')
            z_vars.append(z)
            if sense == 0:      # a x <= b  ->  a x >= b - M z ; need M >= b - min(a x)
                min_ax = sum((c * _bounds(v)[0] if c > 0 else c * _bounds(v)[1]) for c, v in terms)
                big_m = max(1.0, b - min_ax) + 1.0
                solver.add_constraint(expr >= b - big_m * z, name=f'ctc_lo_{rank}')
            else:               # a x >= b  ->  a x <= b + M z ; need M >= max(a x) - b
                max_ax = sum((c * _bounds(v)[1] if c > 0 else c * _bounds(v)[0]) for c, v in terms)
                big_m = max(1.0, max_ax - b) + 1.0
                solver.add_constraint(expr <= b + big_m * z, name=f'ctc_hi_{rank}')
        if z_vars:
            solver.add_constraint(sum(z_vars) <= self.ctc_delta, name="ctc_budget")

        start = time.time()
        results = solver.solve(means=self.solver, log_file=log_path,
                               time_limit=self.time_limits, threads=self.threads)
        return {"solver_time": time.time() - start, "results": results}
