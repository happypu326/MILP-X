"""
EnCore evaluator: solver-informed early-to-final consistency for Predict-and-Search.

Pipeline per instance:
  1. probing solve (short budget) -> early solution(s) X_ES (best-first, up to K);
  2. for each early solution, append its per-variable value as the 7th variable
     feature and run the trained consistency predictor; sign-align the logits to
     X_ES and average -> per-variable consistency probability pbar;
  3. map consistency to a value-conditioned Predict-and-Search score
        p_i = pbar_i        if x_ES_i == 1
              1 - pbar_i     if x_ES_i == 0
     so a variable that is confidently consistent with an early 1 gets a high
     score (pushed to 1) and one consistent with an early 0 gets a low score
     (pushed to 0). This lets the existing trust-region fixing (_fix_pas /
     _fix_soft) fix variables to their early value without any change.
The probing time is charged against the solve budget.
"""

import numpy as np
import torch
from typing import Any, Dict, List

from .ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import get_a_new2, build_edge_features
from src.solver.solver_utils import SOLVER_CLASSES


class EnCoreEvaluator(PSFamilyEvaluator):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.probe_time = config.get("probe_time", 20)
        self.encore_k = config.get("encore_k", 3)
        # base solve budget; probe time is charged PER INSTANCE against this
        # (not cumulatively across the chunk).
        self._base_time_limits = self.time_limits

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        cf = c_nodes.cpu()
        cf[torch.isnan(cf)] = 1
        ei = A._indices()
        ef = build_edge_features(ei, A._values(), use_edge_coeff=self.use_edge_coeff,
                                 num_cons=cf.shape[0])
        batch = torch.zeros(v_nodes.shape[0], dtype=torch.long)
        n_var = len(v_map)

        # (1) probing solve for early solutions
        solver = SOLVER_CLASSES[self.solver]()
        solver.hide_output_to_console()
        solver.load_model(ins_path)
        early_sols, _x_es, t_probe = solver.collect_early_solution(
            time_limit=self.probe_time, threads=self.threads, top_k=self.encore_k)

        probe_names = [solver.varname(v) for v in solver.get_vars()]
        name_to_graph = {n: v_map[n] for n in v_map}

        def to_graph_order(sol):
            d = {n: float(val) for n, val in zip(probe_names, sol)}
            out = np.zeros(n_var, dtype=np.float32)
            for n, gi in name_to_graph.items():
                out[gi] = d.get(n, 0.0)
            return np.round(out)

        early_graph = [to_graph_order(s) for s in early_sols] if early_sols \
            else [np.zeros(n_var, dtype=np.float32)]
        x_es = early_graph[0]

        # (2) sign-aligned logit ensemble over early solutions
        dev = self.device
        agg = None
        for xk in early_graph:
            vf = torch.cat([v_nodes.cpu(), torch.FloatTensor(xk).reshape(-1, 1)], dim=1)
            logit = self.model(cf.to(dev), ei.to(dev), ef.to(dev), vf.to(dev), batch.to(dev))
            if isinstance(logit, tuple):
                logit = logit[0]
            logit = logit.cpu().squeeze()
            tau = torch.FloatTensor(np.where(xk == x_es, 1.0, -1.0))
            agg = tau * logit if agg is None else agg + tau * logit
        pbar = torch.sigmoid(agg / len(early_graph))

        # (3) value-conditioned PS score
        x_es_t = torch.FloatTensor(x_es)
        p = torch.where(x_es_t >= 0.5, pbar, 1.0 - pbar)

        # charge probe time against the solve budget, per instance (not cumulative)
        self.time_limits = max(1, self._base_time_limits - t_probe)

        all_varname = list(v_map)
        binary_name = set(all_varname[i] for i in b_vars)
        scores = []
        for i in range(n_var):
            typ = 'BINARY' if all_varname[i] in binary_name else 'C'
            scores.append([i, all_varname[i], p[i].item(), -1, typ])
        scores.sort(key=lambda x: x[2], reverse=True)
        scores = [x for x in scores if x[4] == 'BINARY']
        return scores
