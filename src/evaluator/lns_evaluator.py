"""
Learned Large-Neighborhood-Search (LNS) harness for MILP.

Covers the destroy-repair family (CL-LNS 2302.01578, LNS-policy 2111.03466,
SPL-LNS 2508.16171, Sonnerat et al. 2107.10201): start from an incumbent, then
repeatedly *destroy* (unfix) a subset of variables, *repair* by re-optimizing
the residual sub-MIP with the solver, and keep the best incumbent.

Neighborhood selection (`lns_mode`):
  * 'random'     -- destroy a random subset (classic LNS baseline).
  * 'prediction' -- learning-guided: destroy the variables where a trained
                    PS-family predictor most disagrees with the incumbent
                    (|p_j - incumbent_j| largest); the solver then re-decides
                    exactly the variables the model is unsure the incumbent got
                    right. Confident agreements stay fixed, shrinking the sub-MIP.

Requires the Gurobi wrapper (copy/reload + per-variable bound setting).
"""

import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any

from .base_evaluator import BaseEvaluator
from src.utils.utils import get_a_new2, build_edge_features, get_initial_best_obj
from src.solver.solver_utils import SOLVER_CLASSES

_MAXIMIZE_TASKS = {"MIS", "CA", "IS"}


class LNSEvaluator(BaseEvaluator):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(config)
        self.model = model
        self.device = config.get("device", "cuda:0")
        self.solver_name = config.get("solver", "gurobi")
        self.gnn_type = config.get("gnn_type", "gcn")
        self.lns_mode = config.get("lns_mode", "prediction")   # 'random' | 'prediction'
        self.n_iters = config.get("n_iters", 5)
        self.destroy_frac = config.get("destroy_frac", 0.3)
        self.init_time = config.get("init_time", 30)
        self.iter_time = config.get("iter_time", 20)
        self.threads = config.get("threads", 1)
        self.use_edge_coeff = config.get("use_edge_coeff", True)
        self.problem = config.get("problem", config.get("task_name", "MVC"))
        self.maximize = config.get("maximize", self.problem in _MAXIMIZE_TASKS)
        self.seed = config.get("seed", 0)

    # ---- predictor marginals (for learning-guided neighborhood) ----------- #
    @torch.no_grad()
    def _predict_marginals(self, ins_path: str) -> Dict[str, float]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)
        cons_f = c_nodes.cpu(); cons_f[torch.isnan(cons_f)] = 1
        edge_idx = A._indices()
        edge_feat = build_edge_features(edge_idx, A._values(),
                                        use_edge_coeff=self.use_edge_coeff,
                                        num_cons=cons_f.shape[0])
        out = self.model(
            cons_f.to(self.device), edge_idx.to(self.device), edge_feat.to(self.device),
            v_nodes.to(self.device), torch.zeros(v_nodes.shape[0], dtype=torch.long, device=self.device),
        )
        if isinstance(out, tuple):
            out = out[0]
        prob = out.sigmoid().cpu().squeeze()
        names = list(v_map.keys())
        return {names[i]: float(prob[i]) for i in b_vars.tolist()}

    # ---- solver helpers --------------------------------------------------- #
    def _solve(self, solver, time_limit, log_path):
        # log_file='' skips the (optional) gurobi_logtools parsing; LNS tracks
        # the objective trajectory itself and does not need the parsed logs.
        solver.solve(means=self.solver_name, log_file='',
                     time_limit=time_limit, threads=self.threads)
        obj = solver.get_obj_val()
        vals = {}
        for v in solver.get_vars():
            try:
                vals[solver.varname(v)] = solver.varval(v)
            except Exception:
                pass
        return obj, vals

    def _better(self, a, b):
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self.maximize else a < b

    # ---- main LNS loop ---------------------------------------------------- #
    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        results = []
        rng = np.random.default_rng(self.seed)

        for idx, ins_path in enumerate(instances):
            ins_name = os.path.basename(ins_path)
            self.logger.info(f"[{idx+1}/{len(instances)}] LNS ({self.lns_mode}) {ins_name}")
            try:
                marg = self._predict_marginals(ins_path) if self.lns_mode == "prediction" else {}

                # --- initial incumbent ---
                solver = SOLVER_CLASSES[self.solver_name]()
                solver.hide_output_to_console(); solver.load_model(ins_path)
                bin_names = [solver.varname(v) for v in solver.get_vars()
                             if str(getattr(v, "VType", "")) == "B"]
                t0 = time.time()
                best_obj, incumbent = self._solve(
                    solver, self.init_time, os.path.join(self.log_dir, f"{ins_name}.init.log"))
                traj = [best_obj]
                total_time = time.time() - t0

                nfix = max(0, int(round((1 - self.destroy_frac) * len(bin_names))))
                for it in range(self.n_iters):
                    if not incumbent:
                        break
                    # priority to DESTROY (higher = more likely unfixed)
                    if self.lns_mode == "prediction":
                        pri = {nm: abs(marg.get(nm, 0.5) - incumbent.get(nm, 0.0))
                               for nm in bin_names}
                    else:
                        pri = {nm: float(rng.random()) for nm in bin_names}
                    # keep-fixed = the lowest-priority nfix variables
                    keep = sorted(bin_names, key=lambda nm: pri[nm])[:nfix]

                    s = SOLVER_CLASSES[self.solver_name]()
                    s.hide_output_to_console(); s.load_model(ins_path)
                    for nm in keep:
                        var = s.get_var_by_name(nm)
                        if var is None:
                            continue
                        val = round(incumbent.get(nm, 0.0))
                        var.lb = val; var.ub = val
                    t1 = time.time()
                    obj, vals = self._solve(
                        s, self.iter_time, os.path.join(self.log_dir, f"{ins_name}.it{it}.log"))
                    total_time += time.time() - t1
                    if self._better(obj, best_obj):
                        best_obj, incumbent = obj, vals
                    traj.append(best_obj)

                results.append({
                    "instance": ins_name, "status": "ok",
                    "best_obj": None if best_obj is None else float(best_obj),
                    "obj_trajectory": [None if o is None else float(o) for o in traj],
                    "total_time": total_time, "n_iters": self.n_iters,
                    "lns_mode": self.lns_mode,
                })
                self.logger.info(f"  best_obj={best_obj} time={total_time:.1f}s")
            except Exception as e:
                self.logger.error(f"  Error: {e}")
                results.append({"instance": ins_name, "status": "error", "error": str(e)})

        ok = [r for r in results if r["status"] == "ok" and r["best_obj"] is not None]
        total_time = float(sum(r.get("total_time", 0.0) for r in results))
        metrics = {
            "method": "LNS", "method_type": self.method_type,
            "task_name": self.task_name, "difficulty": self.difficulty,
            "lns_mode": self.lns_mode,
            "num_instances": len(instances), "num_success": len(ok),
            "avg_best_obj": float(np.mean([r["best_obj"] for r in ok])) if ok else None,
            "avg_time": float(np.mean([r["total_time"] for r in ok])) if ok else None,
            "total_time": total_time,
            "instances": results,   # alias for merge_basic_metrics
            "results": results,
        }
        os.makedirs(self.result_dir, exist_ok=True)
        with open(os.path.join(self.result_dir, "lns_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
