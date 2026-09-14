import os
import pickle
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from .base_evaluator import BaseEvaluator
from src.utils.utils import get_a_new2, build_edge_features
from src.solver.solver_utils import SOLVER_CLASSES
from src.learning.model.bipartite_encoders import GNN_REGISTRY

# encoders that share the plain PS forward signature
# (constraint_features, edge_indices, edge_features, variable_features,
#  batch_indices) -> per-variable logits. Registry-driven so additional
# bipartite encoders (bipartite_gin / rwse / substructure / ...) work without
# editing this file.
PLAIN_FORWARD_GNN_TYPES = ("gcn",) + tuple(GNN_REGISTRY)

class PSFamilyEvaluator(BaseEvaluator):
    """
    Predict-and-Search family of evaluators
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(config)
        self.model = model
        self.gnn_type = config.get("gnn_type", "gcn")
        self.solver = config.get("solver", "gurobi")
        self.fix_strategy = config.get("fix_strategy", "pas")
        self.time_limits = config.get("time_limits", 1000)
        self.threads = config.get("threads", 1)
        self.test_num = config.get("test_num", 100)
        self.device = config.get("device", "cuda:0")
        self.scores_dir = config.get("scores_dir", None)
        self.save_scores = config.get("save_scores", True)
        self.use_constraint_norm = config.get("use_constraint_norm", False)
        self.depth = config.get("depth", 1)
        # Must match the setting used at training time. True -> edges carry
        # [normalized a_ij, sign(a_ij)] (edge_nfeats=2); False -> legacy edge=1.
        self.use_edge_coeff = config.get("use_edge_coeff", True)
        # Neural-Diving-style coverage hard-fixing (fix_strategy='dive').
        self.coverage = config.get("coverage", 0.5)
        self.dive_delta = config.get("dive_delta", 0)

        self.hyperparam_pas = config.get("hyperparam_pas", {
            "IP": (60, 30, 70), "MIS": (600, 600, 100), "WA": (20, 200, 100),
            "CA": (150, 0, 25), "SC": (300, 0, 100), "BP": (100, 0, 50),
            "CFLP": (30, 0, 30), "GISP": (1000, 100, 120), "LB": (300, 100, 150),
            "MVC": (40, 0, 30), "MIRP": (120, 30, 100), "MMCN": (200, 0, 150),
            "NNV": (10, 25, 6), "OTS": (300, 0, 200), "SRPN": (150, 150, 200),
            "IS": (600, 600, 100),
        })

        self.hyperparam_soft = config.get("hyperparam_soft", {
            "CA": (0.99, 0.12), "IP": (0.93, 0.01), "SC": (0.96, 0.02),
            "MIS": (0.95, 0.05), "CFLP": (0.98, 0.02), "GISP": (0.95, 0.05),
            "LB": (0.99, 0.15), "MVC": (0.97, 0.02), "BP": (0.95, 0.05),
            "MIRP": (0.99, 0.20), "MMCN": (0.97, 0.15), "NNV": (0.97, 0.30),
            "OTS": (0.97, 0.15), "SRPN": (0.97, 0.15), "IIS": (0.97, 0.02),
            "IS": (0.95, 0.05),
        })

    @torch.no_grad()
    def _predict_scores(self, ins_path: str) -> List[list]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)

        constraint_features = c_nodes.cpu()
        mask = torch.isnan(constraint_features)
        constraint_features[mask] = 1

        variable_features = v_nodes
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices,
            A._values(),
            use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0],
        )
        batch_indices = torch.zeros(v_nodes.shape[0], dtype=torch.long)

        if self.gnn_type in PLAIN_FORWARD_GNN_TYPES:
            BD = self.model(
                constraint_features.to(self.device),
                edge_indices.to(self.device),
                edge_features.to(self.device),
                variable_features.to(self.device),
                batch_indices.to(self.device),
            )
            if isinstance(BD, tuple):
                BD = BD[0]
            BD = BD.sigmoid().cpu().squeeze()
        elif self.gnn_type == 'moe':
            BD, _ = self.model(
                constraint_features.to(self.device),
                edge_indices.to(self.device),
                edge_features.to(self.device),
                variable_features.to(self.device),
                batch_indices.to(self.device),
            )
            BD = BD.sigmoid().cpu().squeeze()
        elif self.gnn_type == 'coco':
            constraint_features_batch = torch.tensor([0]*len(constraint_features)).to(self.device)
            variable_features_batch = torch.tensor([0]*len(variable_features)).to(self.device)
            BD = self.model(
                constraint_features.to(self.device),
                edge_indices.to(self.device),
                edge_features.to(self.device),
                variable_features.to(self.device),
                torch.tensor([constraint_features.shape[0]]).to(self.device),
                constraint_features_batch.to(self.device),
                variable_features_batch.to(self.device)
            )
            BD = BD.sigmoid().cpu().squeeze()
        else:
            raise ValueError(
                f"Unsupported gnn_type '{self.gnn_type}' for PSFamilyEvaluator")

        all_varname=[]
        for name in v_map:
            all_varname.append(name)
        binary_name=set(all_varname[i] for i in b_vars)
        scores=[]
        for i in range(len(v_map)):
            type="C"
            if all_varname[i] in binary_name:
                type='BINARY'
            scores.append([i, all_varname[i], BD[i].item(), -1, type])

        scores.sort(key=lambda x:x[2],reverse=True)

        scores=[x for x in scores if x[4]=='BINARY']
        return scores

    def _fix_pas(self, scores: List[list], task: str) -> Tuple[List[list], int]:
        if task not in self.hyperparam_pas:
            self.logger.warning(f"Task '{task}' not in hyperparam_pas, using default (0, 0, 0)")
            k0_raw, k1_raw, delta_raw = 0, 0, 0
        else:
            k0_raw, k1_raw, delta_raw = self.hyperparam_pas[task]

        N = len(scores)

        def compute_k(k_raw):
            if isinstance(k_raw, float) and 0 < k_raw < 1:
                return int(k_raw * N)
            return int(k_raw)

        k0 = min(compute_k(k0_raw), N)
        k1 = min(compute_k(k1_raw), N)
        delta = min(compute_k(delta_raw), N)

        scores.sort(key=lambda x: x[2], reverse=True)
        for i in range(min(len(scores), k1)):
            scores[i][3] = 1

        scores.sort(key=lambda x: x[2], reverse=False)
        for i in range(min(len(scores), k0)):
            scores[i][3] = 0

        return scores, delta

    def _fix_soft_confidence(self, scores: List[list], task: str) -> Tuple[List[list], float]:
        if task not in self.hyperparam_soft:
            self.logger.warning(f"Task '{task}' not in hyperparam_soft, using default (0.95, 0.05)")
            threshold, alpha = 0.95, 0.05
        else:
            threshold, alpha = self.hyperparam_soft[task]

        tot_fix = 0
        for i in range(len(scores)):
            if scores[i][2] > threshold:
                scores[i][3] = 1
                tot_fix += 1
        for i in range(len(scores)):
            if scores[i][2] < 1 - threshold:
                scores[i][3] = 0
                tot_fix += 1

        return scores, tot_fix * alpha

    def _fix_dive(self, scores: List[list], task: str) -> Tuple[List[list], float]:
        """
        Neural-Diving-style coverage fixing: fix the most 'reliable' fraction of
        binary variables to their rounded prediction and leave the rest free.

        Reliability priority is a learned selection gate if present (scores[i][5],
        set by NeuralDivingEvaluator) else prediction confidence |p - 0.5|.
        delta = self.dive_delta (0 -> hard fixing, >0 -> soft trust region on the
        fixed set).
        """
        N = len(scores)
        n_fix = min(N, int(round(self.coverage * N)))

        def priority(s):
            return s[5] if len(s) > 5 else abs(s[2] - 0.5)

        order = sorted(range(N), key=lambda i: priority(scores[i]), reverse=True)
        for rank, i in enumerate(order):
            if rank < n_fix:
                scores[i][3] = 1 if scores[i][2] >= 0.5 else 0
            else:
                scores[i][3] = -1
        return scores, float(self.dive_delta)

    def _solve_with_trust_region(
        self, ins_path: str, scores: List[list], delta: float, log_path: str
    ) -> Dict[str, Any]:
        solver = SOLVER_CLASSES[self.solver]()
        solver.hide_output_to_console()
        solver.load_model(ins_path)
        solver.set_aggressive()

        instance_variables = solver.get_vars()
        instance_variables.sort(key=lambda v: solver.varname(v))
        variables_map = {solver.varname(v): v for v in instance_variables}

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

        if len(alphas) > 0:
            all_tmp = 0
            for tmp in alphas:
                all_tmp += tmp
            solver.add_constraint(all_tmp <= delta, name="sum_alpha")

        start = time.time()
        results = solver.solve(
            means=self.solver,
            log_file=log_path,
            time_limit=self.time_limits,
            threads=self.threads
        )
        solver_time = time.time() - start

        return {
            "solver_time": solver_time,
            "results": results,
        }

    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        results = []
        total_time = 0.0

        task = self.task_name

        for idx, ins_path in enumerate(instances):
            ins_name = os.path.basename(ins_path)
            self.logger.info(f"[{idx + 1}/{len(instances)}] Processing {ins_name}")

            try:
                t0 = time.time()
                scores = self._predict_scores(ins_path)
                predict_time = time.time() - t0

                if self.save_scores and self.scores_dir:
                    os.makedirs(self.scores_dir, exist_ok=True)
                    score_path = os.path.join(
                        self.scores_dir, f"scores_{os.path.splitext(ins_name)[0]}.pkl"
                    )
                    with open(score_path, 'wb') as f:
                        pickle.dump(scores, f)

                t1 = time.time()
                if self.fix_strategy == 'pas':
                    scores, delta = self._fix_pas(scores, task)
                elif self.fix_strategy == 'dive':
                    scores, delta = self._fix_dive(scores, task)
                else:
                    scores, delta = self._fix_soft_confidence(scores, task)
                fix_time = time.time() - t1

                log_path = os.path.join(self.log_dir, f"{ins_name}.log")
                solve_result = self._solve_with_trust_region(ins_path, scores, delta, log_path)
                solver_time = solve_result["solver_time"]

                elapsed = predict_time + fix_time + solver_time
                total_time += elapsed

                results.append({
                    "instance": ins_path,
                    "instance_name": ins_name,
                    "predict_time": predict_time,
                    "fix_time": fix_time,
                    "solver_time": solver_time,
                    "total_time": elapsed,
                    "status": "ok",
                })
                self.logger.info(
                    f"  Done: predict={predict_time:.2f}s, fix={fix_time:.2f}s, "
                    f"solve={solver_time:.2f}s"
                )

            except Exception as e:
                self.logger.error(f"  Error processing {ins_name}: {e}")
                results.append({
                    "instance": ins_path,
                    "instance_name": ins_name,
                    "status": "error",
                    "error": str(e),
                })

        num_ok = sum(1 for r in results if r["status"] == "ok")
        avg_time = total_time / max(num_ok, 1)

        metrics = {
            "method_type": self.method_type,
            "task_name": task,
            "difficulty": self.difficulty,
            "num_instances": len(instances),
            "num_success": num_ok,
            "avg_time": avg_time,
            "total_time": total_time,
            "instances": results,
        }

        return metrics
