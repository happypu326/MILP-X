import os
import time
import json
import torch
import gurobipy
from gurobipy import GRB
from typing import List, Dict, Any
import numpy as np
from .base_evaluator import BaseEvaluator
from src.utils.utils import get_a_new2, get_initial_best_obj, build_edge_features
from src.solver.solver_utils import SOLVER_CLASSES
import datetime


class ApolloEvaluator(BaseEvaluator):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(config)
        self.model = model
        self.device = config.get("device", "cuda:0")
        self.solver_name = config.get("solver", "gurobi")
        self.time_limits = config.get("time_limits", [100, 100, 200, 600])
        self.k0_list = config.get("k0_list", [100, 40, 20, 1])
        self.k1_list = config.get("k1_list", [20, 15, 15, 5])
        self.delta_list = config.get("delta_list", [50, 50, 30, 10])
        self.threads = config.get("threads", 1)
        self.mip_focus = config.get("mip_focus", 1)
        self.problem = config.get("problem", "MIS")
        # Must match training-time setting (see build_edge_features).
        self.use_edge_coeff = config.get("use_edge_coeff", True)

    @torch.no_grad()
    def _get_scores(self, ins_path: str, fixing_status: Dict):
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_path)

        constraint_features = c_nodes.cpu()
        constraint_features[torch.isnan(constraint_features)] = 1
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices,
            A._values(),
            use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0],
        )

        BD = self.model(
            constraint_features.to(self.device),
            edge_indices.to(self.device),
            edge_features.to(self.device),
            v_nodes.to(self.device),
        ).sigmoid().cpu().squeeze()

        all_varname=[]
        for name in v_map:
            all_varname.append(name)
            fixing_status[name] = -1
        binary_name=[all_varname[i] for i in b_vars]
        scores=[] # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type="C"
            if all_varname[i] in binary_name:
                type='BINARY'
                fixing_status[all_varname[i]] = 2
            scores.append([i, all_varname[i], BD[i].item(), -1, type])
        
        scores.sort(key=lambda x:x[2],reverse=True)
        scores=[x for x in scores if x[4]=='BINARY'] # get binary
        return scores, fixing_status

    def _fix_variables(self, scores: List, fixing_status: Dict, k0: int, k1: int):
        count1=0
        for i in range(len(scores)):
            if count1<k1:
                scores[i][3] = 1
                count1+=1
                fixing_status[scores[i][1]] = 1

        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < k0:
                scores[i][3] = 0
                count0 += 1
                fixing_status[scores[i][1]] = 1

        return scores, fixing_status

    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        results = []
        obj_list = []
        time_list = []
        node_list = []
        best_obj_list = []

        for idx, ins_path in enumerate(instances):
            ins_name = os.path.basename(ins_path)
            os.makedirs(f'{self.log_dir}/{ins_name}', exist_ok=True)
            self.logger.info(f"[{idx+1}/{len(instances)}] {ins_name}")

            best_obj = get_initial_best_obj(self.problem)
            instance_obj_list = []
            instance_time_list = []
            instance_node_list = []

            try:
                fixing_status = {}
                total_time = 0
                current_ins = ins_path

                for step in range(len(self.time_limits)):
                    scores, fixing_status = self._get_scores(current_ins, fixing_status)
                    scores, fixing_status = self._fix_variables(
                        scores, fixing_status, self.k0_list[step], self.k1_list[step]
                    )

                    solver = SOLVER_CLASSES[self.solver_name]()
                    solver.hide_output_to_console()
                    solver.load_model(current_ins)

                    solver_ps = solver.copy_model()
                    
                    instance_variables = solver_ps.get_vars()
                    instance_variables.sort(key=lambda v: solver_ps.varname(v))
                    variables_map = {solver_ps.varname(v): v for v in instance_variables}

                    alphas = []

                    for i in range(len(scores)):
                        tar_var = variables_map[scores[i][1]]  # target variable <-- variable map
                        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
                        if x_star < 0 or fixing_status[scores[i][1]] != 2:
                            continue
                        
                        tmp_var = solver_ps.create_real_var(
                            name=f'alp_{tar_var}',
                            lower_bound=0.0,
                            upper_bound=GRB.INFINITY
                        )
                        alphas.append(tmp_var)
                        solver_ps.add_constraint(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
                        solver_ps.add_constraint(tmp_var >= x_star - tar_var, f'alpha_down_{i}')

                    all_tmp = 0
                    for tmp in alphas:
                        all_tmp += tmp
                    solver_ps.add_constraint(all_tmp <= self.delta_list[step], name="sum_alpha")

                    t0 = time.time()
                    solver_logs =solver_ps.solve(
                        means=f"Apollo_{self.solver_name}",
                        log_file=f'{self.log_dir}/{ins_name}/{step}.log',
                        time_limit=self.time_limits[step],
                        threads=self.threads,
                    )
                    total_time += time.time() - t0

                    scores.sort(key=lambda x: x[2], reverse=False)
            
                    counting = 0
                    fix_count = 0
                    for i in range(len(scores)):
                        if scores[i][3] != 0:
                            continue
                        
                        if fixing_status[scores[i][1]] == 1 and scores[i][3] == 0:
                            var = solver.get_var_by_name(scores[i][1])
                            if scores[i][3] == solver_ps.get_var_by_name(scores[i][1]).x:
                                fixing_status[scores[i][1]] = 0
                                var.lb = solver_ps.get_var_by_name(scores[i][1]).x
                                var.ub = solver_ps.get_var_by_name(scores[i][1]).x
                                fix_count += 1
                            else:
                                fixing_status[scores[i][1]] = 2
                        counting += 1

                        if counting >= self.k0_list[step]:
                            break

                    counting = 0
                    fix_count = 0
                    scores.sort(key=lambda x: x[2], reverse=True)
                    for i in range(len(scores)):
                        if scores[i][3] != 1:
                            continue
                        
                        # for the vars prefixed to be 0
                        if fixing_status[scores[i][1]] == 1 and scores[i][3] == 1:
                            var = solver.get_var_by_name(scores[i][1])
                            if scores[i][3] == solver_ps.get_var_by_name(scores[i][1]).x:
                                fixing_status[scores[i][1]] = 0
                                var.lb = solver_ps.get_var_by_name(scores[i][1]).x
                                var.ub = solver_ps.get_var_by_name(scores[i][1]).x
                                fix_count += 1
                            else:
                                fixing_status[scores[i][1]] = 2
                        counting += 1

                        if counting >= self.k1_list[step]:
                            break

                    temp_file = os.path.join(self.log_dir, ins_name, f"{ins_name}_step{step}.lp")
                    solver.write_model(temp_file)
                    current_ins = temp_file

                    if self.problem == 'MIS' or self.problem == 'CA':
                        if solver_ps.get_obj_val() > best_obj:
                            best_obj = solver_ps.get_obj_val()
                    else:
                        if solver_ps.get_obj_val() < best_obj:
                            best_obj = solver_ps.get_obj_val()

                    step_obj = solver_ps.get_obj_val()
                    step_time = total_time
                    step_node = solver_ps.get_node_count()

                    instance_obj_list.append(step_obj)
                    instance_time_list.append(step_time)
                    instance_node_list.append(step_node)

                    obj_list.append(step_obj)
                    time_list.append(step_time)
                    node_list.append(step_node)

                best_obj_list.append(best_obj)
                results.append({
                    "instance": ins_name,
                    "total_time": total_time,
                    "best_obj": float(best_obj),
                    "mean_obj": float(np.mean(instance_obj_list)),
                    "mean_time": float(np.mean(instance_time_list)),
                    "mean_node": float(np.mean(instance_node_list)),
                    "status": "ok"
                })

            except Exception as e:
                self.logger.error(f"Error: {e}")
                results.append({"instance": ins_name, "status": "error", "error": str(e)})

        metrics = {
            "method": "Apollo",
            "num_instances": len(instances),
            "num_success": sum(1 for r in results if r["status"] == "ok"),
            "avg_obj": float(np.mean(obj_list)) if obj_list else None,
            "avg_time": float(np.mean(time_list)) if time_list else None,
            "avg_node": float(np.mean(node_list)) if node_list else None,
            "avg_best_obj": float(np.mean(best_obj_list)) if best_obj_list else None,
            "hyperparams": {
                "k0": list(self.k0_list),
                "k1": list(self.k1_list),
                "delta": list(self.delta_list),
                "time_limits": list(self.time_limits)
            },
            "results": results
        }

        os.makedirs(self.result_dir, exist_ok=True)
        with open(os.path.join(self.result_dir, "apollo_results.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
