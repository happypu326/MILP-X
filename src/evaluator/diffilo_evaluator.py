import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from .base_evaluator import BaseEvaluator
from src.utils.utils import get_diffilo_data_test
from src.solver.solver_utils import SOLVER_CLASSES

class DiffILOEvaluator(BaseEvaluator):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(config)
        self.model = model
        self.solver_name = config.get("solver", "gurobi")
        self.time_limit = config.get("time_limit", 1000)
        self.threads = config.get("threads", 1)
        self.test_num = config.get("test_num", 100)
        self.device = config.get("device", "cuda:0")
        self.instance_dir = config.get("instance_dir", "./instance/test")
        self.mip_focus = config.get("mip_focus", 1)

        self.mu = config.get("mu", 0)
        self.delta = config.get("delta", 200)
         

    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        results = []
        total_time = 0.0

        for idx, ins_path in enumerate(instances):
            ins_name = os.path.basename(ins_path)
            self.logger.info(f"[{idx + 1}/{len(instances)}] Processing {ins_name}")

            try:
                t0 = time.time()
                graph, A, b, c = get_diffilo_data_test(ins_path)

                predict_time = time.time() - t0
                
                with torch.no_grad():
                    model = self.model.to(self.device)
                    logits = model.forward(graph)[0]
                model = model.cpu()

                pred = logits.sigmoid().cpu().numpy()
                x = np.random.binomial(1, pred.squeeze(), size=(1000, len(pred)))
                cons = np.maximum(A @ x.T - b, 0).sum(0)
                idx = np.where(cons == 0)[0]
                if len(idx) > 0:
                    best_idx = np.argmin(x[idx] @ c)
                    best_x = x[idx][best_idx]
                else:
                    best_x = np.argmin((x @ c).squeeze() + self.mu * cons)
                    best_x = x[best_x]
                
                solver = SOLVER_CLASSES[self.solver_name]()
                solver.load_model(ins_path)
                solver.hide_output_to_console()
                solver.model.Params.MIPFocus = self.mip_focus
                
                error = 0
                for i, v in enumerate(solver.get_vars()):
                    v_0 = best_x[i]
                    v.Start = v_0
                    
                    tmp_var = solver.create_real_var(name=f'alp_{v}', lower_bound=0.0, upper_bound=1.0)
                    if v_0 == 0:
                        solver.add_constraint(tmp_var == v, name=f"alpha_eq_{i}")
                    elif v_0 == 1:
                        solver.add_constraint(tmp_var == 1 - v, name=f"alpha_eq_{i}")
                    error += tmp_var
                
                solver.add_constraint(error <= self.delta, name="sum_alpha")
                log_path = os.path.join(self.log_dir, f"{ins_name}.log")
                start = time.time()
                solver_logs = solver.solve(
                    means=self.solver_name,
                    log_file=log_path,
                    time_limit=self.time_limit,
                    threads=self.threads
                )
                solver_time = time.time() - start

                elapsed = predict_time + solver_time
                total_time += elapsed

                obj = solver.get_obj_val()
                status_code = int(solver.model.Status)
                sol_count = int(solver.model.SolCount)

                if sol_count > 0:
                    results.append({
                        "instance": ins_path,
                        "instance_name": ins_name,
                        "predict_time": predict_time,
                        "solver_time": solver_time,
                        "total_time": elapsed,
                        "obj": obj,
                        "node_count": solver.get_node_count(),
                        "gap": float(solver.model.MIPGap),
                        "gurobi_status": status_code,
                        "status": "ok",
                    })
                else:
                    results.append({
                        "instance": ins_path,
                        "instance_name": ins_name,
                        "predict_time": predict_time,
                        "solver_time": solver_time,
                        "total_time": elapsed,
                        "obj": None,
                        "node_count": solver.get_node_count(),
                        "gap": None,
                        "gurobi_status": status_code,
                        "status": "infeasible_or_no_solution",
                    })
                self.logger.info(
                    f"  Done: predict={predict_time:.2f}s, "
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
            "method_name": self.method_type,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "num_instances": len(instances),
            "num_success": num_ok,
            "avg_time": avg_time,
            "total_time": total_time,
            "instances": results,
        }

        result_file = os.path.join(self.result_dir, "ps_evaluation_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {result_file}")

        return metrics
