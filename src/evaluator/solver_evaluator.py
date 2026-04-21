import os
import time
import json
import importlib.util
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator
from src.solver.solver_utils import SOLVER_CLASSES

class SolverEvaluator(BaseEvaluator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.solver_name = config.get("solver", "gurobi")
        self.max_time = config.get("max_time", 1000)
        self.threads = config.get("threads", 1)
        self.test_num = config.get("test_num", 100)

    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        total_time = 0.0

        instances_to_test = instances[:self.test_num]

        for idx, ins_path in enumerate(instances_to_test):
            ins_name = os.path.basename(ins_path) if isinstance(ins_path, str) else str(ins_path)
            self.logger.info(f"[{idx + 1}/{len(instances_to_test)}] Solving {ins_name}")

            try:
                solver = SOLVER_CLASSES[self.solver_name]()
                solver.hide_output_to_console()
                solver.load_model(ins_path)
                solver.set_aggressive()

                log_path = os.path.join(self.log_dir, f"{ins_name}.log")

                start = time.time()
                solver.solve(
                    means=self.solver_name,
                    log_file=log_path,
                    time_limit=self.max_time,
                    threads=self.threads,
                )
                elapsed = time.time() - start
                total_time += elapsed

                results.append({
                    "instance": ins_path,
                    "instance_name": ins_name,
                    "solver_time": elapsed,
                    "status": "ok",
                })
                self.logger.info(f"  Done in {elapsed:.2f}s")

            except Exception as e:
                self.logger.error(f"  Error solving {ins_name}: {e}")
                results.append({
                    "instance": ins_path,
                    "instance_name": ins_name,
                    "status": "error",
                    "error": str(e),
                })

        num_ok = sum(1 for r in results if r["status"] == "ok")
        avg_time = total_time / max(num_ok, 1)

        metrics = {
            "method_name": f"Solver_{self.solver_name}",
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "num_instances": len(instances_to_test),
            "num_success": num_ok,
            "avg_time": avg_time,
            "total_time": total_time,
            "instances": results,
        }

        result_file = os.path.join(self.result_dir, "solver_evaluation_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {result_file}")

        return metrics
