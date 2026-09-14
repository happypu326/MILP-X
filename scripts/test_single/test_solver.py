import os
import sys
import argparse

from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluator.solver_evaluator import SolverEvaluator
from src.utils.utils import collect_test_instances


def main():
    cfg = OmegaConf.load('configs/test/test_solver.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    instances = collect_test_instances(cfg.input_dir, cfg.problem_type, cfg.difficulty)
    if not instances:
        raise FileNotFoundError(
            f"No .lp/.mps/.cip instances found under {cfg.input_dir}/{cfg.problem_type}/{cfg.difficulty}"
        )

    print(f"Found {len(instances)} test instances (dir: {cfg.input_dir}/{cfg.problem_type}/{cfg.difficulty})")

    evaluator_cfg = {
        "solver": cfg.solver,
        "max_time": cfg.max_time,
        "threads": cfg.threads,
        "test_num": cfg.test_num,
        "task_name": cfg.problem_type,
        "difficulty": cfg.difficulty,
        "log_dir": cfg.log_dir,
        "result_dir": cfg.result_dir,
        "method_type": cfg.solver,
        "num_workers": cfg.num_workers,
    }

    evaluator = SolverEvaluator(evaluator_cfg)
    metrics = evaluator.evaluate(instances)

    print("\n===== Solver Evaluation Done =====")
    print(f"method_name : {metrics['method_name']}")
    print(f"task_name   : {metrics['task_name']}")
    print(f"difficulty  : {metrics['difficulty']}")
    print(f"num_success : {metrics['num_success']}/{metrics['num_instances']}")
    print(f"avg_time    : {metrics['avg_time']:.4f}")
    print(f"total_time  : {metrics['total_time']:.4f}")


if __name__ == "__main__":
    main()