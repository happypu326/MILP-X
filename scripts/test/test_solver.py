import os
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluator.solver_evaluator import SolverEvaluator
from src.utils.utils import collect_test_instances
from src.utils.parallel_runner import run_parallel_eval, merge_basic_metrics


def build_eval_config(cfg_dict, device, worker_id, worker_log_root, worker_result_root, worker_scores_root, chunk_size):
    return {
        "solver": cfg_dict["solver"],
        "max_time": cfg_dict["max_time"],
        "threads": cfg_dict["threads"],
        "test_num": chunk_size,
        "task_name": cfg_dict["problem_type"],
        "difficulty": cfg_dict["difficulty"],
        "log_dir": worker_log_root,
        "result_dir": worker_result_root,
        "method_type": cfg_dict["solver"],
        "time_flag": cfg_dict.get("time_flag", None),
    }


def main():
    cfg = OmegaConf.load('configs/test/test_solver.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    instances = collect_test_instances(cfg.input_dir, cfg.problem_type, cfg.difficulty)
    if not instances:
        raise FileNotFoundError(
            f"No .lp/.mps/.cip instances found under {cfg.input_dir}/{cfg.problem_type}/{cfg.difficulty}"
        )

    run_parallel_eval(
        cfg_dict=cfg_dict,
        instances=instances,
        evaluator_cls=SolverEvaluator,
        build_model_fn=None,                        
        build_eval_config_fn=build_eval_config,
        merge_fn=merge_basic_metrics,
    )


if __name__ == "__main__":
    main()