import sys
import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import OmegaConf

from src.learning import MoEPolicy
from src.evaluator.ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import collect_test_instances
from src.utils.parallel_runner import run_parallel_eval, merge_basic_metrics


def build_model(cfg_dict, device):
    model = MoEPolicy(
        emb_size=cfg_dict["emb_size"],
        constraint_nfeats=cfg_dict["constraint_nfeats"],
        edge_nfeats=cfg_dict["edge_nfeats"],
        variable_nfeats=cfg_dict["variable_nfeats"],
        num_shared_experts=cfg_dict["num_shared_experts"],
        num_dedicate_experts=cfg_dict["num_dedicate_experts"],
        top_k=cfg_dict["top_k"],
        gate_temperature=cfg_dict["gate_temperature"],
        bias_lr=cfg_dict["bias_lr"],
        dropout=0.1,
        use_dro=cfg_dict["use_dro"],
        eps_wasserstein=cfg_dict["eps_wasserstein"],
        dro_perturb_type=cfg_dict["dro_perturb_type"]
    ).to(device)

    if not os.path.exists(cfg_dict["model_path"]):
        raise FileNotFoundError(f"模型文件未找到: {cfg_dict['model_path']}")

    state_dict = torch.load(cfg_dict["model_path"], map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_eval_config(cfg_dict, device, worker_id, worker_log_root, worker_result_root, worker_scores_root, chunk_size):
    return {
        'method_type': cfg_dict["method_type"],
        "task_name": cfg_dict["test_problem_type"],
        "difficulty": cfg_dict["difficulty"],
        'device': device,
        'log_dir': worker_log_root,
        'result_dir': worker_result_root,
        "gnn_type": cfg_dict["gnn_type"],
        "solver": cfg_dict["solver"],
        "fix_strategy": cfg_dict["fix_strategy"],
        "time_limits": cfg_dict["time_limits"],
        "threads": cfg_dict["threads"],
        'time_flag': cfg_dict.get("time_flag", None),
        "save_scores": cfg_dict["save_scores"],
        "scores_dir": worker_scores_root,
        'hyperparam_pas': cfg_dict["hyperparam_pas"],
        'hyperparam_soft': cfg_dict["hyperparam_soft"],
        'test_num': chunk_size,
    }


def main():
    cfg = OmegaConf.load('configs/test/test_rome.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)

    run_parallel_eval(
        cfg_dict=cfg_dict,
        instances=instances,
        evaluator_cls=PSFamilyEvaluator,
        build_model_fn=build_model,
        build_eval_config_fn=build_eval_config,
        merge_fn=merge_basic_metrics,
    )


if __name__ == '__main__':
    main()