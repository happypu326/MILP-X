import sys
import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import OmegaConf

from src.learning import build_ps_family_model, select_encoder_kwargs
from src.evaluator.ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import collect_test_instances
from src.utils.parallel_runner import run_parallel_eval, merge_basic_metrics


def build_model(cfg_dict, device):
    # Dispatch on gnn_type so any PS-family encoder (gcn / gasse / rwse /
    # substructure / bipartite_gin / ...) can be evaluated. The encoder kwargs
    # come from the SAME selector used at training time, so the checkpoint
    # architecture matches.
    gnn_type = cfg_dict.get("gnn_type", "gcn")
    model = build_ps_family_model(
        gnn_type,
        emb_size=cfg_dict["emb_size"],
        constraint_nfeats=cfg_dict["constraint_nfeats"],
        edge_nfeats=cfg_dict["edge_nfeats"],
        variable_nfeats=cfg_dict["variable_nfeats"],
        **select_encoder_kwargs(gnn_type, cfg_dict),
    ).to(device)

    if not os.path.exists(cfg_dict["model_path"]):
        raise FileNotFoundError(f"Model file not found: {cfg_dict['model_path']}")

    state_dict = torch.load(cfg_dict["model_path"], map_location=device)
    # strict=False tolerates benign buffer differences (e.g. GINE eps), but a
    # gnn_type / hyper-parameter mismatch would otherwise load garbage silently.
    # Surface any non-trivial key mismatch so a wrong checkpoint is caught.
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        n_params = sum(1 for _ in model.state_dict())
        if len(missing) > 0.2 * n_params or len(unexpected) > 0.2 * n_params:
            raise RuntimeError(
                f"Checkpoint mismatch for gnn_type='{gnn_type}': "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys "
                f"(of {n_params}). Does the checkpoint match gnn_type and its "
                f"encoder hyper-parameters (depth / walk_length / ...)?"
            )
        print(f"[test_ps] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys (tolerated).")
    model.eval()
    return model


def build_eval_config(cfg_dict, device, worker_id, worker_log_root, worker_result_root, worker_scores_root, chunk_size):
    return {
        'method_type': cfg_dict["method_type"],
        'task_name': cfg_dict["test_problem_type"],
        'difficulty': cfg_dict["difficulty"],
        'device': device,
        'solver': cfg_dict["solver"],
        'log_dir': worker_log_root,
        'result_dir': worker_result_root,
        "gnn_type": cfg_dict["gnn_type"],
        "fix_strategy": cfg_dict["fix_strategy"],
        "time_limits": cfg_dict["time_limits"],
        "threads": cfg_dict["threads"],
        "scores_dir": worker_scores_root,
        'time_flag': cfg_dict.get("time_flag", None),
        'save_scores': cfg_dict["save_scores"],
        'hyperparam_pas': cfg_dict["hyperparam_pas"],
        'hyperparam_soft': cfg_dict["hyperparam_soft"],
        'use_edge_coeff': cfg_dict.get("use_edge_coeff", True),
        'test_num': chunk_size,
    }


def main():
    cfg = OmegaConf.load('configs/test/test_ps.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    use_edge_coeff = cfg_dict.get("use_edge_coeff", True)
    expected_edge = 2 if use_edge_coeff else 1
    assert cfg_dict["edge_nfeats"] == expected_edge, (
        f"edge_nfeats ({cfg_dict['edge_nfeats']}) must be {expected_edge} for "
        f"use_edge_coeff={use_edge_coeff}; must match the trained model."
    )

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