import sys, os, random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import OmegaConf

from src.learning import build_ps_family_model
from src.evaluator.lns_evaluator import LNSEvaluator
from src.utils.utils import collect_test_instances
from src.utils.parallel_runner import run_parallel_eval, merge_basic_metrics


def build_model(cfg_dict, device):
    # Predictor used by lns_mode in {prediction, cllns}. For random mode (or when
    # no checkpoint is available) an untrained encoder is returned and never used.
    model = build_ps_family_model(
        cfg_dict["gnn_type"], emb_size=cfg_dict["emb_size"],
        constraint_nfeats=cfg_dict["constraint_nfeats"],
        edge_nfeats=cfg_dict["edge_nfeats"], variable_nfeats=cfg_dict["variable_nfeats"],
    ).to(device)
    if cfg_dict.get("lns_mode", "prediction") in ("prediction", "cllns") \
            and os.path.exists(cfg_dict["model_path"]):
        model.load_state_dict(torch.load(cfg_dict["model_path"], map_location=device), strict=False)
    model.eval()
    return model


def build_eval_config(cfg_dict, device, worker_id, worker_log_root, worker_result_root, worker_scores_root, chunk_size):
    return {
        'method_type': cfg_dict["method_type"], 'task_name': cfg_dict["test_problem_type"],
        'difficulty': cfg_dict["difficulty"], 'device': device, 'solver': cfg_dict["solver"],
        'log_dir': worker_log_root, 'result_dir': worker_result_root,
        'gnn_type': cfg_dict["gnn_type"], 'use_edge_coeff': cfg_dict.get("use_edge_coeff", True),
        'lns_mode': cfg_dict["lns_mode"], 'n_iters': cfg_dict["n_iters"],
        'destroy_frac': cfg_dict["destroy_frac"], 'init_time': cfg_dict["init_time"],
        'iter_time': cfg_dict["iter_time"], 'maximize': cfg_dict.get("maximize", False),
        'threads': cfg_dict["threads"], 'problem': cfg_dict["test_problem_type"],
        'seed': cfg_dict.get("seed", 0) + worker_id, 'test_num': chunk_size,
        # CL-LNS destroy-policy options (used only by lns_mode=cllns)
        'window': cfg_dict.get("window", 3),
        'adaptive_gamma': cfg_dict.get("adaptive_gamma", 1.02),
        'adaptive_beta': cfg_dict.get("adaptive_beta", 0.5),
    }


def main():
    cfg = OmegaConf.load('configs/test/test_lns.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)
    run_parallel_eval(cfg_dict=cfg_dict, instances=instances, evaluator_cls=LNSEvaluator,
                      build_model_fn=build_model, build_eval_config_fn=build_eval_config,
                      merge_fn=merge_basic_metrics)


if __name__ == '__main__':
    main()
