import sys, os, random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import OmegaConf

from src.learning.model.shsp import SHSPPolicy
from src.evaluator.shsp_evaluator import SHSPEvaluator
from src.utils.utils import collect_test_instances
from src.utils.parallel_runner import run_parallel_eval, merge_basic_metrics


def build_model(cfg_dict, device):
    model = SHSPPolicy(
        emb_size=cfg_dict["emb_size"], constraint_nfeats=cfg_dict["constraint_nfeats"],
        edge_nfeats=cfg_dict["edge_nfeats"], variable_nfeats=cfg_dict["variable_nfeats"],
        depth=cfg_dict.get("depth", 4), jumping_knowledge=cfg_dict.get("jumping_knowledge", True),
    ).to(device)
    if not os.path.exists(cfg_dict["model_path"]):
        raise FileNotFoundError(f"模型文件未找到: {cfg_dict['model_path']}")
    model.load_state_dict(torch.load(cfg_dict["model_path"], map_location=device), strict=False)
    model.eval()
    return model


def build_eval_config(cfg_dict, device, worker_id, worker_log_root, worker_result_root, worker_scores_root, chunk_size):
    return {
        'method_type': cfg_dict["method_type"], 'task_name': cfg_dict["test_problem_type"],
        'difficulty': cfg_dict["difficulty"], 'device': device, 'solver': cfg_dict["solver"],
        'log_dir': worker_log_root, 'result_dir': worker_result_root,
        'gnn_type': cfg_dict["gnn_type"], 'fix_strategy': cfg_dict["fix_strategy"],
        'time_limits': cfg_dict["time_limits"], 'threads': cfg_dict["threads"],
        'scores_dir': worker_scores_root, 'time_flag': cfg_dict.get("time_flag", None),
        'save_scores': cfg_dict["save_scores"], 'hyperparam_pas': cfg_dict["hyperparam_pas"],
        'hyperparam_soft': cfg_dict["hyperparam_soft"],
        'use_edge_coeff': cfg_dict.get("use_edge_coeff", True),
        'n_levels': cfg_dict["n_levels"], 'repair_frac': cfg_dict["repair_frac"],
        'test_num': chunk_size,
    }


def main():
    cfg = OmegaConf.load('configs/test/test_shsp.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)
    run_parallel_eval(cfg_dict=cfg_dict, instances=instances, evaluator_cls=SHSPEvaluator,
                      build_model_fn=build_model, build_eval_config_fn=build_eval_config,
                      merge_fn=merge_basic_metrics)


if __name__ == '__main__':
    main()
