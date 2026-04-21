import sys
import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from omegaconf import OmegaConf
from datetime import datetime

from src.learning import GNNPolicy
from src.utils.helpers import set_random_seed
from src.evaluator.ps_family_evaluator import PSFamilyEvaluator
from src.utils.utils import collect_test_instances

def main():
    cfg = OmegaConf.load('configs/test/test_ps.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = GNNPolicy(
        emb_size=cfg.emb_size,
        constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats
    ).to(device)

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"模型文件未找到: {cfg.model_path}")
    state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)

    eval_config = {
        'method_type': cfg.method_type,
        'task_name': cfg.test_problem_type,
        'difficulty': cfg.difficulty,
        'device': device,
        'solver': cfg.solver,
        'log_dir': cfg.log_dir,
        'result_dir': cfg.result_dir,
        "gnn_type": cfg.gnn_type,
        "fix_strategy": cfg.fix_strategy,
        "time_limits": cfg.time_limits,
        "threads": cfg.threads,
        "scores_dir": cfg.scores_dir,
        'time_flag':cfg.time_flag,
        'save_scores': cfg.save_scores,
        'hyperparam_pas':cfg.hyperparam_pas,
        'hyperparam_soft':cfg.hyperparam_soft,
    }

    evaluator = PSFamilyEvaluator(
        model=model, 
        config=eval_config
    )

    metrics = evaluator.evaluate(instances)

if __name__ == '__main__':
    main()
