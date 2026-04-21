import sys
import os
from omegaconf import OmegaConf
import torch
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.model.diffilo_gnn import DiffILOGNNPolicy
from src.evaluator.diffilo_evaluator import DiffILOEvaluator
from src.utils.utils import collect_test_instances
def main():
    cfg = OmegaConf.load('configs/test/test_diffilo.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)

    model = DiffILOGNNPolicy(
        emb_size=cfg.emb_size,
        cons_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        var_nfeats=cfg.variable_nfeats,
        depth=cfg.depth
    ).to(device)

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"模型文件未找到: {cfg.model_path}")
    state_dict = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    evaluator_config = {
        'method_type': cfg.method_type,
        'task_name': cfg.test_problem_type,
        'difficulty': cfg.difficulty,
        'device': device,
        'solver': cfg.solver,
        'log_dir': cfg.log_dir,
        'instance_dir': cfg.instance_dir,
        'result_dir': cfg.result_dir,
        'num_samples': cfg.num_samples,
        'delta': cfg.delta,
        'time_limit': cfg.time_limit,
        'threads': cfg.threads,
        'mip_focus': cfg.mip_focus,
        'mu': cfg.mu,
        'time_flag':cfg.time_flag
    }

    evaluator = DiffILOEvaluator(
        model=model,
        config=evaluator_config
    )

    results = evaluator.evaluate(instances)

if __name__ == '__main__':
    main()
