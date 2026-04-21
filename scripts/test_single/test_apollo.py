import sys
import os
from omegaconf import OmegaConf
import random
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.model.gcn import GNNPolicy
from src.evaluator.apollo_evaluator import ApolloEvaluator
from src.utils.utils import collect_test_instances

def main():
    cfg = OmegaConf.load('configs/test/test_apollo.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    instances = collect_test_instances(cfg.instance_dir, cfg.test_problem_type, cfg.difficulty)
    
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

    evaluator_config = {
        'method_type': cfg.method_type,
        'task_name': cfg.test_problem_type,
        'difficulty': cfg.difficulty,
        'device': device,
        'solver': cfg.solver,
        'log_dir': cfg.log_dir,
        'result_dir': cfg.result_dir,
        'time_limits': cfg.time_limits,
        'k0_list': cfg.k0_list,
        'k1_list': cfg.k1_list,
        'delta_list': cfg.delta_list,
        'threads': cfg.threads,
        'mip_focus': cfg.mip_focus,
        'time_flag':cfg.time_flag
    }

    evaluator = ApolloEvaluator(
        model=model,
        config=evaluator_config
    )

    results = evaluator.evaluate(instances)

if __name__ == '__main__':
    main()
