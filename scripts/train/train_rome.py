import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch_geometric
from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader import create_dataloaders
from src.learning import GNNPolicy, MoEPolicy
from src.learning.loss import RoMELossComputer, BCELossComputer
from src.trainer import PS_Family_Trainer
from src.utils.utils import get_group_stats

def main():
    cfg = OmegaConf.load('configs/train/train_rome.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    device = cfg.device

    problem_types = list(cfg.task)
    problem_type_str = '_'.join(problem_types)

    use_edge_coeff = cfg.get('use_edge_coeff', True)
    expected_edge = 2 if use_edge_coeff else 1
    assert cfg.edge_nfeats == expected_edge, (
        f"edge_nfeats ({cfg.edge_nfeats}) must be {expected_edge} for "
        f"use_edge_coeff={use_edge_coeff}"
    )

    train_loader, val_loader = create_dataloaders(
        method_type=cfg.method_type,
        problem_type=cfg.task,
        data_dir=cfg.data_dir,
        difficulty=cfg.difficulty,
        solve_time=cfg.gurobi_time,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_split=cfg.train_split,
        solver_settings=cfg.solver_settings,
        use_edge_coeff=use_edge_coeff,
    )

    train_groups = []
    val_groups = []
    for batch in train_loader:
        train_groups.extend(batch.group.cpu().tolist())
    for batch in val_loader:
        val_groups.extend(batch.group.cpu().tolist())

    train_group_stats = get_group_stats(train_groups, device)
    val_group_stats = get_group_stats(val_groups, device)

    adjustments = [float(c) for c in cfg.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, train_group_stats['n_groups'])
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * train_group_stats['n_groups'])
    else:
        adjustments = np.array(adjustments)

    train_loss_config = {
        'is_robust': cfg.robust,
        'group_stats': train_group_stats,
        'alpha': cfg.alpha,
        'gamma': cfg.gamma,
        'adj': adjustments,
        'step_size': cfg.robust_step_size,
        'normalize_loss': cfg.use_normalized_loss,
        'btl': cfg.btl,
        'min_var_weight': cfg.minimum_variational_weight,
        'device': device
    }
    train_loss_computer = RoMELossComputer(config=train_loss_config)

    val_loss_config = {
        'is_robust': cfg.robust,
        'group_stats': val_group_stats,
        'step_size': cfg.robust_step_size,
        'alpha': cfg.alpha,
        'device': device
    }
    val_loss_computer = BCELossComputer(config=val_loss_config)

    model = MoEPolicy(
        emb_size=cfg.emb_size,
        constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats,
        num_shared_experts=cfg.num_shared_experts,
        num_dedicate_experts=cfg.num_dedicate_experts,
        top_k=cfg.top_k,
        gate_temperature=cfg.gate_temperature,
        bias_lr=cfg.bias_lr,
        dropout=cfg.dropout,
        use_dro=cfg.use_dro,
        eps_wasserstein=cfg.eps_wasserstein,
        dro_perturb_type=cfg.dro_perturb_type
    ).to(device)

    save_name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.gnn_type}"
                 f"_shared_{cfg.num_shared_experts}_dedicate_{cfg.num_dedicate_experts}")
    config = {
        'method_type': cfg.method_type,
        'problem_type': problem_type_str,
        'gnn_type': cfg.gnn_type,
        'other_loss_ratio': cfg.other_loss_ratio,
        'device': device,
        'lr': cfg.lr,
        'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        'save_name': save_name,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta
    }

    trainer = PS_Family_Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_loss_computer=train_loss_computer,
        val_loss_computer=val_loss_computer,
        config=config
    )

    trainer.train()


if __name__ == '__main__':
    main()
