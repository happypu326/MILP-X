"""
Train the discrete Bernoulli diffusion MILP solution predictor (DIFUSCO-style,
adapted to the constraint-variable bipartite graph).

Usage:
    python scripts/train/train_diffusion.py
    python scripts/train/train_diffusion.py task=CA num_timesteps=100 device=cuda:1
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader import create_dataloaders
from src.learning.model.diffusion import DiffusionGNNPolicy
from src.trainer import DiffusionTrainer


def main():
    cfg = OmegaConf.load('configs/train/train_diffusion.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    device = cfg.device

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

    model = DiffusionGNNPolicy(
        emb_size=cfg.emb_size,
        constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats,
        depth=cfg.depth,
        jumping_knowledge=cfg.get('jumping_knowledge', True),
    ).to(device)

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_diffusion"
    config = {
        'method_type': cfg.method_type,
        'problem_type': cfg.task,
        'gnn_type': 'diffusion',
        'device': device,
        'lr': cfg.lr,
        'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        'save_name': save_name,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta,
        'num_timesteps': cfg.num_timesteps,
        'schedule': cfg.schedule,
    }

    trainer = DiffusionTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_loss_computer=None,
        val_loss_computer=None,
        config=config,
    )

    trainer.train()


if __name__ == '__main__':
    main()
