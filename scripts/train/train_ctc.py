"""
Training script for Constraint Matters (dependency-light constraint-reduction PS).

Trains a dual-head bipartite GNN: a per-variable assignment head and a
per-constraint critical-tight-constraint head, with a dual focal loss.

Usage:
    python scripts/train/train_ctc.py task=CA
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader import create_dataloaders
from src.learning.model.ctc_gnn import CTCPolicy
from src.learning.loss.focal_loss import DualFocalLossComputer
from src.trainer import CTC_Trainer


def main():
    cfg = OmegaConf.load('configs/train/train_ctc.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    device = cfg.device

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
        use_edge_coeff=cfg.get('use_edge_coeff', True),
    )

    model = CTCPolicy(
        emb_size=cfg.emb_size,
        constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats,
    ).to(device)

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ctc"
    config = {
        'method_type': cfg.method_type,
        'problem_type': cfg.task,
        'device': device,
        'lr': cfg.lr,
        'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        'save_name': save_name,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta,
        'ctc_lambda': cfg.ctc_lambda,
        'focal_alpha': cfg.focal_alpha,
        'focal_gamma': cfg.focal_gamma,
    }

    trainer = CTC_Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_loss_computer=DualFocalLossComputer(config=config),
        val_loss_computer=DualFocalLossComputer(config=config),
        config=config,
    )
    trainer.train()


if __name__ == '__main__':
    main()
