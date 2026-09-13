"""
Train ConPaS (Contrastive Predict-and-Search, Huang et al. ICML 2024).

Same bipartite encoder + trust-region search as PS; the only change is the
contrastive InfoNCE training objective (src/learning/loss/conpas_loss.py). Any
PS-family encoder can be used via gnn_type.

Usage:
    python scripts/train/train_conpas.py
    python scripts/train/train_conpas.py gnn_type=gasse task=CA tau=0.07
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader import create_dataloaders
from src.learning import build_ps_family_model
from src.trainer import PS_Family_Trainer
from src.learning.loss.conpas_loss import ConPaSLossComputer


def _encoder_kwargs(cfg):
    kw = {}
    gt = cfg.gnn_type
    if gt in ('gasse', 'bipartite_attention', 'random_feature', 'tripartite'):
        kw['depth'] = cfg.get('depth', 4)
        kw['jumping_knowledge'] = cfg.get('jumping_knowledge', True)
    if gt == 'bipartite_attention':
        kw['heads'] = cfg.get('heads', 4)
        kw['dropout'] = cfg.get('dropout', 0.0)
    if gt == 'random_feature':
        kw['n_rand'] = cfg.get('n_rand', 8)
    return kw


def main():
    cfg = OmegaConf.load('configs/train/train_conpas.yaml')
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

    model = build_ps_family_model(
        cfg.gnn_type,
        emb_size=cfg.emb_size,
        constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats,
        **_encoder_kwargs(cfg),
    ).to(device)

    loss_cfg = {'tau': cfg.tau, 'pos_frac': cfg.pos_frac, 'neg_frac': cfg.neg_frac}
    train_loss_computer = ConPaSLossComputer(config=loss_cfg)
    val_loss_computer = ConPaSLossComputer(config=loss_cfg)

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.gnn_type}_conpas"
    config = {
        'method_type': cfg.method_type,
        'problem_type': cfg.task,
        'gnn_type': cfg.gnn_type,
        'device': device,
        'lr': cfg.lr,
        'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        'save_name': save_name,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta,
    }

    trainer = PS_Family_Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_loss_computer=train_loss_computer,
        val_loss_computer=val_loss_computer,
        config=config,
    )

    trainer.train()


if __name__ == '__main__':
    main()
