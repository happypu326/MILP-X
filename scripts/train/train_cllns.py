"""
Training script for CL-LNS (contrastive Large-Neighborhood-Search destroy policy).

Trains a GAT destroy policy with an InfoNCE loss that scores the Local-Branching
expert's destroy subsets above perturbed ones. Consumes the state pickles from
scripts/preprocess/cllns_collect.py; one LNS state per item (batch_size=1).

Usage:
    python scripts/train/train_cllns.py task=MVC
"""

import sys
import os
import glob
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch_geometric
from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader.graph_dataset import GraphDataset
from src.learning import build_ps_family_model, select_encoder_kwargs
from src.trainer import CLLNS_Trainer
from src.learning.loss.cllns_loss import CLLNSLossComputer


def main():
    cfg = OmegaConf.load('configs/train/train_cllns.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    device = cfg.device

    state_files = sorted(glob.glob(os.path.join(cfg.data_dir, '*.pkl')))
    if not state_files:
        raise FileNotFoundError(f"No CL-LNS states in {cfg.data_dir}; run cllns_collect.py first.")
    random.seed(cfg.seed)
    random.shuffle(state_files)
    split = int(cfg.train_split * len(state_files))
    train_files, val_files = state_files[:split], state_files[split:] or state_files[:1]

    train_ds = GraphDataset(train_files, method_type='CLLNS', use_edge_coeff=cfg.get('use_edge_coeff', True))
    val_ds = GraphDataset(val_files, method_type='CLLNS', use_edge_coeff=cfg.get('use_edge_coeff', True))
    train_loader = torch_geometric.loader.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = build_ps_family_model(
        cfg.gnn_type, emb_size=cfg.emb_size, constraint_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats, variable_nfeats=cfg.variable_nfeats,
        **select_encoder_kwargs(cfg.gnn_type, cfg),
    ).to(device)

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cllns"
    config = {
        'method_type': cfg.method_type, 'problem_type': cfg.task, 'device': device,
        'lr': cfg.lr, 'num_epochs': cfg.num_epochs, 'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir, 'save_top_k': cfg.save_top_k, 'save_name': save_name,
        'patience': cfg.patience, 'min_delta': cfg.min_delta, 'tau': cfg.tau,
    }

    trainer = CLLNS_Trainer(
        model=model, train_dataloader=train_loader, val_dataloader=val_loader,
        train_loss_computer=CLLNSLossComputer(config=config),
        val_loss_computer=CLLNSLossComputer(config=config), config=config,
    )
    trainer.train()


if __name__ == '__main__':
    main()
