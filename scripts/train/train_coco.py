import sys
import os
from omegaconf import OmegaConf
import torch
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.model.coco_gnn import CoCoGNNPolicy
from src.learning.loss.coco_loss import CoCoLossComputer
from src.trainer.coco_trainer import CoCoTrainer
from src.dataloader.dataloader import create_dataloaders

def main():
    cfg = OmegaConf.load('configs/train/train_coco.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    torch.manual_seed(cfg.seed)
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

    model = CoCoGNNPolicy(
        emb_size=cfg.emb_size,
        cons_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        var_nfeats=cfg.variable_nfeats,
        depth=cfg.depth,
        Intra_Constraint_Competitive=cfg.Intra_Constraint_Competitive
    ).to(device)

    loss_config = {
        'margin': cfg.margin,
        'alpha': cfg.alpha,
        'tao': cfg.tao,
        'weight_norm': cfg.weight_norm,
        'problem_type': cfg.task
    }
    train_loss_computer = CoCoLossComputer(config=loss_config)
    val_loss_computer = CoCoLossComputer(config=loss_config)

    save_name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Intra_Constraint_Competitive"
                 f"_{cfg.Intra_Constraint_Competitive}_margin_{cfg.margin}_alpha_{cfg.alpha}_tao_{cfg.tao}")
    
    trainer_config = {
        'method_type': cfg.method_type,
        'problem_type': cfg.task,
        'device': cfg.device,
        'lr': cfg.lr,
        'num_epochs': cfg.num_epochs,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        "save_name": save_name
    }

    trainer = CoCoTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_loss_computer=train_loss_computer,
        val_loss_computer=val_loss_computer,
        config=trainer_config
    )

    trainer.train()

if __name__ == '__main__':
    main()
