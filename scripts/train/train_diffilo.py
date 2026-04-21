import sys
import os
from omegaconf import OmegaConf
import torch
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.model.diffilo_gnn import DiffILOGNNPolicy
from src.learning.loss.diffilo_loss import DiffILOLossComputer
from src.trainer.diffilo_trainer import DiffILOTrainer
from src.dataloader.dataloader import create_dataloaders
from src.utils.utils import set_cpu_num, set_seed

def main():
    cfg = OmegaConf.load('configs/train/train_diffilo.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    set_seed(cfg.seed)
    set_cpu_num(cfg.num_workers + 1)
    torch.cuda.set_device(cfg.device)
    device = torch.device(f'{cfg.device}')

    train_loader, val_loader = create_dataloaders(
        method_type=cfg.method_type,
        problem_type=cfg.task,
        data_dir=cfg.data_dir,
        difficulty=cfg.difficulty,
        solve_time=cfg.gurobi_time,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_split=cfg.train_split,
        solver_settings=cfg.solver_settings
    )

    model = DiffILOGNNPolicy(
        emb_size=cfg.emb_size,
        cons_nfeats=cfg.constraint_nfeats,
        edge_nfeats=cfg.edge_nfeats,
        var_nfeats=cfg.variable_nfeats,
        depth=cfg.depth
    ).to(device)

    loss_config = {
        'mu_init': cfg.mu_init,
        'mu_step_size': cfg.mu_step_size,
        'mu_value': cfg.mu_value,
        'mu_max': cfg.mu_max,
        'mu_min': cfg.mu_min,
        'loss_config': cfg.loss_config,
        'num_samples': cfg.num_samples
    }

    train_loss_computer = DiffILOLossComputer(config=loss_config)
    val_loss_computer = DiffILOLossComputer(config=loss_config)

    save_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_emb_{cfg.emb_size}"
        f"_dep_{cfg.depth}"
        f"_ns_{cfg.num_samples}"
        f"_loss_{cfg.loss_config}"
        f"_mu_{cfg.mu_init}_{cfg.mu_step_size}_{cfg.mu_value}_{cfg.mu_min}_{cfg.mu_max}"
        f"_lr_{cfg.lr_i}_{cfg.lr_o}"
    )

    trainer_config = {
        'method_type': cfg.method_type,
        'problem_type': cfg.task,
        'device': cfg.device,
        'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir,
        'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k,
        'optimizer': cfg.optimizer,
        'weight_decay': cfg.weight_decay,
        'momentum': cfg.momentum,
        'lr_scheduler': cfg.lr_scheduler,
        'lr_i': cfg.lr_i,  
        'lr_o': cfg.lr_o,
        'cos_T': cfg.cos_T,
        'cos_min': cfg.cos_min,
        'patience': cfg.patience,
        'min_delta': cfg.min_delta,
        'save_name': save_name,
    }

    trainer = DiffILOTrainer(
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
