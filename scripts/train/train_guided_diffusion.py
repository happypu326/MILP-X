"""Train guided latent-diffusion IP solution generator (arXiv 2406.12349)."""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from datetime import datetime

from src.dataloader import create_dataloaders
from src.learning.model.guided_diffusion import GuidedDiffusionModel
from src.trainer import GuidedDiffusionTrainer


def main():
    cfg = OmegaConf.load('configs/train/train_guided_diffusion.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    device = cfg.device
    use_edge_coeff = cfg.get('use_edge_coeff', True)
    assert cfg.edge_nfeats == (2 if use_edge_coeff else 1)

    train_loader, val_loader = create_dataloaders(
        method_type=cfg.method_type, problem_type=cfg.task, data_dir=cfg.data_dir,
        difficulty=cfg.difficulty, solve_time=cfg.gurobi_time, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, train_split=cfg.train_split,
        solver_settings=cfg.solver_settings, use_edge_coeff=use_edge_coeff,
    )

    model = GuidedDiffusionModel(
        emb_size=cfg.emb_size, latent_dim=cfg.latent_dim,
        constraint_nfeats=cfg.constraint_nfeats, edge_nfeats=cfg.edge_nfeats,
        variable_nfeats=cfg.variable_nfeats,
    ).to(device)

    save_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_guided_diffusion"
    config = {
        'method_type': cfg.method_type, 'problem_type': cfg.task, 'gnn_type': 'guided_diffusion',
        'device': device, 'lr': cfg.lr, 'num_epochs': cfg.num_epochs,
        'model_save_dir': cfg.model_save_dir, 'log_save_dir': cfg.log_save_dir,
        'save_top_k': cfg.save_top_k, 'save_name': save_name,
        'patience': cfg.patience, 'min_delta': cfg.min_delta,
        'num_timesteps': cfg.num_timesteps, 'lambda_recon': cfg.lambda_recon,
        'lambda_align': cfg.lambda_align,
    }
    trainer = GuidedDiffusionTrainer(model, train_loader, val_loader, None, None, config)
    trainer.train()


if __name__ == '__main__':
    main()
