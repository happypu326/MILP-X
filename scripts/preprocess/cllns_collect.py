import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.dataloader.cllns_collect import collect_dataset


def main():
    cfg = OmegaConf.load('configs/preprocess/cllns_collect.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    collect_dataset(
        input_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        solver_name=cfg.solver,
        n_states=cfg.n_states,
        k_frac=cfg.k_frac,
        window=cfg.window,
        init_time=cfg.init_time,
        lb_time=cfg.lb_time,
        n_neg=cfg.n_neg,
        threads=cfg.threads,
        seed=cfg.seed,
    )


if __name__ == '__main__':
    main()
