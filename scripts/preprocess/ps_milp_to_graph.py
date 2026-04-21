import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.preprocessing.milp_preprocessor import MILPPreprocessor

def main():
    cfg = OmegaConf.load('configs/preprocess/preprocess.yaml')
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    preprocessor = MILPPreprocessor(
        solver_name=cfg.solver,
        max_time=cfg.max_time,
        max_solutions=cfg.max_solutions,
        threads=cfg.threads,
        workers=cfg.workers,
    )

    preprocessor.process(cfg.data_dir, cfg.output_dir, mode=cfg.mode)


if __name__ == '__main__':
    main()
