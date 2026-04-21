from abc import ABC, abstractmethod
import torch
from typing import Tuple

class BaseLossComputer(ABC):
    def __init__(self, config: dict):
        """
        config: dict containing hyperparameters for this loss (e.g., weight, margin, temperature etc.)
        """
        self.config = config

    @abstractmethod
    def compute(self, model_output, batch, is_training = False) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss and return (loss_tensor, loss_info_dict)
        
        model_output: model forward output (according to your Model definition structure)
        batch: batch data (including input graph, features, labels (if any), etc.)
        
        Returns:
          loss_tensor: torch.Tensor (scalar, can call backward())
          loss_info_dict: dict (optional, contains sub-loss items, number of positive/negative samples, 
                          predicted probability distributions, etc., for logging)
        """
        pass