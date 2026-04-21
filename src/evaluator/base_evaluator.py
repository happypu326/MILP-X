import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import json

class BaseEvaluator(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Evaluation configuration dictionary, common keys:
                - task_name: Problem type (e.g., 'CA', 'IP', 'MIS')
                - difficulty: Difficulty level (e.g., 'easy', 'hard')
                - log_dir: Log / solver output directory
                - result_dir: Directory to save evaluation results
                - device: Computing device
        """
        self.task_name = config.get("task_name", "CA")
        self.method_type = config.get("method_type", "PS")
        self.difficulty = config.get("difficulty", "hard")
        self.time_flag = config.get("time_flag", None)
        self.log_dir = config.get("log_dir", None)
        self.result_dir = config.get("result_dir", None)
        self.solver = config.get("solver", None)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.task_name}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            fh = logging.FileHandler(os.path.join(self.log_dir, "evaluate.log"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger
    def evaluate(self, instances: Any) -> Dict[str, Any]:
        """
        Unified entry point: perform evaluation on the given instances.

        Args:
            instances: List of instances or a single instance (file path string / object)

        Returns:
            metrics dict, containing at least:
                method_name, num_instances, avg_time, feasible_rate, instances
        """
        if not isinstance(instances, list):
            instances = [instances]
        self.logger.info(f"Starting evaluation on {len(instances)} instances "
                         f"(task={self.task_name}, difficulty={self.difficulty})")
        metrics = self.evaluate_batch(instances)
        self.logger.info(f"Evaluation finished. avg_time={metrics.get('avg_time', 0):.2f}s, "
                         f"feasible_rate={metrics.get('feasible_rate', 0):.2%}")
        
        with open(os.path.join(self.result_dir, 'results.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Results saved to {self.result_dir}")
        return metrics

    @abstractmethod
    def evaluate_batch(self, instances: List[Any]) -> Dict[str, Any]:
        """
        Subclass implementation: batch evaluation logic.

        Args:
            instances: List of instance paths

        Returns:
            metrics dict
        """
        ...