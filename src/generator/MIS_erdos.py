from .base import BaseGenerator
from typing import Optional


class MISErdosRenyiGenerator(BaseGenerator):
    problem_code = "MIS"

    def __init__(
        self,
        *,
        difficulty: str = "very-hard",
        n_nodes: Optional[int] = None,
        avg_degree: float = 4.0,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(seed=seed)

        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.n_nodes = n_nodes if n_nodes is not None else 2000
        elif difficulty == "medium":
            self.n_nodes = n_nodes if n_nodes is not None else 4000
        elif difficulty == "hard":
            self.n_nodes = n_nodes if n_nodes is not None else 5000
        elif difficulty == "very-hard":
            self.n_nodes = n_nodes if n_nodes is not None else 6000
        else:
            self.n_nodes = n_nodes if n_nodes is not None else 6000

        self.avg_degree = avg_degree
        self._generator = None
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        generator = self._get_generator()
        return next(generator)

    def _get_generator(self):
        try:
            import ecole
        except ImportError as exc:
            raise RuntimeError(
                "ecole is required to use MISErdosRenyiGenerator. "
                "Please run: pip install ecole"
            ) from exc

        if self._generator is None:
            p = self.avg_degree / max(self.n_nodes - 1, 1)
            self._generator = ecole.instance.IndependentSetGenerator(
                n_nodes=self.n_nodes,
                graph_type=ecole.instance.IndependentSetGenerator.GraphType.erdos_renyi,
                edge_probability=p,
                random_engine=ecole.RandomEngine(value=self.seed or 0),
            )
        return self._generator

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"mis_er_{self.n_nodes}n_{self.avg_degree}d_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[MIS-ER] Generating instance {idx+1}: {filepath}")
        return filepath