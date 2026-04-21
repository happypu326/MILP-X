from __future__ import annotations

import abc
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class BaseGenerator(abc.ABC):
    default_extension: str = ".lp"
    problem_code: str | None = None

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed

    @abc.abstractmethod
    def build_instance(self, idx: int, **kwargs: Any) -> Any:
        """
        Generate a single instance.
        The return value can be:
        - a pyscipopt / gurobi model
        - an LP/MPS text string
        - a Path
        - or any other object supported by the subclass's write_instance() method
        """
        raise NotImplementedError

    def generate_batch(
        self,
        *,
        n_instances: int,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> list[Path]:
        output_path = self.set_output_dir(output_dir)
        generated_files: list[Path] = []

        for idx in range(n_instances):
            self._reseed(idx)
            instance = self.build_instance(idx=idx, **kwargs)
            filepath = self.persist_instance(instance, output_path, idx=idx, **kwargs)
            generated_files.append(filepath)

        return generated_files

    def ecole_instance_generate(self, **kwargs: Any):
        """
        Exposed to the external ecole InstanceGenerator interface.
        """
        return _EcoleInstanceGenerator(self, kwargs)

    def set_output_dir(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def persist_instance(
        self,
        instance: Any,
        output_dir: Path,
        *,
        idx: int,
        **kwargs: Any,
    ) -> Path:
        actual_output_dir = self._get_output_dir_with_difficulty(output_dir)
        filepath = actual_output_dir / self.make_filename(idx=idx, **kwargs)
        self.write_instance(instance, filepath, **kwargs)
        return filepath

    def make_filename(self, idx: int, **kwargs: Any) -> str:
        return f"instance_{idx + 1:04d}{self.default_extension}"

    def write_instance(self, instance: Any, filepath: Path, **kwargs: Any) -> None:
        write_lp = getattr(self, "_write_lp", None)
        if callable(write_lp):
            write_lp(instance, filepath)
            return

        if hasattr(instance, "writeProblem"):
            instance.writeProblem(str(filepath))
            return

        if hasattr(instance, "write_problem"):
            instance.write_problem(str(filepath))
            return

        if hasattr(instance, "writeMPS"):
            instance.writeMPS(str(filepath))
            return

        if hasattr(instance, "write"):
            instance.write(str(filepath))
            return

        if isinstance(instance, str):
            filepath.write_text(instance, encoding="utf-8")
            return

        if isinstance(instance, bytes):
            filepath.write_bytes(instance)
            return

        if isinstance(instance, Path):
            filepath.write_bytes(instance.read_bytes())
            return

        raise TypeError(
            f"Cannot write {type(instance)}, please implement write_instance() in the subclass."
        )

    def _get_output_dir_with_difficulty(self, base_output_dir: Path) -> Path:
        problem_name = self.problem_code or self.__class__.__name__.replace("Generator", "")
        difficulty = getattr(self, "difficulty", None)

        if difficulty:
            actual_dir = base_output_dir / str(problem_name) / str(difficulty).lower()
        else:
            actual_dir = base_output_dir / str(problem_name)

        actual_dir.mkdir(parents=True, exist_ok=True)
        return actual_dir

    def _reseed(self, idx: int) -> None:
        if self.seed is None:
            return
        iteration_seed = (self.seed + idx) % (2**32 - 1)
        np.random.seed(iteration_seed)
        random.seed(iteration_seed)

    def _instance_to_ecole_model(self, instance: Any, **kwargs: Any):
        try:
            import ecole
        except ImportError as exc:
            raise RuntimeError("ecole is required to use the generate() interface.") from exc

        if isinstance(instance, ecole.scip.Model):
            return instance

        if isinstance(instance, Path):
            return ecole.scip.Model.from_file(str(instance))

        suffix = kwargs.get("extension", self.default_extension)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.write_instance(instance, tmp_path, **kwargs)
            return ecole.scip.Model.from_file(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)


class _EcoleInstanceGenerator:
    def __init__(self, generator: BaseGenerator, kwargs: dict[str, Any]) -> None:
        self.generator = generator
        self.kwargs = kwargs
        self._idx = 0

    def __iter__(self) -> "_EcoleInstanceGenerator":
        return self

    def __next__(self):
        return self()

    def __call__(self):
        self.generator._reseed(self._idx)
        instance = self.generator.build_instance(idx=self._idx, **self.kwargs)
        model = self.generator._instance_to_ecole_model(
            instance,
            idx=self._idx,
            **self.kwargs,
        )
        self._idx += 1
        return model