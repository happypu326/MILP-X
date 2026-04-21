from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse

from .base import BaseGenerator

def generate_setcover(nrows, ncols, density, filename, rng, max_coef=100):
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))


class SetCoverGenerator(BaseGenerator):
    problem_code = "SC"
    def __init__(
        self,
        *,
        difficulty: str = "easy",
        nrows: int | None = None,
        ncols: int | None = None,
        density: float = 0.05,
        max_coef: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        
        difficulty = difficulty.lower()
        if difficulty == "easy":
            self.nrows = nrows if nrows is not None else 500
            self.ncols = ncols if ncols is not None else 1000
        elif difficulty == "medium":
            self.nrows = nrows if nrows is not None else 1000
            self.ncols = ncols if ncols is not None else 1000
        elif difficulty == "hard":
            self.nrows = nrows if nrows is not None else 2000
            self.ncols = ncols if ncols is not None else 1000
        elif difficulty == "very-hard":
            self.nrows = nrows if nrows is not None else 5000
            self.ncols = ncols if ncols is not None else 4000
        elif difficulty == "very-hard2":
            self.nrows = nrows if nrows is not None else 5000
            self.ncols = ncols if ncols is not None else 8000
        else:
            self.nrows = nrows if nrows is not None else 500
            self.ncols = ncols if ncols is not None else 1000
        
        self.density = density
        self.max_coef = max_coef
        self.difficulty = difficulty

    def build_instance(self, idx: int, **kwargs):
        return {"seed": np.random.randint(2**31)}

    def _write_lp(self, instance, filepath: Path) -> None:
        rng = np.random.RandomState(instance["seed"])
        generate_setcover(
            nrows=self.nrows,
            ncols=self.ncols,
            density=self.density,
            filename=filepath,
            rng=rng,
            max_coef=self.max_coef,
        )

    def make_filename(self, idx: int, **kwargs) -> str:
        return f"sc_{self.nrows}r_{self.ncols}c_{idx+1:04d}.lp"

    def persist_instance(self, instance, output_dir: Path, *, idx: int, **kwargs):
        filepath = super().persist_instance(instance, output_dir, idx=idx, **kwargs)
        print(f"[SetCover] Generating instance {idx+1}: {filepath}")
        return filepath

