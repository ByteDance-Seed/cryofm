from dataclasses import dataclass
from typing import Optional

import torch

from cryoseed.data import DataBatch

@dataclass(slots=True)
class SolverInferResult:
    image: torch.Tensor
    ctf: Optional[torch.Tensor]

class Solver:
    def refresh(self):
        raise NotImplementedError

    def infer(self, batch: DataBatch) -> SolverInferResult:
        raise NotImplementedError

    def accumulate(self, result: SolverInferResult):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def zero_accum(self):
        raise NotImplementedError