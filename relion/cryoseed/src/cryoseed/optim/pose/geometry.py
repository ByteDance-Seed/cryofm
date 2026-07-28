from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(slots=True)
class PoseGeometry:
    quat: torch.Tensor | None = None
    rotmat: torch.Tensor | None = None
    trans: torch.Tensor | None = None

    def merged(self, **updates: torch.Tensor | None) -> "PoseGeometry":
        return replace(self, **updates)