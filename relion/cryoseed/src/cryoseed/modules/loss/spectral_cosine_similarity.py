from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from cryoseed.ops.loss import spectral_cosine_similarity

__all__ = ["SpectralCosineSimilarity"]


class SpectralCosineSimilarity(nn.Module):
    def __init__(
        self,
        *,
        weight: Tensor | None = None,
        eps: float = 1e-12,
        reduction: Literal["none", "mean", "sum"] = "none",
    ) -> None:
        super().__init__()
        if weight is not None and not torch.is_tensor(weight):
            raise TypeError(
                f"weight must be a torch.Tensor or None, got {type(weight)!r}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                "reduction must be one of 'none', 'mean', 'sum'; "
                f"got {reduction!r}"
            )
        self.register_buffer("weight", weight if weight is not None else None)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        *,
        input_indices: Tensor | None = None,
        target_indices: Tensor | None = None,
    ) -> Tensor:
        return spectral_cosine_similarity(
            input,
            target,
            weight=self.weight,
            input_indices=input_indices,
            target_indices=target_indices,
            eps=self.eps,
            reduction=self.reduction,
        )