from __future__ import annotations

from typing import Literal

from torch import Tensor

from cryoseed.backends.torch.spectral_cosine_similarity import (
    spectral_cosine_similarity as _torch_impl,
)

__all__ = ["spectral_cosine_similarity"]


def spectral_cosine_similarity(
    input: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    input_indices: Tensor | None = None,
    target_indices: Tensor | None = None,
    out: Tensor | None = None,
    eps: float = 1e-12,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> Tensor:
    return _torch_impl(
        input=input,
        target=target,
        weight=weight,
        input_indices=input_indices,
        target_indices=target_indices,
        out=out,
        eps=eps,
        reduction=reduction,
    )