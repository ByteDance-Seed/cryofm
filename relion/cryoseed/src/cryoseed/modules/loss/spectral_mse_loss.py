"""nn.Module wrapper for :func:`cryoseed.ops.loss.spectral_mse_loss`.

This module provides a convenient ``nn.Module`` interface for training loops.

Notes:
    - ``weight`` is stored as a registered buffer so it follows ``.to(device)`` / ``.cuda()``.
    - If you need dynamic per-step ``weight``, call :func:`cryoseed.ops.loss.spectral_mse_loss`
      directly.
    - If you need low-level output-buffer control via ``out``, call
      :func:`cryoseed.ops.loss.spectral_mse_loss` directly.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from cryoseed.ops.loss import spectral_mse_loss

__all__ = [
    "SpectralMSELoss",
]


class SpectralMSELoss(nn.Module):
    """Spectral MSE loss as an ``nn.Module``.

    Wraps :func:`cryoseed.ops.loss.spectral_mse_loss` and keeps common configuration
    (e.g. ``reduction`` and an optional fixed ``weight``) on the module.
    """

    def __init__(
        self,
        *,
        weight: Tensor | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        spectral_reduction: Literal["auto", "mean", "sum"] = "auto",
    ) -> None:
        super().__init__()

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of 'none', 'mean', 'sum'; got {reduction!r}")
        if spectral_reduction not in ("auto", "mean", "sum"):
            raise ValueError(
                "spectral_reduction must be one of 'auto', 'mean', 'sum'; "
                f"got {spectral_reduction!r}"
            )
        if reduction == "none" and spectral_reduction == "auto":
            raise ValueError(
                "spectral_reduction must be explicitly set to 'mean' or 'sum' "
                "when reduction='none'"
            )

        if weight is not None and not torch.is_tensor(weight):
            raise TypeError(f"weight must be a torch.Tensor or None, got {type(weight)!r}")

        self.reduction: Literal["none", "mean", "sum"] = reduction
        self.spectral_reduction: Literal["auto", "mean", "sum"] = spectral_reduction

        self.register_buffer("weight", weight if weight is not None else None)

    def extra_repr(self) -> str:
        if self.weight is None:
            return f"reduction={self.reduction!r}, spectral_reduction={self.spectral_reduction!r}"
        return (
            f"reduction={self.reduction!r}, spectral_reduction={self.spectral_reduction!r}, "
            f"weight_shape={tuple(self.weight.shape)}, weight_dtype={self.weight.dtype}"
        )

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        *,
        input_indices: Tensor | None = None,
        target_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute the loss.

        Args:
            input: Complex input tensor.
            target: Complex target tensor.
            input_indices: Optional indices for indexed mode.
            target_indices: Optional indices for indexed mode.

        Returns:
            - If ``self.reduction == 'none'``: unreduced loss tensor.
            - Otherwise: scalar loss.
        """

        return spectral_mse_loss(
            input=input,
            target=target,
            weight=self.weight,
            input_indices=input_indices,
            target_indices=target_indices,
            prefer_2stage=None,
            reduction=self.reduction,
            spectral_reduction=self.spectral_reduction,
        )