from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from cryoseed.backends.triton.primitives import weighted_sqdiff_sum


__all__ = [
    "spectral_mse_loss",
]


def _resolve_spectral_reduction(
    reduction: Literal["none", "mean", "sum"],
    spectral_reduction: Literal["auto", "mean", "sum"],
) -> Literal["mean", "sum"]:
    if spectral_reduction == "auto":
        if reduction == "none":
            raise ValueError(
                "spectral_reduction must be explicitly set to 'mean' or 'sum' "
                "when reduction='none'"
            )
        return reduction
    if spectral_reduction not in ("mean", "sum"):
        raise ValueError(
            "spectral_reduction must be one of 'auto', 'mean', 'sum'; "
            f"got {spectral_reduction!r}"
        )
    return spectral_reduction


def spectral_mse_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    input_indices: Tensor | None = None,
    target_indices: Tensor | None = None,
    out: Tensor | None = None,
    prefer_2stage: bool | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
    spectral_reduction: Literal["auto", "mean", "sum"] = "auto",
) -> Tensor:
    """Compute a weighted sum of squared spectral differences (Triton).

    Thin wrapper around :func:`cryoseed.backends.triton.primitives.weighted_sqdiff_sum`.

    The frequency-bin dimension ``D`` is reduced first according to
    ``spectral_reduction``. Any additional ``reduction`` is then applied over the
    per-pair/per-tile loss outputs.

    Two modes are supported:

    - Indexed mode (when indices are provided)::

        loss[i] = sum_d weight[d] * |input[input_indices[i], d] - target[target_indices[i], d]|^2

      Expected input/target shapes: ``(N, ...)``. All feature dimensions after ``N`` are
      flattened into ``D`` internally.

    - Broadcast mode (when indices are omitted)::

        loss[b, ci, co] = sum_d weight[d] * |input[b, ci, d] - target[b, co, d]|^2

      Expected input/target shapes: ``(B, C, ...)``. All feature dimensions after ``C``
      are flattened into ``D`` internally.

      The underlying primitive returns a flat tensor of shape ``(B * C_input * C_target,)``
      unless a strided output view is provided.

    Args:
        input: Input tensor (currently complex CUDA tensors are required).
        target: Tensor to compare against. If on a different device, it is moved to
            ``input.device``.
        weight: Optional real weights for the flattened frequency-bin / feature dimension ``D``.

            If ``None``, uses uniform weights (all ones).

            ``weight`` is flattened internally (via ``weight.reshape(-1)``) and must have
            exactly ``D`` elements.

            ``weight`` must be non-negative.

        input_indices: Optional indices selecting rows from ``input`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        target_indices: Optional indices selecting rows from ``target`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        out: Optional output tensor for the unreduced per-pair/per-tile loss values.
            This buffer is filled before any final ``reduction`` is applied and must already be
            on ``input.device``.
            ``out`` is treated as a preallocated write buffer rather than a differentiable
            input and therefore must have ``requires_grad=False``. If the unreduced loss needs
            to remain in the autograd graph, do not pass ``out`` and use the returned tensor
            instead.
        prefer_2stage: Optional override for the 2-stage Triton reduction path.
        reduction: Specifies the reduction to apply over the per-pair/per-tile loss outputs:
            ``'none'`` | ``'mean'`` | ``'sum'``.

            - ``'none'``: return the unreduced per-pair/per-tile loss tensor after spectral
              reduction. In this mode, ``spectral_reduction`` must be explicitly set to
              ``'mean'`` or ``'sum'``.
            - ``'mean'``: return the mean over all pair/tile losses.
            - ``'sum'``: return the sum over all pair/tile losses.

            Default: ``'mean'``.
        spectral_reduction: Reduction to apply over the spectral / flattened feature dimension
            ``D`` before the final pair/tile reduction:
            ``'auto'`` | ``'mean'`` | ``'sum'``.

            - ``'auto'``: follows ``reduction`` for scalar outputs. Resolves to ``'mean'`` when
              ``reduction='mean'`` and to ``'sum'`` when ``reduction='sum'``. It is invalid
              when ``reduction='none'``.
            - ``'mean'``: normalize ``weight`` to sum to 1 before reducing over ``D``.
            - ``'sum'``: reduce over ``D`` using the raw weights.

            Default: ``'auto'``.

    Returns:
        A float32 tensor.

        - ``reduction='none'``: per-pair/per-tile loss tensor after spectral reduction
          (indexed: ``(N,)``; broadcast: flat ``(B * C_input * C_target,)`` unless
          ``out`` provides a strided view).
        - ``reduction='mean'`` or ``'sum'``: scalar tensor.
    """

    if not torch.is_complex(input) or not torch.is_complex(target):
        raise NotImplementedError(
            "spectral_mse_loss currently requires complex tensors. "
            "Real-valued kernels may be added in a future version."
        )

    if not input.is_cuda:
        raise RuntimeError("Triton spectral_mse_loss requires CUDA tensors")

    device = input.device

    if target.device != device:
        target = target.to(device=device)

    if (input_indices is None) != (target_indices is None):
        raise ValueError("input_indices and target_indices must be both provided or both None")

    if input_indices is not None and input_indices.device != device:
        input_indices = input_indices.to(device=device)
    if target_indices is not None and target_indices.device != device:
        target_indices = target_indices.to(device=device)

    if out is not None and out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if out is not None and out.dtype != torch.float32:
        raise ValueError(f"out must be float32, got {out.dtype}")
    if out is not None and out.dim() == 1 and not out.is_contiguous():
        raise ValueError("out must be contiguous when out is 1D")

    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"reduction must be one of 'none', 'mean', 'sum'; got {reduction!r}")
    spectral_reduction = _resolve_spectral_reduction(reduction, spectral_reduction)

    indexed = input_indices is not None

    if indexed:
        if input.dim() < 2 or target.dim() < 2:
            raise ValueError("indexed mode expects input/target with shape (N, ...)")

        input_flat = input.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        if int(input_flat.shape[1]) != int(target_flat.shape[1]):
            raise ValueError(
                f"D mismatch after flatten: input has {int(input_flat.shape[1])}, target has {int(target_flat.shape[1])}"
            )
        D = int(input_flat.shape[1])

    else:
        if input.dim() < 3 or target.dim() < 3:
            raise ValueError("broadcast mode expects input/target with shape (B, C, ...)")

        if int(target.shape[0]) != int(input.shape[0]):
            raise ValueError("broadcast mode requires input and target to have the same B")

        input_flat = input.flatten(start_dim=2)
        target_flat = target.flatten(start_dim=2)
        if int(input_flat.shape[2]) != int(target_flat.shape[2]):
            raise ValueError(
                f"D mismatch after flatten: input has {int(input_flat.shape[2])}, target has {int(target_flat.shape[2])}"
            )
        D = int(input_flat.shape[2])

    if weight is None:
        weight_1d = torch.ones((D,), device=device, dtype=torch.float32)
    else:
        weight_1d = weight.to(device=device, dtype=torch.float32).reshape(-1)
        if int(weight_1d.numel()) != D:
            raise ValueError(f"weight must have {D} elements after flatten, got {int(weight_1d.numel())}")

    if (weight_1d < 0).any():
        raise ValueError("weight must be non-negative")

    if spectral_reduction == "mean":
        wsum = weight_1d.sum()
        if wsum.item() <= 0.0:
            raise ValueError("weight must have positive sum when spectral_reduction='mean'")
        weight_1d = weight_1d / wsum

    loss = weighted_sqdiff_sum(
        input=input_flat,
        other=target_flat,
        weight=weight_1d,
        input_indices=input_indices,
        other_indices=target_indices,
        out=out,
        prefer_2stage=prefer_2stage,
    )

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()

    raise AssertionError("unreachable")