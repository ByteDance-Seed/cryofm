from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "spectral_mse_loss",
]


def spectral_mse_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    input_indices: Tensor | None = None,
    target_indices: Tensor | None = None,
    out: Tensor | None = None,
    prefer_2stage: bool | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Compute a weighted sum of squared spectral differences (Torch).

    Torch reference backend that mirrors the API/semantics of the Triton implementation.

    The frequency-bin dimension ``D`` is always reduced internally via a weighted sum.
    Any additional ``reduction`` is then applied over the per-pair/per-tile loss outputs.

    Two modes are supported:

    - Indexed mode (when indices are provided)::

        loss[i] = sum_d weight[d] * |input[input_indices[i], d] - target[target_indices[i], d]|^2

      Expected input/target shapes: ``(N, ...)``. All feature dimensions after ``N`` are
      flattened into ``D`` internally.

    - Broadcast mode (when indices are omitted)::

        loss[b, ci, co] = sum_d weight[d] * |input[b, ci, d] - target[b, co, d]|^2

      Expected input/target shapes: ``(B, C, ...)``. All feature dimensions after ``C``
      are flattened into ``D`` internally.

      The output is flattened to ``(B * C_input * C_target,)`` unless a 3D ``out`` view
      of shape ``(B, C_input, C_target)`` is provided.

    Args:
        input: Input tensor (currently complex tensors are required).
        target: Tensor to compare against. If on a different device, it is moved to
            ``input.device``.
        weight: Optional real weights for the flattened frequency-bin / feature dimension ``D``.

            If ``None``, uses uniform weights (all ones).

            ``weight`` is flattened internally (via ``weight.reshape(-1)``) and must have
            exactly ``D`` elements.

            ``weight`` must be non-negative.

            Note:
                When ``reduction='mean'``, ``weight`` is normalized to sum to 1 *before* the
                spectral reduction. In that case, the per-pair/per-tile output is a weighted
                mean over ``D`` (not a weighted sum). A final mean reduction is then applied
                over the per-pair/per-tile outputs.

                Therefore, in general,
                ``spectral_mse_loss(..., reduction='none').mean()`` is not equal to
                ``spectral_mse_loss(..., reduction='mean')`` unless ``weight`` is already
                normalized.
        input_indices: Optional indices selecting rows from ``input`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        target_indices: Optional indices selecting rows from ``target`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        out: Optional output tensor for the unreduced per-pair/per-tile loss values.
            This buffer is filled before any final ``reduction`` is applied and must already be
            on ``input.device``.

            Broadcast mode notes:
                ``out`` can be either a flat contiguous 1D tensor of shape
                ``(B * C_input * C_target,)`` or a 3D view of shape ``(B, C_input, C_target)``.
        prefer_2stage: Present for API compatibility with the Triton backend. Ignored in the
            Torch backend.
        reduction: Specifies the reduction to apply over the per-pair/per-tile loss outputs:
            ``'none'`` | ``'mean'`` | ``'sum'``.

            - ``'none'``: return the unreduced loss tensor.
            - ``'mean'``: return the mean over all pair/tile losses.
            - ``'sum'``: return the sum over all pair/tile losses.

            Default: ``'mean'``.

    Returns:
        A float32 tensor.

        - ``reduction='none'``: unreduced loss tensor (indexed: ``(N,)``; broadcast: flat
          ``(B * C_input * C_target,)`` unless ``out`` provides a 3D view).
        - ``reduction='mean'`` or ``'sum'``: scalar tensor.
    """

    if not torch.is_complex(input) or not torch.is_complex(target):
        raise NotImplementedError(
            "spectral_mse_loss currently requires complex tensors. "
            "Real-valued kernels may be added in a future version."
        )

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

    indexed = input_indices is not None
    _ = prefer_2stage

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

    if reduction == "mean":
        wsum = weight_1d.sum()
        if wsum.item() <= 0.0:
            raise ValueError("weight must have positive sum when reduction='mean'")
        weight_1d = weight_1d / wsum

    sqrtw = weight_1d.sqrt()

    if indexed:
        assert input_indices is not None and target_indices is not None
        ii = input_indices.reshape(-1).to(device=device, dtype=torch.long)
        ti = target_indices.reshape(-1).to(device=device, dtype=torch.long)

        a = input_flat.index_select(0, ii)
        b = target_flat.index_select(0, ti)

        diff = (a - b) * sqrtw
        loss_flat = (diff.real.square() + diff.imag.square()).sum(dim=-1)

        if out is not None:
            if out.numel() != loss_flat.numel():
                raise ValueError(
                    f"out.numel() must match unreduced loss size ({int(loss_flat.numel())}), got {int(out.numel())}"
                )
            out.copy_(loss_flat.reshape(out.shape))
            loss_unreduced = out
        else:
            loss_unreduced = loss_flat

    else:
        B = int(input_flat.shape[0])
        Ci = int(input_flat.shape[1])
        Co = int(target_flat.shape[1])

        if out is None:
            out_view = torch.empty((B, Ci, Co), device=device, dtype=torch.float32)
            return_flat = True
        else:
            return_flat = False
            if out.dim() == 3 and tuple(out.shape) == (B, Ci, Co):
                out_view = out
            elif out.dim() == 1 and out.numel() == B * Ci * Co:
                if not out.is_contiguous():
                    raise ValueError("broadcast mode requires contiguous flat out when out is 1D")
                out_view = out.view(B, Ci, Co)
            else:
                raise ValueError(
                    "broadcast mode out must be (B, C_input, C_target) or flat contiguous (B*C_input*C_target,)"
                )

        a_w = input_flat * sqrtw
        b_w = target_flat * sqrtw

        a2 = (a_w.real.square() + a_w.imag.square()).sum(dim=-1)
        b2 = (b_w.real.square() + b_w.imag.square()).sum(dim=-1)

        a_conj = a_w.conj()
        chunk = 1024
        for co0 in range(0, Co, chunk):
            co1 = min(co0 + chunk, Co)
            b_chunk = b_w[:, co0:co1]
            cross = torch.matmul(a_conj, b_chunk.transpose(1, 2)).real
            out_view[:, :, co0:co1] = a2[:, :, None] + b2[:, None, co0:co1] - 2.0 * cross

        if out is None:
            loss_unreduced = out_view.reshape(-1)
        else:
            loss_unreduced = out

    if reduction == "none":
        return loss_unreduced
    if reduction == "mean":
        return loss_unreduced.mean()
    if reduction == "sum":
        return loss_unreduced.sum()

    raise ValueError(f"reduction must be one of 'none', 'mean', 'sum'; got {reduction!r}")