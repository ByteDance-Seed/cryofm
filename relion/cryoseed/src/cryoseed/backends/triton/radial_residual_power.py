"""Triton backend for radial residual power."""

from __future__ import annotations

import torch
from torch import Tensor

from cryoseed.backends.triton.primitives import weighted_sqdiff_group_sum_indexed_complex

__all__ = [
    "radial_residual_power",
]


def radial_residual_power(
    input: Tensor,
    target: Tensor,
    *,
    radial_indices: Tensor,
    radial_weight: Tensor,
    num_radial_bins: int,
    input_indices: Tensor,
    target_indices: Tensor,
    out: Tensor | None = None,
    prefer_2stage: bool | None = None,
) -> Tensor:
    """Compute indexed radial residual power profiles with Triton.

    This backend forwards the radial reduction to the grouped primitive
    :func:`weighted_sqdiff_group_sum_indexed_complex`.

    Args:
        input: Complex tensor of shape ``(N_input, P)`` containing flattened
            valid radial points.
        target: Complex tensor of shape ``(N_target, P)`` containing flattened
            valid radial points.
        radial_indices: Integer tensor of shape ``(P,)`` mapping each valid
            point to a radial bin.
        radial_weight: Real tensor of shape ``(P,)`` containing per-point
            weights that convert bin accumulation into a radial mean.
        num_radial_bins: Number of radial bins in the output.
        input_indices: Integer tensor selecting rows from ``input``.
        target_indices: Integer tensor selecting rows from ``target``.
        out: Optional float32 output tensor of shape ``(N, R)``.
        prefer_2stage: Present for API compatibility with other backends. The
            current grouped Triton kernel ignores this flag.

    Returns:
        A float32 tensor of shape ``(N, num_radial_bins)``.
    """
    _ = prefer_2stage
    if not input.is_cuda:
        raise RuntimeError("Triton radial_residual_power requires CUDA tensors")
    if not torch.is_complex(input) or not torch.is_complex(target):
        raise NotImplementedError("radial_residual_power currently requires complex tensors")
    if input.dim() != 2 or target.dim() != 2:
        raise ValueError("radial_residual_power expects flattened 2D tensors with shape (N, P)")
    if input.shape[1] != target.shape[1]:
        raise ValueError("input and target must have the same flattened feature size")

    device = input.device
    if target.device != device:
        target = target.to(device=device)
    if input_indices.device != device:
        input_indices = input_indices.to(device=device)
    if target_indices.device != device:
        target_indices = target_indices.to(device=device)
    if radial_indices.device != device:
        radial_indices = radial_indices.to(device=device)
    if radial_weight.device != device:
        radial_weight = radial_weight.to(device=device)

    return weighted_sqdiff_group_sum_indexed_complex(
        input=input,
        other=target,
        weight=radial_weight,
        input_indices=input_indices,
        other_indices=target_indices,
        group_index=radial_indices,
        num_groups=int(num_radial_bins),
        out=out,
    )