"""Torch reference backend for radial residual power."""

from __future__ import annotations

import torch
from torch import Tensor

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
) -> Tensor:
    """Compute indexed radial residual power profiles with Torch ops.

    This backend mirrors the semantics of the Triton implementation while using
    straightforward ``index_select`` and ``scatter_add_`` operations.

    For each selected pair ``i``, this function computes::

        out[i, r] = sum_{d: radial_indices[d] == r}
            radial_weight[d]
            * |input[input_indices[i], d] - target[target_indices[i], d]|^2

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
        out: Optional float32 tensor of shape ``(N, R)`` where ``N`` is the
            number of selected pairs and ``R`` is ``num_radial_bins``.

    Returns:
        A float32 tensor of shape ``(N, num_radial_bins)``.
    """
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

    input_indices = input_indices.reshape(-1).to(dtype=torch.long)
    target_indices = target_indices.reshape(-1).to(dtype=torch.long)
    radial_indices = radial_indices.reshape(-1).to(dtype=torch.long)
    radial_weight = radial_weight.reshape(-1).to(dtype=torch.float32)

    if int(input_indices.numel()) != int(target_indices.numel()):
        raise ValueError("input_indices and target_indices must have the same number of elements")
    if int(radial_indices.numel()) != int(input.shape[1]):
        raise ValueError("radial_indices must match the flattened feature size")
    if int(radial_weight.numel()) != int(input.shape[1]):
        raise ValueError("radial_weight must match the flattened feature size")

    N = int(input_indices.numel())
    num_radial_bins = int(num_radial_bins)
    if out is not None:
        if out.requires_grad:
            raise ValueError("out must not require grad")
        if out.dtype != torch.float32 or out.device != device or out.shape != (N, num_radial_bins):
            raise ValueError("out must be float32 on the same device with shape (N, num_radial_bins)")
        out.zero_()
    else:
        out = torch.zeros((N, num_radial_bins), device=device, dtype=torch.float32)

    proj_sel = input.index_select(0, input_indices)
    target_sel = target.index_select(0, target_indices)
    weighted_residual = (proj_sel - target_sel).abs().square().to(torch.float32) * radial_weight.view(1, -1)
    out.scatter_add_(
        1,
        radial_indices.view(1, -1).expand(N, -1),
        weighted_residual,
    )
    return out