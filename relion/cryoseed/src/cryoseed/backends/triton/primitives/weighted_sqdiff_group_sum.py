"""Grouped weighted squared-difference Triton primitives.

This module provides the low-level indexed grouped reduction used by
radial-profile style operators. Unlike ``weighted_sqdiff_sum``, the output keeps
one reduction slot per group instead of collapsing the full feature dimension to
a single scalar per pair.
"""

import torch
import triton

from cryoseed.backends.triton.primitives.kernels.weighted_sqdiff_group_sum_indexed import (
    weighted_sqdiff_group_sum_indexed_cplx_fwd_kernel as _weighted_sqdiff_group_sum_indexed_cplx_fwd_kernel,
)
from cryoseed.backends.triton.primitives.weighted_sqdiff_sum import d_bucket

__all__ = [
    "weighted_sqdiff_group_sum_indexed_complex",
]


def _maybe_int64(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` as int64 without copying when the dtype already matches."""
    return x if x.dtype == torch.int64 else x.to(torch.int64)


def weighted_sqdiff_group_sum_indexed_complex(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    input_indices: torch.Tensor,
    other_indices: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
    out: torch.Tensor | None = None,
):
    """Compute grouped weighted squared differences for indexed pairs.

    For each selected pair ``i``, this primitive computes::

        out[i, g] = sum_{d: group_index[d] == g}
            weight[d] * |input[input_indices[i], d] - other[other_indices[i], d]|^2

    This is a low-level Triton primitive used by higher-level ops such as
    :func:`cryoseed.ops.radial.radial_residual_power`.

    Args:
        input: Complex tensor of shape ``(N_input, D)``.
        other: Complex tensor of shape ``(N_other, D)``.
        weight: Real tensor of shape ``(D,)`` after flattening.
        input_indices: Integer tensor selecting rows from ``input``.
        other_indices: Integer tensor selecting rows from ``other``.
        group_index: Integer tensor of shape ``(D,)`` assigning each feature to
            a reduction group in ``[0, num_groups)``.
        num_groups: Number of output groups.
        out: Optional float32 output tensor of shape ``(N, num_groups)`` where
            ``N`` is the number of selected pairs. If provided, the buffer is
            zeroed before use and overwritten in place.

    Returns:
        A float32 tensor of shape ``(N, num_groups)``.

    Note:
        Autograd is not implemented for this primitive yet.
    """
    if not input.is_cuda:
        raise RuntimeError("Triton weighted_sqdiff_group_sum requires CUDA tensors")
    if input.dim() != 2 or other.dim() != 2:
        raise ValueError(
            "indexed grouped variant requires input and other to be 2D tensors: "
            "input.shape=(N_input, D), other.shape=(N_other, D)"
        )
    if (not input.is_complex()) or (not other.is_complex()):
        raise ValueError("input/other must be complex tensors")
    if input.shape[1] != other.shape[1]:
        raise ValueError("input/other must have the same D")
    if input.device != other.device or weight.device != input.device:
        raise ValueError("input/other/weight must be on the same device")

    D = int(input.shape[1])
    weight_1d = weight.reshape(-1)
    group_index_1d = group_index.reshape(-1)
    if int(weight_1d.numel()) != D:
        raise ValueError(f"weight must have {D} elements after flatten, got {int(weight_1d.numel())}")
    if int(group_index_1d.numel()) != D:
        raise ValueError(f"group_index must have {D} elements after flatten, got {int(group_index_1d.numel())}")
    if int(input_indices.numel()) != int(other_indices.numel()):
        raise ValueError("input_indices and other_indices must have the same number of elements")

    if torch.is_grad_enabled() and any(t.requires_grad for t in (input, other, weight)):
        raise NotImplementedError("weighted_sqdiff_group_sum_indexed_complex does not support autograd yet")

    num_groups = int(num_groups)
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")

    # Normalize dtypes/layout before dispatch so the Triton kernel sees a stable
    # memory layout and does not need to handle dtype conversions internally.
    input = input.contiguous()
    other = other.contiguous()
    weight = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)
    input_indices = _maybe_int64(input_indices).contiguous()
    other_indices = _maybe_int64(other_indices).contiguous()
    group_index = _maybe_int64(group_index.to(device=input.device)).contiguous().reshape(-1)

    if torch.any(group_index < 0) or torch.any(group_index >= num_groups):
        raise ValueError("group_index contains out-of-range values")

    N = int(input_indices.numel())
    if out is not None:
        if out.requires_grad:
            raise ValueError("out must not require grad")
        if out.dtype != torch.float32 or out.device != input.device or out.shape != (N, num_groups):
            raise ValueError("out must be float32 on the same device with shape (N, num_groups)")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        # The grouped kernel uses atomic accumulation, so caller-provided output
        # buffers must be cleared before launch.
        out.zero_()
    else:
        out = torch.zeros((N, num_groups), device=input.device, dtype=torch.float32)

    # Triton kernels operate on explicit real/imag channels rather than PyTorch
    # complex views, so convert once here and pass the derived strides through.
    x_ri = torch.view_as_real(input)
    y_ri = torch.view_as_real(other)
    D_BUCKET = d_bucket(D)

    # Launch shape:
    # - axis 0 enumerates indexed pairs
    # - axis 1 enumerates feature tiles of size BLOCK_D
    grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
    _weighted_sqdiff_group_sum_indexed_cplx_fwd_kernel[grid](
        x_ri,
        y_ri,
        weight,
        input_indices,
        other_indices,
        group_index,
        out,
        G=num_groups,
        D=D,
        D_BUCKET=D_BUCKET,
        x_stride0=x_ri.stride(0),
        x_stride1=x_ri.stride(1),
        x_stride2=x_ri.stride(2),
        y_stride0=y_ri.stride(0),
        y_stride1=y_ri.stride(1),
        y_stride2=y_ri.stride(2),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
    )
    return out