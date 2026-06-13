import torch
import triton

from cryoseed.backends.triton.primitives.kernels.weighted_sqdiff_sum_indexed import (
    weighted_sqdiff_sum_cplx_kernel as _weighted_sqdiff_sum_cplx_kernel,
    weighted_sqdiff_sum_partial_cplx_kernel as _weighted_sqdiff_sum_partial_cplx_kernel,
    reduce_partial_kernel as _reduce_partial_kernel,
)
from cryoseed.backends.triton.primitives.kernels.weighted_sqdiff_sum_broadcast import (
    weighted_sqdiff_sum_broadcast_cplx_tile_kernel as _weighted_sqdiff_sum_broadcast_cplx_tile_kernel,
)

__all__ = [
    "weighted_sqdiff_sum",
    "weighted_sqdiff_sum_indexed_complex",
    "weighted_sqdiff_sum_broadcast_complex",
]

# 8 buckets by D range: <2^11, <2^12, <2^13, <2^14, <2^15, <2^16, <2^17, >=2^17
_D0 = 2 ** 11
_D1 = 2 ** 12
_D2 = 2 ** 13
_D3 = 2 ** 14
_D4 = 2 ** 15
_D5 = 2 ** 16
_D6 = 2 ** 17


def d_bucket(D) -> int:
    D = int(D)
    return (
        0 if D < _D0 else
        1 if D < _D1 else
        2 if D < _D2 else
        3 if D < _D3 else
        4 if D < _D4 else
        5 if D < _D5 else
        6 if D < _D6 else
        7
    )


def _maybe_int32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(torch.int32)


def _should_use_2stage(N: int, D: int, device, BLOCK_D: int = 512) -> bool:
    props = torch.cuda.get_device_properties(device)
    SM = props.multi_processor_count
    target_ctas = SM * 12
    P = (D + BLOCK_D - 1) // BLOCK_D
    return (N < target_ctas) and (N * P >= target_ctas)


def _weighted_sqdiff_sum_1stage(x, y, w, x_idx, y_idx, out, D_BUCKET: int):
    x_ri = torch.view_as_real(x)
    y_ri = torch.view_as_real(y)

    N = x_idx.numel()
    D = w.shape[0]

    grid = (N,)
    _weighted_sqdiff_sum_cplx_kernel[grid](
        x_ri,
        y_ri,
        w,
        x_idx,
        y_idx,
        out,
        D=D,
        D_BUCKET=D_BUCKET,
        x_stride0=x_ri.stride(0),
        x_stride1=x_ri.stride(1),
        x_stride2=x_ri.stride(2),
        y_stride0=y_ri.stride(0),
        y_stride1=y_ri.stride(1),
        y_stride2=y_ri.stride(2),
    )
    return out


def _weighted_sqdiff_sum_2stage(x, y, w, x_idx, y_idx, out, D_BUCKET: int, BLOCK_D=512, BLOCK_P=128):
    x_ri = torch.view_as_real(x)
    y_ri = torch.view_as_real(y)

    N = x_idx.numel()
    D = w.shape[0]
    P = (D + BLOCK_D - 1) // BLOCK_D

    partial = torch.empty((N, P), device=x.device, dtype=torch.float32)

    grid1 = (N, P)
    _weighted_sqdiff_sum_partial_cplx_kernel[grid1](
        x_ri,
        y_ri,
        w,
        x_idx,
        y_idx,
        partial,
        D=D,
        D_BUCKET=D_BUCKET,
        x_stride0=x_ri.stride(0),
        x_stride1=x_ri.stride(1),
        x_stride2=x_ri.stride(2),
        y_stride0=y_ri.stride(0),
        y_stride1=y_ri.stride(1),
        y_stride2=y_ri.stride(2),
        p_stride0=partial.stride(0),
        p_stride1=partial.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=8 if BLOCK_D >= 512 else 4,
    )

    grid2 = (N,)
    _reduce_partial_kernel[grid2](
        partial,
        out,
        P,
        p_stride0=partial.stride(0),
        p_stride1=partial.stride(1),
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    return out

def weighted_sqdiff_sum_indexed_complex(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    input_indices: torch.Tensor,
    other_indices: torch.Tensor,
    out: torch.Tensor | None = None,
    prefer_2stage: bool | None = None,
):
    """
    Indexed variant:

        out[i] = sum_d weight[d] * |input[input_indices[i], d] - other[other_indices[i], d]|^2

    Args:
        input: complex tensor, [N_input, D]
        other: complex tensor, [N_other, D]
        weight: real tensor, [D]
        input_indices: int tensor, [N]
        other_indices: int tensor, [N]
        out: optional float32 tensor, [N]
        prefer_2stage: optional override for 2-stage path in the Triton backend

    Returns:
        out: float32 tensor, [N]
    """
    # ---- shape checks (indexed)
    if input.dim() != 2 or other.dim() != 2:
        raise ValueError(
            "indexed variant requires input and other to be 2D tensors: "
            "input.shape=(N_input, D), other.shape=(N_other, D)"
        )
    if (not input.is_complex()) or (not other.is_complex()):
        raise ValueError("input/other must be complex tensors")
    if input.device != other.device or weight.device != input.device:
        raise ValueError("input/other/weight must be on the same device")

    input = input.contiguous()
    other = other.contiguous()
    weight = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)

    input_indices = _maybe_int32(input_indices).contiguous()
    other_indices = _maybe_int32(other_indices).contiguous()

    N = int(input_indices.numel())
    D = int(weight.shape[0])
    D_BUCKET = d_bucket(D)

    if out is None:
        out = torch.empty((N,), device=input.device, dtype=torch.float32)

    if prefer_2stage is None:
        prefer_2stage = _should_use_2stage(N, D, input.device, BLOCK_D=512)

    if prefer_2stage:
        return _weighted_sqdiff_sum_2stage(input, other, weight, input_indices, other_indices, out, D_BUCKET=D_BUCKET)

    return _weighted_sqdiff_sum_1stage(input, other, weight, input_indices, other_indices, out, D_BUCKET=D_BUCKET)


def weighted_sqdiff_sum_broadcast_complex(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
):
    """
    Broadcast variant (tile):

        out[b, ci, co] = sum_d weight[d] * |input[b, ci, d] - other[b, co, d]|^2

    Args:
        input: complex tensor, [B, C_input, D]
        other: complex tensor, [B, C_other, D]
        weight: real tensor, [D]
        out: optional float32 tensor, [B*C_input*C_other] (flattened)

    Returns:
        out_flat: float32 tensor, [B*C_input*C_other]
    """
    # ---- shape checks (broadcast)
    if input.dim() != 3 or other.dim() != 3:
        raise ValueError(
            "broadcast variant requires input and other to be 3D tensors: "
            "input.shape=(B, C_input, D), other.shape=(B, C_other, D)"
        )
    if (not input.is_complex()) or (not other.is_complex()):
        raise ValueError("input/other must be complex tensors")
    if input.shape[0] != other.shape[0] or input.shape[2] != other.shape[2]:
        raise ValueError("input/other must have same B and D")
    if weight.ndim != 1 or weight.shape[0] != input.shape[2]:
        raise ValueError("weight must be 1D with length D")
    if input.device != other.device or weight.device != input.device:
        raise ValueError("input/other/weight must be on the same device")

    x_ri = torch.view_as_real(input)
    y_ri = torch.view_as_real(other)

    B, C_input, D = input.shape
    C_other = other.shape[1]
    N = B * C_input * C_other

    weight = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)
    D_BUCKET = d_bucket(int(D))

    if out is None:
        out_ptr = torch.empty((N,), device=input.device, dtype=torch.float32)
        out_sb = C_input * C_other
        out_sc = C_other
        out_so = 1
    else:
        if out.dtype != torch.float32 or out.device != input.device:
            raise ValueError("out must be float32 on the same device")
        if out.ndim == 1:
            if out.shape != (N,):
                raise ValueError("out 1D shape must be (B*C_input*C_other,)")
            out_ptr = out
            out_sb = C_input * C_other
            out_sc = C_other
            out_so = 1
        elif out.ndim == 3:
            if out.shape != (B, C_input, C_other):
                raise ValueError("out 3D shape must be (B,C_input,C_other)")
            out_ptr = out
            out_sb, out_sc, out_so = out.stride()
        else:
            raise ValueError("out must be a 1D flat tensor or a 3D strided view")

    grid = lambda meta: (
        B * triton.cdiv(C_input, meta["BLOCK_CI"]) * triton.cdiv(C_other, meta["BLOCK_CO"]),
    )

    _weighted_sqdiff_sum_broadcast_cplx_tile_kernel[grid](
        x_ri,
        y_ri,
        weight,
        out_ptr,
        B,
        C_input,
        C_other,
        out_sb=out_sb,
        out_sc=out_sc,
        out_so=out_so,
        D=int(D),
        D_BUCKET=D_BUCKET,
        x_sb=x_ri.stride(0),
        x_sc=x_ri.stride(1),
        x_sd=x_ri.stride(2),
        x_sri=x_ri.stride(3),
        y_sb=y_ri.stride(0),
        y_sc=y_ri.stride(1),
        y_sd=y_ri.stride(2),
        y_sri=y_ri.stride(3),
    )

    return out_ptr

def weighted_sqdiff_sum(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    input_indices: torch.Tensor | None = None,
    other_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    prefer_2stage: bool | None = None,
):
    """
    Compute a weighted sum of squared differences with indexed selection.

    For each i, this function computes::

        out[i] = sum_d weight[d] * |input[input_indices[i], d]
                                   - other[other_indices[i], d]|^2

    Convenience wrapper.

    - If input_indices/other_indices are provided -> indexed variant.
    - Otherwise -> broadcast variant.

    This keeps call sites simple while still allowing explicit APIs above.
 
    Args:
        input: Input tensor. Commonly shaped (N_input, D).
        other: The other tensor to compare against. Commonly shaped (N_other, D).
        weight: Weights applied to the squared difference. Must be broadcastable to the
            final elementwise result.
        input_indices: Indices selecting rows/elements from ``input``.
        other_indices: Indices selecting rows/elements from ``other``. Typically has the
            same shape as ``input_indices`` and defines pairwise mapping, i.e.
            ``input[input_indices[i]]`` is compared with ``other[other_indices[i]]``.
        out: Optional output tensor to write results into. If provided, its shape and
            dtype must match the computed output.
 
    Returns:
        A 1D tensor of shape (N,), where each element is the weighted sum of squared
        differences over the last dimension.
    """
    if (input_indices is None) != (other_indices is None):
        raise ValueError("input_indices and other_indices must be both provided or both None")

    # ---- broadcast path
    if input_indices is None:
        if input.dim() != 3 or other.dim() != 3:
            raise ValueError(
                "broadcast path selected (no indices provided), "
                "but input/other are not 3D tensors. "
                "Expected input.shape=(B, C_input, D), other.shape=(B, C_other, D)."
            )
        return weighted_sqdiff_sum_broadcast_complex(input, other, weight, out=out)

    # ---- indexed path
    if input.dim() != 2 or other.dim() != 2:
        raise ValueError(
            "indexed path selected (indices provided), "
            "but input/other are not 2D tensors. "
            "Expected input.shape=(N_input, D), other.shape=(N_other, D)."
        )

    return weighted_sqdiff_sum_indexed_complex(
        input, other, weight, input_indices, other_indices, out=out, prefer_2stage=prefer_2stage
    )