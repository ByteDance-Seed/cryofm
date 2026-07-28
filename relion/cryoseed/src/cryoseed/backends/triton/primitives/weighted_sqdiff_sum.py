import torch
import triton

from cryoseed.backends.triton.primitives.kernels.weighted_sqdiff_sum_indexed import (
    weighted_sqdiff_sum_indexed_cplx_bwd_input_other_kernel as _weighted_sqdiff_sum_indexed_cplx_bwd_input_other_kernel,
    weighted_sqdiff_sum_indexed_cplx_bwd_weight_kernel as _weighted_sqdiff_sum_indexed_cplx_bwd_weight_kernel,
    weighted_sqdiff_sum_indexed_cplx_fwd_kernel as _weighted_sqdiff_sum_indexed_cplx_fwd_kernel,
    weighted_sqdiff_sum_indexed_cplx_partial_fwd_kernel as _weighted_sqdiff_sum_indexed_cplx_partial_fwd_kernel,
    weighted_sqdiff_sum_indexed_partial_reduce_fwd_kernel as _weighted_sqdiff_sum_indexed_partial_reduce_fwd_kernel,
)
from cryoseed.backends.triton.primitives.kernels.weighted_sqdiff_sum_broadcast import (
    weighted_sqdiff_sum_broadcast_cplx_tile_bwd_input_kernel as _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_input_kernel,
    weighted_sqdiff_sum_broadcast_cplx_tile_bwd_other_kernel as _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_other_kernel,
    weighted_sqdiff_sum_broadcast_cplx_tile_bwd_weight_kernel as _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_weight_kernel,
    weighted_sqdiff_sum_broadcast_cplx_tile_fwd_kernel as _weighted_sqdiff_sum_broadcast_cplx_tile_fwd_kernel,
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


def _maybe_int64(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int64 else x.to(torch.int64)


def _should_use_2stage(N: int, D: int, device, BLOCK_D: int = 512) -> bool:
    props = torch.cuda.get_device_properties(device)
    SM = props.multi_processor_count
    target_ctas = SM * 12
    P = (D + BLOCK_D - 1) // BLOCK_D
    return (N < target_ctas) and (N * P >= target_ctas)


def _use_custom_backward(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _weighted_sqdiff_sum_1stage(x, y, w, x_idx, y_idx, out, D_BUCKET: int):
    x_ri = torch.view_as_real(x)
    y_ri = torch.view_as_real(y)

    N = x_idx.numel()
    D = w.shape[0]

    grid = (N,)
    _weighted_sqdiff_sum_indexed_cplx_fwd_kernel[grid](
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
    _weighted_sqdiff_sum_indexed_cplx_partial_fwd_kernel[grid1](
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
    _weighted_sqdiff_sum_indexed_partial_reduce_fwd_kernel[grid2](
        partial,
        out,
        P,
        p_stride0=partial.stride(0),
        p_stride1=partial.stride(1),
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    return out


def _weighted_sqdiff_sum_indexed_forward_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    out: torch.Tensor,
    prefer_2stage: bool | None,
):
    D_BUCKET = d_bucket(int(w.shape[0]))

    if prefer_2stage is None:
        prefer_2stage = _should_use_2stage(int(x_idx.numel()), int(w.shape[0]), x.device, BLOCK_D=512)

    if prefer_2stage:
        return _weighted_sqdiff_sum_2stage(x, y, w, x_idx, y_idx, out, D_BUCKET=D_BUCKET)

    return _weighted_sqdiff_sum_1stage(x, y, w, x_idx, y_idx, out, D_BUCKET=D_BUCKET)


def _weighted_sqdiff_sum_indexed_backward_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    x_idx: torch.Tensor,
    y_idx: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    needs_grad_x: bool,
    needs_grad_y: bool,
    needs_grad_w: bool,
):
    x_ri = torch.view_as_real(x)
    y_ri = torch.view_as_real(y)
    N = int(x_idx.numel())
    D = int(w.shape[0])
    D_BUCKET = d_bucket(D)

    grad_output = grad_output.reshape(-1).contiguous()

    grad_x = torch.zeros_like(x) if needs_grad_x else None
    grad_y = torch.zeros_like(y) if needs_grad_y else None
    grad_w = torch.zeros_like(w) if needs_grad_w else None

    if needs_grad_x or needs_grad_y:
        # Use dedicated scratch buffers for optional outputs so Triton autotune
        # can safely reset side-effect targets to zero without touching inputs.
        grad_x_ri = torch.view_as_real(grad_x) if grad_x is not None else torch.empty_like(x_ri)
        grad_y_ri = torch.view_as_real(grad_y) if grad_y is not None else torch.empty_like(y_ri)

        grid_xy = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
        _weighted_sqdiff_sum_indexed_cplx_bwd_input_other_kernel[grid_xy](
            x_ri,
            y_ri,
            w,
            x_idx,
            y_idx,
            grad_output,
            grad_x_ri,
            grad_y_ri,
            D,
            D_BUCKET=D_BUCKET,
            x_stride0=x_ri.stride(0),
            x_stride1=x_ri.stride(1),
            x_stride2=x_ri.stride(2),
            y_stride0=y_ri.stride(0),
            y_stride1=y_ri.stride(1),
            y_stride2=y_ri.stride(2),
            gx_stride0=grad_x_ri.stride(0),
            gx_stride1=grad_x_ri.stride(1),
            gx_stride2=grad_x_ri.stride(2),
            gy_stride0=grad_y_ri.stride(0),
            gy_stride1=grad_y_ri.stride(1),
            gy_stride2=grad_y_ri.stride(2),
            HAS_GRAD_X=needs_grad_x,
            HAS_GRAD_Y=needs_grad_y,
        )

    if needs_grad_w:
        grid_w = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
        _weighted_sqdiff_sum_indexed_cplx_bwd_weight_kernel[grid_w](
            x_ri,
            y_ri,
            x_idx,
            y_idx,
            grad_output,
            grad_w,
            D,
            D_BUCKET=D_BUCKET,
            x_stride0=x_ri.stride(0),
            x_stride1=x_ri.stride(1),
            x_stride2=x_ri.stride(2),
            y_stride0=y_ri.stride(0),
            y_stride1=y_ri.stride(1),
            y_stride2=y_ri.stride(2),
        )

    return grad_x, grad_y, grad_w


def _weighted_sqdiff_sum_broadcast_forward_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    out_ptr: torch.Tensor,
    out_sb: int,
    out_sc: int,
    out_so: int,
):
    x_ri = torch.view_as_real(input)
    y_ri = torch.view_as_real(other)

    B, C_input, D = input.shape
    C_other = other.shape[1]
    D_BUCKET = d_bucket(int(D))

    grid = lambda meta: (
        B * triton.cdiv(C_input, meta["BLOCK_CI"]) * triton.cdiv(C_other, meta["BLOCK_CO"]),
    )

    _weighted_sqdiff_sum_broadcast_cplx_tile_fwd_kernel[grid](
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


def _weighted_sqdiff_sum_broadcast_backward_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    needs_grad_input: bool,
    needs_grad_other: bool,
    needs_grad_weight: bool,
):
    x_ri = torch.view_as_real(input)
    y_ri = torch.view_as_real(other)

    B, C_input, D = input.shape
    C_other = other.shape[1]
    D_BUCKET = d_bucket(int(D))
    grad_output_3d = grad_output.reshape(B, C_input, C_other).contiguous()

    grad_input = torch.zeros_like(input) if needs_grad_input else None
    grad_other = torch.zeros_like(other) if needs_grad_other else None
    grad_weight = torch.zeros_like(weight) if needs_grad_weight else None

    if needs_grad_input:
        grad_input_ri = torch.view_as_real(grad_input)
        grid_x = lambda meta: (
            B * triton.cdiv(C_input, meta["BLOCK_CI"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_input_kernel[grid_x](
            x_ri,
            y_ri,
            weight,
            grad_output_3d,
            grad_input_ri,
            B,
            C_input,
            C_other,
            D,
            g_sb=grad_output_3d.stride(0),
            g_sc=grad_output_3d.stride(1),
            g_so=grad_output_3d.stride(2),
            x_sb=x_ri.stride(0),
            x_sc=x_ri.stride(1),
            x_sd=x_ri.stride(2),
            x_sri=x_ri.stride(3),
            y_sb=y_ri.stride(0),
            y_sc=y_ri.stride(1),
            y_sd=y_ri.stride(2),
            y_sri=y_ri.stride(3),
            gx_sb=grad_input_ri.stride(0),
            gx_sc=grad_input_ri.stride(1),
            gx_sd=grad_input_ri.stride(2),
            gx_sri=grad_input_ri.stride(3),
            D_BUCKET=D_BUCKET,
        )

    if needs_grad_other:
        grad_other_ri = torch.view_as_real(grad_other)
        grid_y = lambda meta: (
            B * triton.cdiv(C_other, meta["BLOCK_CO"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_other_kernel[grid_y](
            x_ri,
            y_ri,
            weight,
            grad_output_3d,
            grad_other_ri,
            B,
            C_input,
            C_other,
            D,
            g_sb=grad_output_3d.stride(0),
            g_sc=grad_output_3d.stride(1),
            g_so=grad_output_3d.stride(2),
            x_sb=x_ri.stride(0),
            x_sc=x_ri.stride(1),
            x_sd=x_ri.stride(2),
            x_sri=x_ri.stride(3),
            y_sb=y_ri.stride(0),
            y_sc=y_ri.stride(1),
            y_sd=y_ri.stride(2),
            y_sri=y_ri.stride(3),
            gy_sb=grad_other_ri.stride(0),
            gy_sc=grad_other_ri.stride(1),
            gy_sd=grad_other_ri.stride(2),
            gy_sri=grad_other_ri.stride(3),
            D_BUCKET=D_BUCKET,
        )

    if needs_grad_weight:
        grid_w = lambda meta: (
            B * triton.cdiv(C_input, meta["BLOCK_CI"]) * triton.cdiv(C_other, meta["BLOCK_CO"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        _weighted_sqdiff_sum_broadcast_cplx_tile_bwd_weight_kernel[grid_w](
            x_ri,
            y_ri,
            grad_output_3d,
            grad_weight,
            B,
            C_input,
            C_other,
            D,
            g_sb=grad_output_3d.stride(0),
            g_sc=grad_output_3d.stride(1),
            g_so=grad_output_3d.stride(2),
            x_sb=x_ri.stride(0),
            x_sc=x_ri.stride(1),
            x_sd=x_ri.stride(2),
            x_sri=x_ri.stride(3),
            y_sb=y_ri.stride(0),
            y_sc=y_ri.stride(1),
            y_sd=y_ri.stride(2),
            y_sri=y_ri.stride(3),
            D_BUCKET=D_BUCKET,
        )

    return grad_input, grad_other, grad_weight


class _WeightedSqdiffSumIndexedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        other: torch.Tensor,
        weight: torch.Tensor,
        input_indices: torch.Tensor,
        other_indices: torch.Tensor,
        out: torch.Tensor | None,
        prefer_2stage: bool | None,
    ):
        input_c = input.contiguous()
        other_c = other.contiguous()
        weight_c = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)
        input_indices_c = _maybe_int64(input_indices).contiguous()
        other_indices_c = _maybe_int64(other_indices).contiguous()

        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(input_c, other_c, weight_c, input_indices_c, other_indices_c)

        result = out
        if result is None:
            result = torch.empty((int(input_indices_c.numel()),), device=input.device, dtype=torch.float32)
        else:
            ctx.mark_dirty(result)

        return _weighted_sqdiff_sum_indexed_forward_impl(
            input_c,
            other_c,
            weight_c,
            input_indices_c,
            other_indices_c,
            result,
            prefer_2stage=prefer_2stage,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_c, other_c, weight_c, input_indices_c, other_indices_c = ctx.saved_tensors
        grad_input, grad_other, grad_weight = _weighted_sqdiff_sum_indexed_backward_impl(
            input_c,
            other_c,
            weight_c,
            input_indices_c,
            other_indices_c,
            grad_output,
            needs_grad_x=ctx.needs_input_grad[0],
            needs_grad_y=ctx.needs_input_grad[1],
            needs_grad_w=ctx.needs_input_grad[2],
        )

        if grad_weight is not None:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype).reshape(ctx.weight_shape)

        return grad_input, grad_other, grad_weight, None, None, None, None


class _WeightedSqdiffSumBroadcastFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        other: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor | None,
    ):
        input_c = input.contiguous()
        other_c = other.contiguous()
        weight_c = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)

        B, C_input, _ = input_c.shape
        C_other = other_c.shape[1]
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(input_c, other_c, weight_c)

        result = out
        if result is None:
            result = torch.empty((B * C_input * C_other,), device=input.device, dtype=torch.float32)
            out_sb = C_input * C_other
            out_sc = C_other
            out_so = 1
        else:
            ctx.mark_dirty(result)
            if result.ndim == 1:
                out_sb = C_input * C_other
                out_sc = C_other
                out_so = 1
            else:
                out_sb, out_sc, out_so = result.stride()

        return _weighted_sqdiff_sum_broadcast_forward_impl(
            input_c,
            other_c,
            weight_c,
            result,
            out_sb=out_sb,
            out_sc=out_sc,
            out_so=out_so,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_c, other_c, weight_c = ctx.saved_tensors
        grad_input, grad_other, grad_weight = _weighted_sqdiff_sum_broadcast_backward_impl(
            input_c,
            other_c,
            weight_c,
            grad_output,
            needs_grad_input=ctx.needs_input_grad[0],
            needs_grad_other=ctx.needs_input_grad[1],
            needs_grad_weight=ctx.needs_input_grad[2],
        )

        if grad_weight is not None:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype).reshape(ctx.weight_shape)

        return grad_input, grad_other, grad_weight, None

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
        out: optional float32 tensor, [N]. Treated as a preallocated write buffer,
            so it must have ``requires_grad=False``. If the output needs to remain
            in the autograd graph, do not pass ``out`` and use the returned tensor
            instead.
        prefer_2stage: optional override for 2-stage path in the Triton backend

    Returns:
        out: float32 tensor, [N]
    """
    if not input.is_cuda:
        raise RuntimeError("Triton weighted_sqdiff_sum requires CUDA tensors")
    if input.dim() != 2 or other.dim() != 2:
        raise ValueError(
            "indexed variant requires input and other to be 2D tensors: "
            "input.shape=(N_input, D), other.shape=(N_other, D)"
        )
    if (not input.is_complex()) or (not other.is_complex()):
        raise ValueError("input/other must be complex tensors")
    if input.shape[1] != other.shape[1]:
        raise ValueError("input/other must have the same D")
    if input.device != other.device or weight.device != input.device:
        raise ValueError("input/other/weight must be on the same device")
    weight_1d = weight.reshape(-1)
    if int(weight_1d.numel()) != int(input.shape[1]):
        raise ValueError(f"weight must have {int(input.shape[1])} elements after flatten, got {int(weight_1d.numel())}")

    if int(input_indices.numel()) != int(other_indices.numel()):
        raise ValueError("input_indices and other_indices must have the same number of elements")

    N = int(input_indices.numel())
    if out is not None:
        if out.requires_grad:
            raise ValueError("out must not require grad")
        if out.dtype != torch.float32 or out.device != input.device or out.shape != (N,):
            raise ValueError("out must be float32 on the same device with shape (N,)")

    if _use_custom_backward(input, other, weight):
        return _WeightedSqdiffSumIndexedFn.apply(
            input,
            other,
            weight,
            input_indices,
            other_indices,
            out,
            prefer_2stage,
        )

    input = input.contiguous()
    other = other.contiguous()
    weight = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)
    input_indices = _maybe_int64(input_indices).contiguous()
    other_indices = _maybe_int64(other_indices).contiguous()
    if out is None:
        out = torch.empty((N,), device=input.device, dtype=torch.float32)

    return _weighted_sqdiff_sum_indexed_forward_impl(
        input,
        other,
        weight,
        input_indices,
        other_indices,
        out,
        prefer_2stage=prefer_2stage,
    )


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
        out: optional float32 tensor, [B*C_input*C_other] (flattened). Treated as
            a preallocated write buffer, so it must have ``requires_grad=False``.
            If the output needs to remain in the autograd graph, do not pass
            ``out`` and use the returned tensor instead.

    Returns:
        out_flat: float32 tensor, [B*C_input*C_other]
    """
    if not input.is_cuda:
        raise RuntimeError("Triton weighted_sqdiff_sum requires CUDA tensors")
    if input.dim() != 3 or other.dim() != 3:
        raise ValueError(
            "broadcast variant requires input and other to be 3D tensors: "
            "input.shape=(B, C_input, D), other.shape=(B, C_other, D)"
        )
    if (not input.is_complex()) or (not other.is_complex()):
        raise ValueError("input/other must be complex tensors")
    if input.shape[0] != other.shape[0] or input.shape[2] != other.shape[2]:
        raise ValueError("input/other must have same B and D")
    if input.device != other.device or weight.device != input.device:
        raise ValueError("input/other/weight must be on the same device")
    if int(weight.reshape(-1).numel()) != int(input.shape[2]):
        raise ValueError(f"weight must have {int(input.shape[2])} elements after flatten, got {int(weight.reshape(-1).numel())}")

    B, C_input, _ = input.shape
    C_other = other.shape[1]
    N = B * C_input * C_other

    if out is not None:
        if out.requires_grad:
            raise ValueError("out must not require grad")
        if out.dtype != torch.float32 or out.device != input.device:
            raise ValueError("out must be float32 on the same device")
        if out.ndim == 1:
            if out.shape != (N,):
                raise ValueError("out 1D shape must be (B*C_input*C_other,)")
        elif out.ndim == 3:
            if out.shape != (B, C_input, C_other):
                raise ValueError("out 3D shape must be (B,C_input,C_other)")
        else:
            raise ValueError("out must be a 1D flat tensor or a 3D strided view")

    if _use_custom_backward(input, other, weight):
        return _WeightedSqdiffSumBroadcastFn.apply(input, other, weight, out)

    input = input.contiguous()
    other = other.contiguous()
    weight = weight.to(device=input.device, dtype=torch.float32).contiguous().reshape(-1)

    if out is None:
        out_ptr = torch.empty((N,), device=input.device, dtype=torch.float32)
        out_sb = C_input * C_other
        out_sc = C_other
        out_so = 1
    else:
        out_ptr = out
        if out.ndim == 1:
            out_sb = C_input * C_other
            out_sc = C_other
            out_so = 1
        else:
            out_sb, out_sc, out_so = out.stride()

    return _weighted_sqdiff_sum_broadcast_forward_impl(
        input,
        other,
        weight,
        out_ptr,
        out_sb=out_sb,
        out_sc=out_sc,
        out_so=out_so,
    )

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
            dtype must match the computed output. ``out`` is treated as a preallocated
            write buffer rather than a differentiable input and therefore must have
            ``requires_grad=False``. If the output needs to remain in the autograd
            graph, do not pass ``out`` and use the returned tensor instead.
 
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