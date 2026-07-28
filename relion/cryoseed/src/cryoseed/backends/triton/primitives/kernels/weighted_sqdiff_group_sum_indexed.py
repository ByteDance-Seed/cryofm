"""Triton kernels for grouped indexed weighted squared-difference reduction."""

import triton
import triton.language as tl

__all__ = [
    "weighted_sqdiff_group_sum_indexed_cplx_fwd_kernel",
]


_AUTOTUNE = [
    triton.Config({"BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 256}, num_warps=4),
    triton.Config({"BLOCK_D": 512}, num_warps=8),
]


@triton.autotune(
    configs=_AUTOTUNE,
    key=["D_BUCKET"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def weighted_sqdiff_group_sum_indexed_cplx_fwd_kernel(
    x_ri_ptr,
    y_ri_ptr,
    w_ptr,
    x_idx_ptr,
    y_idx_ptr,
    group_idx_ptr,
    out_ptr,
    G,
    D: tl.constexpr,
    D_BUCKET: tl.constexpr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    x_stride2: tl.constexpr,
    y_stride0: tl.constexpr,
    y_stride1: tl.constexpr,
    y_stride2: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program layout:
    # - pid(0) selects the indexed pair i
    # - pid(1) selects a contiguous feature tile along D
    i = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1)

    xi = tl.load(x_idx_ptr + i).to(tl.int64)
    yi = tl.load(y_idx_ptr + i).to(tl.int64)

    x_base = x_ri_ptr + xi * x_stride0
    y_base = y_ri_ptr + yi * y_stride0

    d = p * BLOCK_D + tl.arange(0, BLOCK_D)
    m = d < D

    x_re = tl.load(x_base + d * x_stride1 + 0 * x_stride2, mask=m, other=0.0).to(tl.float32)
    x_im = tl.load(x_base + d * x_stride1 + 1 * x_stride2, mask=m, other=0.0).to(tl.float32)
    y_re = tl.load(y_base + d * y_stride1 + 0 * y_stride2, mask=m, other=0.0).to(tl.float32)
    y_im = tl.load(y_base + d * y_stride1 + 1 * y_stride2, mask=m, other=0.0).to(tl.float32)

    w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(group_idx_ptr + d, mask=m, other=0).to(tl.int64)

    dr = x_re - y_re
    di = x_im - y_im
    contrib = w * (dr * dr + di * di)

    # Multiple feature positions can map to the same output group, so the
    # grouped reduction is accumulated with atomics.
    out_base = out_ptr + i * out_stride0 + g * out_stride1
    tl.atomic_add(out_base, contrib, mask=m & (g >= 0) & (g < G))