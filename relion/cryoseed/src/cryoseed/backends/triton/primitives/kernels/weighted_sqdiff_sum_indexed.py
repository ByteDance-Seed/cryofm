import triton
import triton.language as tl


__all__ = [
    "weighted_sqdiff_sum_cplx_kernel",
    "weighted_sqdiff_sum_partial_cplx_kernel",
    "reduce_partial_kernel",
]


# ----------------------------
# Autotune configs (1-stage)
# ----------------------------
_AUTOTUNE_1STAGE = [
    triton.Config({"BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 256}, num_warps=4),
    triton.Config({"BLOCK_D": 512}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_1STAGE, key=["D_BUCKET"])
@triton.jit
def weighted_sqdiff_sum_cplx_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    x_idx_ptr, y_idx_ptr,
    out_ptr,
    D: tl.constexpr,
    D_BUCKET: tl.constexpr,  # only for autotune key
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    y_stride0: tl.constexpr, y_stride1: tl.constexpr, y_stride2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)

    xi = tl.load(x_idx_ptr + i).to(tl.int32)
    yi = tl.load(y_idx_ptr + i).to(tl.int32)

    x_base = x_ri_ptr + xi * x_stride0
    y_base = y_ri_ptr + yi * y_stride0

    acc = tl.zeros((), dtype=tl.float32)
    for d0 in range(0, tl.cdiv(D, BLOCK_D)):
        d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        m = d < D

        x_re = tl.load(x_base + d * x_stride1 + 0 * x_stride2, mask=m, other=0.0).to(tl.float32)
        x_im = tl.load(x_base + d * x_stride1 + 1 * x_stride2, mask=m, other=0.0).to(tl.float32)
        y_re = tl.load(y_base + d * y_stride1 + 0 * y_stride2, mask=m, other=0.0).to(tl.float32)
        y_im = tl.load(y_base + d * y_stride1 + 1 * y_stride2, mask=m, other=0.0).to(tl.float32)

        w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)

        dr = x_re - y_re
        di = x_im - y_im
        acc += tl.sum(w * (dr * dr + di * di), axis=0)

    tl.store(out_ptr + i, acc)


@triton.jit
def weighted_sqdiff_sum_partial_cplx_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    x_idx_ptr, y_idx_ptr,
    partial_ptr,                # [N, P]
    D: tl.constexpr,
    D_BUCKET: tl.constexpr,     # keep aligned with 1-stage (not used)
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    y_stride0: tl.constexpr, y_stride1: tl.constexpr, y_stride2: tl.constexpr,
    p_stride0: tl.constexpr, p_stride1: tl.constexpr,
    BLOCK_D: tl.constexpr,       # FIXED for 2-stage
):
    i = tl.program_id(0)
    p = tl.program_id(1)

    xi = tl.load(x_idx_ptr + i).to(tl.int32)
    yi = tl.load(y_idx_ptr + i).to(tl.int32)

    x_base = x_ri_ptr + xi * x_stride0
    y_base = y_ri_ptr + yi * y_stride0

    d = p * BLOCK_D + tl.arange(0, BLOCK_D)
    m = d < D

    x_re = tl.load(x_base + d * x_stride1 + 0 * x_stride2, mask=m, other=0.0).to(tl.float32)
    x_im = tl.load(x_base + d * x_stride1 + 1 * x_stride2, mask=m, other=0.0).to(tl.float32)
    y_re = tl.load(y_base + d * y_stride1 + 0 * y_stride2, mask=m, other=0.0).to(tl.float32)
    y_im = tl.load(y_base + d * y_stride1 + 1 * y_stride2, mask=m, other=0.0).to(tl.float32)

    w = tl.load(w_ptr + d, mask=m, other=0.0).to(tl.float32)

    dr = x_re - y_re
    di = x_im - y_im
    acc = tl.sum(w * (dr * dr + di * di), axis=0)

    tl.store(partial_ptr + i * p_stride0 + p * p_stride1, acc)


@triton.jit
def reduce_partial_kernel(
    partial_ptr, out_ptr,
    P,                               # runtime
    p_stride0: tl.constexpr, p_stride1: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    for k in range(0, tl.cdiv(P, BLOCK_P)):
        p = k * BLOCK_P + offs
        m = p < P
        acc += tl.load(partial_ptr + i * p_stride0 + p * p_stride1, mask=m, other=0.0)

    tl.store(out_ptr + i, tl.sum(acc, axis=0))