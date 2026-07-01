import triton
import triton.language as tl


__all__ = [
    "weighted_sqdiff_sum_indexed_cplx_fwd_kernel",
    "weighted_sqdiff_sum_indexed_cplx_partial_fwd_kernel",
    "weighted_sqdiff_sum_indexed_partial_reduce_fwd_kernel",
    "weighted_sqdiff_sum_indexed_cplx_bwd_input_other_kernel",
    "weighted_sqdiff_sum_indexed_cplx_bwd_weight_kernel",
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
def weighted_sqdiff_sum_indexed_cplx_fwd_kernel(
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
def weighted_sqdiff_sum_indexed_cplx_partial_fwd_kernel(
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
def weighted_sqdiff_sum_indexed_partial_reduce_fwd_kernel(
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


@triton.autotune(
    configs=_AUTOTUNE_1STAGE,
    key=["D_BUCKET"],
    # Triton autotune runs real candidate kernels. These gradients are built via
    # atomic accumulation into fresh per-backward buffers, so reset_to_zero is
    # needed to keep probe runs from polluting grad_x/grad_y.
    reset_to_zero=["grad_x_ri_ptr", "grad_y_ri_ptr"],
)
@triton.jit
def weighted_sqdiff_sum_indexed_cplx_bwd_input_other_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    x_idx_ptr, y_idx_ptr,
    grad_out_ptr,
    grad_x_ri_ptr,
    grad_y_ri_ptr,
    D,
    D_BUCKET: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    y_stride0: tl.constexpr, y_stride1: tl.constexpr, y_stride2: tl.constexpr,
    gx_stride0: tl.constexpr, gx_stride1: tl.constexpr, gx_stride2: tl.constexpr,
    gy_stride0: tl.constexpr, gy_stride1: tl.constexpr, gy_stride2: tl.constexpr,
    HAS_GRAD_X: tl.constexpr,
    HAS_GRAD_Y: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    grad_out = tl.load(grad_out_ptr + i).to(tl.float32)
    scale = 2.0 * grad_out * w

    grad_re = scale * (x_re - y_re)
    grad_im = scale * (x_im - y_im)

    if HAS_GRAD_X:
        grad_x_base = grad_x_ri_ptr + xi * gx_stride0
        tl.atomic_add(grad_x_base + d * gx_stride1 + 0 * gx_stride2, grad_re, mask=m)
        tl.atomic_add(grad_x_base + d * gx_stride1 + 1 * gx_stride2, grad_im, mask=m)

    if HAS_GRAD_Y:
        grad_y_base = grad_y_ri_ptr + yi * gy_stride0
        tl.atomic_add(grad_y_base + d * gy_stride1 + 0 * gy_stride2, -grad_re, mask=m)
        tl.atomic_add(grad_y_base + d * gy_stride1 + 1 * gy_stride2, -grad_im, mask=m)


@triton.autotune(
    configs=_AUTOTUNE_1STAGE,
    key=["D_BUCKET"],
    # Triton autotune runs real candidate kernels. grad_w is a fresh
    # per-backward atomic accumulation buffer, so reset_to_zero prevents probe
    # runs from leaking into the final gradient.
    reset_to_zero=["grad_w_ptr"],
)
@triton.jit
def weighted_sqdiff_sum_indexed_cplx_bwd_weight_kernel(
    x_ri_ptr, y_ri_ptr,
    x_idx_ptr, y_idx_ptr,
    grad_out_ptr,
    grad_w_ptr,
    D,
    D_BUCKET: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    y_stride0: tl.constexpr, y_stride1: tl.constexpr, y_stride2: tl.constexpr,
    BLOCK_D: tl.constexpr,
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

    grad_out = tl.load(grad_out_ptr + i).to(tl.float32)
    grad_w = grad_out * ((x_re - y_re) * (x_re - y_re) + (x_im - y_im) * (x_im - y_im))
    tl.atomic_add(grad_w_ptr + d, grad_w, mask=m)