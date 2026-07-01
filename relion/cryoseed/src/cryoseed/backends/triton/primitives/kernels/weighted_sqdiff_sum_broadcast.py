import triton
import triton.language as tl


__all__ = [
    "weighted_sqdiff_sum_broadcast_cplx_tile_fwd_kernel",
    "weighted_sqdiff_sum_broadcast_cplx_tile_bwd_input_kernel",
    "weighted_sqdiff_sum_broadcast_cplx_tile_bwd_other_kernel",
    "weighted_sqdiff_sum_broadcast_cplx_tile_bwd_weight_kernel",
]


# ----------------------------
# Autotune configs (tile)
# ----------------------------
_AUTOTUNE_TILE = [
    triton.Config({"BLOCK_CI": 8,  "BLOCK_CO": 8,  "BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_CI": 8,  "BLOCK_CO": 16, "BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_CI": 16, "BLOCK_CO": 8,  "BLOCK_D": 128}, num_warps=4),

    triton.Config({"BLOCK_CI": 8,  "BLOCK_CO": 8,  "BLOCK_D": 256}, num_warps=4),
    triton.Config({"BLOCK_CI": 8,  "BLOCK_CO": 16, "BLOCK_D": 256}, num_warps=8),
    triton.Config({"BLOCK_CI": 16, "BLOCK_CO": 8,  "BLOCK_D": 256}, num_warps=8),

    triton.Config({"BLOCK_CI": 8,  "BLOCK_CO": 8,  "BLOCK_D": 512}, num_warps=8),
]

@triton.autotune(configs=_AUTOTUNE_TILE, key=["D_BUCKET"])
@triton.jit
def weighted_sqdiff_sum_broadcast_cplx_tile_fwd_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    out_ptr,
    B, C_input, C_other,
    out_sb: tl.constexpr, out_sc: tl.constexpr, out_so: tl.constexpr,
    D: tl.constexpr,
    D_BUCKET: tl.constexpr,  # only for autotune key
    x_sb: tl.constexpr, x_sc: tl.constexpr, x_sd: tl.constexpr, x_sri: tl.constexpr,
    y_sb: tl.constexpr, y_sc: tl.constexpr, y_sd: tl.constexpr, y_sri: tl.constexpr,
    BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    ntiles_ci = tl.cdiv(C_input, BLOCK_CI)
    ntiles_co = tl.cdiv(C_other, BLOCK_CO)
    tiles_per_batch = ntiles_ci * ntiles_co

    pid_b = pid // tiles_per_batch
    pid_t = pid - pid_b * tiles_per_batch

    pid_b_mask = pid_b < B

    pid_b_i64 = pid_b.to(tl.int64)

    tile_ci = pid_t // ntiles_co
    tile_co = pid_t - tile_ci * ntiles_co

    ci0 = tile_ci * BLOCK_CI
    co0 = tile_co * BLOCK_CO

    ci = ci0 + tl.arange(0, BLOCK_CI)[:, None]  # [BCI, 1]
    co = co0 + tl.arange(0, BLOCK_CO)[None, :]  # [1, BCO]

    ci_mask = (ci < C_input) & pid_b_mask
    co_mask = (co < C_other) & pid_b_mask
    out_mask = ci_mask & co_mask

    acc = tl.zeros((BLOCK_CI, BLOCK_CO), dtype=tl.float32)

    for d0 in range(0, tl.cdiv(D, BLOCK_D)):
        d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = d < D
        d_i64 = d.to(tl.int64)

        # x: [BCI, BD]
        x_base = x_ri_ptr + pid_b_i64 * x_sb + ci.to(tl.int64) * x_sc + d_i64[None, :] * x_sd
        x_re = tl.load(x_base + 0 * x_sri, mask=ci_mask & dm[None, :], other=0.0).to(tl.float32)
        x_im = tl.load(x_base + 1 * x_sri, mask=ci_mask & dm[None, :], other=0.0).to(tl.float32)

        # y: [BCO, BD]
        co_vec = (co0 + tl.arange(0, BLOCK_CO))[:, None]
        y_base = y_ri_ptr + pid_b_i64 * y_sb + co_vec.to(tl.int64) * y_sc + d_i64[None, :] * y_sd
        y_mask = (co_vec < C_other) & dm[None, :] & pid_b_mask
        y_re = tl.load(y_base + 0 * y_sri, mask=y_mask, other=0.0).to(tl.float32)
        y_im = tl.load(y_base + 1 * y_sri, mask=y_mask, other=0.0).to(tl.float32)

        w = tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32)

        dr = x_re[:, None, :] - y_re[None, :, :]
        di = x_im[:, None, :] - y_im[None, :, :]
        acc += tl.sum((dr * dr + di * di) * w[None, None, :], axis=2)

    out_idx = (
        pid_b_i64 * out_sb
        + ci.to(tl.int64) * out_sc
        + co.to(tl.int64) * out_so
    )
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.autotune(configs=_AUTOTUNE_TILE, key=["D_BUCKET"])
@triton.jit
def weighted_sqdiff_sum_broadcast_cplx_tile_bwd_input_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    grad_out_ptr,
    grad_x_ri_ptr,
    B, C_input, C_other, D,
    g_sb: tl.constexpr, g_sc: tl.constexpr, g_so: tl.constexpr,
    x_sb: tl.constexpr, x_sc: tl.constexpr, x_sd: tl.constexpr, x_sri: tl.constexpr,
    y_sb: tl.constexpr, y_sc: tl.constexpr, y_sd: tl.constexpr, y_sri: tl.constexpr,
    gx_sb: tl.constexpr, gx_sc: tl.constexpr, gx_sd: tl.constexpr, gx_sri: tl.constexpr,
    D_BUCKET: tl.constexpr,
    BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    ntiles_ci = tl.cdiv(C_input, BLOCK_CI)
    pid_b = pid0 // ntiles_ci
    tile_ci = pid0 - pid_b * ntiles_ci

    ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    d = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)
    ci_mask = ci < C_input
    d_mask = d < D

    pid_b_i64 = pid_b.to(tl.int64)
    d_i64 = d.to(tl.int64)

    x_base = x_ri_ptr + pid_b_i64 * x_sb + ci[:, None].to(tl.int64) * x_sc + d_i64[None, :] * x_sd
    x_re = tl.load(x_base + 0 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    x_im = tl.load(x_base + 1 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    w = 2.0 * tl.load(w_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    acc_re = tl.zeros((BLOCK_CI, BLOCK_D), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_CI, BLOCK_D), dtype=tl.float32)

    for co0 in range(0, tl.cdiv(C_other, BLOCK_CO)):
        co = co0 * BLOCK_CO + tl.arange(0, BLOCK_CO)
        co_mask = co < C_other

        y_base = y_ri_ptr + pid_b_i64 * y_sb + co[:, None].to(tl.int64) * y_sc + d_i64[None, :] * y_sd
        y_re = tl.load(y_base + 0 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        y_im = tl.load(y_base + 1 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        grad_ptr = grad_out_ptr + pid_b_i64 * g_sb + ci[:, None].to(tl.int64) * g_sc + co[None, :].to(tl.int64) * g_so
        grad = tl.load(grad_ptr, mask=ci_mask[:, None] & co_mask[None, :], other=0.0).to(tl.float32)
        scale = grad[:, :, None] * w[None, None, :]

        acc_re += tl.sum(scale * (x_re[:, None, :] - y_re[None, :, :]), axis=1)
        acc_im += tl.sum(scale * (x_im[:, None, :] - y_im[None, :, :]), axis=1)

    out_base = grad_x_ri_ptr + pid_b_i64 * gx_sb + ci[:, None].to(tl.int64) * gx_sc + d_i64[None, :] * gx_sd
    out_mask = ci_mask[:, None] & d_mask[None, :]
    tl.store(out_base + 0 * gx_sri, acc_re, mask=out_mask)
    tl.store(out_base + 1 * gx_sri, acc_im, mask=out_mask)


@triton.autotune(configs=_AUTOTUNE_TILE, key=["D_BUCKET"])
@triton.jit
def weighted_sqdiff_sum_broadcast_cplx_tile_bwd_other_kernel(
    x_ri_ptr, y_ri_ptr, w_ptr,
    grad_out_ptr,
    grad_y_ri_ptr,
    B, C_input, C_other, D,
    g_sb: tl.constexpr, g_sc: tl.constexpr, g_so: tl.constexpr,
    x_sb: tl.constexpr, x_sc: tl.constexpr, x_sd: tl.constexpr, x_sri: tl.constexpr,
    y_sb: tl.constexpr, y_sc: tl.constexpr, y_sd: tl.constexpr, y_sri: tl.constexpr,
    gy_sb: tl.constexpr, gy_sc: tl.constexpr, gy_sd: tl.constexpr, gy_sri: tl.constexpr,
    D_BUCKET: tl.constexpr,
    BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    ntiles_co = tl.cdiv(C_other, BLOCK_CO)
    pid_b = pid0 // ntiles_co
    tile_co = pid0 - pid_b * ntiles_co

    co = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    d = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)
    co_mask = co < C_other
    d_mask = d < D

    pid_b_i64 = pid_b.to(tl.int64)
    d_i64 = d.to(tl.int64)

    y_base = y_ri_ptr + pid_b_i64 * y_sb + co[:, None].to(tl.int64) * y_sc + d_i64[None, :] * y_sd
    y_re = tl.load(y_base + 0 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    y_im = tl.load(y_base + 1 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    w = 2.0 * tl.load(w_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    acc_re = tl.zeros((BLOCK_CO, BLOCK_D), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_CO, BLOCK_D), dtype=tl.float32)

    for ci0 in range(0, tl.cdiv(C_input, BLOCK_CI)):
        ci = ci0 * BLOCK_CI + tl.arange(0, BLOCK_CI)
        ci_mask = ci < C_input

        x_base = x_ri_ptr + pid_b_i64 * x_sb + ci[:, None].to(tl.int64) * x_sc + d_i64[None, :] * x_sd
        x_re = tl.load(x_base + 0 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        x_im = tl.load(x_base + 1 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        grad_ptr = grad_out_ptr + pid_b_i64 * g_sb + ci[:, None].to(tl.int64) * g_sc + co[None, :].to(tl.int64) * g_so
        grad = tl.load(grad_ptr, mask=ci_mask[:, None] & co_mask[None, :], other=0.0).to(tl.float32)
        scale = grad[:, :, None] * w[None, None, :]

        acc_re += tl.sum(scale * (y_re[None, :, :] - x_re[:, None, :]), axis=0)
        acc_im += tl.sum(scale * (y_im[None, :, :] - x_im[:, None, :]), axis=0)

    out_base = grad_y_ri_ptr + pid_b_i64 * gy_sb + co[:, None].to(tl.int64) * gy_sc + d_i64[None, :] * gy_sd
    out_mask = co_mask[:, None] & d_mask[None, :]
    tl.store(out_base + 0 * gy_sri, acc_re, mask=out_mask)
    tl.store(out_base + 1 * gy_sri, acc_im, mask=out_mask)


@triton.autotune(
    configs=_AUTOTUNE_TILE,
    key=["D_BUCKET"],
    # Triton autotune runs real candidate kernels. grad_w is a fresh
    # per-backward atomic accumulation buffer, so reset_to_zero prevents probe
    # runs from leaking into the final gradient.
    reset_to_zero=["grad_w_ptr"],
)
@triton.jit
def weighted_sqdiff_sum_broadcast_cplx_tile_bwd_weight_kernel(
    x_ri_ptr, y_ri_ptr,
    grad_out_ptr,
    grad_w_ptr,
    B, C_input, C_other, D,
    g_sb: tl.constexpr, g_sc: tl.constexpr, g_so: tl.constexpr,
    x_sb: tl.constexpr, x_sc: tl.constexpr, x_sd: tl.constexpr, x_sri: tl.constexpr,
    y_sb: tl.constexpr, y_sc: tl.constexpr, y_sd: tl.constexpr, y_sri: tl.constexpr,
    D_BUCKET: tl.constexpr,
    BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    ntiles_ci = tl.cdiv(C_input, BLOCK_CI)
    ntiles_co = tl.cdiv(C_other, BLOCK_CO)
    tiles_per_batch = ntiles_ci * ntiles_co
    pid_b = pid0 // tiles_per_batch
    pid_t = pid0 - pid_b * tiles_per_batch
    tile_ci = pid_t // ntiles_co
    tile_co = pid_t - tile_ci * ntiles_co

    ci = tile_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    co = tile_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    d = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)
    ci_mask = ci < C_input
    co_mask = co < C_other
    d_mask = d < D

    pid_b_i64 = pid_b.to(tl.int64)
    d_i64 = d.to(tl.int64)

    x_base = x_ri_ptr + pid_b_i64 * x_sb + ci[:, None].to(tl.int64) * x_sc + d_i64[None, :] * x_sd
    x_re = tl.load(x_base + 0 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    x_im = tl.load(x_base + 1 * x_sri, mask=ci_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    y_base = y_ri_ptr + pid_b_i64 * y_sb + co[:, None].to(tl.int64) * y_sc + d_i64[None, :] * y_sd
    y_re = tl.load(y_base + 0 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    y_im = tl.load(y_base + 1 * y_sri, mask=co_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    grad_ptr = grad_out_ptr + pid_b_i64 * g_sb + ci[:, None].to(tl.int64) * g_sc + co[None, :].to(tl.int64) * g_so
    grad = tl.load(grad_ptr, mask=ci_mask[:, None] & co_mask[None, :], other=0.0).to(tl.float32)

    sq = (x_re[:, None, :] - y_re[None, :, :]) * (x_re[:, None, :] - y_re[None, :, :])
    sq += (x_im[:, None, :] - y_im[None, :, :]) * (x_im[:, None, :] - y_im[None, :, :])
    acc = tl.sum(tl.sum(grad[:, :, None] * sq, axis=0), axis=0)

    tl.atomic_add(grad_w_ptr + d, acc, mask=d_mask)