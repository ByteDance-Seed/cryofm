import triton
import triton.language as tl


__all__ = [
    "weighted_sqdiff_sum_broadcast_cplx_tile_kernel",
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
def weighted_sqdiff_sum_broadcast_cplx_tile_kernel(
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