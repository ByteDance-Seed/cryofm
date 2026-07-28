import triton
import triton.language as tl

__all__ = ["central_slice_embed_indexed_kernel"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["L", "P"],
    # Triton autotune runs real candidate kernels, so this accumulation buffer
    # can be polluted during config search. Backprojection reuses the same
    # output volumes across chunks, meaning they may already contain partial
    # sums from earlier launches. Use restore_value so autotune probes see the
    # same initial state without zeroing or double-accumulating the volume.
    restore_value=["out_numer_cplx_ptr", "out_denom_ptr"],
)
# Keep N/K as runtime scalars. Triton may otherwise specialize them into
# constexpr/meta parameters even though this kernel only uses them for runtime
# indexing and bounds checks.
@triton.jit(do_not_specialize=["N", "K"])
def central_slice_embed_indexed_kernel(
    input_cplx_ptr,
    modulation_ptr,
    pixel_weight_ptr,
    input_idx_ptr,
    rot_ptr,
    shift_ptr,
    pose_weight_ptr,
    x_grid_ptr,
    y_grid_ptr,
    out_idx_ptr,
    out_numer_cplx_ptr,
    out_denom_ptr,
    N,
    K,
    L: tl.constexpr,
    P: tl.constexpr,
    CENTER: tl.constexpr,
    input_stride_b: tl.constexpr,
    input_stride_p: tl.constexpr,
    input_stride_cplx: tl.constexpr,
    modulation_stride_b: tl.constexpr,
    modulation_stride_p: tl.constexpr,
    pose_weight_stride_b: tl.constexpr,
    rot_stride_b: tl.constexpr,
    rot_stride_k: tl.constexpr,
    shift_stride_b: tl.constexpr,
    shift_stride_d: tl.constexpr,
    input_idx_stride_b: tl.constexpr,
    out_idx_stride_b: tl.constexpr,
    numer_stride_k: tl.constexpr,
    numer_stride_v: tl.constexpr,
    numer_stride_cplx: tl.constexpr,
    denom_stride_k: tl.constexpr,
    denom_stride_v: tl.constexpr,
    HAS_MODULATION: tl.constexpr,
    HAS_OUT_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    linear = pid * BLOCK + tl.arange(0, BLOCK)

    n = N * P
    mask = linear < n

    s = linear // P
    p = linear % P

    if HAS_OUT_INDEX:
        k = tl.load(out_idx_ptr + s * out_idx_stride_b, mask=mask, other=0).to(tl.int32)
    else:
        k = tl.zeros([BLOCK], dtype=tl.int32)
    mask = mask & (k >= 0) & (k < K)
    k_i64 = k.to(tl.int64)

    i = tl.load(input_idx_ptr + s * input_idx_stride_b, mask=mask, other=0).to(tl.int32)

    x_grid_i32 = tl.load(x_grid_ptr + p, mask=mask, other=0).to(tl.int32)
    y_grid_i32 = tl.load(y_grid_ptr + p, mask=mask, other=0).to(tl.int32)
    x_grid = x_grid_i32.to(tl.float32)
    y_grid = y_grid_i32.to(tl.float32)

    invL = 1.0 / tl.full((), L, tl.float32)
    x_norm = x_grid * invL
    y_norm = y_grid * invL

    shift_base = shift_ptr + s * shift_stride_b
    dx = tl.load(shift_base + 0 * shift_stride_d, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(shift_base + 1 * shift_stride_d, mask=mask, other=0.0).to(tl.float32)

    angle = (-6.283185307179586) * (dx * x_norm + dy * y_norm)
    phase_re = tl.cos(angle)
    phase_im = tl.sin(angle)

    rot_base = rot_ptr + s * rot_stride_b
    r00 = tl.load(rot_base + 0 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)
    r01 = tl.load(rot_base + 1 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)
    r02 = tl.load(rot_base + 2 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)
    r10 = tl.load(rot_base + 3 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)
    r11 = tl.load(rot_base + 4 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)
    r12 = tl.load(rot_base + 5 * rot_stride_k, mask=mask, other=0.0).to(tl.float32)

    x_rot = x_grid * r00 + y_grid * r10
    y_rot = x_grid * r01 + y_grid * r11
    z_rot = x_grid * r02 + y_grid * r12

    x0 = tl.floor(x_rot).to(tl.int32)
    y0 = tl.floor(y_rot).to(tl.int32)
    z0 = tl.floor(z_rot).to(tl.int32)

    fx = x_rot - x0.to(tl.float32)
    fy = y_rot - y0.to(tl.float32)
    fz = z_rot - z0.to(tl.float32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    one = 1.0
    mfx = one - fx
    mfy = one - fy
    mfz = one - fz

    w000 = mfz * mfy * mfx
    w001 = mfz * mfy * fx
    w010 = mfz * fy * mfx
    w011 = mfz * fy * fx
    w100 = fz * mfy * mfx
    w101 = fz * mfy * fx
    w110 = fz * fy * mfx
    w111 = fz * fy * fx

    xi = (x_grid_i32 + CENTER).to(tl.int32)
    yi = (y_grid_i32 + CENTER).to(tl.int32)
    pix_lin = yi * L + xi

    # pixel_weight_ptr stores a full (L*L,) flattened weight map; sampled pixels are gathered via pix_lin.
    pixel_weight = tl.load(pixel_weight_ptr + pix_lin.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
    pose_weight = tl.load(pose_weight_ptr + s * pose_weight_stride_b, mask=mask, other=0.0).to(tl.float32)
    w = pose_weight * pixel_weight

    input_base = (
        input_cplx_ptr
        + i.to(tl.int64) * input_stride_b
        + pix_lin.to(tl.int64) * input_stride_p
    )
    input0_re = tl.load(input_base + 0 * input_stride_cplx, mask=mask, other=0.0).to(tl.float32)
    input0_im = tl.load(input_base + 1 * input_stride_cplx, mask=mask, other=0.0).to(tl.float32)

    input_re = input0_re * phase_re - input0_im * phase_im
    input_im = input0_re * phase_im + input0_im * phase_re

    if HAS_MODULATION:
        modulation_base = (
            modulation_ptr
            + i.to(tl.int64) * modulation_stride_b
            + pix_lin.to(tl.int64) * modulation_stride_p
        )
        modulation = tl.load(modulation_base, mask=mask, other=0.0).to(tl.float32)
        numer_re = w * modulation * input_re
        numer_im = w * modulation * input_im
        denom = w * modulation * modulation
    else:
        numer_re = w * input_re
        numer_im = w * input_im
        denom = w

    vx0 = x0 + CENTER
    vx1 = x1 + CENTER
    vy0 = y0 + CENTER
    vy1 = y1 + CENTER
    vz0 = z0 + CENTER
    vz1 = z1 + CENTER

    in_bounds = (
        (vx0 >= 0) & (vx1 < L)
        & (vy0 >= 0) & (vy1 < L)
        & (vz0 >= 0) & (vz1 < L)
    )
    mask = mask & in_bounds

    m000 = mask
    m001 = mask
    m010 = mask
    m011 = mask

    m100 = mask
    m101 = mask
    m110 = mask
    m111 = mask

    out_numer_base = out_numer_cplx_ptr + k_i64 * numer_stride_k
    out_denom_base = out_denom_ptr + k_i64 * denom_stride_k
    has_out_denom = denom_stride_v != 0

    zero_i64 = tl.zeros([BLOCK], dtype=tl.int64)

    v_lin_raw = (vz0 * (L * L) + vy0 * L + vx0).to(tl.int64)
    v_lin = tl.where(m000, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w000 * numer_re, mask=m000)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w000 * numer_im, mask=m000)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w000 * denom, mask=m000)

    v_lin_raw = (vz0 * (L * L) + vy0 * L + vx1).to(tl.int64)
    v_lin = tl.where(m001, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w001 * numer_re, mask=m001)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w001 * numer_im, mask=m001)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w001 * denom, mask=m001)

    v_lin_raw = (vz0 * (L * L) + vy1 * L + vx0).to(tl.int64)
    v_lin = tl.where(m010, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w010 * numer_re, mask=m010)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w010 * numer_im, mask=m010)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w010 * denom, mask=m010)

    v_lin_raw = (vz0 * (L * L) + vy1 * L + vx1).to(tl.int64)
    v_lin = tl.where(m011, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w011 * numer_re, mask=m011)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w011 * numer_im, mask=m011)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w011 * denom, mask=m011)

    v_lin_raw = (vz1 * (L * L) + vy0 * L + vx0).to(tl.int64)
    v_lin = tl.where(m100, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w100 * numer_re, mask=m100)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w100 * numer_im, mask=m100)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w100 * denom, mask=m100)

    v_lin_raw = (vz1 * (L * L) + vy0 * L + vx1).to(tl.int64)
    v_lin = tl.where(m101, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w101 * numer_re, mask=m101)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w101 * numer_im, mask=m101)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w101 * denom, mask=m101)

    v_lin_raw = (vz1 * (L * L) + vy1 * L + vx0).to(tl.int64)
    v_lin = tl.where(m110, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w110 * numer_re, mask=m110)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w110 * numer_im, mask=m110)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w110 * denom, mask=m110)

    v_lin_raw = (vz1 * (L * L) + vy1 * L + vx1).to(tl.int64)
    v_lin = tl.where(m111, v_lin_raw, zero_i64)
    v_base = out_numer_base + v_lin * numer_stride_v
    tl.atomic_add(v_base + 0 * numer_stride_cplx, w111 * numer_re, mask=m111)
    tl.atomic_add(v_base + 1 * numer_stride_cplx, w111 * numer_im, mask=m111)
    if has_out_denom:
        tl.atomic_add(out_denom_base + v_lin * denom_stride_v, w111 * denom, mask=m111)