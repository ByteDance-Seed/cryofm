import triton
import triton.language as tl

__all__ = [
    "central_slice_sample_ndhwc_ch2_kernel",
]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["L", "P"],
)
@triton.jit
def central_slice_sample_ndhwc_ch2_kernel(
    input_ptr,
    rotation_ptr,
    output_ptr,
    N_POSE,
    Q: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    SHIFT: tl.constexpr,
    input_stride_n: tl.constexpr,
    input_stride_d: tl.constexpr,
    input_stride_h: tl.constexpr,
    input_stride_w: tl.constexpr,
    input_stride_c: tl.constexpr,
    rotation_stride_pose: tl.constexpr,
    output_stride_pose: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_w: tl.constexpr,
    output_stride_c: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    linear = pid * BLOCK + tl.arange(0, BLOCK)

    N_ELEM = N_POSE * P
    mask = linear < N_ELEM

    pose = linear // P
    pix = linear - pose * P

    batch = pose // Q

    iy = pix // L
    ix = pix - iy * L

    denom = tl.full((), L - 1, tl.float32)
    x = (-1.0 + 2.0 * ix.to(tl.float32) / denom) - SHIFT
    y = (-1.0 + 2.0 * iy.to(tl.float32) / denom) - SHIFT
    z = 0.0

    rot_base = rotation_ptr + pose.to(tl.int64) * rotation_stride_pose
    r00 = tl.load(rot_base + 0, mask=mask, other=0.0).to(tl.float32)
    r01 = tl.load(rot_base + 1, mask=mask, other=0.0).to(tl.float32)
    r02 = tl.load(rot_base + 2, mask=mask, other=0.0).to(tl.float32)
    r10 = tl.load(rot_base + 3, mask=mask, other=0.0).to(tl.float32)
    r11 = tl.load(rot_base + 4, mask=mask, other=0.0).to(tl.float32)
    r12 = tl.load(rot_base + 5, mask=mask, other=0.0).to(tl.float32)
    r20 = tl.load(rot_base + 6, mask=mask, other=0.0).to(tl.float32)
    r21 = tl.load(rot_base + 7, mask=mask, other=0.0).to(tl.float32)
    r22 = tl.load(rot_base + 8, mask=mask, other=0.0).to(tl.float32)

    x_rot = x * r00 + y * r10 + z * r20
    y_rot = x * r01 + y * r11 + z * r21
    z_rot = x * r02 + y * r12 + z * r22

    x_rot = x_rot + SHIFT
    y_rot = y_rot + SHIFT
    z_rot = z_rot + SHIFT

    in_range = (
        (x_rot >= -1.0)
        & (x_rot <= 1.0)
        & (y_rot >= -1.0)
        & (y_rot <= 1.0)
        & (z_rot >= -1.0)
        & (z_rot <= 1.0)
    )
    sample_mask = mask & in_range

    xf = (x_rot + 1.0) * (0.5 * denom)
    yf = (y_rot + 1.0) * (0.5 * denom)
    zf = (z_rot + 1.0) * (0.5 * denom)

    x0 = tl.floor(xf).to(tl.int32)
    y0 = tl.floor(yf).to(tl.int32)
    z0 = tl.floor(zf).to(tl.int32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    fx = xf - x0.to(tl.float32)
    fy = yf - y0.to(tl.float32)
    fz = zf - z0.to(tl.float32)

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

    inb_x0 = (x0 >= 0) & (x0 < L)
    inb_x1 = (x1 >= 0) & (x1 < L)
    inb_y0 = (y0 >= 0) & (y0 < L)
    inb_y1 = (y1 >= 0) & (y1 < L)
    inb_z0 = (z0 >= 0) & (z0 < L)
    inb_z1 = (z1 >= 0) & (z1 < L)

    base_n = batch.to(tl.int64) * input_stride_n
    d0o = z0.to(tl.int64) * input_stride_d
    d1o = z1.to(tl.int64) * input_stride_d
    h0o = y0.to(tl.int64) * input_stride_h
    h1o = y1.to(tl.int64) * input_stride_h
    w0o = x0.to(tl.int64) * input_stride_w
    w1o = x1.to(tl.int64) * input_stride_w

    base_pose = pose.to(tl.int64) * output_stride_pose
    oh = iy.to(tl.int64) * output_stride_h
    ow = ix.to(tl.int64) * output_stride_w
    out_base = base_pose + oh + ow

    m000 = sample_mask & inb_z0 & inb_y0 & inb_x0
    m001 = sample_mask & inb_z0 & inb_y0 & inb_x1
    m010 = sample_mask & inb_z0 & inb_y1 & inb_x0
    m011 = sample_mask & inb_z0 & inb_y1 & inb_x1
    m100 = sample_mask & inb_z1 & inb_y0 & inb_x0
    m101 = sample_mask & inb_z1 & inb_y0 & inb_x1
    m110 = sample_mask & inb_z1 & inb_y1 & inb_x0
    m111 = sample_mask & inb_z1 & inb_y1 & inb_x1

    base000 = base_n + d0o + h0o + w0o
    base001 = base_n + d0o + h0o + w1o
    base010 = base_n + d0o + h1o + w0o
    base011 = base_n + d0o + h1o + w1o
    base100 = base_n + d1o + h0o + w0o
    base101 = base_n + d1o + h0o + w1o
    base110 = base_n + d1o + h1o + w0o
    base111 = base_n + d1o + h1o + w1o

    v000_re = tl.load(input_ptr + base000 + 0 * input_stride_c, mask=m000, other=0.0).to(tl.float32)
    v001_re = tl.load(input_ptr + base001 + 0 * input_stride_c, mask=m001, other=0.0).to(tl.float32)
    v010_re = tl.load(input_ptr + base010 + 0 * input_stride_c, mask=m010, other=0.0).to(tl.float32)
    v011_re = tl.load(input_ptr + base011 + 0 * input_stride_c, mask=m011, other=0.0).to(tl.float32)
    v100_re = tl.load(input_ptr + base100 + 0 * input_stride_c, mask=m100, other=0.0).to(tl.float32)
    v101_re = tl.load(input_ptr + base101 + 0 * input_stride_c, mask=m101, other=0.0).to(tl.float32)
    v110_re = tl.load(input_ptr + base110 + 0 * input_stride_c, mask=m110, other=0.0).to(tl.float32)
    v111_re = tl.load(input_ptr + base111 + 0 * input_stride_c, mask=m111, other=0.0).to(tl.float32)
    out_re = (
        w000 * v000_re
        + w001 * v001_re
        + w010 * v010_re
        + w011 * v011_re
        + w100 * v100_re
        + w101 * v101_re
        + w110 * v110_re
        + w111 * v111_re
    )

    v000_im = tl.load(input_ptr + base000 + 1 * input_stride_c, mask=m000, other=0.0).to(tl.float32)
    v001_im = tl.load(input_ptr + base001 + 1 * input_stride_c, mask=m001, other=0.0).to(tl.float32)
    v010_im = tl.load(input_ptr + base010 + 1 * input_stride_c, mask=m010, other=0.0).to(tl.float32)
    v011_im = tl.load(input_ptr + base011 + 1 * input_stride_c, mask=m011, other=0.0).to(tl.float32)
    v100_im = tl.load(input_ptr + base100 + 1 * input_stride_c, mask=m100, other=0.0).to(tl.float32)
    v101_im = tl.load(input_ptr + base101 + 1 * input_stride_c, mask=m101, other=0.0).to(tl.float32)
    v110_im = tl.load(input_ptr + base110 + 1 * input_stride_c, mask=m110, other=0.0).to(tl.float32)
    v111_im = tl.load(input_ptr + base111 + 1 * input_stride_c, mask=m111, other=0.0).to(tl.float32)
    out_im = (
        w000 * v000_im
        + w001 * v001_im
        + w010 * v010_im
        + w011 * v011_im
        + w100 * v100_im
        + w101 * v101_im
        + w110 * v110_im
        + w111 * v111_im
    )

    tl.store(output_ptr + out_base + 0 * output_stride_c, out_re, mask=mask)
    tl.store(output_ptr + out_base + 1 * output_stride_c, out_im, mask=mask)