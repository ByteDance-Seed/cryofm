import triton
import triton.language as tl

__all__ = [
    "central_slice_sample_ncdhw_ch2_fwd_kernel",
    "central_slice_sample_ncdhw_ch2_bwd_input_kernel",
]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["L", "P"],
)
# Keep Q as a runtime scalar. Triton may otherwise specialize it into a
# constexpr/meta parameter, which breaks the explicit tl.int32 cast below.
@triton.jit(do_not_specialize=["Q"])
def central_slice_sample_ncdhw_ch2_fwd_kernel(
    input_ptr,
    rotation_ptr,
    output_ptr,
    N_POSE,
    Q,
    L: tl.constexpr,
    P: tl.constexpr,
    SHIFT: tl.constexpr,
    input_stride_n: tl.constexpr,
    input_stride_c: tl.constexpr,
    input_stride_d: tl.constexpr,
    input_stride_h: tl.constexpr,
    input_stride_w: tl.constexpr,
    rotation_stride_pose: tl.constexpr,
    output_stride_pose: tl.constexpr,
    output_stride_c: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    linear = pid * BLOCK + tl.arange(0, BLOCK)
    Q_i32 = Q.to(tl.int32)

    N_ELEM = N_POSE * P
    mask = linear < N_ELEM

    pose = linear // P
    pix = linear - pose * P

    batch = pose // Q_i32

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

    # Match grid_sample(padding_mode="zeros"): even if the sampling location
    # lies slightly outside [-1, 1], any in-bounds neighbors still contribute.
    sample_mask = mask

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

    m000 = sample_mask & inb_z0 & inb_y0 & inb_x0
    m001 = sample_mask & inb_z0 & inb_y0 & inb_x1
    m010 = sample_mask & inb_z0 & inb_y1 & inb_x0
    m011 = sample_mask & inb_z0 & inb_y1 & inb_x1
    m100 = sample_mask & inb_z1 & inb_y0 & inb_x0
    m101 = sample_mask & inb_z1 & inb_y0 & inb_x1
    m110 = sample_mask & inb_z1 & inb_y1 & inb_x0
    m111 = sample_mask & inb_z1 & inb_y1 & inb_x1

    base_c0 = base_n + 0 * input_stride_c
    v0000 = tl.load(input_ptr + base_c0 + d0o + h0o + w0o, mask=m000, other=0.0).to(tl.float32)
    v0010 = tl.load(input_ptr + base_c0 + d0o + h0o + w1o, mask=m001, other=0.0).to(tl.float32)
    v0100 = tl.load(input_ptr + base_c0 + d0o + h1o + w0o, mask=m010, other=0.0).to(tl.float32)
    v0110 = tl.load(input_ptr + base_c0 + d0o + h1o + w1o, mask=m011, other=0.0).to(tl.float32)
    v1000 = tl.load(input_ptr + base_c0 + d1o + h0o + w0o, mask=m100, other=0.0).to(tl.float32)
    v1010 = tl.load(input_ptr + base_c0 + d1o + h0o + w1o, mask=m101, other=0.0).to(tl.float32)
    v1100 = tl.load(input_ptr + base_c0 + d1o + h1o + w0o, mask=m110, other=0.0).to(tl.float32)
    v1110 = tl.load(input_ptr + base_c0 + d1o + h1o + w1o, mask=m111, other=0.0).to(tl.float32)
    out0 = w000 * v0000 + w001 * v0010 + w010 * v0100 + w011 * v0110 + w100 * v1000 + w101 * v1010 + w110 * v1100 + w111 * v1110

    base_c1 = base_n + 1 * input_stride_c
    v0001 = tl.load(input_ptr + base_c1 + d0o + h0o + w0o, mask=m000, other=0.0).to(tl.float32)
    v0011 = tl.load(input_ptr + base_c1 + d0o + h0o + w1o, mask=m001, other=0.0).to(tl.float32)
    v0101 = tl.load(input_ptr + base_c1 + d0o + h1o + w0o, mask=m010, other=0.0).to(tl.float32)
    v0111 = tl.load(input_ptr + base_c1 + d0o + h1o + w1o, mask=m011, other=0.0).to(tl.float32)
    v1001 = tl.load(input_ptr + base_c1 + d1o + h0o + w0o, mask=m100, other=0.0).to(tl.float32)
    v1011 = tl.load(input_ptr + base_c1 + d1o + h0o + w1o, mask=m101, other=0.0).to(tl.float32)
    v1101 = tl.load(input_ptr + base_c1 + d1o + h1o + w0o, mask=m110, other=0.0).to(tl.float32)
    v1111 = tl.load(input_ptr + base_c1 + d1o + h1o + w1o, mask=m111, other=0.0).to(tl.float32)
    out1 = w000 * v0001 + w001 * v0011 + w010 * v0101 + w011 * v0111 + w100 * v1001 + w101 * v1011 + w110 * v1101 + w111 * v1111

    out_base = base_pose + oh + ow
    tl.store(output_ptr + out_base + 0 * output_stride_c, out0, mask=mask)
    tl.store(output_ptr + out_base + 1 * output_stride_c, out1, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["L", "P"],
    # Triton autotune runs real candidate kernels. grad_input is a fresh
    # per-backward atomic scatter buffer, so reset_to_zero keeps probe runs from
    # polluting the final gradient.
    reset_to_zero=["grad_input_ptr"],
)
# Keep Q as a runtime scalar. Triton may otherwise specialize it into a
# constexpr/meta parameter, which breaks the explicit tl.int32 cast below.
@triton.jit(do_not_specialize=["Q"])
def central_slice_sample_ncdhw_ch2_bwd_input_kernel(
    grad_output_ptr,
    rotation_ptr,
    grad_input_ptr,
    N_POSE,
    Q,
    L: tl.constexpr,
    P: tl.constexpr,
    SHIFT: tl.constexpr,
    grad_output_stride_pose: tl.constexpr,
    grad_output_stride_c: tl.constexpr,
    grad_output_stride_h: tl.constexpr,
    grad_output_stride_w: tl.constexpr,
    rotation_stride_pose: tl.constexpr,
    grad_input_stride_n: tl.constexpr,
    grad_input_stride_c: tl.constexpr,
    grad_input_stride_d: tl.constexpr,
    grad_input_stride_h: tl.constexpr,
    grad_input_stride_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    linear = pid * BLOCK + tl.arange(0, BLOCK)
    Q_i32 = Q.to(tl.int32)

    N_ELEM = N_POSE * P
    mask = linear < N_ELEM

    pose = linear // P
    pix = linear - pose * P

    batch = pose // Q_i32

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

    # Match grid_sample backward: partially out-of-bounds samples still scatter
    # to any in-bounds neighbors with the corresponding interpolation weights.
    sample_mask = mask

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

    base_pose = pose.to(tl.int64) * grad_output_stride_pose
    oh = iy.to(tl.int64) * grad_output_stride_h
    ow = ix.to(tl.int64) * grad_output_stride_w
    go_base = base_pose + oh + ow
    grad0 = tl.load(grad_output_ptr + go_base + 0 * grad_output_stride_c, mask=mask, other=0.0).to(tl.float32)
    grad1 = tl.load(grad_output_ptr + go_base + 1 * grad_output_stride_c, mask=mask, other=0.0).to(tl.float32)

    base_n = batch.to(tl.int64) * grad_input_stride_n
    d0o = z0.to(tl.int64) * grad_input_stride_d
    d1o = z1.to(tl.int64) * grad_input_stride_d
    h0o = y0.to(tl.int64) * grad_input_stride_h
    h1o = y1.to(tl.int64) * grad_input_stride_h
    w0o = x0.to(tl.int64) * grad_input_stride_w
    w1o = x1.to(tl.int64) * grad_input_stride_w

    base_c0 = base_n + 0 * grad_input_stride_c
    base_c1 = base_n + 1 * grad_input_stride_c

    m000 = sample_mask & inb_z0 & inb_y0 & inb_x0
    m001 = sample_mask & inb_z0 & inb_y0 & inb_x1
    m010 = sample_mask & inb_z0 & inb_y1 & inb_x0
    m011 = sample_mask & inb_z0 & inb_y1 & inb_x1
    m100 = sample_mask & inb_z1 & inb_y0 & inb_x0
    m101 = sample_mask & inb_z1 & inb_y0 & inb_x1
    m110 = sample_mask & inb_z1 & inb_y1 & inb_x0
    m111 = sample_mask & inb_z1 & inb_y1 & inb_x1

    tl.atomic_add(grad_input_ptr + base_c0 + d0o + h0o + w0o, w000 * grad0, mask=m000)
    tl.atomic_add(grad_input_ptr + base_c1 + d0o + h0o + w0o, w000 * grad1, mask=m000)
    tl.atomic_add(grad_input_ptr + base_c0 + d0o + h0o + w1o, w001 * grad0, mask=m001)
    tl.atomic_add(grad_input_ptr + base_c1 + d0o + h0o + w1o, w001 * grad1, mask=m001)
    tl.atomic_add(grad_input_ptr + base_c0 + d0o + h1o + w0o, w010 * grad0, mask=m010)
    tl.atomic_add(grad_input_ptr + base_c1 + d0o + h1o + w0o, w010 * grad1, mask=m010)
    tl.atomic_add(grad_input_ptr + base_c0 + d0o + h1o + w1o, w011 * grad0, mask=m011)
    tl.atomic_add(grad_input_ptr + base_c1 + d0o + h1o + w1o, w011 * grad1, mask=m011)
    tl.atomic_add(grad_input_ptr + base_c0 + d1o + h0o + w0o, w100 * grad0, mask=m100)
    tl.atomic_add(grad_input_ptr + base_c1 + d1o + h0o + w0o, w100 * grad1, mask=m100)
    tl.atomic_add(grad_input_ptr + base_c0 + d1o + h0o + w1o, w101 * grad0, mask=m101)
    tl.atomic_add(grad_input_ptr + base_c1 + d1o + h0o + w1o, w101 * grad1, mask=m101)
    tl.atomic_add(grad_input_ptr + base_c0 + d1o + h1o + w0o, w110 * grad0, mask=m110)
    tl.atomic_add(grad_input_ptr + base_c1 + d1o + h1o + w0o, w110 * grad1, mask=m110)
    tl.atomic_add(grad_input_ptr + base_c0 + d1o + h1o + w1o, w111 * grad0, mask=m111)
    tl.atomic_add(grad_input_ptr + base_c1 + d1o + h1o + w1o, w111 * grad1, mask=m111)