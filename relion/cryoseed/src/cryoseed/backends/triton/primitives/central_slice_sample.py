"""Triton primitives for sampling a z=0 plane from a rotated 3D volume.

Given a 3D input tensor and a batch of rotation matrices, these functions sample the
input on the plane z=0 using trilinear interpolation (out-of-bounds values are
filled with 0). Internally, coordinates are treated as row-vectors (x, y, z) and
right-multiplied by the rotation matrix: ``coords_rot = coords @ R``.

This is equivalent to calling ``torch.nn.functional.grid_sample`` with a specific
grid construction (align-corners mapping), but implemented in Triton.
"""

import torch
from torch import Tensor
import triton

from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_ch2 import (
    central_slice_sample_ncdhw_ch2_kernel as _central_slice_sample_ncdhw_ch2_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_gen import (
    central_slice_sample_ncdhw_gen_kernel as _central_slice_sample_ncdhw_gen_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_ch2 import (
    central_slice_sample_ndhwc_ch2_kernel as _central_slice_sample_ndhwc_ch2_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_gen import (
    central_slice_sample_ndhwc_gen_kernel as _central_slice_sample_ndhwc_gen_kernel,
)

__all__ = [
    "central_slice_sample",
    "central_slice_sample_ncdhw",
    "central_slice_sample_ndhwc",
]

def _check_rotation(rotation: Tensor, N: int) -> int:
    """Validate rotation tensor.

    Rotation convention matches the Torch backend:
    - rotation has shape (N, Q, 3, 3)
    - coordinates are treated as row-vectors (x, y, z) and right-multiplied by R
      inside the kernels.
    """
    if rotation.ndim != 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must be (N, Q, 3, 3), got {tuple(rotation.shape)}")
    if rotation.shape[0] != N:
        raise ValueError(f"rotation.shape[0] must equal N, got {rotation.shape[0]} vs {N}")
    return int(rotation.shape[1])


def central_slice_sample_ncdhw(
    input: Tensor,
    rotation: Tensor,
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Sample a z=0 plane from a rotated 3D volume (NCDHW layout).

    Args:
        input: Volume tensor of shape ``(N, C, D, H, W)``. Must be cubic.
        rotation: Rotation matrices of shape ``(N, Q, 3, 3)``.
        use_ch2:
            - ``None``: automatically enable the specialized ``C==2`` kernel when
              applicable.
            - ``True``: force the specialized kernel (requires ``C==2``).
        align_dc:
            If ``True`` and ``L`` is even, applies a small shift in normalized
            coordinates so rotations are performed about the continuous origin.

    Returns:
        Output tensor with shape ``(N, Q, C, L, L)``, where ``L == W``.
    """

    if input.ndim != 5:
        raise ValueError(f"input must be 5D (N, C, D, H, W), got {tuple(input.shape)}")

    N, C, D, H, W = input.shape
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    if use_ch2 is None:
        use_ch2 = C == 2
    elif use_ch2 and C != 2:
        raise ValueError(f"use_ch2=True requires C==2, got C={C}")

    Q = _check_rotation(rotation, N)
    L = int(W)

    if L < 2:
        return torch.zeros((N, Q, C, L, L), device=input.device, dtype=input.dtype)

    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()
    output = torch.empty((N * Q, C, L, L), device=input.device, dtype=input.dtype)

    shift = ((1.0 / (L - 1)) if (L % 2 == 0) else 0.0) if align_dc else 0.0
    # For even L under the align-corners mapping, the continuous origin lies
    # between pixel centers. If align_dc=True, we rotate about the continuous origin
    # by applying a small SHIFT before/after rotation in normalized [-1, 1] coords.
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ncdhw_ch2_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            Q=Q,
            L=L,
            P=P,
            SHIFT=shift,
            input_stride_n=input.stride(0),
            input_stride_c=input.stride(1),
            input_stride_d=input.stride(2),
            input_stride_h=input.stride(3),
            input_stride_w=input.stride(4),
            rotation_stride_pose=rotation_flat.stride(0),
            output_stride_pose=output.stride(0),
            output_stride_c=output.stride(1),
            output_stride_h=output.stride(2),
            output_stride_w=output.stride(3),
        )
    else:
        def grid(meta):
            return (triton.cdiv(N_pose * P * C, meta["BLOCK"]),)

        _central_slice_sample_ncdhw_gen_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            C,
            Q=Q,
            L=L,
            P=P,
            SHIFT=shift,
            input_stride_n=input.stride(0),
            input_stride_c=input.stride(1),
            input_stride_d=input.stride(2),
            input_stride_h=input.stride(3),
            input_stride_w=input.stride(4),
            rotation_stride_pose=rotation_flat.stride(0),
            output_stride_pose=output.stride(0),
            output_stride_c=output.stride(1),
            output_stride_h=output.stride(2),
            output_stride_w=output.stride(3),
        )

    return output.view(N, Q, C, L, L)


def central_slice_sample_ndhwc(
    input: Tensor,
    rotation: Tensor,
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Sample a z=0 plane from a rotated 3D volume (NDHWC layout).

    Args:
        input: Volume tensor of shape ``(N, D, H, W, C)``. Must be cubic.
        rotation: Rotation matrices of shape ``(N, Q, 3, 3)``.
        use_ch2: Same semantics as :func:`central_slice_sample_ncdhw`.
        align_dc: Same semantics as :func:`central_slice_sample_ncdhw`.

    Returns:
        Output tensor with shape ``(N, Q, L, L, C)``, where ``L == W``.
    """

    if input.ndim != 5:
        raise ValueError(f"input must be 5D (N, D, H, W, C), got {tuple(input.shape)}")

    N, D, H, W, C = input.shape
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    if use_ch2 is None:
        use_ch2 = C == 2
    elif use_ch2 and C != 2:
        raise ValueError(f"use_ch2=True requires C==2, got C={C}")

    Q = _check_rotation(rotation, N)
    L = int(W)

    if L < 2:
        return torch.zeros((N, Q, L, L, C), device=input.device, dtype=input.dtype)

    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()
    output = torch.empty((N * Q, L, L, C), device=input.device, dtype=input.dtype)

    shift = ((1.0 / (L - 1)) if (L % 2 == 0) else 0.0) if align_dc else 0.0
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ndhwc_ch2_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            Q=Q,
            L=L,
            P=P,
            SHIFT=shift,
            input_stride_n=input.stride(0),
            input_stride_d=input.stride(1),
            input_stride_h=input.stride(2),
            input_stride_w=input.stride(3),
            input_stride_c=input.stride(4),
            rotation_stride_pose=rotation_flat.stride(0),
            output_stride_pose=output.stride(0),
            output_stride_h=output.stride(1),
            output_stride_w=output.stride(2),
            output_stride_c=output.stride(3),
        )
    else:
        def grid(meta):
            return (triton.cdiv(N_pose * P * C, meta["BLOCK"]),)

        _central_slice_sample_ndhwc_gen_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            C,
            Q=Q,
            L=L,
            P=P,
            SHIFT=shift,
            input_stride_n=input.stride(0),
            input_stride_d=input.stride(1),
            input_stride_h=input.stride(2),
            input_stride_w=input.stride(3),
            input_stride_c=input.stride(4),
            rotation_stride_pose=rotation_flat.stride(0),
            output_stride_pose=output.stride(0),
            output_stride_h=output.stride(1),
            output_stride_w=output.stride(2),
            output_stride_c=output.stride(3),
        )

    return output.view(N, Q, L, L, C)


def central_slice_sample(
    input: Tensor,
    rotation: Tensor,
    *,
    channel_last: bool = False,
    align_dc: bool = False,
) -> Tensor:
    """Layout-dispatch wrapper for :func:`central_slice_sample_ncdhw` / NDHWC.

    Args:
        input:
            - ``channel_last=True``: ``(N, D, H, W, C)``
            - ``channel_last=False``: ``(N, C, D, H, W)``
        rotation: ``(N, Q, 3, 3)``.
        channel_last: Select NDHWC (True) or NCDHW (False).
        align_dc: See :func:`central_slice_sample_ncdhw`.

    Returns:
        - ``channel_last=True``: ``(N, Q, L, L, C)``
        - ``channel_last=False``: ``(N, Q, C, L, L)``
    """

    if channel_last:
        return central_slice_sample_ndhwc(input, rotation, align_dc=align_dc)
    return central_slice_sample_ncdhw(input, rotation, align_dc=align_dc)