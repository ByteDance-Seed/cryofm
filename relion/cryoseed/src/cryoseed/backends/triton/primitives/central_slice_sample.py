"""Triton primitives for sampling a z=0 plane from a rotated 3D volume.

Given a 3D input tensor and a batch of rotation matrices, these functions sample the
input on the plane z=0 using trilinear interpolation (out-of-bounds values are
filled with 0). Internally, coordinates are treated as row-vectors (x, y, z) and
right-multiplied by the rotation matrix: ``coords_rot = coords @ R``.

This is equivalent to calling ``torch.nn.functional.grid_sample`` with a specific
grid construction (align-corners mapping), but implemented in Triton.
"""

from __future__ import annotations

import torch
from torch import Tensor
import triton

from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_ch2 import (
    central_slice_sample_ncdhw_ch2_bwd_input_kernel as _central_slice_sample_ncdhw_ch2_bwd_input_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_ch2 import (
    central_slice_sample_ncdhw_ch2_fwd_kernel as _central_slice_sample_ncdhw_ch2_fwd_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_gen import (
    central_slice_sample_ncdhw_gen_bwd_input_kernel as _central_slice_sample_ncdhw_gen_bwd_input_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ncdhw_gen import (
    central_slice_sample_ncdhw_gen_fwd_kernel as _central_slice_sample_ncdhw_gen_fwd_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_ch2 import (
    central_slice_sample_ndhwc_ch2_bwd_input_kernel as _central_slice_sample_ndhwc_ch2_bwd_input_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_ch2 import (
    central_slice_sample_ndhwc_ch2_fwd_kernel as _central_slice_sample_ndhwc_ch2_fwd_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_gen import (
    central_slice_sample_ndhwc_gen_bwd_input_kernel as _central_slice_sample_ndhwc_gen_bwd_input_kernel,
)
from cryoseed.backends.triton.primitives.kernels.central_slice_sample_ndhwc_gen import (
    central_slice_sample_ndhwc_gen_fwd_kernel as _central_slice_sample_ndhwc_gen_fwd_kernel,
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


def _resolve_shift(L: int, align_dc: bool) -> float:
    """Return the normalized SHIFT used to rotate about the continuous origin."""
    if (not align_dc) or (L % 2 != 0):
        return 0.0
    return 1.0 / (L - 1)


def _resolve_use_ch2(C: int, use_ch2: bool | None) -> bool:
    """Resolve whether to use the specialized ``C==2`` kernel variant."""
    if use_ch2 is None:
        return C == 2
    if use_ch2 and C != 2:
        raise ValueError(f"use_ch2=True requires C==2, got C={C}")
    return use_ch2


def _central_slice_sample_ncdhw_forward(
    input: Tensor,
    rotation: Tensor,
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Forward helper for the NCDHW layout.

    This preserves the original Triton forward path and is shared by the autograd
    wrapper so backward can be added without changing the sampling implementation.
    """
    if input.ndim != 5:
        raise ValueError(f"input must be 5D (N, C, D, H, W), got {tuple(input.shape)}")

    N, C, D, H, W = input.shape
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    use_ch2 = _resolve_use_ch2(C, use_ch2)
    Q = _check_rotation(rotation, N)
    L = int(W)

    if L < 2:
        return torch.zeros((N, Q, C, L, L), device=input.device, dtype=input.dtype)

    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()
    output = torch.empty((N * Q, C, L, L), device=input.device, dtype=input.dtype)

    shift = _resolve_shift(L, align_dc)
    # For even L under the align-corners mapping, the continuous origin lies
    # between pixel centers. If align_dc=True, we rotate about the continuous origin
    # by applying a small SHIFT before/after rotation in normalized [-1, 1] coords.
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ncdhw_ch2_fwd_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            Q,
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

        _central_slice_sample_ncdhw_gen_fwd_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            C,
            Q,
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


def _central_slice_sample_ndhwc_forward(
    input: Tensor,
    rotation: Tensor,
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Forward helper for the NDHWC layout."""
    if input.ndim != 5:
        raise ValueError(f"input must be 5D (N, D, H, W, C), got {tuple(input.shape)}")

    N, D, H, W, C = input.shape
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    use_ch2 = _resolve_use_ch2(C, use_ch2)
    Q = _check_rotation(rotation, N)
    L = int(W)

    if L < 2:
        return torch.zeros((N, Q, L, L, C), device=input.device, dtype=input.dtype)

    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()
    output = torch.empty((N * Q, L, L, C), device=input.device, dtype=input.dtype)

    shift = _resolve_shift(L, align_dc)
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ndhwc_ch2_fwd_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            Q,
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

        _central_slice_sample_ndhwc_gen_fwd_kernel[grid](
            input,
            rotation_flat,
            output,
            N_pose,
            C,
            Q,
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


def _central_slice_sample_ncdhw_backward_input(
    grad_output: Tensor,
    rotation: Tensor,
    input_shape: tuple[int, int, int, int, int],
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Backward helper that computes only ``grad_input`` for the NCDHW layout.

    The operation is the adjoint of the trilinear sampling performed in forward:
    each output gradient value is scattered back to the 8 contributing voxels using
    the same interpolation weights.
    """
    N, C, D, H, W = input_shape
    if grad_output.ndim != 5:
        raise ValueError(f"grad_output must be 5D (N, Q, C, L, L), got {tuple(grad_output.shape)}")
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    use_ch2 = _resolve_use_ch2(C, use_ch2)
    Q = _check_rotation(rotation, N)
    L = int(W)
    if grad_output.shape != (N, Q, C, L, L):
        raise ValueError(
            "grad_output must match sampled output shape "
            f"({N}, {Q}, {C}, {L}, {L}), "
            f"got {tuple(grad_output.shape)}"
        )

    grad_input = torch.zeros(input_shape, device=grad_output.device, dtype=grad_output.dtype)
    if L < 2:
        return grad_input

    # Flatten (N, Q) to the same pose-major layout used by the forward kernels.
    grad_output_flat = grad_output.contiguous().view(N * Q, C, L, L)
    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()

    shift = _resolve_shift(L, align_dc)
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ncdhw_ch2_bwd_input_kernel[grid](
            grad_output_flat,
            rotation_flat,
            grad_input,
            N_pose,
            Q,
            L=L,
            P=P,
            SHIFT=shift,
            grad_output_stride_pose=grad_output_flat.stride(0),
            grad_output_stride_c=grad_output_flat.stride(1),
            grad_output_stride_h=grad_output_flat.stride(2),
            grad_output_stride_w=grad_output_flat.stride(3),
            rotation_stride_pose=rotation_flat.stride(0),
            grad_input_stride_n=grad_input.stride(0),
            grad_input_stride_c=grad_input.stride(1),
            grad_input_stride_d=grad_input.stride(2),
            grad_input_stride_h=grad_input.stride(3),
            grad_input_stride_w=grad_input.stride(4),
        )
    else:
        def grid(meta):
            return (triton.cdiv(N_pose * P * C, meta["BLOCK"]),)

        _central_slice_sample_ncdhw_gen_bwd_input_kernel[grid](
            grad_output_flat,
            rotation_flat,
            grad_input,
            N_pose,
            C,
            Q,
            L=L,
            P=P,
            SHIFT=shift,
            grad_output_stride_pose=grad_output_flat.stride(0),
            grad_output_stride_c=grad_output_flat.stride(1),
            grad_output_stride_h=grad_output_flat.stride(2),
            grad_output_stride_w=grad_output_flat.stride(3),
            rotation_stride_pose=rotation_flat.stride(0),
            grad_input_stride_n=grad_input.stride(0),
            grad_input_stride_c=grad_input.stride(1),
            grad_input_stride_d=grad_input.stride(2),
            grad_input_stride_h=grad_input.stride(3),
            grad_input_stride_w=grad_input.stride(4),
        )

    return grad_input


def _central_slice_sample_ndhwc_backward_input(
    grad_output: Tensor,
    rotation: Tensor,
    input_shape: tuple[int, int, int, int, int],
    *,
    use_ch2: bool | None = None,
    align_dc: bool = False,
) -> Tensor:
    """Backward helper that computes only ``grad_input`` for the NDHWC layout."""
    N, D, H, W, C = input_shape
    if grad_output.ndim != 5:
        raise ValueError(f"grad_output must be 5D (N, Q, L, L, C), got {tuple(grad_output.shape)}")
    if not (D == H == W):
        raise ValueError(f"input must be cubic, got (D,H,W)=({D},{H},{W})")

    use_ch2 = _resolve_use_ch2(C, use_ch2)
    Q = _check_rotation(rotation, N)
    L = int(W)
    if grad_output.shape != (N, Q, L, L, C):
        raise ValueError(
            "grad_output must match sampled output shape "
            f"({N}, {Q}, {L}, {L}, {C}), "
            f"got {tuple(grad_output.shape)}"
        )

    grad_input = torch.zeros(input_shape, device=grad_output.device, dtype=grad_output.dtype)
    if L < 2:
        return grad_input

    grad_output_flat = grad_output.contiguous().view(N * Q, L, L, C)
    rotation_flat = rotation.to(torch.float32).reshape(N * Q, 9).contiguous()

    shift = _resolve_shift(L, align_dc)
    P = L * L
    N_pose = N * Q

    if use_ch2:
        def grid(meta):
            return (triton.cdiv(N_pose * P, meta["BLOCK"]),)

        _central_slice_sample_ndhwc_ch2_bwd_input_kernel[grid](
            grad_output_flat,
            rotation_flat,
            grad_input,
            N_pose,
            Q,
            L=L,
            P=P,
            SHIFT=shift,
            grad_output_stride_pose=grad_output_flat.stride(0),
            grad_output_stride_h=grad_output_flat.stride(1),
            grad_output_stride_w=grad_output_flat.stride(2),
            grad_output_stride_c=grad_output_flat.stride(3),
            rotation_stride_pose=rotation_flat.stride(0),
            grad_input_stride_n=grad_input.stride(0),
            grad_input_stride_d=grad_input.stride(1),
            grad_input_stride_h=grad_input.stride(2),
            grad_input_stride_w=grad_input.stride(3),
            grad_input_stride_c=grad_input.stride(4),
        )
    else:
        def grid(meta):
            return (triton.cdiv(N_pose * P * C, meta["BLOCK"]),)

        _central_slice_sample_ndhwc_gen_bwd_input_kernel[grid](
            grad_output_flat,
            rotation_flat,
            grad_input,
            N_pose,
            C,
            Q,
            L=L,
            P=P,
            SHIFT=shift,
            grad_output_stride_pose=grad_output_flat.stride(0),
            grad_output_stride_h=grad_output_flat.stride(1),
            grad_output_stride_w=grad_output_flat.stride(2),
            grad_output_stride_c=grad_output_flat.stride(3),
            rotation_stride_pose=rotation_flat.stride(0),
            grad_input_stride_n=grad_input.stride(0),
            grad_input_stride_d=grad_input.stride(1),
            grad_input_stride_h=grad_input.stride(2),
            grad_input_stride_w=grad_input.stride(3),
            grad_input_stride_c=grad_input.stride(4),
        )

    return grad_input


def _central_slice_sample_backward_rotation_placeholder(*, channel_last: bool) -> Tensor:
    """Explicit placeholder for the future ``grad_rotation`` implementation."""
    layout = "NDHWC" if channel_last else "NCDHW"
    raise NotImplementedError(
        f"central_slice_sample backward for rotation is not implemented yet ({layout}); "
        "grad_input is supported, grad_rotation is a placeholder for now"
    )


class _CentralSliceSampleFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        rotation: Tensor,
        channel_last: bool,
        use_ch2: bool | None,
        align_dc: bool,
    ) -> Tensor:
        """Run the Triton forward kernel and stash metadata for backward."""
        ctx.channel_last = channel_last
        ctx.use_ch2 = use_ch2
        ctx.align_dc = align_dc
        ctx.input_shape = tuple(input.shape)
        ctx.save_for_backward(rotation)

        if channel_last:
            return _central_slice_sample_ndhwc_forward(input, rotation, use_ch2=use_ch2, align_dc=align_dc)
        return _central_slice_sample_ncdhw_forward(input, rotation, use_ch2=use_ch2, align_dc=align_dc)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Backward currently supports ``grad_input`` and reserves ``grad_rotation``."""
        (rotation,) = ctx.saved_tensors
        grad_input = None
        grad_rotation = None

        if ctx.needs_input_grad[0]:
            if ctx.channel_last:
                grad_input = _central_slice_sample_ndhwc_backward_input(
                    grad_output.contiguous(),
                    rotation,
                    ctx.input_shape,
                    use_ch2=ctx.use_ch2,
                    align_dc=ctx.align_dc,
                )
            else:
                grad_input = _central_slice_sample_ncdhw_backward_input(
                    grad_output.contiguous(),
                    rotation,
                    ctx.input_shape,
                    use_ch2=ctx.use_ch2,
                    align_dc=ctx.align_dc,
                )

        if ctx.needs_input_grad[1]:
            grad_rotation = _central_slice_sample_backward_rotation_placeholder(channel_last=ctx.channel_last)

        return grad_input, grad_rotation, None, None, None


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

    Note:
        Autograd currently implements ``grad_input`` and leaves ``grad_rotation``
        as an explicit placeholder.
    """

    return _CentralSliceSampleFn.apply(input, rotation, False, use_ch2, align_dc)


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

    Note:
        Autograd currently implements ``grad_input`` and leaves ``grad_rotation``
        as an explicit placeholder.
    """

    return _CentralSliceSampleFn.apply(input, rotation, True, use_ch2, align_dc)


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