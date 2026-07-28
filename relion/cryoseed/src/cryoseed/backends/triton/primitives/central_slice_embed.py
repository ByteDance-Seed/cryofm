"""Triton primitives for splatting rotated 2D slices into a 3D grid.

These functions accumulate (atomic add) complex-valued samples from one or more
2D central slices into a cubic 3D volume using trilinear splatting in voxel
space.

Rotation convention matches the Torch backend: coordinates are treated as
row-vectors ``(x, y, 0)`` and right-multiplied by the rotation matrix:
``coords_rot = coords @ R``.

The wrappers below mainly perform dtype/device/shape checks and prepare
contiguous, flattened views expected by the underlying Triton kernels.
"""

import torch
from torch import Tensor
from typing import Optional
import triton

from cryoseed.backends.triton.primitives.kernels.central_slice_embed_batched import central_slice_embed_batched_kernel as _central_slice_embed_batched_kernel
from cryoseed.backends.triton.primitives.kernels.central_slice_embed_indexed import central_slice_embed_indexed_kernel as _central_slice_embed_indexed_kernel


__all__ = [
    "central_slice_embed_batched",
    "central_slice_embed_indexed",
]

def central_slice_embed_batched(
    input: Tensor,
    modulation: Optional[Tensor],
    pixel_weight: Tensor,
    rot: Tensor,
    pose_weight: Tensor,
    x_grid: Tensor,
    y_grid: Tensor,
    out_index: Optional[Tensor],
    out_numer: Tensor,
    out_denom: Optional[Tensor] = None,
) -> None:
    """Accumulate a batch of 2D slices into a 3D grid (in-place).

    Each sampled pixel ``(x_grid[p], y_grid[p])`` is rotated into 3D using the
    per-slice rotation matrix and then splatted to the 8 neighboring voxels with
    trilinear weights. Accumulation is done via atomics into ``out_numer`` and
    optionally ``out_denom``.

    Args:
        input: Complex input slices of shape ``(B, L, L)`` and dtype ``complex64``.
        modulation:
            Optional per-pixel real modulation of shape ``(B, L, L)``.
            If provided, numerator accumulates ``w * modulation * input`` and
            denominator accumulates ``w * modulation**2``. If omitted, numerator
            accumulates ``w * input`` and denominator accumulates ``w``.
        pixel_weight:
            Real per-pixel weights with ``numel == L * L`` (e.g. ``(L, L)`` or
            ``(L * L,)``). Sampled pixels are gathered using ``x_grid/y_grid``.
        rot:
            Rotation matrices with ``numel == B * 9`` (reshaped to ``(B, 9)``).
            Coordinates are treated as row-vectors and right-multiplied by ``R``.
        pose_weight: Real weights of shape ``(B,)`` applied per slice.
        x_grid: X offsets for sampled pixels, shape ``(P,)``.
        y_grid: Y offsets for sampled pixels, shape ``(P,)`` (must match ``x_grid``).
        out_index:
            Optional int tensor of shape ``(B,)`` selecting which output volume
            ``k`` each slice contributes to. Required when ``K > 1``. If ``None``,
            all slices write to ``k = 0``.
        out_numer:
            Output numerator buffer of shape ``(K, L, L, L)``, dtype ``complex64``.
            Must be contiguous. Updated in-place.
        out_denom:
            Optional output denominator buffer of shape ``(K, L, L, L)``, dtype
            ``float32``. Must be contiguous. If ``None``, the denominator is not
            written.

    Returns:
        None. Outputs are written in-place.

    Note:
        This is a low-level primitive. ``x_grid`` and ``y_grid`` are assumed to
        index valid input pixels after shifting by ``L // 2``; out-of-range values
        can lead to invalid memory accesses.
    """

    if not input.is_cuda:
        raise RuntimeError("central_slice_embed_batched requires CUDA tensors")
    if input.dtype != torch.complex64:
        raise ValueError(f"input must be complex64, got {input.dtype}")
    if input.dim() != 3 or input.shape[1] != input.shape[2]:
        raise ValueError(f"input must be (B,L,L) with square last-2 dims, got {tuple(input.shape)}")

    B, L, _ = input.shape
    device = input.device

    input = input.contiguous()

    if pixel_weight.numel() != L * L:
        raise ValueError(f"pixel_weight must have {L * L} elements, got {pixel_weight.numel()}")
    pixel_weight = pixel_weight.to(device=device, dtype=torch.float32).contiguous().reshape(L * L)

    if pose_weight.numel() != B:
        raise ValueError(f"pose_weight must have {B} elements, got {pose_weight.numel()}")
    pose_weight = pose_weight.to(device=device, dtype=torch.float32).contiguous().reshape(B)

    if rot.numel() != B * 9:
        raise ValueError(f"rot must have {B * 9} elements, got {rot.numel()}")
    rot = rot.to(device=device, dtype=torch.float32).contiguous().reshape(B, 9)

    x_grid = x_grid.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    y_grid = y_grid.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    if x_grid.numel() != y_grid.numel():
        raise ValueError(f"x_grid and y_grid must have same numel, got {x_grid.numel()} and {y_grid.numel()}")

    if (not out_numer.is_cuda) or (out_numer.device != device) or (out_numer.dtype != torch.complex64):
        raise ValueError(
            f"out_numer must be complex64 on {device}, got dtype={out_numer.dtype}, device={out_numer.device}"
        )

    if out_numer.dim() != 4 or out_numer.shape[1:] != (L, L, L):
        raise ValueError(
            f"out_numer must be (K,L,L,L) with (L,L,L)=({L},{L},{L}), got {tuple(out_numer.shape)}"
        )
    K = int(out_numer.shape[0])

    if K > 1 and out_index is None:
        raise ValueError("out_index must be provided when out_numer has K>1")

    if out_index is None:
        has_out_index = False
        out_index_i32 = torch.empty((1,), device=device, dtype=torch.int32)
    else:
        has_out_index = True
        if out_index.shape != (B,):
            raise ValueError(f"out_index must be (B,)=({B},), got {tuple(out_index.shape)}")
        out_index_i32 = out_index.to(device=device, dtype=torch.int32).contiguous()

    if has_out_index:
        if torch.any((out_index_i32 < 0) | (out_index_i32 >= K)).item():
            raise ValueError("out_index contains out-of-range values")

    if not out_numer.is_contiguous():
        raise ValueError("out_numer must be contiguous")

    if out_denom is None:
        out_denom_flat = torch.empty((1,), device=device, dtype=torch.float32)
        denom_stride_k = 0
        denom_stride_v = 0
    else:
        if (not out_denom.is_cuda) or (out_denom.device != device):
            raise ValueError(f"out_denom must be on {device}, got device={out_denom.device}")
        if out_denom.shape != (K, L, L, L):
            raise ValueError(
                f"out_denom must be (K,L,L,L)=({K},{L},{L},{L}), got {tuple(out_denom.shape)}"
            )
        if out_denom.dtype != torch.float32:
            raise ValueError(f"out_denom must be float32, got {out_denom.dtype}")
        if not out_denom.is_contiguous():
            raise ValueError("out_denom must be contiguous")
        out_denom_flat = out_denom.reshape(K, L * L * L)
        denom_stride_k = out_denom_flat.stride(0)
        denom_stride_v = out_denom_flat.stride(1)

    center = L // 2
    P = int(x_grid.numel())

    input_cplx = torch.view_as_real(input).reshape(B, L * L, 2)
    out_numer_cplx = torch.view_as_real(out_numer).reshape(K, L * L * L, 2)

    if modulation is None:
        has_modulation = False
        modulation_flat = torch.empty((1,), device=device, dtype=torch.float32)
    else:
        if modulation.shape != (B, L, L):
            raise ValueError(
                f"modulation must be (B,L,L)=({B},{L},{L}), got {tuple(modulation.shape)}"
            )
        has_modulation = True
        modulation_flat = modulation.to(device=device, dtype=torch.float32).contiguous().reshape(B, L * L)

    def grid(meta):
        return (triton.cdiv(B * P, meta["BLOCK"]),)

    _central_slice_embed_batched_kernel[grid](
        input_cplx,
        modulation_flat,
        pixel_weight,
        rot,
        pose_weight,
        x_grid,
        y_grid,
        out_index_i32,
        out_numer_cplx,
        out_denom_flat,
        B,
        K,
        L=L,
        P=P,
        CENTER=center,
        input_stride_b=input_cplx.stride(0),
        input_stride_p=input_cplx.stride(1),
        input_stride_cplx=input_cplx.stride(2),
        modulation_stride_b=modulation_flat.stride(0) if has_modulation else 0,
        modulation_stride_p=modulation_flat.stride(1) if has_modulation else 0,
        rot_stride_b=rot.stride(0),
        rot_stride_k=rot.stride(1),
        out_idx_stride_b=out_index_i32.stride(0),
        numer_stride_k=out_numer_cplx.stride(0),
        numer_stride_v=out_numer_cplx.stride(1),
        numer_stride_cplx=out_numer_cplx.stride(2),
        denom_stride_k=denom_stride_k,
        denom_stride_v=denom_stride_v,
        HAS_MODULATION=has_modulation,
        HAS_OUT_INDEX=has_out_index,
    )


def central_slice_embed_indexed(
    input: Tensor,
    modulation: Optional[Tensor],
    pixel_weight: Tensor,
    input_index: Tensor,
    rot: Tensor,
    shift: Tensor,
    pose_weight: Tensor,
    x_grid: Tensor,
    y_grid: Tensor,
    out_index: Optional[Tensor],
    out_numer: Tensor,
    out_denom: Optional[Tensor] = None,
) -> None:
    """Accumulate selected slices into a 3D grid with per-sample phase shifts.

    Compared to :func:`central_slice_embed_batched`, this variant takes ``N``
    samples from a larger pool of ``B`` input slices via ``input_index`` and
    applies an additional in-plane translation using the Fourier shift theorem:
    ``shift[s] = (dx, dy)`` introduces a phase factor
    ``exp(-2π i (dx * x/L + dy * y/L))`` per pixel.

    Args:
        input: Complex input slices of shape ``(B, L, L)`` and dtype ``complex64``.
        modulation: Optional per-pixel real modulation of shape ``(B, L, L)``.
        pixel_weight: Real per-pixel weights with ``numel == L * L``.
        input_index:
            Indices into the first dimension of ``input`` with shape ``(N,)``.
            Each value must be in ``[0, B)``.
        rot:
            Rotation matrices with ``numel == N * 9`` (reshaped to ``(N, 9)``).
            Rotation convention matches :func:`central_slice_embed_batched`.
        shift: Real shifts of shape ``(N, 2)`` storing ``(dx, dy)``.
        pose_weight: Real weights of shape ``(N,)`` applied per sample.
        x_grid: X offsets for sampled pixels, shape ``(P,)``.
        y_grid: Y offsets for sampled pixels, shape ``(P,)`` (must match ``x_grid``).
        out_index:
            Optional int tensor of shape ``(N,)`` selecting which output volume
            ``k`` each sample contributes to. Required when ``K > 1``. If ``None``,
            all samples write to ``k = 0``.
        out_numer:
            Output numerator buffer of shape ``(K, L, L, L)``, dtype ``complex64``.
            Must be contiguous. Updated in-place.
        out_denom:
            Optional output denominator buffer of shape ``(K, L, L, L)``, dtype
            ``float32``. Must be contiguous. If ``None``, the denominator is not
            written.

    Returns:
        None. Outputs are written in-place.

    Note:
        This is a low-level primitive. ``x_grid`` and ``y_grid`` are assumed to
        index valid input pixels after shifting by ``L // 2``.
    """

    if not input.is_cuda:
        raise RuntimeError("central_slice_embed_indexed requires CUDA tensors")
    if input.dtype != torch.complex64:
        raise ValueError(f"input must be complex64, got {input.dtype}")
    if input.dim() != 3 or input.shape[1] != input.shape[2]:
        raise ValueError(f"input must be (B,L,L), got {tuple(input.shape)}")

    B, L, _ = input.shape
    device = input.device

    input = input.contiguous()

    if pixel_weight.numel() != L * L:
        raise ValueError(f"pixel_weight must have {L * L} elements, got {pixel_weight.numel()}")
    pixel_weight = pixel_weight.to(device=device, dtype=torch.float32).contiguous().reshape(L * L)

    input_index = input_index.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    N = int(input_index.numel())

    if torch.any((input_index < 0) | (input_index >= B)).item():
        raise ValueError("input_index contains out-of-range values")

    if shift.shape != (N, 2):
        raise ValueError(f"shift must be (N,2)=({N},2), got {tuple(shift.shape)}")
    shift = shift.to(device=device, dtype=torch.float32).contiguous().reshape(N, 2)

    if pose_weight.numel() != N:
        raise ValueError(f"pose_weight must have {N} elements, got {pose_weight.numel()}")
    pose_weight = pose_weight.to(device=device, dtype=torch.float32).contiguous().reshape(N)

    if rot.numel() != N * 9:
        raise ValueError(f"rot must have {N * 9} elements, got {rot.numel()}")
    rot = rot.to(device=device, dtype=torch.float32).contiguous().reshape(N, 9)

    x_grid = x_grid.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    y_grid = y_grid.to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    if x_grid.numel() != y_grid.numel():
        raise ValueError(f"x_grid and y_grid must have same numel, got {x_grid.numel()} and {y_grid.numel()}")

    if modulation is None:
        has_modulation = False
        modulation_flat = torch.empty((1,), device=device, dtype=torch.float32)
    else:
        if modulation.shape != (B, L, L):
            raise ValueError(
                f"modulation must be (B,L,L)=({B},{L},{L}), got {tuple(modulation.shape)}"
            )
        has_modulation = True
        modulation_flat = modulation.to(device=device, dtype=torch.float32).contiguous().reshape(B, L * L)

    if (not out_numer.is_cuda) or (out_numer.device != device) or (out_numer.dtype != torch.complex64):
        raise ValueError(
            f"out_numer must be complex64 on {device}, got dtype={out_numer.dtype}, device={out_numer.device}"
        )

    if out_numer.dim() != 4 or out_numer.shape[1:] != (L, L, L):
        raise ValueError(
            f"out_numer must be (K,L,L,L) with (L,L,L)=({L},{L},{L}), got {tuple(out_numer.shape)}"
        )
    K = int(out_numer.shape[0])

    if K > 1 and out_index is None:
        raise ValueError("out_index must be provided when out_numer has K>1")

    if out_index is None:
        has_out_index = False
        out_index_i32 = torch.empty((1,), device=device, dtype=torch.int32)
    else:
        has_out_index = True
        if out_index.shape != (N,):
            raise ValueError(f"out_index must be (N,)=({N},), got {tuple(out_index.shape)}")
        out_index_i32 = out_index.to(device=device, dtype=torch.int32).contiguous()

    if has_out_index:
        if torch.any((out_index_i32 < 0) | (out_index_i32 >= K)).item():
            raise ValueError("out_index contains out-of-range values")

    if not out_numer.is_contiguous():
        raise ValueError("out_numer must be contiguous")
    out_numer_cplx = torch.view_as_real(out_numer).reshape(K, L * L * L, 2)

    if out_denom is None:
        out_denom_flat = torch.empty((1,), device=device, dtype=torch.float32)
        denom_stride_k = 0
        denom_stride_v = 0
    else:
        if (not out_denom.is_cuda) or (out_denom.device != device):
            raise ValueError(f"out_denom must be on {device}, got device={out_denom.device}")
        if out_denom.shape != (K, L, L, L):
            raise ValueError(
                f"out_denom must be (K,L,L,L)=({K},{L},{L},{L}), got {tuple(out_denom.shape)}"
            )
        if out_denom.dtype != torch.float32:
            raise ValueError(f"out_denom must be float32, got {out_denom.dtype}")
        if not out_denom.is_contiguous():
            raise ValueError("out_denom must be contiguous")
        out_denom_flat = out_denom.reshape(K, L * L * L)
        denom_stride_k = out_denom_flat.stride(0)
        denom_stride_v = out_denom_flat.stride(1)

    center = L // 2
    P = int(x_grid.numel())

    input_cplx = torch.view_as_real(input).reshape(B, L * L, 2)

    def grid(meta):
        return (triton.cdiv(N * P, meta["BLOCK"]),)

    _central_slice_embed_indexed_kernel[grid](
        input_cplx,
        modulation_flat,
        pixel_weight,
        input_index,
        rot,
        shift,
        pose_weight,
        x_grid,
        y_grid,
        out_index_i32,
        out_numer_cplx,
        out_denom_flat,
        N,
        K,
        L=L,
        P=P,
        CENTER=center,
        input_stride_b=input_cplx.stride(0),
        input_stride_p=input_cplx.stride(1),
        input_stride_cplx=input_cplx.stride(2),
        modulation_stride_b=modulation_flat.stride(0) if has_modulation else 0,
        modulation_stride_p=modulation_flat.stride(1) if has_modulation else 0,
        pose_weight_stride_b=pose_weight.stride(0),
        rot_stride_b=rot.stride(0),
        rot_stride_k=rot.stride(1),
        shift_stride_b=shift.stride(0),
        shift_stride_d=shift.stride(1),
        input_idx_stride_b=input_index.stride(0),
        out_idx_stride_b=out_index_i32.stride(0),
        numer_stride_k=out_numer_cplx.stride(0),
        numer_stride_v=out_numer_cplx.stride(1),
        numer_stride_cplx=out_numer_cplx.stride(2),
        denom_stride_k=denom_stride_k,
        denom_stride_v=denom_stride_v,
        HAS_MODULATION=has_modulation,
        HAS_OUT_INDEX=has_out_index,
    )