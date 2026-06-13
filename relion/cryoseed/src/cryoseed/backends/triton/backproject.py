"""Triton backend for Fourier-space backprojection.

This module implements a CUDA/Triton version of the reference
``cryoseed.backends.torch.backproject.backproject``.

It accumulates a set of complex Fourier images (central slices) into a cubic 3D
Fourier volume by rotating each 2D frequency coordinate ``(x, y, 0)`` into 3D
and splatting its contribution to the 8 neighboring voxels (trilinear splat).
Accumulation is performed with atomic adds.

Conventions
    - 2D tensors use storage order (y, x) but coordinates are treated as (x, y).
    - 3D volumes use storage order (z, y, x) but coordinates are treated as (x, y, z).
    - ``rotation`` is applied to row-vector coordinates: ``coords_rot = coords @ R``.
    - ``translation`` is applied as a Fourier phase ramp: ``exp(-2π i * k · t)``.

The Triton implementation delegates the heavy lifting to
``central_slice_embed_batched`` / ``central_slice_embed_indexed``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from cryoseed.backends.triton.primitives.central_slice_embed import (
    central_slice_embed_batched,
    central_slice_embed_indexed,
)
from cryoseed.fft.coords import fftindex_components2d_radial, fftfreq_coords2d


__all__ = [
    "backproject",
]


def _canonicalize_noise_spectrum(
    noise_spectrum: Tensor,
    *,
    L: int,
    device: torch.device,
    pix_lin: Optional[Tensor] = None,
) -> Tensor:
    """Validate and canonicalize the 2D noise power spectrum.

    Args:
        noise_spectrum: Tensor of shape ``(L, L)``.
        L: Image size.
        device: Target CUDA device.
        pix_lin: Optional flattened pixel indices that will actually be sampled.
            If provided, ``noise_spectrum`` must be strictly positive at those indices.

    Returns:
        ``noise_spectrum`` converted to contiguous ``float32`` on ``device``.

    Raises:
        ValueError: If shape is wrong, contains NaNs/Infs, has negative values, or has
            non-positive values at sampled pixels.
    """
    if noise_spectrum.dim() != 2 or noise_spectrum.shape != (L, L):
        raise ValueError(
            f"noise_spectrum must be (L,L)=({L},{L}), got {tuple(noise_spectrum.shape)}"
        )

    noise_spectrum_f32 = noise_spectrum.to(device=device, dtype=torch.float32).contiguous()

    if not torch.isfinite(noise_spectrum_f32).all():
        raise ValueError("noise_spectrum must be finite")
    if torch.any(noise_spectrum_f32 < 0):
        raise ValueError("noise_spectrum must be non-negative")

    if pix_lin is None:
        if torch.any(noise_spectrum_f32 <= 0):
            raise ValueError("noise_spectrum must be strictly positive")
    else:
        noise_spectrum_flat = noise_spectrum_f32.reshape(-1)
        pix_lin_i64 = pix_lin.to(device=device, dtype=torch.int64).reshape(-1)
        used = noise_spectrum_flat.index_select(0, pix_lin_i64)
        if torch.any(used <= 0):
            raise ValueError("noise_spectrum must be strictly positive within the sampled radius")

    return noise_spectrum_f32


def _prepare_volume_numerator(
    volume_numerator: Optional[Tensor],
    *,
    K: int,
    L: int,
    device: torch.device,
) -> Tensor:
    """Allocate or validate the complex numerator accumulation buffer."""
    if volume_numerator is None:
        return torch.zeros((K, L, L, L), device=device, dtype=torch.complex64)

    if (not volume_numerator.is_cuda) or (volume_numerator.device != device):
        raise ValueError(
            f"volume_numerator must be a CUDA tensor on {device}, got device={volume_numerator.device}"
        )

    if volume_numerator.dtype != torch.complex64:
        raise ValueError(f"volume_numerator must be complex64, got {volume_numerator.dtype}")

    if not volume_numerator.is_contiguous():
        raise ValueError("volume_numerator must be contiguous")

    if volume_numerator.dim() != 4 or volume_numerator.shape != (K, L, L, L):
        raise ValueError(
            f"volume_numerator must be (K,L,L,L)=({K},{L},{L},{L}), got {tuple(volume_numerator.shape)}"
        )

    return volume_numerator


def _prepare_volume_denominator(
    volume_denominator: Optional[Tensor],
    *,
    K: int,
    L: int,
    device: torch.device,
    return_denom: bool,
) -> Optional[Tensor]:
    """Allocate or validate the float denominator accumulation buffer."""
    if not return_denom:
        return None

    if volume_denominator is None:
        return torch.zeros((K, L, L, L), device=device, dtype=torch.float32)

    if (not volume_denominator.is_cuda) or (volume_denominator.device != device):
        raise ValueError(
            f"volume_denominator must be a CUDA tensor on {device}, got device={volume_denominator.device}"
        )

    if volume_denominator.dtype != torch.float32:
        raise ValueError(f"volume_denominator must be float32, got {volume_denominator.dtype}")

    if not volume_denominator.is_contiguous():
        raise ValueError("volume_denominator must be contiguous")

    if volume_denominator.dim() != 4 or volume_denominator.shape != (K, L, L, L):
        raise ValueError(
            f"volume_denominator must be (K,L,L,L)=({K},{L},{L},{L}), got {tuple(volume_denominator.shape)}"
        )

    return volume_denominator

def backproject(
    image: Tensor,
    ctf: Optional[Tensor],
    noise_spectrum: Tensor,
    *,
    image_index: Optional[Tensor],
    volume_index: Optional[Tensor] = None,
    rotation: Tensor,
    translation: Tensor,
    probability: Optional[Tensor],
    radius: float,
    volume_numerator: Optional[Tensor] = None,
    volume_denominator: Optional[Tensor] = None,
    return_denom: bool = True,
) -> tuple[Tensor, Optional[Tensor]]:
    """Backproject 2D Fourier images into a 3D Fourier volume (Triton).

    This function accumulates complex Fourier images (in fftshift convention)
    into one or more 3D Fourier volumes.

    Conceptually, for each pose ``n`` and each sampled 2D frequency coordinate
    ``(x, y)``, we:
        1) apply an in-plane translation via a Fourier phase ramp
           ``exp(-2π i * (dx * x/L + dy * y/L))``,
        2) optionally apply a real modulation (CTF),
        3) rotate ``(x, y, 0)`` into 3D using ``rotation[n]``,
        4) splat the result into the 3D grid with trilinear weights.

    Args:
        image:
            Complex Fourier images in fftshift convention, shape ``(B, L, L)``,
            dtype ``complex64``.
        ctf:
            Optional real modulation in the same layout as ``image``, shape
            ``(B, L, L)``. Converted to contiguous ``float32``.
        noise_spectrum:
            Per-pixel noise power spectrum, shape ``(L, L)``, must be finite and
            non-negative.

            Values at pixels within ``radius`` (i.e. the pixels actually backprojected)
            must be strictly positive. Values outside the radius may be 0 (e.g. when
            produced by ``radial_broadcast(..., padding_mode='zeros')``).

            Used to derive per-pixel weights ``1 / noise_spectrum`` (with 0 weight where
            ``noise_spectrum == 0``).
        image_index:
            Optional indices mapping each pose to an image, shape ``(N,)``.
            If ``None``, uses the identity mapping and requires ``N == B``.
        volume_index:
            Optional indices mapping each pose to an output volume id, shape ``(N,)``.
            If provided, values must be in ``[0, K)``.
        rotation:
            Rotation matrices, shape ``(N, 3, 3)`` or flattened ``(N, 9)``.
            Applied as ``coords_rot = coords @ R``.
        translation:
            In-plane translations ``(dx, dy)`` in pixel units, shape ``(N, 2)``.
        probability:
            Optional non-negative per-pose weights, shape ``(N,)``. If ``None``,
            uses ones.
        radius:
            Max radius (in FFT index units) of pixels to backproject.
        volume_numerator:
            Optional accumulation buffer, shape ``(K, L, L, L)``, dtype ``complex64``.
            Must be contiguous if provided.
        volume_denominator:
            Optional accumulation buffer, shape ``(K, L, L, L)``, dtype ``float32``.
            Must be contiguous if provided and ``return_denom=True``.
        return_denom:
            If ``True``, also accumulate/return the denominator volume.

    Returns:
        ``(volume_numerator, volume_denominator)``. If ``return_denom=False``,
        the denominator is ``None``.

    Note:
        This is a low-level backend routine. Accumulation uses atomic adds and is
        non-deterministic across runs due to parallel floating-point reduction.
    """

    radius = float(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    if not image.is_cuda:
        raise RuntimeError("backproject requires CUDA tensors")
    if image.dtype != torch.complex64:
        raise ValueError(f"image must be complex64, got {image.dtype}")
    if image.dim() != 3 or image.shape[1] != image.shape[2]:
        raise ValueError(f"image must be (B,L,L) with square last-2 dims, got {tuple(image.shape)}")

    B, L, _ = image.shape
    device = image.device

    if not return_denom and volume_denominator is not None:
        raise ValueError("volume_denominator must be None when return_denom=False")

    if ctf is None:
        ctf_f32 = None
    else:
        if ctf.shape != (B, L, L):
            raise ValueError(f"ctf must be (B,L,L)=({B},{L},{L}), got {tuple(ctf.shape)}")
        ctf_f32 = ctf.to(device=device, dtype=torch.float32).contiguous()

    x_grid, y_grid = fftindex_components2d_radial(
        L,
        max_radius=radius,
        device=device,
        dtype=torch.int32,
    )

    center = L // 2
    x_i64 = x_grid.to(device=device, dtype=torch.int64) + center
    y_i64 = y_grid.to(device=device, dtype=torch.int64) + center
    pix_lin = (y_i64 * L + x_i64).contiguous()

    noise_spectrum_2d = _canonicalize_noise_spectrum(
        noise_spectrum,
        L=L,
        device=device,
        pix_lin=pix_lin,
    )

    noise_spectrum_flat = noise_spectrum_2d.reshape(L * L)
    pixel_weight_flat = torch.zeros_like(noise_spectrum_flat)
    positive_mask = noise_spectrum_flat > 0
    pixel_weight_flat[positive_mask] = noise_spectrum_flat[positive_mask].reciprocal()
    pixel_weight_flat = pixel_weight_flat.contiguous()

    if volume_numerator is not None:
        if volume_numerator.dim() != 4:
            raise ValueError(f"volume_numerator must be (K,L,L,L), got {tuple(volume_numerator.shape)}")
        K = int(volume_numerator.shape[0])
    elif volume_denominator is not None:
        if volume_denominator.dim() != 4:
            raise ValueError(
                f"volume_denominator must be (K,L,L,L) when volume_numerator is None, got {tuple(volume_denominator.shape)}"
            )
        K = int(volume_denominator.shape[0])
    elif volume_index is not None:
        if torch.any(volume_index < 0):
            raise ValueError("volume_index contains negative values")
        K = int(volume_index.max().item()) + 1
    else:
        K = 1

    if K > 1 and volume_index is None:
        raise ValueError("volume_index must be provided when K>1")

    if volume_index is None:
        volume_index_i64 = None
    else:
        volume_index_i64 = volume_index.to(device=device, dtype=torch.int64).contiguous().reshape(-1)
        if torch.any((volume_index_i64 < 0) | (volume_index_i64 >= K)):
            raise ValueError("volume_index contains out-of-range values")

    volume_numerator = _prepare_volume_numerator(
        volume_numerator,
        K=K,
        L=L,
        device=device,
    )

    volume_denominator = _prepare_volume_denominator(
        volume_denominator,
        K=K,
        L=L,
        device=device,
        return_denom=return_denom,
    )

    if rotation.dim() == 3:
        N = int(rotation.shape[0])
        if rotation.shape != (N, 3, 3):
            raise ValueError(f"rotation must be (N,3,3)=({N},3,3), got {tuple(rotation.shape)}")
        rot_flat = rotation.to(device=device, dtype=torch.float32).contiguous().reshape(N, 9)
    elif rotation.dim() == 2:
        N = int(rotation.shape[0])
        if rotation.shape != (N, 9):
            raise ValueError(f"rotation must be (N,9)=({N},9), got {tuple(rotation.shape)}")
        rot_flat = rotation.to(device=device, dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"rotation must be (N,3,3) or (N,9), got {tuple(rotation.shape)}")

    if translation.dim() != 2 or translation.shape != (N, 2):
        raise ValueError(f"translation must be (N,2)=({N},2), got {tuple(translation.shape)}")
    trans_f32 = translation.to(device=device, dtype=torch.float32).contiguous().reshape(N, 2)

    if probability is None:
        prob_flat = torch.ones((N,), device=device, dtype=torch.float32)
    else:
        if probability.shape != (N,):
            raise ValueError(f"probability must be (N,)=({N},), got {tuple(probability.shape)}")
        prob_flat = probability.to(device=device, dtype=torch.float32).contiguous().reshape(N)

    if not torch.isfinite(prob_flat).all():
        raise ValueError("probability must be finite")
    if torch.any(prob_flat < 0):
        raise ValueError("probability must be non-negative")

    if volume_index is None:
        volume_index_i32 = None
    else:
        if volume_index.shape != (N,):
            raise ValueError(f"volume_index must be (N,)=({N},), got {tuple(volume_index.shape)}")
        volume_index_i32 = volume_index.to(device=device, dtype=torch.int32).contiguous().reshape(N)

    if image_index is None:
        if B != N:
            raise ValueError(f"image_index must be provided when B={B} != N={N}")
        image_flat = image.contiguous().reshape(B, L * L)
        coords = fftfreq_coords2d(L, device=device, dtype=torch.float32)
        coords_p = coords.index_select(0, pix_lin)
        delta = trans_f32 @ coords_p.t()
        phase = torch.exp((-2j * torch.pi) * delta).to(dtype=torch.complex64)
        image_shifted_flat = image_flat.clone()
        image_shifted_flat[:, pix_lin] = image_shifted_flat.index_select(1, pix_lin) * phase
        image_shifted = image_shifted_flat.reshape(B, L, L)
        central_slice_embed_batched(
            input=image_shifted,
            modulation=ctf_f32,
            pixel_weight=pixel_weight_flat,
            rot=rot_flat,
            pose_weight=prob_flat,
            x_grid=x_grid,
            y_grid=y_grid,
            out_index=volume_index_i32,
            out_numer=volume_numerator,
            out_denom=volume_denominator,
        )
    else:
        if image_index.shape != (N,):
            raise ValueError(f"image_index must be (N,)=({N},), got {tuple(image_index.shape)}")
        image_index_i64 = image_index.to(device=device, dtype=torch.int64).contiguous().reshape(-1)
        if torch.any((image_index_i64 < 0) | (image_index_i64 >= B)):
            raise ValueError("image_index contains out-of-range values")
        image_index_i32 = image_index_i64.to(torch.int32)
        central_slice_embed_indexed(
            input=image,
            modulation=ctf_f32,
            pixel_weight=pixel_weight_flat,
            input_index=image_index_i32,
            rot=rot_flat,
            shift=trans_f32,
            pose_weight=prob_flat,
            x_grid=x_grid,
            y_grid=y_grid,
            out_index=volume_index_i32,
            out_numer=volume_numerator,
            out_denom=volume_denominator,
        )

    return volume_numerator, volume_denominator