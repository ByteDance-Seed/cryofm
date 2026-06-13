from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

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

    Returns a contiguous float32 tensor on ``device``.

    If ``pix_lin`` is provided, values at those pixels must be strictly positive.
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


def _canonicalize_rotation(
    rotation: Tensor,
    *,
    N: int,
    device: torch.device,
) -> Tensor:
    """Canonicalize rotations to a contiguous (N, 3, 3) float32 tensor on ``device``."""

    if rotation.dim() == 3:
        if rotation.shape != (N, 3, 3):
            raise ValueError(
                f"rotation must be (N,3,3)=({N},3,3), got {tuple(rotation.shape)}"
            )
        return rotation.to(device=device, dtype=torch.float32).contiguous()

    if rotation.dim() == 2:
        if rotation.shape != (N, 9):
            raise ValueError(f"rotation must be (N,9)=({N},9), got {tuple(rotation.shape)}")
        return rotation.to(device=device, dtype=torch.float32).contiguous().reshape(N, 3, 3)

    raise ValueError(f"rotation must be (N,3,3) or (N,9), got {tuple(rotation.shape)}")


def _canonicalize_pose_inputs(
    *,
    image: Tensor,
    image_index: Optional[Tensor],
    volume_index: Optional[Tensor],
    rotation: Tensor,
    translation: Tensor,
    probability: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Canonicalize per-pose inputs.

    Returns:
        (image_index_i64, volume_index_i64, rotmat_f32, trans_f32, prob_f32)
    """

    device = image.device
    B = int(image.shape[0])

    if rotation.ndim < 2:
        raise ValueError(f"rotation must be (N,3,3) or (N,9), got {tuple(rotation.shape)}")
    N = int(rotation.shape[0])

    rotmat = _canonicalize_rotation(
        rotation,
        N=N,
        device=device,
    )

    if translation.dim() != 2 or translation.shape != (N, 2):
        raise ValueError(f"translation must be (N,2)=({N},2), got {tuple(translation.shape)}")
    trans_f32 = translation.to(device=device, dtype=torch.float32).contiguous()

    if probability is None:
        prob_f32 = torch.ones((N,), device=device, dtype=torch.float32)
    else:
        if probability.shape != (N,):
            raise ValueError(f"probability must be (N,)=({N},), got {tuple(probability.shape)}")
        prob_f32 = probability.to(device=device, dtype=torch.float32).contiguous()

    if not torch.isfinite(prob_f32).all():
        raise ValueError("probability must be finite")
    if torch.any(prob_f32 < 0):
        raise ValueError("probability must be non-negative")

    if image_index is None:
        if N != B:
            raise ValueError(f"image_index must be provided when B={B} != N={N}")
        image_index_i64 = torch.arange(N, device=device, dtype=torch.int64)
    else:
        if image_index.shape != (N,):
            raise ValueError(f"image_index must be (N,)=({N},), got {tuple(image_index.shape)}")
        image_index_i64 = image_index.to(device=device, dtype=torch.int64).contiguous().reshape(-1)
        if torch.any((image_index_i64 < 0) | (image_index_i64 >= B)):
            raise ValueError("image_index contains out-of-range values")

    if volume_index is None:
        volume_index_i64 = torch.zeros((N,), device=device, dtype=torch.int64)
    else:
        if volume_index.shape != (N,):
            raise ValueError(f"volume_index must be (N,)=({N},), got {tuple(volume_index.shape)}")
        volume_index_i64 = volume_index.to(device=device, dtype=torch.int64).contiguous().reshape(-1)

    return image_index_i64, volume_index_i64, rotmat, trans_f32, prob_f32


def _prepare_volume_numerator(
    volume_numerator: Optional[Tensor],
    *,
    K: int,
    L: int,
    device: torch.device,
) -> Tensor:
    if volume_numerator is None:
        return torch.zeros((K, L, L, L), device=device, dtype=torch.complex64)

    if not volume_numerator.is_contiguous():
        raise ValueError("volume_numerator must be contiguous")

    if volume_numerator.device != device or volume_numerator.dtype != torch.complex64:
        raise ValueError(
            f"volume_numerator must be complex64 on {device}, got dtype={volume_numerator.dtype}, device={volume_numerator.device}"
        )

    if volume_numerator.dim() != 4 or volume_numerator.shape != (K, L, L, L):
        raise ValueError(f"volume_numerator must be (K,L,L,L)=({K},{L},{L},{L}), got {tuple(volume_numerator.shape)}")

    return volume_numerator


def _prepare_volume_denominator(
    volume_denominator: Optional[Tensor],
    *,
    K: int,
    L: int,
    device: torch.device,
    return_denom: bool,
) -> Optional[Tensor]:
    if not return_denom:
        return None

    if volume_denominator is None:
        return torch.zeros((K, L, L, L), device=device, dtype=torch.float32)

    if not volume_denominator.is_contiguous():
        raise ValueError("volume_denominator must be contiguous")

    if volume_denominator.device != device or volume_denominator.dtype != torch.float32:
        raise ValueError(
            f"volume_denominator must be float32 on {device}, got dtype={volume_denominator.dtype}, device={volume_denominator.device}"
        )

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
    """Backproject 2D Fourier images into a 3D Fourier volume.

    This is a pure PyTorch reference backend (no custom CUDA/Triton kernels).

    Conventions
        - 2D tensors use storage order (y, x) but coordinates are treated as (x, y).
        - 3D volumes use storage order (z, y, x) but coordinates are treated as (x, y, z).
        - ``rotation`` is applied to row-vector coordinates: ``coords_rot = coords @ R``.
        - ``translation`` is applied as a Fourier phase ramp: ``exp(-2π i * k · t)``.

    Args:
        image: Complex Fourier images in fftshift convention, shape (B, L, L), dtype complex64.
        ctf: Optional CTF in the same layout as ``image``: (B, L, L), converted to float32.
        noise_spectrum: Per-pixel noise power spectrum, shape (L, L), must be finite and non-negative.
            Values at pixels within ``radius`` must be strictly positive (values outside may be 0).
        image_index: Optional indices mapping each pose to an image, shape (N,). If ``None``,
            uses the identity mapping and requires N == B.
        volume_index: Optional indices mapping each pose to a volume id in [0, K), shape (N,).
        rotation: Rotation matrices, shape (N, 3, 3) or flattened (N, 9).
        translation: In-plane translations (x, y) in pixel units, shape (N, 2).
        probability: Optional non-negative per-pose weights, shape (N,). If ``None``, uses ones.
        radius: Max radius (in FFT index units) of pixels to backproject.
        volume_numerator: Optional accumulation buffer, shape (K, L, L, L), dtype complex64.
        volume_denominator: Optional accumulation buffer, shape (K, L, L, L), dtype float32.
        return_denom: If ``True``, also accumulate/return the denominator volume.

    Returns:
        (volume_numerator, volume_denominator). If ``return_denom=False``, the denominator is ``None``.
    """

    device = image.device

    if not return_denom and volume_denominator is not None:
        raise ValueError("volume_denominator must be None when return_denom=False")

    if image.dtype != torch.complex64:
        raise ValueError(f"image must be complex64, got {image.dtype}")
    if image.dim() != 3 or image.shape[1] != image.shape[2]:
        raise ValueError(f"image must be (B,L,L), got {tuple(image.shape)}")

    B, L, _ = image.shape

    radius = float(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    x_grid, y_grid = fftindex_components2d_radial(
        L,
        max_radius=radius,
        device=device,
        dtype=torch.int32,
    )
    center = L // 2

    x_i64 = x_grid.to(torch.int64) + center
    y_i64 = y_grid.to(torch.int64) + center
    pix_lin = (y_i64 * L + x_i64).contiguous()

    noise_spectrum_2d = _canonicalize_noise_spectrum(
        noise_spectrum,
        L=L,
        device=device,
        pix_lin=pix_lin,
    )

    image_index_i64, volume_index_i64, rotmat, trans_f32, prob_f32 = _canonicalize_pose_inputs(
        image=image,
        image_index=image_index,
        volume_index=volume_index,
        rotation=rotation,
        translation=translation,
        probability=probability,
    )
    N = int(image_index_i64.numel())

    image_flat = image.contiguous().reshape(B, L * L)

    noise_spectrum_flat = noise_spectrum_2d.reshape(L * L)
    noise_spectrum_p = noise_spectrum_flat.index_select(0, pix_lin)
    pixel_weight = noise_spectrum_p.reciprocal().contiguous()

    if ctf is None:
        ctf_flat = None
    else:
        if ctf.shape != (B, L, L):
            raise ValueError(f"ctf must be (B,L,L), got {tuple(ctf.shape)}")
        ctf_flat = ctf.to(device=device, dtype=torch.float32).contiguous().reshape(B, L * L)

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
        if torch.any(volume_index_i64 < 0):
            raise ValueError("volume_index contains negative values")
        K = int(volume_index_i64.max().item()) + 1
    else:
        K = 1

    if K > 1 and volume_index is None:
        raise ValueError("volume_index must be provided when K>1")
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

    vol_num_flat = volume_numerator.reshape(K * L * L * L)
    vol_den_flat = volume_denominator.reshape(K * L * L * L) if return_denom else None
    vol_stride = L * L * L

    x_f32_all = x_grid.to(device=device, dtype=torch.float32)
    y_f32_all = y_grid.to(device=device, dtype=torch.float32)

    image_pose_flat = image_flat.index_select(0, image_index_i64)
    ctf_pose_flat = ctf_flat.index_select(0, image_index_i64) if ctf_flat is not None else None

    # Apply in-plane translations via a Fourier phase ramp.
    # Compute the phase only for the sampled pixels within ``radius`` to avoid
    # materializing an (N, L*L) phase tensor.
    coords = fftfreq_coords2d(L, device=device, dtype=torch.float32)  # (L*L, 2) in (x, y) order
    coords_p = coords.index_select(0, pix_lin)  # (P, 2)
    delta = trans_f32 @ coords_p.t()  # (N, P)
    phase = torch.exp((-2j * torch.pi) * delta).to(dtype=torch.complex64)

    image_p = image_pose_flat.index_select(1, pix_lin) * phase
    w = prob_f32[:, None] * pixel_weight[None, :]

    if ctf_pose_flat is None:
        numerator = w * image_p
        denominator = w
    else:
        ctf_p = ctf_pose_flat.index_select(1, pix_lin)
        numerator = w * ctf_p * image_p
        denominator = w * (ctf_p * ctf_p)

    r00 = rotmat[:, 0, 0]
    r01 = rotmat[:, 0, 1]
    r02 = rotmat[:, 0, 2]
    r10 = rotmat[:, 1, 0]
    r11 = rotmat[:, 1, 1]
    r12 = rotmat[:, 1, 2]

    x_rot = x_f32_all[None, :] * r00[:, None] + y_f32_all[None, :] * r10[:, None]
    y_rot = x_f32_all[None, :] * r01[:, None] + y_f32_all[None, :] * r11[:, None]
    z_rot = x_f32_all[None, :] * r02[:, None] + y_f32_all[None, :] * r12[:, None]

    x0 = torch.floor(x_rot).to(torch.int64)
    y0 = torch.floor(y_rot).to(torch.int64)
    z0 = torch.floor(z_rot).to(torch.int64)

    fx = x_rot - x0.to(x_rot.dtype)
    fy = y_rot - y0.to(y_rot.dtype)
    fz = z_rot - z0.to(z_rot.dtype)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    interp_weights = torch.stack(
        [
            (1 - fz) * (1 - fy) * (1 - fx),
            (1 - fz) * (1 - fy) * fx,
            (1 - fz) * fy * (1 - fx),
            (1 - fz) * fy * fx,
            fz * (1 - fy) * (1 - fx),
            fz * (1 - fy) * fx,
            fz * fy * (1 - fx),
            fz * fy * fx,
        ],
        dim=1,
    )

    vx = torch.stack([x0, x1, x0, x1, x0, x1, x0, x1], dim=1) + center
    vy = torch.stack([y0, y0, y1, y1, y0, y0, y1, y1], dim=1) + center
    vz = torch.stack([z0, z0, z0, z0, z1, z1, z1, z1], dim=1) + center

    voxel_coords = torch.stack([vz, vy, vx], dim=-1)
    in_bound = (voxel_coords >= 0) & (voxel_coords < L)
    all_in_bound = in_bound.all(dim=-1).all(dim=1)
    interp_weights = interp_weights * all_in_bound[:, None, :].to(interp_weights.dtype)

    vx = vx.clamp(0, L - 1)
    vy = vy.clamp(0, L - 1)
    vz = vz.clamp(0, L - 1)

    lin_idx = vz * (L * L) + vy * L + vx
    vol_offset = volume_index_i64[:, None, None] * vol_stride
    global_lin_idx = (vol_offset + lin_idx).reshape(-1)

    contrib_num = (interp_weights * numerator[:, None, :]).reshape(-1)
    vol_num_flat.index_add_(0, global_lin_idx, contrib_num)

    if return_denom:
        contrib_den = (interp_weights * denominator[:, None, :]).to(torch.float32).reshape(-1)
        vol_den_flat.index_add_(0, global_lin_idx, contrib_den)

    return volume_numerator, volume_denominator