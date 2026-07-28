"""Mask utilities used in cryo-EM preprocessing.

This module groups together NumPy and torch geometry masks, soft-mask
application helpers, and Fourier-space masking utilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "circular_mask_numpy",
    "spherical_mask_numpy",
    "circular_mask",
    "spherical_mask",
    "radial_mask",
    "masked_lerp",
    "particle_mask",
    "center_fit_mask",
    "load_mask_mrc",
    "lowpass_mask",
]

# NumPy Geometry Masks

def circular_mask_numpy(
    h: int,
    w: int,
    center: tuple[int, int] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    """Create a 2D circular mask (NumPy).

    Args:
        h: Image height.
        w: Image width.
        center: Circle center in (x, y) index order. Defaults to the image center.
        radius: Radius in pixels. Defaults to the largest radius that stays in-bounds.

    Returns:
        A boolean NumPy array of shape (H, W).
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return dist_from_center <= radius


def spherical_mask_numpy(
    d: int,
    h: int,
    w: int,
    center: tuple[int, int, int] | None = None,
    radius: int | None = None,
) -> np.ndarray:
    """Create a 3D spherical mask (NumPy).

    Args:
        d: Volume depth.
        h: Volume height.
        w: Volume width.
        center: Sphere center in (x, y, z) index order. Defaults to the volume center.
        radius: Radius in pixels/voxels. Defaults to the largest radius that stays in-bounds.

    Returns:
        A boolean NumPy array of shape (D, H, W).
    """
    if center is None:
        center = (int(w / 2), int(h / 2), int(d / 2))
    if radius is None:
        radius = min(
            center[0],
            center[1],
            center[2],
            w - center[0],
            h - center[1],
            d - center[2],
        )

    z, y, x = np.ogrid[:d, :h, :w]
    dist_from_center = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )
    return dist_from_center <= radius


# Torch Geometry Masks

def circular_mask(
    h: int,
    w: int,
    center: Sequence[float] | None = None,
    radius: float | None = None,
    soft_edge_pixels: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a 2D circular mask.
 
    Args:
        h, w: Image size in (H, W) order.
        center: Circle center in (x, y) index order. Defaults to the image center.
        radius: Radius in pixels. Defaults to the largest radius that stays in-bounds.
        soft_edge_pixels: Width of the soft edge in pixels. A value of 0
            returns a hard boolean mask.
        device: Device for the returned mask. Defaults to CPU.
        dtype: Dtype for intermediate coordinate computation.
 
    Returns:
        A boolean torch.Tensor of shape (H, W) when ``soft_edge_pixels == 0``.
        Otherwise returns a float mask with values in ``[0, 1]``.
    """
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got {dtype}")
    if center is None:
        center = (int(w / 2), int(h / 2))
    else:
        center = (float(center[0]), float(center[1]))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    radius = float(radius)
    soft_edge_pixels = float(soft_edge_pixels)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if soft_edge_pixels < 0:
        raise ValueError(
            f"soft_edge_pixels must be non-negative, got {soft_edge_pixels}"
        )

    y = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, w)

    cx, cy = float(center[0]), float(center[1])
    distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if soft_edge_pixels == 0:
        return distance <= radius

    outer_radius = radius + soft_edge_pixels
    transition = 0.5 - 0.5 * torch.cos(
        math.pi * (outer_radius - distance) / soft_edge_pixels
    )
    return torch.where(
        distance <= radius,
        torch.ones_like(distance),
        torch.where(
            distance >= outer_radius,
            torch.zeros_like(distance),
            transition,
        ),
    )


def spherical_mask(
    d: int,
    h: int,
    w: int,
    center: Sequence[float] | None = None,
    radius: float | None = None,
    soft_edge_pixels: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a 3D spherical mask.
 
    Args:
        d, h, w: Volume size in (D, H, W) order.
        center: Sphere center in (x, y, z) index order. Defaults to the volume center.
        radius: Radius in pixels/voxels. Defaults to the largest radius that stays in-bounds.
        soft_edge_pixels: Width of the soft edge in voxels. A value of 0
            returns a hard boolean mask.
        device: Device for the returned mask. Defaults to CPU.
        dtype: Dtype for intermediate coordinate computation.
 
    Returns:
        A boolean torch.Tensor of shape (D, H, W) when ``soft_edge_pixels == 0``.
        Otherwise returns a float mask with values in ``[0, 1]``.
    """
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got {dtype}")
    if center is None:
        center = (int(w / 2), int(h / 2), int(d / 2))
    else:
        center = (float(center[0]), float(center[1]), float(center[2]))
    if radius is None:
        radius = min(
            center[0],
            center[1],
            center[2],
            w - center[0],
            h - center[1],
            d - center[2],
        )
    radius = float(radius)
    soft_edge_pixels = float(soft_edge_pixels)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if soft_edge_pixels < 0:
        raise ValueError(
            f"soft_edge_pixels must be non-negative, got {soft_edge_pixels}"
        )

    z = torch.arange(d, device=device, dtype=dtype).view(d, 1, 1)
    y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)

    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    distance = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    if soft_edge_pixels == 0:
        return distance <= radius

    outer_radius = radius + soft_edge_pixels
    transition = 0.5 - 0.5 * torch.cos(
        math.pi * (outer_radius - distance) / soft_edge_pixels
    )
    return torch.where(
        distance <= radius,
        torch.ones_like(distance),
        torch.where(
            distance >= outer_radius,
            torch.zeros_like(distance),
            transition,
        ),
    )


def radial_mask(
    shape: Sequence[int],
    *,
    radius: float | None = None,
    center: Sequence[float] | None = None,
    soft_edge_pixels: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Dispatch to the 2D or 3D torch radial mask implementation."""
    shape = tuple(int(size) for size in shape)
    if len(shape) == 2:
        return circular_mask(
            shape[0],
            shape[1],
            center=center,
            radius=radius,
            soft_edge_pixels=soft_edge_pixels,
            device=device,
            dtype=dtype,
        )
    if len(shape) == 3:
        return spherical_mask(
            shape[0],
            shape[1],
            shape[2],
            center=center,
            radius=radius,
            soft_edge_pixels=soft_edge_pixels,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"shape must be 2D or 3D, got {shape}")


# Mask Application Helpers

def center_fit_mask(volume: Tensor, side_length: int) -> Tensor:
    """Center-crop or zero-pad a cubic 3D mask to ``side_length`` voxels."""
    side_length = int(side_length)
    if side_length <= 0:
        raise ValueError(f"side_length must be positive, got {side_length}")
    if volume.ndim != 3 or not (volume.shape[0] == volume.shape[1] == volume.shape[2]):
        raise ValueError(
            f"volume must be a cubic 3D tensor, got shape={tuple(volume.shape)}"
        )

    source_size = int(volume.shape[-1])
    if source_size == side_length:
        return volume
    if source_size > side_length:
        start = (source_size - side_length) // 2
        end = start + side_length
        return volume[start:end, start:end, start:end]

    output = volume.new_zeros((side_length, side_length, side_length))
    start = (side_length - source_size) // 2
    end = start + source_size
    output[start:end, start:end, start:end] = volume
    return output


def load_mask_mrc(
    path: str | Path,
    *,
    side_length: int,
    angpix: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Load a solvent mask from MRC and align it to the current grid."""
    side_length = int(side_length)
    angpix = float(angpix)
    if side_length <= 0:
        raise ValueError(f"side_length must be positive, got {side_length}")
    if angpix <= 0:
        raise ValueError(f"angpix must be positive, got {angpix}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"dtype must be floating point, got {dtype}")

    path = Path(path)
    if not path.name:
        raise ValueError("a solvent-mask MRC path is required")
    if not path.is_file():
        raise FileNotFoundError(path)

    with mrcfile.open(path, permissive=True) as mrc:
        data = torch.as_tensor(mrc.data.copy(), dtype=dtype)
        mask_angpix = float(mrc.voxel_size.x)

    if data.ndim != 3 or not (data.shape[0] == data.shape[1] == data.shape[2]):
        raise ValueError(
            f"solvent mask must be a cubic 3D volume, got {tuple(data.shape)}"
        )
    if not torch.isfinite(data).all():
        raise ValueError("solvent mask contains non-finite values")

    if mask_angpix > 0 and abs(mask_angpix - angpix) > 1e-3:
        rescaled_size = max(
            1,
            int(round(int(data.shape[-1]) * mask_angpix / angpix)),
        )
        data = F.interpolate(
            data[None, None],
            size=(rescaled_size, rescaled_size, rescaled_size),
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    data = center_fit_mask(data, side_length)
    return data.clamp(0.0, 1.0).to(device=device, dtype=dtype)


def masked_lerp(input: Tensor, mask: Tensor, other: Tensor | float = 0) -> Tensor:

    """Blend ``other`` and ``input`` using a broadcastable soft mask.

    Args:
        input: Input tensor that should be kept where the mask is near ``1``.
        other: Background tensor or scalar blended in where the mask is near ``0``.

    Returns:
        A tensor with the same shape and dtype as ``input``.
    """
    mask = mask.to(device=input.device, dtype=input.real.dtype)
    if not torch.is_tensor(other):
        other = input.new_tensor(other)
    else:
        other = other.to(device=input.device, dtype=input.dtype)
    return torch.lerp(other, input, mask)


def particle_mask(
    h: int,
    w: int,
    *,
    particle_diameter: float,
    angpix: float,
    center: Sequence[float] | Tensor | None = None,
    soft_edge_pixels: float = 5.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a 2D soft particle mask in real space.

    Args:
        h: Image height.
        w: Image width.
        particle_diameter: Particle diameter in physical units.
        angpix: Pixel size in the same physical units as ``particle_diameter``.
        center: Particle center in ``(x, y)`` index order. Accepts either a
            single center of shape ``(2,)`` or batched centers of shape ``(B, 2)``.
            Defaults to the image center.
        soft_edge_pixels: Width of the raised-cosine soft edge in pixels.
        device: Device for the returned mask. Defaults to CPU.
        dtype: Floating-point dtype used to build the mask.

    Returns:
        A float mask of shape ``(H, W)`` for a single center, or ``(B, H, W)``
        for batched centers, with values in ``[0, 1]``.
    """
    h = int(h)
    w = int(w)
    if h != w:
        raise ValueError(
            f"particle_mask expects square particles, got ({h}, {w})"
        )
    particle_diameter = float(particle_diameter)
    angpix = float(angpix)
    soft_edge_pixels = float(soft_edge_pixels)
    if particle_diameter <= 0:
        raise ValueError(
            f"particle_diameter must be positive, got {particle_diameter}"
        )
    if angpix <= 0:
        raise ValueError(f"angpix must be positive, got {angpix}")
    if soft_edge_pixels <= 0:
        raise ValueError(
            f"soft_edge_pixels must be positive, got {soft_edge_pixels}"
        )

    if torch.is_tensor(center):
        if center.ndim != 2 or center.shape[-1] != 2:
            raise ValueError(
                f"batched center must have shape (B, 2), got {tuple(center.shape)}"
            )

        center = center.to(device=device, dtype=dtype)
        y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
        x = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
        cx = center[:, 0].view(-1, 1, 1)
        cy = center[:, 1].view(-1, 1, 1)
        distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = particle_diameter / (2.0 * angpix)
        outer_radius = radius + soft_edge_pixels
        transition = 0.5 - 0.5 * torch.cos(
            math.pi * (outer_radius - distance) / soft_edge_pixels
        )
        return torch.where(
            distance <= radius,
            torch.ones_like(distance),
            torch.where(
                distance >= outer_radius,
                torch.zeros_like(distance),
                transition,
            ),
        )

    return circular_mask(
        h,
        w,
        center=center,
        radius=particle_diameter / (2.0 * angpix),
        soft_edge_pixels=soft_edge_pixels,
        device=device,
        dtype=dtype,
    )


# Fourier-Space Masks

def lowpass_mask(x: Tensor, side_length: int, ndim: int) -> Tensor:
    """Create a centered low-pass mask in the Fourier domain.

    The mask keeps a centered ``(L, L)`` or ``(L, L, L)`` window in
    frequency space.

    Args:
        x: Reference tensor. The returned mask matches ``x`` in shape, device,
            and dtype.
        side_length: Side length ``L`` of the retained central frequency window.
        ndim: Spatial dimensionality of the mask. Must be ``2`` or ``3``.

    Returns:
        A tensor mask with the same shape and dtype as ``x``.
    """
    L = int(side_length)
    D = int(x.shape[-1])
    if L <= 0:
        raise ValueError(f"side_length must be > 0, got {side_length}")
    if L > D:
        raise ValueError(f"side_length ({L}) must be <= x.shape[-1] ({D})")

    dc = D // 2
    dc_left = L // 2
    start = dc - dc_left
    end = start + L

    mask = torch.zeros_like(x)
    if ndim == 2:
        small_mask = circular_mask(L, L, radius=L // 2, device=x.device)
        mask[..., start:end, start:end] = small_mask
    elif ndim == 3:
        small_mask = spherical_mask(L, L, L, radius=L // 2, device=x.device)
        mask[..., start:end, start:end, start:end] = small_mask
    else:
        raise ValueError(f"Invalid ndim: {ndim}")

    return mask