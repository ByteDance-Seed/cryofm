"""Mask utilities used in cryo-EM preprocessing.

This module provides small helpers to build circular / spherical masks in NumPy
or PyTorch, and a RELION-style soft mask for batched tensors.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

_all__ = [
    "circular_mask_numpy",
    "spherical_mask_numpy",
    "circular_mask",
    "spherical_mask",
    "soft_mask_background",
    "lowpass_filter",
]

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


def circular_mask(
    h, w, center=None, radius=None, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Create a 2D circular mask.
 
    Args:
        h, w: Image size in (H, W) order.
        center: Circle center in (x, y) index order. Defaults to the image center.
        radius: Radius in pixels. Defaults to the largest radius that stays in-bounds.
        device: Device for the returned mask. Defaults to CPU.
        dtype: Dtype for intermediate coordinate computation.
 
    Returns:
        A boolean torch.Tensor of shape (H, W).
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, w)

    cx, cy = float(center[0]), float(center[1])
    r2 = float(radius) * float(radius)

    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    return dist2 <= r2


def spherical_mask(
    d, h, w, center=None, radius=None, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Create a 3D spherical mask.
 
    Args:
        d, h, w: Volume size in (D, H, W) order.
        center: Sphere center in (x, y, z) index order. Defaults to the volume center.
        radius: Radius in pixels/voxels. Defaults to the largest radius that stays in-bounds.
        device: Device for the returned mask. Defaults to CPU.
        dtype: Dtype for intermediate coordinate computation.
 
    Returns:
        A boolean torch.Tensor of shape (D, H, W).
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

    z = torch.arange(d, device=device, dtype=dtype).view(d, 1, 1)
    y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)

    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    r2 = float(radius) * float(radius)

    dist2 = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
    return dist2 <= r2


def soft_mask_background(
    vol: Tensor,
    radius: float,
    cosine_width: float = 5.0,
    noise: Tensor | None = None,
) -> Tensor:
    """Apply a RELION-style soft mask outside a radius.

    Values outside ``radius + cosine_width`` are replaced by background. Values
    in the transition band are blended using a raised cosine.

    Args:
        vol: Batched tensor of shape (B, D, H, W) or (B, H, W).
        radius: Inner radius in pixels/voxels.
        cosine_width: Width of the raised-cosine transition band.
        noise: Optional background tensor with the same shape as ``vol``. If not
            provided, the background is estimated from the outer region.

    Returns:
        Masked tensor with the same shape and dtype as ``vol``.
    """
    device = vol.device
    dtype = vol.dtype

    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if cosine_width <= 0:
        raise ValueError(f"cosine_width must be > 0, got {cosine_width}")
    if noise is not None and noise.shape != vol.shape:
        raise ValueError(f"noise must have the same shape as vol, got {noise.shape} vs {vol.shape}")

    # Determine spatial dimensions
    if vol.dim() == 4:  # Batched 3D: (batch_size, D, H, W)
        batch_size = vol.shape[0]
        spatial_dims = vol.shape[1:]
        is_3d = True
    elif vol.dim() == 3:  # Batched 2D: (batch_size, H, W)
        batch_size = vol.shape[0]
        spatial_dims = vol.shape[1:]
        is_3d = False
    else:
        raise ValueError(
            "Volume must be batched: (batch_size, D, H, W) or (batch_size, H, W)"
        )

    radius_p = radius + cosine_width

    # Create coordinate grids for spatial dimensions
    if is_3d:  # 3D spatial dimensions
        d_coords = (
            torch.arange(spatial_dims[0], device=device, dtype=dtype)
            - spatial_dims[0] // 2
        )
        h_coords = (
            torch.arange(spatial_dims[1], device=device, dtype=dtype)
            - spatial_dims[1] // 2
        )
        w_coords = (
            torch.arange(spatial_dims[2], device=device, dtype=dtype)
            - spatial_dims[2] // 2
        )

        d_grid, h_grid, w_grid = torch.meshgrid(
            d_coords, h_coords, w_coords, indexing="ij"
        )
        r = torch.sqrt(d_grid**2 + h_grid**2 + w_grid**2)

    else:  # 2D spatial dimensions
        h_coords = (
            torch.arange(spatial_dims[0], device=device, dtype=dtype)
            - spatial_dims[0] // 2
        )
        w_coords = (
            torch.arange(spatial_dims[1], device=device, dtype=dtype)
            - spatial_dims[1] // 2
        )

        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        r = torch.sqrt(h_grid**2 + w_grid**2)

    # Expand r to match batch dimension
    r = r.unsqueeze(0).expand(batch_size, *r.shape)

    # Create masks for different regions
    outer_mask = r > radius_p
    transition_mask = (r >= radius) & (r <= radius_p)

    # Calculate background value if noise is not provided
    if noise is None:
        # Calculate weighted average of background values for each item in batch
        transition_weights = torch.where(
            transition_mask,
            0.5 + 0.5 * torch.cos(math.pi * (radius_p - r) / cosine_width),
            0.0,
        )

        outer_weights = outer_mask.float()

        # Sum over spatial dimensions, keep batch dim
        spatial_dims_to_sum = list(range(1, vol.dim()))
        total_weights = outer_weights + transition_weights
        weighted_sum = (vol * outer_weights + vol * transition_weights).sum(
            dim=spatial_dims_to_sum, keepdim=True
        )
        weight_sum = total_weights.sum(dim=spatial_dims_to_sum, keepdim=True)

        # Avoid division by zero
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        sum_bg = weighted_sum / weight_sum

        # Expand to full volume shape
        background = sum_bg.expand_as(vol)
    else:
        background = noise

    # Apply the mask
    result = vol.clone()

    # Outer region: replace with background
    result = torch.where(outer_mask, background, result)

    # Transition region: blend with raised cosine
    raisedcos = 0.5 + 0.5 * torch.cos(math.pi * (radius_p - r) / cosine_width)
    blended = (1 - raisedcos) * vol + raisedcos * background
    result = torch.where(transition_mask, blended, result)

    return result

def lowpass_mask(x: Tensor, side_length: int, ndim: int) -> Tensor:
    """Build a centered low-pass mask in the Fourier domain.

    This helper mirrors the behavior of the legacy implementation that crops a
    centered (L, L) or (L, L, L) window in frequency space.

    Args:
        x: Input tensor. The mask is created with the same shape as ``x``.
        side_length: Side length ``L`` of the kept central frequency window.
        ndim: Spatial dimensionality (2 or 3).

    Returns:
        A tensor mask with the same shape as ``x``.
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