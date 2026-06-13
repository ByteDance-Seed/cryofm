from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cryoseed.fft.coords import fftfreq_slice_coords3d, fftfreq_to_grid


__all__ = [
    "project",
]


def project(
    volume: Tensor,
    rotation: Tensor,
    *,
    channel_last: bool = True,
) -> Tensor:
    """Projection via ``grid_sample``.

    Conventions
        - Volume storage order is (z, y, x). ``grid_sample`` expects grid coordinates
          in (x, y, z) order, so all coordinates in this function are (x, y, z).
        - The projection is a central Fourier slice at z=0 (detector plane).
        - ``rotation`` represents the rotation of the *volume* relative to the
          detector frame.

          We implement the equivalent sampling operation by rotating the sampling
          coordinates in the opposite sense (relative motion). Coordinates are
          treated as row-vectors and right-multiplied:

              coords = coords @ R

          If you prefer column-vector notation, this corresponds to applying
          ``R.T`` to column vectors, so no explicit transpose is needed here.

    Args:
        volume:
            - ``channel_last=True``: (B, D, H, W, C)
            - ``channel_last=False``: (B, C, D, H, W)
        rotation: (B, Q, 3, 3)

    Returns:
        - ``channel_last=True``: (B, Q, H, W, C)
        - ``channel_last=False``: (B, Q, C, H, W)
    """

    if volume.ndim != 5:
        raise ValueError(f"volume must be 5D, got {tuple(volume.shape)}")
    if rotation.ndim != 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must be (B, Q, 3, 3), got {tuple(rotation.shape)}")

    if channel_last:
        B, D, H, W, C = volume.shape
    else:
        B, C, D, H, W = volume.shape

    if rotation.shape[0] != B:
        raise ValueError(f"rotation.shape[0] must equal B, got {rotation.shape[0]} vs {B}")
    if not (D == H == W):
        raise ValueError(f"volume must be cubic, got (D,H,W)=({D},{H},{W})")

    Q = int(rotation.shape[1])
    L = int(D)

    if L < 2:
        if channel_last:
            return torch.zeros((B, Q, L, L, C), device=volume.device, dtype=volume.dtype)
        return torch.zeros((B, Q, C, L, L), device=volume.device, dtype=volume.dtype)

    if channel_last:
        volume_ncdhw = volume.permute(0, 4, 1, 2, 3)
    else:
        volume_ncdhw = volume

    coords = fftfreq_slice_coords3d(L, device=volume.device, dtype=torch.float32)  # (L*L, 3)

    rotation = rotation.to(device=volume.device, dtype=torch.float32).reshape(B * Q, 3, 3)

    coords = coords.unsqueeze(0) @ rotation  # (B*Q, L*L, 3), row-vector @ R
    coords = coords.view(B, Q, L, L, 3)  # (B, Q, H=y, W=x, 3)

    grid = fftfreq_to_grid(coords, L, align_corners=True)

    sampled = F.grid_sample(volume_ncdhw, grid, align_corners=True)  # (B, C, Q, L, L)

    if channel_last:
        proj = sampled.permute(0, 2, 3, 4, 1)  # (B, Q, L, L, C)
    else:
        proj = sampled.permute(0, 2, 1, 3, 4)  # (B, Q, C, L, L)
    return proj