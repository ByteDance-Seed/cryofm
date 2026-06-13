"""Coordinate helpers for FFT-based ops.

Axis convention

3D

- Tensor storage order: (z, y, x)
- Spatial coordinate order: (x, y, z)

i.e. tensor[z, y, x] corresponds to coordinate (x, y, z).

2D

- Tensor storage order: (y, x)
- Spatial coordinate order: (x, y)

i.e. tensor[y, x] corresponds to coordinate (x, y).

Caching

Some helpers return cached tensors for common (n_fft, n_out, device, dtype) tuples.
These cached tensors are shared across calls: do not modify them in-place.
If you switch resolution/device frequently or want to release GPU memory, call
:func:`clear_coord_caches`.
"""
from functools import lru_cache
from typing import Optional

import torch

from cryoseed.utils.torch_utils import _norm_device, _norm_dtype

__all__ = [
    "fftfreq",
    "fftfreq_coords2d",
    "fftfreq_coords3d",
    "fftfreq_slice_coords3d",
    "fftfreq_to_grid",
    "fftindex",
    "fftindex_coords2d",
    "fftindex_coords3d",
    "fftindex_coords2d_radial",
    "fftindex_components2d_radial",
    "fftindex_radial2d",
    "fftindex_radial3d",
    "clear_coord_caches",
]

# ========== Helper Functions ==========

def clear_coord_caches() -> None:
    """Clear internal LRU caches used by coordinate helpers."""

    _cached_fftfreq.cache_clear()
    _cached_fftfreq_coords2d.cache_clear()
    _cached_fftfreq_coords3d.cache_clear()
    _cached_fftfreq_slice_coords3d.cache_clear()

    _cached_fftindex.cache_clear()
    _cached_fftindex_coords2d.cache_clear()
    _cached_fftindex_coords3d.cache_clear()

    _cached_fftindex_coords2d_radial.cache_clear()
    _cached_fftindex_components2d_radial.cache_clear()

    _cached_fftindex_radial2d.cache_clear()
    _cached_fftindex_radial3d.cache_clear()


def _isqrt(x: torch.Tensor) -> torch.Tensor:
    if not x.dtype in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }:
        raise TypeError("_isqrt expects an integer tensor")

    if torch.any(x < 0):
        raise ValueError("_isqrt expects non-negative integers")

    x_int64 = x.to(torch.int64)
    y = torch.sqrt(x_int64.to(torch.float64)).floor().to(torch.int64)

    y = torch.where(y * y > x_int64, y - 1, y)
    yp1 = y + 1
    y = torch.where(yp1 * yp1 <= x_int64, yp1, y)

    return y.to(x.dtype)


# ========== FFT Frequencies ==========

@lru_cache(maxsize=16)
def _cached_fftfreq(
    n_fft: int,
    n_out: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_fft = int(n_fft)
    if n_fft <= 0:
        raise ValueError(f"n_fft must be > 0, got {n_fft}")

    if n_out is not None:
        n_out = int(n_out)
        if n_out <= 0:
            raise ValueError(f"n_out must be > 0, got {n_out}")
        if n_out > n_fft:
            raise ValueError(f"n_out must be <= n_fft, got n_out={n_out}, n_fft={n_fft}")

    freq = torch.fft.fftfreq(n_fft, device=device, dtype=dtype)
    freq = torch.fft.fftshift(freq)
    if n_out is not None:
        dc = n_fft // 2
        left = n_out // 2
        right = n_out - left
        freq = freq[dc - left : dc + right]
    return freq.contiguous()


def fftfreq(
    n_fft: int,
    n_out: Optional[int] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Centered FFT frequencies in the same convention as ``torch.fft.fftshift``.

    Returns a 1D tensor of frequencies in [-0.5, 0.5). If ``n_out`` is provided, the
    centered window of length ``n_out`` is returned.

    Note: the returned tensor may be cached and shared across calls; do not modify it
    in-place.
    """

    n_fft = int(n_fft)
    n_out = None if n_out is None else int(n_out)
    device = _norm_device(device)
    dtype = _norm_dtype(dtype)
    return _cached_fftfreq(n_fft, n_out, device, dtype)


@lru_cache(maxsize=8)
def _cached_fftfreq_coords2d(
    n_fft: int,
    n_out: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freq = _cached_fftfreq(int(n_fft), n_out, device, dtype)
    if hasattr(torch, "cartesian_prod"):
        yx = torch.cartesian_prod(freq, freq)
        return yx[:, [1, 0]].contiguous()
    yy, xx = torch.meshgrid(freq, freq, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).contiguous()


def fftfreq_coords2d(
    n_fft: int,
    n_out: Optional[int] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """2D frequency coordinates flattened to shape (N, 2) in (x, y) order."""

    n_fft = int(n_fft)
    n_out = None if n_out is None else int(n_out)
    device = _norm_device(device)
    dtype = _norm_dtype(dtype)
    return _cached_fftfreq_coords2d(n_fft, n_out, device, dtype)


@lru_cache(maxsize=4)
def _cached_fftfreq_coords3d(
    n_fft: int,
    n_out: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    freq = _cached_fftfreq(int(n_fft), n_out, device, dtype)
    if hasattr(torch, "cartesian_prod"):
        zyx = torch.cartesian_prod(freq, freq, freq)
        return zyx[:, [2, 1, 0]].contiguous()
    zz, yy, xx = torch.meshgrid(freq, freq, freq, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).contiguous()


def fftfreq_coords3d(
    n_fft: int,
    n_out: Optional[int] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """3D frequency coordinates flattened to shape (N, 3) in (x, y, z) order."""

    n_fft = int(n_fft)
    n_out = None if n_out is None else int(n_out)
    device = _norm_device(device)
    dtype = _norm_dtype(dtype)
    return _cached_fftfreq_coords3d(n_fft, n_out, device, dtype)


@lru_cache(maxsize=8)
def _cached_fftfreq_slice_coords3d(
    n_fft: int,
    n_out: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords2d = _cached_fftfreq_coords2d(n_fft, n_out, device, dtype)
    out = torch.empty((coords2d.shape[0], 3), device=device, dtype=dtype)
    out[:, :2] = coords2d
    out[:, 2].zero_()
    return out.contiguous()


def fftfreq_slice_coords3d(
    n_fft: int,
    n_out: Optional[int] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """3D central slice coordinates (z=0) flattened to shape (N, 3) in (x, y, z) order."""

    n_fft = int(n_fft)
    n_out = None if n_out is None else int(n_out)
    device = _norm_device(device)
    dtype = _norm_dtype(dtype)
    return _cached_fftfreq_slice_coords3d(n_fft, n_out, device, dtype)


def fftfreq_to_grid(
    freq: torch.Tensor,
    n_fft: int,
    align_corners: bool = True,
) -> torch.Tensor:
    """Convert centered FFT frequencies to ``grid_sample`` coordinates.

    ``freq`` is assumed to be in the same centered convention as :func:`fftfreq`.
    The output is normalized to [-1, 1] according to ``align_corners``.
    """

    n_fft = int(n_fft)
    idx = freq * n_fft + (n_fft // 2)
    if align_corners:
        return idx * (2.0 / (n_fft - 1)) - 1.0
    return (2.0 * idx + 1.0) / n_fft - 1.0

# ========== FFT Indices ==========

@lru_cache(maxsize=16)
def _cached_fftindex(
    side_length: int,
    device: torch.device,
) -> torch.Tensor:
    side_length = int(side_length)
    if side_length <= 0:
        raise ValueError(f"side_length must be > 0, got {side_length}")

    start = -(side_length // 2)
    idx = torch.arange(start, start + side_length, device=device, dtype=torch.int64)
    return idx.contiguous()


def fftindex(
    side_length: int,
    device=None,
) -> torch.Tensor:
    """Centered FFT integer indices in the same convention as ``torch.fft.fftshift``.

    Returns a 1D tensor of integer indices of length ``side_length``:

    - even L:  [-L/2, ..., -1, 0, 1, ..., L/2-1]
    - odd L:   [-(L//2), ..., -1, 0, 1, ..., (L//2)]

    Note: the returned tensor may be cached and shared across calls; do not modify it
    in-place.
    """

    side_length = int(side_length)
    device = _norm_device(device)
    return _cached_fftindex(side_length, device)


@lru_cache(maxsize=8)
def _cached_fftindex_coords2d(
    side_length: int,
    device: torch.device,
) -> torch.Tensor:
    freq = _cached_fftindex(int(side_length), device)
    if hasattr(torch, "cartesian_prod"):
        yx = torch.cartesian_prod(freq, freq)
        return yx[:, [1, 0]].contiguous()
    yy, xx = torch.meshgrid(freq, freq, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).contiguous()


def fftindex_coords2d(
    side_length: int,
    device=None,
) -> torch.Tensor:
    """2D integer index coordinates flattened to shape (N, 2) in (x, y) order."""

    side_length = int(side_length)
    device = _norm_device(device)
    return _cached_fftindex_coords2d(side_length, device)


@lru_cache(maxsize=4)
def _cached_fftindex_coords3d(
    side_length: int,
    device: torch.device,
) -> torch.Tensor:
    side_length = int(side_length)
    freq = _cached_fftindex(side_length, device)
    if hasattr(torch, "cartesian_prod"):
        zyx = torch.cartesian_prod(freq, freq, freq)
        return zyx[:, [2, 1, 0]].contiguous()
    zz, yy, xx = torch.meshgrid(freq, freq, freq, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).contiguous()


def fftindex_coords3d(
    side_length: int,
    device=None,
) -> torch.Tensor:
    """3D integer index coordinates flattened to shape (N, 3) in (x, y, z) order."""

    side_length = int(side_length)
    device = _norm_device(device)
    return _cached_fftindex_coords3d(side_length, device)


@lru_cache(maxsize=8)
def _cached_fftindex_coords2d_radial(
    side_length: int,
    max_radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    side_length = int(side_length)
    max_radius = float(max_radius)
    if max_radius < 0:
        raise ValueError(f"max_radius must be >= 0, got {max_radius}")

    coords = _cached_fftindex_coords2d(side_length, device)
    x = coords[:, 0]
    y = coords[:, 1]

    r2 = (x * x) + (y * y)
    max_r2 = max_radius * max_radius
    mask = r2 <= max_r2

    coords = coords[mask].contiguous()
    if coords.dtype != dtype:
        coords = coords.to(dtype=dtype)
    return coords


def fftindex_coords2d_radial(
    side_length: int,
    max_radius: float,
    device=None,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """2D integer index coordinates within a radial mask, flattened to shape (N, 2).

    Returns integer coordinates in ``(x, y)`` order from a centered 2D FFT grid,
    keeping only points satisfying ``x^2 + y^2 <= max_radius^2``.

    Note: the returned tensor may be cached and shared across calls; do not modify it
    in-place.
    """

    side_length = int(side_length)
    max_radius = float(max_radius)
    device = _norm_device(device)
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"dtype must be torch.int32 or torch.int64, got {dtype}")
    return _cached_fftindex_coords2d_radial(side_length, max_radius, device, dtype)


@lru_cache(maxsize=4)
def _cached_fftindex_components2d_radial(
    side_length: int,
    max_radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    side_length = int(side_length)
    max_radius = float(max_radius)
    if max_radius < 0:
        raise ValueError(f"max_radius must be >= 0, got {max_radius}")

    coords = _cached_fftindex_coords2d(side_length, device)
    x = coords[:, 0]
    y = coords[:, 1]

    r2 = (x * x) + (y * y)
    mask = r2 <= max_radius * max_radius

    x = x[mask].to(dtype).contiguous()
    y = y[mask].to(dtype).contiguous()
    return x, y


def fftindex_components2d_radial(
    side_length: int,
    max_radius: float,
    device=None,
    dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D FFT-grid coordinate components within a radial mask.

    Returns cached ``(x, y)`` integer coordinate components, each with shape ``(N,)``.
    The returned tensors are contiguous and suitable for low-level kernel use.

    Args:
        side_length: Grid side length ``L``.
        max_radius: Radial cutoff in centered FFT index units.
        device: Target device.
        dtype: Output dtype for ``x`` and ``y`` (default: ``torch.int32``).
    """

    side_length = int(side_length)
    max_radius = float(max_radius)
    device = _norm_device(device)
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"dtype must be torch.int32 or torch.int64, got {dtype}")
    return _cached_fftindex_components2d_radial(side_length, max_radius, device, dtype)


@lru_cache(maxsize=8)
def _cached_fftindex_radial2d(
    side_length: int,
    device: torch.device,
) -> torch.Tensor:
    side_length = int(side_length)
    idx = _cached_fftindex(side_length, device)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    r2 = (xx * xx) + (yy * yy)
    return _isqrt(r2).reshape(-1).contiguous()


def fftindex_radial2d(
    side_length: int,
    device=None,
) -> torch.Tensor:
    """Flattened integer radius bins on a centered 2D FFT grid.

    Returns a 1D int64 tensor of shape ``(side_length * side_length,)``.
    """

    side_length = int(side_length)
    device = _norm_device(device)
    return _cached_fftindex_radial2d(side_length, device)


@lru_cache(maxsize=4)
def _cached_fftindex_radial3d(
    side_length: int,
    device: torch.device,
) -> torch.Tensor:
    side_length = int(side_length)
    idx = _cached_fftindex(side_length, device)
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    r2 = (xx * xx) + (yy * yy) + (zz * zz)
    return _isqrt(r2).reshape(-1).contiguous()


def fftindex_radial3d(
    side_length: int,
    device=None,
) -> torch.Tensor:
    """Flattened integer radius bins on a centered 3D FFT grid.

    Returns a 1D int64 tensor of shape ``(side_length**3,)``.
    """

    side_length = int(side_length)
    device = _norm_device(device)
    return _cached_fftindex_radial3d(side_length, device)