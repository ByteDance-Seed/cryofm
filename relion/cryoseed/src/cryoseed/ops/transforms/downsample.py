from __future__ import annotations

from torch import Tensor

__all__ = [
    "downsample2d",
    "downsample3d",
]


def _crop_center_freq_nd(x: Tensor, side_length: int, ndim: int) -> Tensor:
    """Center-crop the last `ndim` dimensions to a size-L window.

    This helper assumes the cropped dimensions are all equal-length and
    interpreted on a centered discrete grid (e.g. Fourier grid with DC-centered
    convention).
    """
    if x.ndim < ndim:
        raise ValueError(f"Expected x.ndim >= {ndim}, got {x.ndim}")

    L = int(side_length)
    if L <= 0:
        raise ValueError(f"Expected side_length > 0, got {side_length}")

    spatial_shape = x.shape[-ndim:]
    D = int(spatial_shape[-1])

    if any(int(s) != D for s in spatial_shape):
        raise ValueError(
            f"Expected last {ndim} dims to be equal, got {spatial_shape}"
        )
    if L > D:
        raise ValueError(
            f"Expected side_length <= input size, got {L} > {D}"
        )
    if L == D:
        return x

    dc = D // 2
    left = L // 2
    right = L - left

    s = dc - left
    e = dc + right

    slices = [slice(None)] * (x.ndim - ndim) + [slice(s, e)] * ndim
    return x[tuple(slices)]


def downsample2d(image: Tensor, side_length: int) -> Tensor:
    """Downsample a 2D Fourier image by center-cropping its frequency window.

    Args:
        image: Tensor with shape (..., D, D).
        side_length: Output side length L.

    Returns:
        Tensor with shape (..., L, L).
    """
    return _crop_center_freq_nd(image, side_length, ndim=2)


def downsample3d(volume: Tensor, side_length: int) -> Tensor:
    """Downsample a 3D Fourier volume by center-cropping its frequency window.

    Args:
        volume: Tensor with shape (..., D, D, D).
        side_length: Output side length L.

    Returns:
        Tensor with shape (..., L, L, L).
    """
    return _crop_center_freq_nd(volume, side_length, ndim=3)
