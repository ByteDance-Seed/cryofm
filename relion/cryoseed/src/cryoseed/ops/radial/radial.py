"""Fourier-space radial (shell-wise) utilities.

Radius bins are defined on the centered (fftshifted) FFT grid via
`cryoseed.fft.coords.fftindex_radial2d/3d`.

Note:
    `max_radius` here is an arbitrary cutoff chosen by the caller (e.g. a low-pass
    limit). It is not inherently tied to the Nyquist radius.
"""

import math
from functools import lru_cache
from typing import Literal

import torch

from cryoseed.fft.coords import fftindex_radial2d, fftindex_radial3d


def radial_broadcast(
    input: torch.Tensor,
    ndim: int,
    *,
    out_len: int | None = None,
    padding_mode: Literal["zeros", "border"] = "zeros",
) -> torch.Tensor:
    """Broadcast a radial profile back to a 2D/3D FFT grid.

    This maps a 1D radial profile defined on integer radius bins back onto a
    centered FFT grid (i.e. the same convention as ``torch.fft.fftshift``).

    Args:
        input (torch.Tensor): Tensor of shape ``batch_shape + (R,)``, where ``R`` is the
            number of radius bins.
        ndim (int): Spatial dimensionality of the output grid. Must be 2 or 3.
        out_len (int, optional): Output grid side length ``L``. If ``None``, defaults to
            ``2 * (R - 1)``.
        padding_mode (str, optional): How to handle grid points with radius greater than
            ``R - 1``. Must be one of:

            - ``"zeros"``: fill with 0
            - ``"border"``: reuse the last bin

            Default: ``"zeros"``.

    Returns:
        torch.Tensor: Tensor of shape ``batch_shape + (L,)*ndim``.

    Note:
        Radius indices on the FFT grid are computed via :func:`~cryoseed.fft.coords.fftindex_radial2d`
        and :func:`~cryoseed.fft.coords.fftindex_radial3d`. When ``out_len`` is ``None``, the
        default is an even side length ``L = 2 * (R - 1)``.
    """
    ndim = int(ndim)
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")

    if input.ndim < 1:
        raise ValueError("input must have at least 1 dimension")

    batch_shape = input.shape[:-1]
    num_radial_bins = int(input.shape[-1])
    if num_radial_bins <= 0:
        raise ValueError("input.shape[-1] must be > 0")
    max_radius = num_radial_bins - 1

    if out_len is None:
        out_len = 2 * max_radius if max_radius > 0 else 1
    out_len = int(out_len)
    if out_len <= 0:
        raise ValueError("out_len must be > 0")

    batch_size = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1
    input_flat = input.reshape(batch_size, num_radial_bins)

    if ndim == 3:
        r_flat = fftindex_radial3d(out_len, device=input.device)
    else:
        r_flat = fftindex_radial2d(out_len, device=input.device)

    if padding_mode == "border":
        r_flat = r_flat.clamp(max=max_radius)
        src_flat = input_flat
    elif padding_mode == "zeros":
        fill_col = torch.zeros((batch_size, 1), device=input.device, dtype=input.dtype)
        src_flat = torch.cat((input_flat, fill_col), dim=1)
        fill_bin = max_radius + 1
        r_flat = torch.where(
            r_flat > max_radius,
            torch.full_like(r_flat, fill_bin),
            r_flat,
        )
    else:
        raise ValueError('padding_mode must be "zeros" or "border"')

    r_batch = r_flat.unsqueeze(0).expand(batch_size, -1)
    out_flat = src_flat.gather(1, r_batch)
    out = out_flat.view((batch_size,) + (out_len,) * ndim)

    if len(batch_shape) == 0:
        return out.squeeze(0)
    return out.reshape(batch_shape + (out_len,) * ndim)


@lru_cache(maxsize=4)
def _cached_radial_avg_cache(
    device_type: str,
    device_index: int | None,
    ndim: int,
    side_length: int,
    max_radius: int,
):
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    return _build_radial_avg_cache(
        device=device,
        side_length=int(side_length),
        max_radius=int(max_radius),
        ndim=int(ndim),
    )


def clear_radial_average_cache():
    """Clear the internal index cache used by `radial_average(use_cache=True)`."""
    _cached_radial_avg_cache.cache_clear()


def _build_radial_avg_cache(
    device: torch.device,
    side_length: int,
    max_radius: int,
    ndim: int,
):
    if ndim == 3:
        r_flat = fftindex_radial3d(side_length, device=device)
    elif ndim == 2:
        r_flat = fftindex_radial2d(side_length, device=device)
    else:
        raise ValueError("ndim must be 2 or 3")

    num_points = int(r_flat.numel())

    valid_mask = r_flat <= max_radius
    valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)

    num_radial_bins = max_radius + 1
    radial_indices = r_flat.index_select(0, valid_indices)
    radial_bin_counts = torch.bincount(radial_indices, minlength=num_radial_bins)

    return {
        "num_points": num_points,
        "valid_indices": valid_indices,
        "radial_indices": radial_indices,
        "radial_bin_counts": radial_bin_counts,
        "num_radial_bins": num_radial_bins,
    }


def radial_average(
    input: torch.Tensor,
    max_radius: int,
    ndim: int,
    use_cache: bool = False,
) -> torch.Tensor:
    """Compute a radial (shell-wise) average over the last ``ndim`` spatial dimensions.

    Args:
        input (torch.Tensor): Tensor of shape ``batch_shape + (L,)*ndim``.
        max_radius (int): Radial cutoff in integer radius bins. Only points with
            ``r <= max_radius`` contribute to the average.
        ndim (int): Spatial dimensionality. Must be 2 or 3.
        use_cache (bool, optional): If ``True``, caches index metadata on the same device for reuse.
            Default: ``False``.

    Returns:
        torch.Tensor: Tensor of shape ``batch_shape + (max_radius + 1,)``.
    """
    ndim = int(ndim)
    max_radius = int(max_radius)
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")

    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")

    if input.ndim < ndim:
        raise ValueError("input must have at least ndim spatial dimensions")

    side_length = int(input.shape[-1])
    if side_length <= 0:
        raise ValueError("input spatial dimensions must be > 0")
    if any(input.shape[-k] != side_length for k in range(1, ndim + 1)):
        raise ValueError("input spatial dimensions must all be equal to side_length")

    batch_shape = input.shape[:-ndim]
    batch_size = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1

    cache = (
        _cached_radial_avg_cache(
            input.device.type,
            input.device.index,
            ndim,
            side_length,
            max_radius,
        )
        if use_cache
        else _build_radial_avg_cache(input.device, side_length, max_radius, ndim)
    )

    valid_indices = cache["valid_indices"]
    radial_indices = cache["radial_indices"]
    radial_bin_counts = cache["radial_bin_counts"]
    num_radial_bins = int(cache["num_radial_bins"])

    num_points = int(cache["num_points"])
    input_flat = input.reshape(batch_size, num_points)

    src = input_flat.index_select(1, valid_indices)
    idx = radial_indices.unsqueeze(0).expand(batch_size, -1)

    bin_sums = torch.zeros(
        batch_size,
        num_radial_bins,
        dtype=input.dtype,
        device=input.device,
    )
    bin_sums.scatter_add_(1, idx, src)

    denom = radial_bin_counts.to(dtype=input.dtype).clamp(min=1).unsqueeze(0)
    radial_avg_flat = bin_sums / denom
    radial_avg = radial_avg_flat.reshape(*batch_shape, num_radial_bins)
    return radial_avg