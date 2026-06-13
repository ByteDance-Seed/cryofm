from __future__ import annotations

import einops
import torch
from torch import Tensor

from cryoseed.fft.coords import fftfreq_coords2d

__all__ = ["translate_image"]


def translate_image(image: Tensor, translation: Tensor) -> Tensor:
    """Translate 2D Fourier-domain images by applying a translation-induced phase factor.

    Axis/order conventions:
    - ``image`` is stored in tensor order ``(y, x)`` (i.e. ``image[..., y, x]``).
    - ``translation`` is in spatial coordinate order ``(x, y)`` measured in pixels.

    Args:
        image:
            Fourier-domain images of shape ``(B, H, W)``.
        translation:
            Translation tensor in ``(x, y)`` order.

            Supported shapes:
            - ``(B, 2)``: one translation per image
            - ``(B, T, 2)``: ``T`` translations per image

    Returns:
        Translated images:
        - If ``translation.shape == (B, 2)``, returns shape ``(B, H, W)``
        - If ``translation.shape == (B, T, 2)``, returns shape ``(B, T, H, W)``

    Notes:
        Uses centered FFT frequencies in ``[-0.5, 0.5)`` from ``fftfreq_coords2d``.
        The implementation generates a real-valued phase angle and converts it to a
        unit-magnitude phase factor.
    """
    if image.ndim != 3:
        raise ValueError(f"image must have shape (B,H,W), got {tuple(image.shape)}")

    B, H, W = image.shape
    if H != W:
        raise ValueError(f"image must be square, got (H,W)=({H},{W})")

    if translation.ndim == 2:
        if translation.shape != (B, 2):
            raise ValueError(
                f"translation must have shape (B,2) with B={B}, got {tuple(translation.shape)}"
            )
        translation = translation[:, None, :]
        squeeze_t = True
    elif translation.ndim == 3:
        if translation.shape[0] != B or translation.shape[-1] != 2:
            raise ValueError(
                f"translation must have shape (B,T,2) with B={B}, got {tuple(translation.shape)}"
            )
        squeeze_t = False
    else:
        raise ValueError(
            f"translation must be 2D or 3D, got translation.ndim={translation.ndim}"
        )

    coord_dtype = image.real.dtype
    if translation.dtype.is_floating_point:
        coord_dtype = torch.promote_types(coord_dtype, translation.dtype)
    coords = fftfreq_coords2d(H, device=image.device, dtype=coord_dtype)  # (H*W, 2) in (x, y)

    image = einops.rearrange(image, "B H W -> B 1 (H W)")
    delta = (translation.to(dtype=coords.dtype) @ coords.t())  # (B,T,H*W)

    image = torch.exp(-2j * torch.pi * delta) * image
    image = einops.rearrange(image, "B T (H W) -> B T H W", H=H, W=W)

    if squeeze_t:
        image = image.squeeze(1)
    return image