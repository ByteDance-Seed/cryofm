from __future__ import annotations

import warnings
from typing import Literal, Optional

from torch import Tensor


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is unavailable in the current environment."""


BackendName = Literal["unknown", "triton", "torch"]
ConcreteBackendName = Literal["triton", "torch"]

__all__ = [
    "BackendName",
    "BackendUnavailableError",
    "backproject",
    "backend_status",
]

_BACKEND_SELECTION: BackendName = "unknown"
_BACKEND_ERROR: Optional[str] = None
_WARNED_FALLBACK = False


def backend_status() -> tuple[BackendName, Optional[str]]:
    """Return current backproject backend selection state.

    Returns:
        A tuple ``(backend, error)`` where:

        - ``backend`` is one of ``{"unknown", "triton", "torch"}``.
        - ``error`` stores the last backend availability error
          encountered during auto-selection fallback, otherwise ``None``.
    """
    return _BACKEND_SELECTION, _BACKEND_ERROR


def _warn_triton_fallback_once(message: str) -> None:
    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _WARNED_FALLBACK = True


def _backproject_with_backend(
    backend: ConcreteBackendName,
    image: Tensor,
    ctf: Optional[Tensor],
    noise_spectrum: Tensor,
    *,
    probability: Optional[Tensor],
    image_index: Optional[Tensor],
    volume_index: Optional[Tensor],
    rotation: Tensor,
    translation: Tensor,
    radius: float,
    volume_numerator: Optional[Tensor],
    volume_denominator: Optional[Tensor],
    return_denom: bool,
) -> tuple[Tensor, Optional[Tensor]]:
    if backend == "triton":
        try:
            from cryoseed.backends.triton.backproject import backproject as backproject_triton
        except Exception as e:
            raise BackendUnavailableError(
                f"Failed to import Triton backproject backend: {e!r}"
            ) from e

        return backproject_triton(
            image=image,
            ctf=ctf,
            noise_spectrum=noise_spectrum,
            probability=probability,
            image_index=image_index,
            volume_index=volume_index,
            rotation=rotation,
            translation=translation,
            radius=radius,
            volume_numerator=volume_numerator,
            volume_denominator=volume_denominator,
            return_denom=return_denom,
        )

    if backend == "torch":
        from cryoseed.backends.torch.backproject import backproject as backproject_torch

        return backproject_torch(
            image=image,
            ctf=ctf,
            noise_spectrum=noise_spectrum,
            probability=probability,
            image_index=image_index,
            volume_index=volume_index,
            rotation=rotation,
            translation=translation,
            radius=radius,
            volume_numerator=volume_numerator,
            volume_denominator=volume_denominator,
            return_denom=return_denom,
        )

    raise ValueError(f"Unknown backend: {backend}")


def backproject(
    image: Tensor,
    ctf: Optional[Tensor],
    noise_spectrum: Tensor,
    *,
    probability: Optional[Tensor],
    image_index: Optional[Tensor],
    volume_index: Optional[Tensor] = None,
    rotation: Tensor,
    translation: Tensor,
    radius: float,
    volume_numerator: Optional[Tensor] = None,
    volume_denominator: Optional[Tensor] = None,
    return_denom: bool = True,
) -> tuple[Tensor, Optional[Tensor]]:
    """Backproject 2D Fourier images into a 3D Fourier volume.

    This function prefers the Triton backend when available and falls back to
    the Torch implementation otherwise.

    Args:
        image: Input Fourier images with shape ``(B, L, L)``.
        ctf: Optional per-image CTF modulation with shape ``(B, L, L)``.
        noise_spectrum: Per-pixel noise spectrum map with shape ``(L, L)``.
        probability: Optional per-pose probability/weight with shape ``(N,)``.
        image_index: Optional pose-to-image index mapping with shape ``(N,)``.
            If ``None``, requires ``N == B``.
        volume_index: Optional pose-to-volume index mapping with shape ``(N,)``.
            If ``None``, all poses are accumulated into volume 0.
        rotation: Per-pose rotation matrices with shape ``(N, 3, 3)`` or
            flattened shape ``(N, 9)``.
        translation: Per-pose translations with shape ``(N, 2)``.
            Must be provided; pass zeros for no shift. The first dimension
            must match ``rotation``.
        radius: Maximum radial support in Fourier pixels.
        volume_numerator: Optional output numerator buffer.
        volume_denominator: Optional output denominator buffer.
        return_denom: Whether to accumulate and return denominator.

    Returns:
        A tuple ``(volume_numerator, volume_denominator)``.

    See Also:
        :func:`backend_status` to inspect the cached backend selection state.
    """
    global _BACKEND_SELECTION, _BACKEND_ERROR

    if _BACKEND_SELECTION == "triton":
        return _backproject_with_backend(
            "triton",
            image,
            ctf,
            noise_spectrum,
            probability=probability,
            image_index=image_index,
            volume_index=volume_index,
            rotation=rotation,
            translation=translation,
            radius=radius,
            volume_numerator=volume_numerator,
            volume_denominator=volume_denominator,
            return_denom=return_denom,
        )

    if _BACKEND_SELECTION == "unknown":
        try:
            out = _backproject_with_backend(
                "triton",
                image,
                ctf,
                noise_spectrum,
                probability=probability,
                image_index=image_index,
                volume_index=volume_index,
                rotation=rotation,
                translation=translation,
                radius=radius,
                volume_numerator=volume_numerator,
                volume_denominator=volume_denominator,
                return_denom=return_denom,
            )
            _BACKEND_SELECTION = "triton"
            _BACKEND_ERROR = None
            return out
        except BackendUnavailableError as e:
            _BACKEND_SELECTION = "torch"
            _BACKEND_ERROR = repr(e)
            _warn_triton_fallback_once(
                f"Triton backproject backend is unavailable; falling back to torch. {e}"
            )

    return _backproject_with_backend(
        "torch",
        image,
        ctf,
        noise_spectrum,
        probability=probability,
        image_index=image_index,
        volume_index=volume_index,
        rotation=rotation,
        translation=translation,
        radius=radius,
        volume_numerator=volume_numerator,
        volume_denominator=volume_denominator,
        return_denom=return_denom,
    )