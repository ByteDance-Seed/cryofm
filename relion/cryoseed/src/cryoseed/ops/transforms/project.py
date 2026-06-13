from __future__ import annotations

from typing import Optional, Literal

from torch import Tensor


def _is_user_error(e: Exception) -> bool:
    return isinstance(e, (ValueError, TypeError, NotImplementedError))

BackendName = Literal["unknown", "cuda", "triton", "torch"]

__all__ = [
    "BackendName",
    "backend_status",
    "project",
]

_BACKEND_SELECTION: BackendName = "unknown"
_BACKEND_ERROR: Optional[str] = None


def backend_status() -> tuple[str, Optional[str]]:
    """Return current projection backend selection state.

    Returns:
        A tuple ``(backend, error)`` where:

        - ``backend`` is one of ``{"unknown", "triton", "torch"}``.
        - ``error`` stores the last backend initialization error (``repr(e)``) when
          auto-selection failed, otherwise ``None``.
    """

    return _BACKEND_SELECTION, _BACKEND_ERROR


def _project_with_backend(
    backend: BackendName,
    volume: Tensor,
    rotation: Tensor,
    *,
    channel_last: bool,
) -> Tensor:
    """Dispatch projection to a concrete backend implementation.

    Args:
        backend: Backend name.
        volume:
            - ``channel_last=True``: ``(B, D, H, W, C)``
            - ``channel_last=False``: ``(B, C, D, H, W)``
        rotation: ``(B, Q, 3, 3)``.
        channel_last: Whether ``volume`` uses channel-last layout.

    Returns:
        - ``channel_last=True``: ``(B, Q, H, W, C)``
        - ``channel_last=False``: ``(B, Q, C, H, W)``
    """
    # if backend == "cuda":
    #     from cryoseed.backends.cuda.project import project as project_cuda

    #     return project_cuda(volume, rotation, channel_last=channel_last)
    
    if backend == "triton":
        from cryoseed.backends.triton.project import project as project_triton

        return project_triton(volume, rotation, channel_last=channel_last)

    if backend == "torch":
        from cryoseed.backends.torch.project import project as project_torch

        return project_torch(volume, rotation, channel_last=channel_last)

    raise ValueError(f"Unknown backend: {backend}")


def project(volume: Tensor, rotation: Tensor, *, channel_last: bool = True) -> Tensor:
    """Project a 3D Fourier volume by sampling its z=0 central slice.

    This function auto-selects the fastest available backend (currently Triton on
    CUDA) and falls back to the Torch implementation on error.

    Args:
        volume:
            - ``channel_last=True``: ``(B, D, H, W, C)``
            - ``channel_last=False``: ``(B, C, D, H, W)``

            Typically ``C=2`` stores complex numbers as ``(re, im)``.
        rotation: ``(B, Q, 3, 3)`` rotation matrices.
        channel_last: Whether ``volume`` uses channel-last layout.

    Returns:
        - ``channel_last=True``: ``(B, Q, H, W, C)``
        - ``channel_last=False``: ``(B, Q, C, H, W)``

    See Also:
        :func:`backend_status` to inspect which backend was selected.
    """

    global _BACKEND_SELECTION, _BACKEND_ERROR

    if _BACKEND_SELECTION in ("triton",):
        try:
            return _project_with_backend(
                _BACKEND_SELECTION,
                volume,
                rotation,
                channel_last=channel_last,
            )
        except Exception as e:
            if _is_user_error(e):
                raise
            _BACKEND_SELECTION = "torch"
            _BACKEND_ERROR = repr(e)

    if _BACKEND_SELECTION == "unknown":
        for backend in ("triton",):
            try:
                out = _project_with_backend(
                    backend,
                    volume,
                    rotation,
                    channel_last=channel_last,
                )
                _BACKEND_SELECTION = backend
                _BACKEND_ERROR = None
                return out
            except Exception as e:
                if _is_user_error(e):
                    raise
                _BACKEND_ERROR = repr(e)

        _BACKEND_SELECTION = "torch"

    return _project_with_backend(
        "torch",
        volume,
        rotation,
        channel_last=channel_last,
    )