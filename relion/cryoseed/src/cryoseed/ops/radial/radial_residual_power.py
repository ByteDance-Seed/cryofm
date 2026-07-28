"""Public radial_residual_power API with backend auto-selection.

Most users should call :func:`radial_residual_power` from this module instead of
calling backend implementations directly.

This wrapper resolves shared radial metadata from the cache in
``cryoseed.ops.radial.radial`` and dispatches to the fastest available backend
(currently Triton on CUDA for supported inputs), falling back to the Torch
reference implementation when needed.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor
import torch

from .radial import _get_radial_cache

__all__ = [
    "radial_residual_power",
]

_BACKEND_SELECTION: str = "unknown"
_BACKEND_ERROR: Optional[str] = None


def _is_user_error(e: Exception) -> bool:
    return isinstance(e, (ValueError, TypeError, NotImplementedError))


def _radial_residual_power_with_backend(
    backend: str,
    input: Tensor,
    target: Tensor,
    *,
    radial_indices: Tensor,
    radial_weight: Tensor,
    num_radial_bins: int,
    input_indices: Tensor,
    target_indices: Tensor,
    out: Tensor | None,
    prefer_2stage: bool | None,
) -> Tensor:
    if backend == "triton":
        from cryoseed.backends.triton.radial_residual_power import radial_residual_power as radial_residual_power_triton

        return radial_residual_power_triton(
            input=input,
            target=target,
            radial_indices=radial_indices,
            radial_weight=radial_weight,
            num_radial_bins=num_radial_bins,
            input_indices=input_indices,
            target_indices=target_indices,
            out=out,
            prefer_2stage=prefer_2stage,
        )

    if backend == "torch":
        from cryoseed.backends.torch.radial_residual_power import radial_residual_power as radial_residual_power_torch

        return radial_residual_power_torch(
            input=input,
            target=target,
            radial_indices=radial_indices,
            radial_weight=radial_weight,
            num_radial_bins=num_radial_bins,
            input_indices=input_indices,
            target_indices=target_indices,
            out=out,
        )

    raise ValueError(f"Unknown backend: {backend}")


def _prepare_flat_valid_input(
    input: Tensor,
    *,
    valid_indices: Tensor,
    num_points: int,
    num_valid_points: int,
) -> Tensor:
    input_flat = input.flatten(start_dim=1)
    D = int(input_flat.shape[1])
    if D == num_points:
        return input_flat.index_select(1, valid_indices)
    if D == num_valid_points:
        return input_flat
    raise ValueError(
        f"Flattened feature size must match either full radial grid ({num_points}) "
        f"or valid radial points ({num_valid_points}), got {D}"
    )


def radial_residual_power(
    input: Tensor,
    target: Tensor,
    *,
    input_indices: Tensor,
    target_indices: Tensor,
    side_length: int,
    max_radius: int,
    ndim: int = 2,
    use_cache: bool = False,
    out: Tensor | None = None,
    prefer_2stage: bool | None = None,
) -> Tensor:
    """Compute pairwise radial residual power profiles.

    This is the recommended public entry point. It accepts either full spatial /
    Fourier grids or tensors that are already flattened to the valid radial
    points defined by ``side_length`` and ``max_radius``.

    For each selected pair ``i``, this function computes::

        out[i, r] = sum_{d: radial_indices[d] == r}
            radial_weight[d]
            * |input[input_indices[i], d] - target[target_indices[i], d]|^2

    where ``radial_weight`` is derived from cached radial metadata so that each
    bin stores the mean residual power over the valid points assigned to that
    radius.

    Args:
        input: Complex input tensor. The leading dimension indexes candidate
            rows. Remaining dimensions are flattened internally.
        target: Complex tensor to compare against. If on a different device, it
            is moved to ``input.device``.
        input_indices: Indices selecting rows from ``input``.
        target_indices: Indices selecting rows from ``target``. Must have the
            same number of elements as ``input_indices``.
        side_length: Side length of the underlying square grid used to derive
            radial metadata.
        max_radius: Maximum radial bin to include.
        ndim: Radial indexing dimensionality. Default: ``2``.
        use_cache: If ``True``, reuse cached radial metadata for the given
            device and geometry. Default: ``False``.
        out: Optional float32 output tensor of shape ``(N, R)`` where ``N`` is
            the number of selected pairs and ``R`` is the number of radial bins.
            If provided, it is treated as a write buffer and must already be on
            ``input.device`` with ``requires_grad=False``.
        prefer_2stage: Optional Triton backend hint kept for API consistency.
            Ignored by the Torch backend.

    Returns:
        A float32 tensor of shape ``(N, R)`` containing per-pair radial residual
        power profiles.

    Backend policy:
        - If a Triton backend is available for the given inputs, it is tried
          first on CUDA.
        - If Triton fails with a backend/runtime error, this function falls
          back to the Torch reference implementation and remembers that choice.
        - User input errors are re-raised and do not trigger backend fallback.
    """
    global _BACKEND_SELECTION, _BACKEND_ERROR

    if (input_indices is None) != (target_indices is None):
        raise ValueError("input_indices and target_indices must be both provided or both None")
    if input_indices is None or target_indices is None:
        raise ValueError("radial_residual_power currently requires indexed inputs")
    if not torch.is_complex(input) or not torch.is_complex(target):
        raise NotImplementedError("radial_residual_power currently requires complex tensors")

    device = input.device
    if target.device != device:
        target = target.to(device=device)
    if input_indices.device != device:
        input_indices = input_indices.to(device=device)
    if target_indices.device != device:
        target_indices = target_indices.to(device=device)

    cache = _get_radial_cache(
        device,
        int(side_length),
        int(max_radius),
        int(ndim),
        use_cache=use_cache,
    )
    valid_indices = cache["valid_indices"]
    radial_indices = cache["radial_indices"]
    radial_weight = cache["radial_weight"]
    num_radial_bins = int(cache["num_radial_bins"])
    num_points = int(cache["num_points"])
    num_valid_points = int(radial_indices.numel())

    input_flat = _prepare_flat_valid_input(
        input,
        valid_indices=valid_indices,
        num_points=num_points,
        num_valid_points=num_valid_points,
    )
    target_flat = _prepare_flat_valid_input(
        target,
        valid_indices=valid_indices,
        num_points=num_points,
        num_valid_points=num_valid_points,
    )

    try_triton = bool(device.type == "cuda")

    if _BACKEND_SELECTION == "triton" and try_triton:
        try:
            return _radial_residual_power_with_backend(
                "triton",
                input_flat,
                target_flat,
                radial_indices=radial_indices,
                radial_weight=radial_weight,
                num_radial_bins=num_radial_bins,
                input_indices=input_indices,
                target_indices=target_indices,
                out=out,
                prefer_2stage=prefer_2stage,
            )
        except Exception as e:
            if _is_user_error(e):
                raise
            _BACKEND_SELECTION = "torch"
            _BACKEND_ERROR = repr(e)

    if _BACKEND_SELECTION == "unknown":
        if try_triton:
            try:
                result = _radial_residual_power_with_backend(
                    "triton",
                    input_flat,
                    target_flat,
                    radial_indices=radial_indices,
                    radial_weight=radial_weight,
                    num_radial_bins=num_radial_bins,
                    input_indices=input_indices,
                    target_indices=target_indices,
                    out=out,
                    prefer_2stage=prefer_2stage,
                )
                _BACKEND_SELECTION = "triton"
                _BACKEND_ERROR = None
                return result
            except Exception as e:
                if _is_user_error(e):
                    raise
                _BACKEND_ERROR = repr(e)
        _BACKEND_SELECTION = "torch"

    return _radial_residual_power_with_backend(
        "torch",
        input_flat,
        target_flat,
        radial_indices=radial_indices,
        radial_weight=radial_weight,
        num_radial_bins=num_radial_bins,
        input_indices=input_indices,
        target_indices=target_indices,
        out=out,
        prefer_2stage=prefer_2stage,
    )