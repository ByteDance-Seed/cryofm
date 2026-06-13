from __future__ import annotations

from typing import Optional, Literal

from torch import Tensor

"""Public spectral_mse_loss API with backend auto-selection.

Most users should call :func:`spectral_mse_loss` from this module instead of calling
backend implementations directly.

This wrapper picks the fastest available backend (currently Triton on CUDA for complex
inputs) and falls back to the Torch reference implementation when the selected backend
is unavailable or fails internally.

Use :func:`backend_status` to inspect which backend is currently selected and whether
there was a backend initialization/runtime error during auto-selection.
"""


def _is_user_error(e: Exception) -> bool:
    return isinstance(e, (ValueError, TypeError, NotImplementedError))

BackendName = Literal["unknown", "cuda", "triton", "torch"]

__all__ = [
    "BackendName",
    "backend_status",
    "spectral_mse_loss",
]

_BACKEND_SELECTION: BackendName = "unknown"
_BACKEND_ERROR: Optional[str] = None


def backend_status() -> tuple[str, Optional[str]]:
    """Return current backend selection state for :func:`spectral_mse_loss`.

    Returns:
        A tuple ``(backend, error)`` where:

        - ``backend`` is one of ``{"unknown", "triton", "torch"}``.
        - ``error`` stores the last backend initialization/runtime error (``repr(e)``)
          when auto-selection failed, otherwise ``None``.
    """

    return _BACKEND_SELECTION, _BACKEND_ERROR


def _spectral_mse_loss_with_backend(
    backend: BackendName,
    input: Tensor,
    target: Tensor,
    weight: Tensor | None,
    input_indices: Tensor | None,
    target_indices: Tensor | None,
    out: Tensor | None,
    prefer_2stage: bool | None,
    reduction: Literal["none", "mean", "sum"],
) -> Tensor:
    if backend == "triton":
        from cryoseed.backends.triton.spectral_mse_loss import spectral_mse_loss as spectral_mse_loss_triton

        return spectral_mse_loss_triton(
            input=input,
            target=target,
            weight=weight,
            input_indices=input_indices,
            target_indices=target_indices,
            out=out,
            prefer_2stage=prefer_2stage,
            reduction=reduction,
        )

    if backend == "torch":
        from cryoseed.backends.torch.spectral_mse_loss import spectral_mse_loss as spectral_mse_loss_torch

        return spectral_mse_loss_torch(
            input=input,
            target=target,
            weight=weight,
            input_indices=input_indices,
            target_indices=target_indices,
            out=out,
            prefer_2stage=prefer_2stage,
            reduction=reduction,
        )

    raise ValueError(f"Unknown backend: {backend}")


def spectral_mse_loss(
    input: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    input_indices: Tensor | None = None,
    target_indices: Tensor | None = None,
    out: Tensor | None = None,
    prefer_2stage: bool | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Compute a weighted sum of squared spectral differences.

    This is the recommended public entry point. It mirrors the API/semantics of the
    Triton backend, while automatically selecting an available backend.

    The frequency-bin dimension ``D`` is always reduced internally via a weighted sum.
    Any additional ``reduction`` is then applied over the per-pair/per-tile loss outputs.

    Two modes are supported:

    - Indexed mode (when indices are provided)::

        loss[i] = sum_d weight[d] * |input[input_indices[i], d] - target[target_indices[i], d]|^2

      Expected input/target shapes: ``(N, ...)``. All feature dimensions after ``N`` are
      flattened into ``D`` internally.

    - Broadcast mode (when indices are omitted)::

        loss[b, ci, co] = sum_d weight[d] * |input[b, ci, d] - target[b, co, d]|^2

      Expected input/target shapes: ``(B, C, ...)``. All feature dimensions after ``C``
      are flattened into ``D`` internally.

      The unreduced output is flattened to ``(B * C_input * C_target,)`` unless a 3D
      ``out`` view of shape ``(B, C_input, C_target)`` is provided.

    Args:
        input: Input tensor. Currently complex tensors are required.
        target: Tensor to compare against. If on a different device, it is moved to
            ``input.device``.
        weight: Optional real weights for the flattened frequency-bin / feature dimension ``D``.

            If ``None``, uses uniform weights (all ones).

            ``weight`` is flattened internally (via ``weight.reshape(-1)``) and must have
            exactly ``D`` elements.

            Note:
                When ``reduction='mean'``, ``weight`` is normalized to sum to 1 *before* the
                spectral reduction. In that case, the per-pair/per-tile output is a weighted
                mean over ``D`` (not a weighted sum). A final mean reduction is then applied
                over the per-pair/per-tile outputs.

                Therefore, in general,
                ``spectral_mse_loss(..., reduction='none').mean()`` is not equal to
                ``spectral_mse_loss(..., reduction='mean')`` unless ``weight`` is already
                normalized.
        input_indices: Optional indices selecting rows from ``input`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        target_indices: Optional indices selecting rows from ``target`` (indexed mode). If on a
            different device, it is moved to ``input.device``.
        out: Optional output tensor for the unreduced per-pair/per-tile loss values.
            This buffer is filled before any final ``reduction`` is applied and must already be
            on ``input.device``.

            Broadcast mode notes:
                ``out`` can be either a flat contiguous 1D tensor of shape
                ``(B * C_input * C_target,)`` or a 3D view of shape ``(B, C_input, C_target)``.

            Note:
                If ``reduction != 'none'`` and ``out`` is provided, ``out`` is still populated
                with the unreduced values, and the return value is the reduced scalar.
        prefer_2stage: Optional override for the 2-stage Triton reduction path. Ignored by the
            Torch backend.
        reduction: Specifies the reduction to apply over the per-pair/per-tile loss outputs:
            ``'none'`` | ``'mean'`` | ``'sum'``.

            - ``'none'``: return the unreduced loss tensor.
            - ``'mean'``: return the mean over all pair/tile losses.
            - ``'sum'``: return the sum over all pair/tile losses.

            Default: ``'mean'``.

    Returns:
        A float32 tensor.

        - ``reduction='none'``: unreduced loss tensor (indexed: ``(N,)``; broadcast: flat
          ``(B * C_input * C_target,)`` unless ``out`` provides a 3D view).
        - ``reduction='mean'`` or ``'sum'``: scalar tensor.

    Backend policy:
        - If a Triton backend is available for the given inputs, it is tried first.
        - If Triton fails with a backend/runtime error, this function falls back to the
          Torch reference implementation and remembers that choice.
        - User input errors (e.g. shape/dtype/device mismatches) are re-raised and do not
          trigger a backend fallback.

    See Also:
        :func:`backend_status`.
    """

    global _BACKEND_SELECTION, _BACKEND_ERROR

    try_triton = bool(input.is_cuda and input.is_complex())

    if _BACKEND_SELECTION in ("triton",) and try_triton:
        try:
            return _spectral_mse_loss_with_backend(
                _BACKEND_SELECTION,
                input,
                target,
                weight,
                input_indices,
                target_indices,
                out,
                prefer_2stage,
                reduction,
            )
        except Exception as e:
            if _is_user_error(e):
                raise
            _BACKEND_SELECTION = "torch"
            _BACKEND_ERROR = repr(e)

    if _BACKEND_SELECTION == "unknown":
        if try_triton:
            for backend in ("triton",):
                try:
                    result = _spectral_mse_loss_with_backend(
                        backend,
                        input,
                        target,
                        weight,
                        input_indices,
                        target_indices,
                        out,
                        prefer_2stage,
                        reduction,
                    )
                    _BACKEND_SELECTION = backend
                    _BACKEND_ERROR = None
                    return result
                except Exception as e:
                    if _is_user_error(e):
                        raise
                    _BACKEND_ERROR = repr(e)

        _BACKEND_SELECTION = "torch"

    return _spectral_mse_loss_with_backend(
        "torch",
        input,
        target,
        weight,
        input_indices,
        target_indices,
        out,
        prefer_2stage,
        reduction,
    )