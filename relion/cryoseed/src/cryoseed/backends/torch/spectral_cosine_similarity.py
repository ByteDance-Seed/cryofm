from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = ["spectral_cosine_similarity"]


def spectral_cosine_similarity(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    input_indices: Tensor | None = None,
    target_indices: Tensor | None = None,
    out: Tensor | None = None,
    eps: float = 1e-12,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> Tensor:
    if not torch.is_complex(input) or not torch.is_complex(target):
        raise NotImplementedError(
            "spectral_cosine_similarity requires complex tensors"
        )
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of 'none', 'mean', 'sum'; "
            f"got {reduction!r}"
        )

    device = input.device
    target = target.to(device=device)
    if (input_indices is None) != (target_indices is None):
        raise ValueError(
            "input_indices and target_indices must be both provided or both None"
        )
    if out is not None:
        if out.device != device:
            raise ValueError(f"out must be on {device}, got {out.device}")
        if out.dtype != torch.float32:
            raise ValueError(f"out must be float32, got {out.dtype}")

    indexed = input_indices is not None
    if indexed:
        if input.dim() < 2 or target.dim() < 2:
            raise ValueError("indexed mode expects input/target with shape (N, ...)")
        input_flat = input.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        feature_dim = 1
    else:
        if input.dim() < 3 or target.dim() < 3:
            raise ValueError(
                "broadcast mode expects input/target with shape (B, C, ...)"
            )
        if int(input.shape[0]) != int(target.shape[0]):
            raise ValueError(
                "broadcast mode requires input and target to have the same B"
            )
        input_flat = input.flatten(start_dim=2)
        target_flat = target.flatten(start_dim=2)
        feature_dim = 2

    D = int(input_flat.shape[feature_dim])
    if int(target_flat.shape[feature_dim]) != D:
        raise ValueError(
            "D mismatch after flatten: "
            f"input has {D}, target has {int(target_flat.shape[feature_dim])}"
        )
    if weight is None:
        weight_1d = torch.ones((D,), device=device, dtype=torch.float32)
    else:
        weight_1d = weight.to(device=device, dtype=torch.float32).reshape(-1)
        if int(weight_1d.numel()) != D:
            raise ValueError(
                f"weight must have {D} elements after flatten, "
                f"got {int(weight_1d.numel())}"
            )
    if bool((weight_1d < 0).any()):
        raise ValueError("weight must be non-negative")
    if float(weight_1d.sum().item()) <= 0:
        raise ValueError("weight must have positive sum")

    if indexed:
        assert input_indices is not None and target_indices is not None
        ii = input_indices.reshape(-1).to(device=device, dtype=torch.long)
        ti = target_indices.reshape(-1).to(device=device, dtype=torch.long)
        if int(ii.numel()) != int(ti.numel()):
            raise ValueError("input_indices and target_indices must have equal size")
        a = input_flat.index_select(0, ii)
        b = target_flat.index_select(0, ti)
        cross = (a.conj() * b * weight_1d).sum(dim=-1).real
        a2 = (a.abs().square() * weight_1d).sum(dim=-1)
        b2 = (b.abs().square() * weight_1d).sum(dim=-1)
        corr = cross / torch.sqrt((a2 * b2).clamp_min(float(eps)))
        corr = corr.to(torch.float32)
        if out is not None:
            if out.numel() != corr.numel():
                raise ValueError(
                    f"out.numel() must be {int(corr.numel())}, got {int(out.numel())}"
                )
            out.copy_(corr.reshape(out.shape))
            result = out
        else:
            result = corr
    else:
        B, Ci = int(input_flat.shape[0]), int(input_flat.shape[1])
        Co = int(target_flat.shape[1])
        if out is None:
            out_view = torch.empty((B, Ci, Co), device=device, dtype=torch.float32)
        elif out.dim() == 3 and tuple(out.shape) == (B, Ci, Co):
            out_view = out
        elif out.dim() == 1 and out.numel() == B * Ci * Co and out.is_contiguous():
            out_view = out.view(B, Ci, Co)
        else:
            raise ValueError(
                "broadcast mode out must be (B, C_input, C_target) or "
                "flat contiguous (B*C_input*C_target,)"
            )

        weighted_input_conj = input_flat.conj() * weight_1d
        input_power = (input_flat.abs().square() * weight_1d).sum(dim=-1)
        target_power = (target_flat.abs().square() * weight_1d).sum(dim=-1)
        chunk = 1024
        for start in range(0, Co, chunk):
            end = min(start + chunk, Co)
            target_chunk = target_flat[:, start:end]
            cross = torch.matmul(
                weighted_input_conj, target_chunk.transpose(1, 2)
            ).real
            denom = torch.sqrt(
                (
                    input_power[:, :, None]
                    * target_power[:, None, start:end]
                ).clamp_min(float(eps))
            )
            out_view[:, :, start:end] = cross / denom
        result = out if out is not None else out_view.reshape(-1)

    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    return result.sum()