from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "norm_device",
    "norm_dtype",
    "_norm_device",
    "_norm_dtype",
]


def _norm_device(device: torch.device | str | torch.Tensor | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.Tensor):
        device = device.device
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _norm_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    return dtype


def norm_device(device: torch.device | str | torch.Tensor | None) -> torch.device:
    return _norm_device(device)


def norm_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    return _norm_dtype(dtype)