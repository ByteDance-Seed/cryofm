from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
from torch import Tensor, nn

from cryoseed.config import MainConfig

__all__ = [
    "Volume",
]

TVolume = TypeVar("TVolume", bound="Volume")


class Volume(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls: type[TVolume], config: MainConfig) -> TVolume:
        raise NotImplementedError

    @abstractmethod
    def project(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def backproject(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def zero_accum(self, *, set_to_none: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError