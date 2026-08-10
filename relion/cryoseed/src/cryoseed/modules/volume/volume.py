from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

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

    def backward(self, loss: Tensor) -> None:
        """Backpropagate a loss through the current computation graph."""
        loss.backward()

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update_lr(self) -> None:
        """Advance the representation-owned learning-rate policy."""

    def reset_lr(
        self,
        learning_rate: float | Sequence[float] | None = None,
    ) -> None:
        """Reset the representation-owned learning-rate policy."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError