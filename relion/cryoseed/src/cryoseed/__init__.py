from __future__ import annotations

import importlib
from typing import Any

from ._version import __version__

__all__ = [
    "__version__",
    "backends",
    "config",
    "cryoem",
    "data",
    "fft",
    "metrics",
    "modules",
    "ops",
    "optim",
    "plugins",
    "runtime",
    "state",
    "utils",
]

_SUBMODULES = {name for name in __all__ if name != "__version__"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))