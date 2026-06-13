from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "errors",
    "registry",
    "spec",
]

_SUBMODULES = set(__all__)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))