from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BackendName",
    "backend_status",
    "spectral_mse_loss",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BackendName": ("spectral_mse_loss", "BackendName"),
    "backend_status": ("spectral_mse_loss", "backend_status"),
    "spectral_mse_loss": ("spectral_mse_loss", "spectral_mse_loss"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))