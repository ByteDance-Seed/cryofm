from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "clear_radial_cache",
    "clear_radial_average_cache",
    "radial_average",
    "radial_broadcast",
    "radial_residual_power",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "clear_radial_cache": ("radial", "clear_radial_cache"),
    "clear_radial_average_cache": ("radial", "clear_radial_average_cache"),
    "radial_average": ("radial", "radial_average"),
    "radial_broadcast": ("radial", "radial_broadcast"),
    "radial_residual_power": ("radial_residual_power", "radial_residual_power"),
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