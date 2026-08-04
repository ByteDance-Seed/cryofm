from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AbInitioEngine",
    "HeteroRefineEngine",
    "HomoRefineEngine",
    "abinitio",
    "external",
    "heterorefine",
    "homorefine",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AbInitioEngine": ("abinitio", "AbInitioEngine"),
    "HeteroRefineEngine": ("heterorefine", "HeteroRefineEngine"),
    "HomoRefineEngine": ("homorefine", "HomoRefineEngine"),
}


def __getattr__(name: str) -> Any:
    if name in {"abinitio", "external", "heterorefine", "homorefine"}:
        target_module = "homorefine" if name == "external" else name
        module = importlib.import_module(f"{__name__}.{target_module}")
        globals()[name] = module
        return module

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