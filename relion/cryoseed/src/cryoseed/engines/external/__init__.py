from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "protocol",
    "ExternalReconstructJob",
    "ExternalReconstructManager",
    "ExternalReconstructResult",
    "ExternalReconstructLayout",
    "build_external_reconstruct_job",
    "build_external_reconstruct_layout",
    "write_external_reconstruct_metadata",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ExternalReconstructJob": ("manager", "ExternalReconstructJob"),
    "ExternalReconstructManager": ("manager", "ExternalReconstructManager"),
    "ExternalReconstructResult": ("manager", "ExternalReconstructResult"),
    "ExternalReconstructLayout": ("protocol", "ExternalReconstructLayout"),
    "build_external_reconstruct_job": ("protocol", "build_external_reconstruct_job"),
    "build_external_reconstruct_layout": ("protocol", "build_external_reconstruct_layout"),
    "write_external_reconstruct_metadata": ("protocol", "write_external_reconstruct_metadata"),
}


def __getattr__(name: str) -> Any:
    if name == "protocol":
        module = importlib.import_module(f"{__name__}.protocol")
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