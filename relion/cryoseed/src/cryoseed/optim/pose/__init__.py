from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "EulerPoseSearcher",
    "HEALPixPoseSearcher",
    "PoseSearcher",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "EulerPoseSearcher": ("euler_searcher", "EulerPoseSearcher"),
    "HEALPixPoseSearcher": ("healpix_searcher", "HEALPixPoseSearcher"),
    "PoseSearcher": ("pose_searcher", "PoseSearcher"),
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