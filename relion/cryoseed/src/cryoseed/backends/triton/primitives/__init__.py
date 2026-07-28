from __future__ import annotations

import importlib
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "central_slice_embed_batched": ("central_slice_embed", "central_slice_embed_batched"),
    "central_slice_embed_indexed": ("central_slice_embed", "central_slice_embed_indexed"),
    "central_slice_sample": ("central_slice_sample", "central_slice_sample"),
    "central_slice_sample_ncdhw": ("central_slice_sample", "central_slice_sample_ncdhw"),
    "central_slice_sample_ndhwc": ("central_slice_sample", "central_slice_sample_ndhwc"),
    "weighted_sqdiff_group_sum_indexed_complex": ("weighted_sqdiff_group_sum", "weighted_sqdiff_group_sum_indexed_complex"),
    "weighted_sqdiff_sum": ("weighted_sqdiff_sum", "weighted_sqdiff_sum"),
    "weighted_sqdiff_sum_indexed_complex": ("weighted_sqdiff_sum", "weighted_sqdiff_sum_indexed_complex"),
    "weighted_sqdiff_sum_broadcast_complex": ("weighted_sqdiff_sum", "weighted_sqdiff_sum_broadcast_complex"),
}

__all__ = [
    *sorted(_EXPORTS.keys()),
    "kernels",
]


def __getattr__(name: str) -> Any:
    if name == "kernels":
        return importlib.import_module(f"{__name__}.kernels")

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