"""cryoseed.data

Data layer of cryoSeed.

Public API:

- :class:`~cryoseed.data.ParticleDataset`: Reads particle images and metadata from a RELION STAR.
- :class:`~cryoseed.data.DataBatch`: Standard batch protocol consumed by solvers.
- :func:`~cryoseed.data.data_collate_fn`: Collates per-sample dicts into a :class:`~cryoseed.data.DataBatch`.
- DataLoader builders in :mod:`cryoseed.data.dataloader`.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DataBatch",
    "ParticleDataset",
    "data_collate_fn",
    "build_dataset",
    "build_dataloader",
    "build_distributed_dataloader",
    "build_distributed_sampler",
    "single_thread_worker_init_fn",
    "split_dataset_in_half",
    "split_dataset_debug",
    "build_half_dataloaders",
    "build_distributed_half_dataloaders",
    "build_debug_distributed_half_dataloaders",
    "build_half_dataloaders_from_star",
    "build_distributed_half_dataloaders_from_star",
    "build_debug_distributed_half_dataloaders_from_star",
    "save_mrc",
    "read_starfile",
    "parse_optics_parameters",
    "merge_optics_to_particles",
    "parse_stack_entries",
    "save_starfile",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DataBatch": ("batch", "DataBatch"),
    "data_collate_fn": ("collate", "data_collate_fn"),
    "build_dataloader": ("dataloader", "build_dataloader"),
    "build_dataset": ("dataloader", "build_dataset"),
    "build_distributed_dataloader": ("dataloader", "build_distributed_dataloader"),
    "single_thread_worker_init_fn": ("dataloader", "single_thread_worker_init_fn"),
    "build_debug_distributed_half_dataloaders": (
        "dataloader",
        "build_debug_distributed_half_dataloaders",
    ),
    "build_debug_distributed_half_dataloaders_from_star": (
        "dataloader",
        "build_debug_distributed_half_dataloaders_from_star",
    ),
    "build_distributed_half_dataloaders": ("dataloader", "build_distributed_half_dataloaders"),
    "build_distributed_half_dataloaders_from_star": (
        "dataloader",
        "build_distributed_half_dataloaders_from_star",
    ),
    "build_distributed_sampler": ("dataloader", "build_distributed_sampler"),
    "build_half_dataloaders": ("dataloader", "build_half_dataloaders"),
    "build_half_dataloaders_from_star": ("dataloader", "build_half_dataloaders_from_star"),
    "split_dataset_debug": ("dataloader", "split_dataset_debug"),
    "split_dataset_in_half": ("dataloader", "split_dataset_in_half"),
    "ParticleDataset": ("dataset", "ParticleDataset"),
    "save_mrc": ("mrc", "save_mrc"),
    "merge_optics_to_particles": ("star", "merge_optics_to_particles"),
    "parse_optics_parameters": ("star", "parse_optics_parameters"),
    "parse_stack_entries": ("star", "parse_stack_entries"),
    "read_starfile": ("star", "read_starfile"),
    "save_starfile": ("star", "save_starfile"),
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