from __future__ import annotations

"""DataLoader builder utilities for :mod:`cryoseed.data`.

This module provides small, composable helpers to:

- Build a :class:`~cryoseed.data.ParticleDataset` from a RELION STAR file.
- Split any dataset into two disjoint halves (or small debug subsets).
- Construct standard or distributed :class:`torch.utils.data.DataLoader` objects
  yielding :class:`~cryoseed.data.DataBatch`.

The API is intentionally split into two layers:

- Dataset-based functions, e.g. :func:`build_half_dataloaders`, that accept an
  already-constructed dataset.
- STAR convenience wrappers, e.g. :func:`build_half_dataloaders_from_star`, that
  build the dataset internally.
"""

from typing import Any

import torch

from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset

from .collate import data_collate_fn
from .dataset import ParticleDataset

__all__ = [
    "build_dataset",
    "build_dataloader",
    "build_distributed_dataloader",
    "build_distributed_sampler",
    "split_dataset_in_half",
    "split_dataset_debug",
    "build_half_dataloaders",
    "build_distributed_half_dataloaders",
    "build_debug_distributed_half_dataloaders",
    "build_half_dataloaders_from_star",
    "build_distributed_half_dataloaders_from_star",
    "build_debug_distributed_half_dataloaders_from_star",
]


def build_dataset(
    star_path: str,
    data_prefix: str = "",
    num_particles: int | None = None,
    selection_seed: int = 0,
    image_size: int | None = None,
    angpix: float | None = None,
    default_optic_params: dict | None = None,
    default_particle_params: dict | None = None,
) -> ParticleDataset:
    """Build a :class:`~cryoseed.data.ParticleDataset` from a STAR file.

    Args:
        star_path (str): Path to the STAR file.
        data_prefix (str, optional): Prefix prepended to MRC/MRCS paths parsed
            from the STAR file. Defaults to ``""``.
        selection_seed (int, optional): Seed mixed into deterministic
            pseudo-random subset selection when ``num_particles`` truncates the
            dataset.
        default_optic_params (dict | None, optional): Fallback values for missing
            optics-level CTF fields.
        default_particle_params (dict | None, optional): Fallback values for missing
            particle-level CTF fields.

    Returns:
        ParticleDataset: The constructed dataset.
    """
    return ParticleDataset(
        star_path=star_path,
        data_prefix=data_prefix,
        num_particles=num_particles,
        selection_seed=selection_seed,
        image_size=image_size,
        angpix=angpix,
        default_optic_params=default_optic_params,
        default_particle_params=default_particle_params,
    )


def split_dataset_in_half(
    ds: Dataset,
    *,
    seed: int = 42,
) -> tuple[Subset, Subset, torch.Tensor, torch.Tensor]:
    """Randomly split a dataset into two disjoint halves.

    Args:
        ds: Any indexable dataset.
        seed: RNG seed used to generate the split.

    Returns:
        ``(ds_half0, ds_half1, idx_half0, idx_half1)`` where the subsets are
        :class:`torch.utils.data.Subset` and the indices are 1D ``LongTensor``.
    """
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    half = n // 2
    idx_half0 = perm[:half]
    idx_half1 = perm[half:]

    ds_half0 = Subset(ds, idx_half0)
    ds_half1 = Subset(ds, idx_half1)
    return ds_half0, ds_half1, idx_half0, idx_half1


def split_dataset_debug(
    ds: Dataset,
    *,
    seed: int = 42,
    half_size: int = 100,
) -> tuple[Subset, Subset, torch.Tensor, torch.Tensor]:
    """Build two small debug subsets.

    Args:
        ds: Any indexable dataset.
        seed: RNG seed used to generate the split.
        half_size: Size of each debug subset.

    Returns:
        ``(ds_half0, ds_half1, idx_half0, idx_half1)``.

    Raises:
        ValueError: If the dataset is too small for two debug subsets.
    """
    n = len(ds)
    if 2 * half_size > n:
        raise ValueError(
            f"half_size={half_size} is too large for dataset of size {n}; "
            f"need at least {2 * half_size} samples"
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    idx_half0 = perm[:half_size]
    idx_half1 = perm[half_size : 2 * half_size]

    ds_half0 = Subset(ds, idx_half0)
    ds_half1 = Subset(ds, idx_half1)
    return ds_half0, ds_half1, idx_half0, idx_half1


def build_dataloader(
    ds: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
    sampler: Sampler | None = None,
    drop_last: bool = True,
) -> DataLoader:
    """Build a standard :class:`torch.utils.data.DataLoader`.

    Args:
        ds: Dataset or subset.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes.
        prefetch_factor: Number of batches prefetched per worker.
            Only used when ``num_workers > 0``.
        persistent_workers: Keep workers alive across epochs.
            Only used when ``num_workers > 0``.
        pin_memory: Pin CPU memory for faster host-to-device transfer.
        generator: RNG generator used by the loader.
        sampler: Custom sampler. When provided, ``shuffle`` is disabled.
        drop_last: Drop the last incomplete batch.

    Returns:
        Loader yielding :class:`~cryoseed.data.DataBatch`.
    """
    if sampler is not None:
        shuffle = False

    kwargs: dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers if num_workers > 0 else False),
        drop_last=drop_last,
        collate_fn=data_collate_fn,
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    if generator is not None:
        kwargs["generator"] = generator

    return DataLoader(**kwargs)


def build_distributed_dataloader(
    ds: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    drop_last: bool = True,
    device_mesh=None,
) -> tuple[DataLoader, DistributedSampler]:
    """Build a distributed dataloader and its sampler for a dataset.

    Args:
        ds: Dataset or subset.
        batch_size: Batch size.
        shuffle: Whether the sampler shuffles indices.
        num_workers: Number of worker processes.
        prefetch_factor: Prefetch factor when ``num_workers > 0``.
        persistent_workers: Keep workers alive when ``num_workers > 0``.
        device: If CUDA, enables ``pin_memory``.
        seed: RNG seed used by the distributed sampler.
        drop_last: Whether to drop the last incomplete batch.
        device_mesh: Device mesh used to infer rank and world size.

    Returns:
        ``(dataloader, sampler)``.
    """
    sampler = build_distributed_sampler(
        ds,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )
    pin_memory = device is not None and device.type == "cuda"
    dataloader = build_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=drop_last,
    )
    return dataloader, sampler


def _get_rank_world_size(device_mesh) -> tuple[int, int]:
    """Infer ``(rank, world_size)`` from ``device_mesh.get_group(0)``."""
    group = device_mesh.get_group(0)
    return group.rank(), group.size()


def build_distributed_sampler(
    ds: Dataset,
    *,
    shuffle: bool = False,
    seed: int = 42,
    drop_last: bool = True,
    device_mesh=None,
) -> DistributedSampler:
    """Build a :class:`torch.utils.data.DistributedSampler` for a dataset.

    Args:
        ds: Dataset or subset.
        shuffle: Whether the sampler shuffles indices each epoch.
        seed: RNG seed used by the sampler.
        drop_last: Whether to drop the tail so every rank sees the same number
            of samples.
        device_mesh: Device mesh used to infer rank and world size.

    Returns:
        The constructed distributed sampler.
    """
    if device_mesh is None:
        raise ValueError("device_mesh must be provided for distributed sampling")

    rank, world_size = _get_rank_world_size(device_mesh)

    return DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def build_half_dataloaders(
    ds: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Build two dataloaders from two random halves of a dataset.

    Args:
        ds: Dataset to split.
        batch_size: Batch size.
        shuffle: Whether to shuffle each subset.
        num_workers: Number of worker processes.
        prefetch_factor: Prefetch factor when ``num_workers > 0``.
        persistent_workers: Keep workers alive when ``num_workers > 0``.
        device: If CUDA, enables ``pin_memory``.
        seed: RNG seed used for splitting and shuffling.
        drop_last: Drop the last incomplete batch.

    Returns:
        ``(dl_half0, dl_half1, idx_half0, idx_half1)``.
    """
    ds_half0, ds_half1, idx_half0, idx_half1 = split_dataset_in_half(ds, seed=seed)

    pin_memory = device is not None and device.type == "cuda"
    g_loader = torch.Generator().manual_seed(seed) if shuffle else None

    dl_half0 = build_dataloader(
        ds_half0,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        generator=g_loader,
        drop_last=drop_last,
    )
    dl_half1 = build_dataloader(
        ds_half1,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        generator=g_loader,
        drop_last=drop_last,
    )

    return dl_half0, dl_half1, idx_half0, idx_half1


def build_distributed_half_dataloaders(
    ds: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    drop_last: bool = True,
    device_mesh=None,
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Build two distributed dataloaders from two random halves of a dataset.

    This utility constructs per-half :class:`torch.utils.data.DistributedSampler`
    and returns both the loaders and samplers.

    Args:
        ds: Dataset to split.
        batch_size: Batch size.
        shuffle: Whether samplers shuffle indices.
        num_workers: Number of worker processes.
        prefetch_factor: Prefetch factor when ``num_workers > 0``.
        persistent_workers: Keep workers alive when ``num_workers > 0``.
        device: If CUDA, enables ``pin_memory``.
        seed: RNG seed used for splitting and sampler shuffling.
        drop_last: Whether to drop the last incomplete batch.
        device_mesh: Device mesh used to infer rank and world size.

    Returns:
        ``(dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1)``.
    """
    ds_half0, ds_half1, idx_half0, idx_half1 = split_dataset_in_half(ds, seed=seed)

    sampler_half0 = build_distributed_sampler(
        ds_half0,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )
    sampler_half1 = build_distributed_sampler(
        ds_half1,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )

    pin_memory = device is not None and device.type == "cuda"

    dl_half0 = build_dataloader(
        ds_half0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler_half0,
        drop_last=drop_last,
    )
    dl_half1 = build_dataloader(
        ds_half1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler_half1,
        drop_last=drop_last,
    )

    return dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1


def build_debug_distributed_half_dataloaders(
    ds: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    device_mesh=None,
    half_size: int = 100,
    drop_last: bool = True,
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Build two small distributed dataloaders for debugging.

    Args:
        ds: Dataset to split.
        batch_size: Batch size.
        shuffle: Whether samplers shuffle indices.
        num_workers: Number of worker processes.
        prefetch_factor: Prefetch factor when ``num_workers > 0``.
        persistent_workers: Keep workers alive when ``num_workers > 0``.
        device: If CUDA, enables ``pin_memory``.
        seed: RNG seed used for splitting and sampler shuffling.
        device_mesh: Device mesh used to infer rank and world size.
        half_size: Size of each debug subset.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        ``(dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1)``.
    """
    ds_half0, ds_half1, idx_half0, idx_half1 = split_dataset_debug(
        ds,
        seed=seed,
        half_size=half_size,
    )

    sampler_half0 = build_distributed_sampler(
        ds_half0,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )
    sampler_half1 = build_distributed_sampler(
        ds_half1,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )

    pin_memory = device is not None and device.type == "cuda"

    dl_half0 = build_dataloader(
        ds_half0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler_half0,
        drop_last=drop_last,
    )
    dl_half1 = build_dataloader(
        ds_half1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler_half1,
        drop_last=drop_last,
    )

    return dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1


def build_half_dataloaders_from_star(
    star_path: str,
    *,
    data_prefix: str = "",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    drop_last: bool = True,
    default_optic_params: dict | None = None,
    default_particle_params: dict | None = None,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Build two dataloaders from two random halves of a STAR-defined dataset.

    Args:
        star_path (str): Path to the STAR file.
        data_prefix (str, optional): Prefix prepended to MRC/MRCS paths.
            Defaults to ``""``.
        batch_size (int, optional): Batch size. Defaults to ``32``.
        shuffle (bool, optional): Whether to shuffle each subset. Defaults to ``False``.
        num_workers (int, optional): Number of worker processes. Defaults to ``4``.
        prefetch_factor (int, optional): Prefetch factor (when ``num_workers > 0``).
            Defaults to ``2``.
        persistent_workers (bool, optional): Keep workers alive (when ``num_workers > 0``).
            Defaults to ``True``.
        device (torch.device | None, optional): If CUDA, enables ``pin_memory``.
        seed (int, optional): RNG seed used for splitting and shuffling. Defaults to ``42``.
        drop_last (bool, optional): Drop the last incomplete batch. Defaults to ``True``.
        default_optic_params (dict | None, optional): Fallback optics parameters.
        default_particle_params (dict | None, optional): Fallback particle parameters.

    Returns:
        tuple: ``(dl_half0, dl_half1, idx_half0, idx_half1)``.
    """
    ds = build_dataset(
        star_path=star_path,
        data_prefix=data_prefix,
        default_optic_params=default_optic_params,
        default_particle_params=default_particle_params,
    )
    return build_half_dataloaders(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        device=device,
        seed=seed,
        drop_last=drop_last,
    )


def build_distributed_half_dataloaders_from_star(
    star_path: str,
    *,
    data_prefix: str = "",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    drop_last: bool = True,
    device_mesh=None,
    default_optic_params: dict | None = None,
    default_particle_params: dict | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Build two distributed dataloaders from two random halves of a STAR dataset.

    Args:
        star_path (str): Path to the STAR file.
        data_prefix (str, optional): Prefix prepended to MRC/MRCS paths.
            Defaults to ``""``.
        batch_size (int, optional): Batch size. Defaults to ``32``.
        shuffle (bool, optional): Whether samplers shuffle indices. Defaults to ``False``.
        num_workers (int, optional): Number of worker processes. Defaults to ``4``.
        prefetch_factor (int, optional): Prefetch factor (when ``num_workers > 0``).
            Defaults to ``2``.
        persistent_workers (bool, optional): Keep workers alive (when ``num_workers > 0``).
            Defaults to ``True``.
        device (torch.device | None, optional): If CUDA, enables ``pin_memory``.
        seed (int, optional): RNG seed used for splitting and sampler shuffling.
            Defaults to ``42``.
        drop_last (bool, optional): Drop the last incomplete batch. Defaults to ``True``.
        device_mesh: Device mesh used to infer rank and world size.
        default_optic_params (dict | None, optional): Fallback optics parameters.
        default_particle_params (dict | None, optional): Fallback particle parameters.

    Returns:
        tuple: ``(dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1)``.

    Raises:
        ValueError: If ``device_mesh`` is not provided.
    """
    ds = build_dataset(
        star_path=star_path,
        data_prefix=data_prefix,
        default_optic_params=default_optic_params,
        default_particle_params=default_particle_params,
    )
    return build_distributed_half_dataloaders(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        device=device,
        seed=seed,
        drop_last=drop_last,
        device_mesh=device_mesh,
    )


def build_debug_distributed_half_dataloaders_from_star(
    star_path: str,
    *,
    data_prefix: str = "",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    device_mesh=None,
    half_size: int = 100,
    drop_last: bool = True,
    default_optic_params: dict | None = None,
    default_particle_params: dict | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
    torch.Tensor,
    torch.Tensor,
]:
    """Build two small distributed dataloaders from a STAR-defined dataset.

    Args:
        star_path (str): Path to the STAR file.
        data_prefix (str, optional): Prefix prepended to MRC/MRCS paths.
            Defaults to ``""``.
        batch_size (int, optional): Batch size. Defaults to ``32``.
        shuffle (bool, optional): Whether samplers shuffle indices. Defaults to ``False``.
        num_workers (int, optional): Number of worker processes. Defaults to ``4``.
        prefetch_factor (int, optional): Prefetch factor (when ``num_workers > 0``).
            Defaults to ``2``.
        persistent_workers (bool, optional): Keep workers alive (when ``num_workers > 0``).
            Defaults to ``True``.
        device (torch.device | None, optional): If CUDA, enables ``pin_memory``.
        seed (int, optional): RNG seed used for splitting and sampler shuffling.
            Defaults to ``42``.
        device_mesh: Device mesh used to infer rank and world size.
        half_size (int, optional): Size of each debug subset. Defaults to ``100``.
        drop_last (bool, optional): Drop the last incomplete batch. Defaults to ``True``.
        default_optic_params (dict | None, optional): Fallback optics parameters.
        default_particle_params (dict | None, optional): Fallback particle parameters.

    Returns:
        tuple: ``(dl_half0, dl_half1, sampler_half0, sampler_half1, idx_half0, idx_half1)``.

    Raises:
        ValueError: If ``device_mesh`` is not provided.
        ValueError: If ``half_size`` is too large for the dataset.
    """
    ds = build_dataset(
        star_path=star_path,
        data_prefix=data_prefix,
        default_optic_params=default_optic_params,
        default_particle_params=default_particle_params,
    )
    return build_debug_distributed_half_dataloaders(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        device=device,
        seed=seed,
        device_mesh=device_mesh,
        half_size=half_size,
        drop_last=drop_last,
    )