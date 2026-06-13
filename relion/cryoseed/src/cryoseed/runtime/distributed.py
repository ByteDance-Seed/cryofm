# cryoseed/runtime/distributed.py
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from cryoseed.utils.torch_utils import _norm_device


class _FallbackDeviceMesh:
    def __init__(self, mesh_shape: tuple[int, int], *, group0, group1):
        self.mesh_shape = tuple(int(x) for x in mesh_shape)
        self._groups = {0: group0, 1: group1}

    def get_group(self, dim: int):
        return self._groups[int(dim)]


def _infer_parallel_sizes(
    *,
    world_size: int,
    data_parallel_size: int | None,
    compute_parallel_size: int | None,
) -> tuple[int, int]:
    if data_parallel_size is None and compute_parallel_size is None:
        data_parallel_size = int(world_size)
        compute_parallel_size = 1

    elif data_parallel_size is None:
        if compute_parallel_size is None:
            raise ValueError("compute_parallel_size must be provided when data_parallel_size is None")
        if world_size % compute_parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by compute_parallel_size ({compute_parallel_size})"
            )
        data_parallel_size = world_size // compute_parallel_size

    elif compute_parallel_size is None:
        if world_size % data_parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by data_parallel_size ({data_parallel_size})"
            )
        compute_parallel_size = world_size // data_parallel_size

    data_parallel_size = int(data_parallel_size)
    compute_parallel_size = int(compute_parallel_size)

    if data_parallel_size <= 0 or compute_parallel_size <= 0:
        raise ValueError("parallel sizes must be positive integers")

    if data_parallel_size * compute_parallel_size != world_size:
        raise ValueError(
            "Invalid mesh: data_parallel_size * compute_parallel_size must equal world_size, "
            f"got {data_parallel_size} * {compute_parallel_size} != {world_size}"
        )

    return data_parallel_size, compute_parallel_size


def _build_fallback_device_mesh(
    *,
    data_parallel_size: int,
    compute_parallel_size: int,
    rank: int,
):
    dp = int(data_parallel_size)
    cp = int(compute_parallel_size)
    r = int(rank)

    # Mesh layout assumption (row-major): rank = dp_index * cp + cp_index
    # - get_group(0): same cp_index, varying dp_index, size = dp
    # - get_group(1): same dp_index, varying cp_index, size = cp
    col_this = r % cp
    row_this = r // cp

    group0 = None
    group1 = None

    for col in range(cp):
        ranks = [i * cp + col for i in range(dp)]
        g = dist.new_group(ranks=ranks)
        if col == col_this:
            group0 = g

    for row in range(dp):
        ranks = [row * cp + j for j in range(cp)]
        g = dist.new_group(ranks=ranks)
        if row == row_this:
            group1 = g

    if group0 is None or group1 is None:
        raise RuntimeError("Failed to construct fallback device mesh groups")

    return _FallbackDeviceMesh((dp, cp), group0=group0, group1=group1)


@dataclass
class RuntimeContext:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    device_mesh: object | None


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def is_rank0() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_runtime(
    data_parallel_size: int | None = None,
    compute_parallel_size: int | None = None,
    *,
    local_rank: int | None = None,
) -> RuntimeContext:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = _norm_device("cuda" if torch.cuda.is_available() else "cpu")
        return RuntimeContext(
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
            device_mesh=None,
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    use_nccl = torch.cuda.is_available() and dist.is_nccl_available()

    if use_nccl:
        if local_rank is None:
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
            else:
                ndev = torch.cuda.device_count()
                if ndev <= 0:
                    raise RuntimeError("No CUDA devices found")
                local_rank = rank % ndev

        torch.cuda.set_device(int(local_rank))
        device = _norm_device(f"cuda:{int(local_rank)}")
        backend = "nccl"
    else:
        local_rank = 0 if local_rank is None else int(local_rank)
        device = _norm_device("cpu")
        backend = "gloo"

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=300),
        )

    data_parallel_size, compute_parallel_size = _infer_parallel_sizes(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        compute_parallel_size=compute_parallel_size,
    )

    device_mesh = None
    try:
        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh(
            device.type,
            mesh_shape=(data_parallel_size, compute_parallel_size),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError) as e:
        warnings.warn(
            f"init_device_mesh failed ({type(e).__name__}: {e}); falling back to manual process groups",
            RuntimeWarning,
        )
        device_mesh = None

    if device_mesh is None:
        device_mesh = _build_fallback_device_mesh(
            data_parallel_size=data_parallel_size,
            compute_parallel_size=compute_parallel_size,
            rank=rank,
        )

    return RuntimeContext(
        is_distributed=True,
        rank=rank,
        world_size=world_size,
        local_rank=int(local_rank),
        device=device,
        device_mesh=device_mesh,
    )


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()