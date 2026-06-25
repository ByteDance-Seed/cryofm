from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cryoseed.config import MainConfig
from cryoseed.utils.torch_utils import _norm_device


class Pose(nn.Module):
    @classmethod
    def from_config(
        cls,
        config: MainConfig,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        requires_grad: bool | None = None,
    ):
        if requires_grad is None:
            requires_grad = bool(config.reconstruction.requires_grad)
        return cls(
            int(config.data.num_particles),
            device=device,
            device_mesh=device_mesh,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        num_particles: int,
        *,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.num_particles = num_particles
        dev = _norm_device(device)
        self.register_buffer("_device_anchor", torch.empty(0, device=dev), persistent=False)
        self.device_mesh = device_mesh

        quat = torch.randn(num_particles, 4, device=dev)
        quat = F.normalize(quat, dim=1)

        self.quat = nn.Parameter(quat, requires_grad=requires_grad)
        self.trans = nn.Parameter(
            torch.zeros(num_particles, 2, device=dev),
            requires_grad=requires_grad,
        )
        self.register_buffer(
            "trans_update_rms",
            torch.zeros(1, dtype=self.trans.dtype, device=dev),
            persistent=False,
        )

        self.register_buffer(
            "valid_count",
            torch.zeros(num_particles, dtype=torch.long, device=dev),
            persistent=True,
        )
        self.register_buffer(
            "quat_accum",
            torch.zeros(num_particles, 4, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "trans_accum",
            torch.zeros(num_particles, 2, dtype=self.trans.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "trans_update_rms_accum",
            torch.zeros(1, dtype=self.trans.dtype, device=dev),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def quaternion(self, index):
        return F.normalize(self.quat[index], dim=-1)

    def translation(self, index):
        return self.trans[index]

    @torch.no_grad()
    def accumulate(
        self,
        index: torch.LongTensor,
        *,
        quaternion: torch.Tensor,
        translation: torch.Tensor,
    ):
        self.valid_count[index] += 1
        self.trans_update_rms_accum += (translation - self.trans[index]).pow(2).sum()

        self.quat_accum[index] = quaternion
        self.trans_accum[index] = translation

    @torch.no_grad()
    def zero_accum(self):
        self.valid_count.zero_()
        self.quat_accum.zero_()
        self.trans_accum.zero_()
        self.trans_update_rms_accum.zero_()

    @torch.no_grad()
    def update(self, all_reduce: bool = True):
        dist = getattr(torch, "distributed", None)
        if all_reduce and dist is not None and dist.is_available() and dist.is_initialized():
            if self.device_mesh is not None:
                group = (
                    self.device_mesh.get_group(0)
                    if hasattr(self.device_mesh, "get_group")
                    else self.device_mesh
                )
            else:
                group = dist.group.WORLD

            data_parallel_size = dist.get_world_size(group=group)
            if data_parallel_size > 1:
                dist.all_reduce(self.valid_count, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(self.quat_accum, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(self.trans_accum, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(self.trans_update_rms_accum, op=dist.ReduceOp.SUM, group=group)

        if (self.valid_count > 1).any():
            raise ValueError("Some particles were updated more than once.")

        mask = self.valid_count > 0
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            self.quat.index_copy_(0, idx, F.normalize(self.quat_accum[idx], dim=-1))
            self.trans.index_copy_(0, idx, self.trans_accum[idx])
            self.trans_update_rms.copy_(
                torch.sqrt(self.trans_update_rms_accum / self.valid_count.sum())
            )