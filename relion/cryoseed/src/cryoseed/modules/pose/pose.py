from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cryoseed.config import MainConfig
from cryoseed.utils.torch_utils import _norm_device


# Particle-state code follows rot -> trans -> vol. This intentionally differs
# from search-space order because Pose stores the current state of each
# particle, not the traversal order of search candidates.
class Pose(nn.Module):
    @classmethod
    def from_config(
        cls,
        config: MainConfig,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        requires_grad: bool = False,
    ):
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
        self.requires_grad = bool(requires_grad)

        quat = torch.randn(num_particles, 4, device=dev)
        quat = F.normalize(quat, dim=1)

        self.quat = nn.Parameter(quat, requires_grad=requires_grad)
        self.trans = nn.Parameter(
            torch.zeros(num_particles, 2, device=dev),
            requires_grad=requires_grad,
        )
        self.register_buffer(
            "volume_index",
            torch.zeros(num_particles, dtype=torch.long, device=dev),
            persistent=True,
        )
        self.register_buffer(
            "confidence",
            torch.zeros(num_particles, dtype=self.quat.dtype, device=dev),
            persistent=True,
        )
        self.register_buffer(
            "volume_class_confidence",
            torch.zeros(num_particles, dtype=self.quat.dtype, device=dev),
            persistent=True,
        )
        self.register_buffer(
            "rot_update_rms",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "trans_update_rms",
            torch.zeros(1, dtype=self.trans.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "volume_class_change_rate",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "avg_confidence",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "avg_volume_class_confidence",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
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
            "volume_index_accum",
            torch.zeros(num_particles, dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "confidence_accum",
            torch.zeros(num_particles, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "volume_class_confidence_accum",
            torch.zeros(num_particles, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "rot_update_rms_accum",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "trans_update_rms_accum",
            torch.zeros(1, dtype=self.trans.dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "assignment_change_accum",
            torch.zeros(1, dtype=self.quat.dtype, device=dev),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def requires_grad_(self, requires_grad: bool = True) -> Pose:
        """Set autograd participation for all pose Parameters."""
        self.requires_grad = bool(requires_grad)
        self.quat.requires_grad_(self.requires_grad)
        self.trans.requires_grad_(self.requires_grad)
        return self

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
        volume_index: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        volume_class_confidence: torch.Tensor | None = None,
    ):
        self.valid_count[index] += 1
        current_quat = F.normalize(self.quat[index], dim=-1)
        updated_quat = F.normalize(quaternion, dim=-1)
        quat_dot = (current_quat * updated_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
        rot_update = 2.0 * torch.acos(quat_dot)
        self.rot_update_rms_accum += rot_update.pow(2).sum()
        self.trans_update_rms_accum += (translation - self.trans[index]).pow(2).sum()

        self.quat_accum[index] = quaternion
        self.trans_accum[index] = translation
        if volume_index is not None:
            self.volume_index_accum[index] = volume_index.to(
                device=self.device,
                dtype=self.volume_index_accum.dtype,
            )
        if confidence is not None:
            self.confidence_accum[index] = confidence.to(
                device=self.device,
                dtype=self.confidence_accum.dtype,
            )
        if volume_class_confidence is not None:
            self.volume_class_confidence_accum[index] = volume_class_confidence.to(
                device=self.device,
                dtype=self.volume_class_confidence_accum.dtype,
            )

    @torch.no_grad()
    def zero_accum(self):
        self.valid_count.zero_()
        self.quat_accum.zero_()
        self.trans_accum.zero_()
        self.volume_index_accum.zero_()
        self.confidence_accum.zero_()
        self.volume_class_confidence_accum.zero_()
        self.rot_update_rms_accum.zero_()
        self.trans_update_rms_accum.zero_()
        self.assignment_change_accum.zero_()

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
                dist.all_reduce(self.volume_index_accum, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(self.confidence_accum, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(
                    self.volume_class_confidence_accum, op=dist.ReduceOp.SUM, group=group
                )
                dist.all_reduce(self.rot_update_rms_accum, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(self.trans_update_rms_accum, op=dist.ReduceOp.SUM, group=group)

        if (self.valid_count > 1).any():
            raise ValueError("Some particles were updated more than once.")

        mask = self.valid_count > 0
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            valid_total = self.valid_count.sum()
            valid_total_float = valid_total.to(dtype=self.confidence_accum.dtype)
            assignment_changed = (
                self.volume_index[idx] != self.volume_index_accum[idx]
            ).sum()
            self.assignment_change_accum.copy_(
                assignment_changed.to(
                    dtype=self.assignment_change_accum.dtype,
                    device=self.assignment_change_accum.device,
                )
            )
            self.quat.index_copy_(0, idx, F.normalize(self.quat_accum[idx], dim=-1))
            self.trans.index_copy_(0, idx, self.trans_accum[idx])
            self.volume_index.index_copy_(0, idx, self.volume_index_accum[idx])
            self.confidence.index_copy_(0, idx, self.confidence_accum[idx])
            self.volume_class_confidence.index_copy_(
                0, idx, self.volume_class_confidence_accum[idx]
            )
            self.avg_confidence.copy_(
                self.confidence_accum[idx].sum().reshape_as(self.avg_confidence)
                / valid_total_float
            )
            self.avg_volume_class_confidence.copy_(
                self.volume_class_confidence_accum[idx]
                .sum()
                .reshape_as(self.avg_volume_class_confidence)
                / valid_total_float
            )
            self.volume_class_change_rate.copy_(
                self.assignment_change_accum
                / valid_total.to(dtype=self.assignment_change_accum.dtype)
            )
            self.rot_update_rms.copy_(
                torch.sqrt(
                    self.rot_update_rms_accum
                    / valid_total.to(dtype=self.rot_update_rms_accum.dtype)
                )
            )
            self.trans_update_rms.copy_(
                torch.sqrt(
                    self.trans_update_rms_accum
                    / valid_total.to(dtype=self.trans_update_rms_accum.dtype)
                )
            )
        else:
            self.avg_confidence.zero_()
            self.avg_volume_class_confidence.zero_()