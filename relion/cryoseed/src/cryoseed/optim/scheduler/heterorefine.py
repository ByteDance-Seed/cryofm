from __future__ import annotations

import torch

from cryoseed.config import MainConfig
from cryoseed.state import OptimState


class HeteroRefineScheduler:
    """Update only the Fourier side length from per-class DVP curves."""

    def __init__(
        self,
        state: OptimState,
        *,
        image_size: int,
        angpix: float,
        increase_radius_step: int = 10,
        dvp_threshold: float = 1.0,
    ):
        self.state = state
        self.image_size = int(image_size)
        self.angpix = float(angpix)
        self.increase_radius_step = int(increase_radius_step)
        self.dvp_threshold = float(dvp_threshold)

    @classmethod
    def from_config(cls, state: OptimState, config: MainConfig):
        scheduler = cls(
            state,
            image_size=int(config.data.image_size),
            angpix=float(config.data.angpix),
            increase_radius_step=int(
                config.heterorefine.scheduler.increase_radius_step
            ),
            dvp_threshold=float(config.heterorefine.scheduler.dvp_threshold),
        )
        scheduler._resolve_pose_translation_center()
        return scheduler

    def _resolve_pose_translation_center(self) -> None:
        self.state.schedule.use_pose_translation_as_center = (
            self.state.schedule.pose_translation_center_mode != "never"
        )

    def _crossing_radii(
        self,
        dvp_scores: torch.Tensor,
        weight_spectrum: torch.Tensor,
    ) -> tuple[list[int], list[bool]]:
        if dvp_scores.ndim != 2 or weight_spectrum.shape != dvp_scores.shape:
            raise ValueError("dvp_scores and weight_spectrum must both have shape (K,R)")

        radii: list[int] = []
        valid_classes: list[bool] = []
        current_radius = min(
            int(self.state.schedule.side_length) // 2,
            int(dvp_scores.shape[1]) - 1,
        )
        for class_dvp, class_weight in zip(dvp_scores, weight_spectrum):
            valid = bool(torch.any(class_weight[1 : current_radius + 1] > 0))
            valid_classes.append(valid)
            if not valid:
                radii.append(0)
                continue
            below = class_dvp[1 : current_radius + 1] < self.dvp_threshold
            if bool(torch.any(below)):
                first_below = int(torch.nonzero(below, as_tuple=False)[0].item()) + 1
                radii.append(max(1, first_below - 1))
            else:
                radii.append(max(1, current_radius))
        return radii, valid_classes

    def step(
        self,
        dvp_scores: torch.Tensor,
        weight_spectrum: torch.Tensor,
    ) -> None:
        radii, valid_classes = self._crossing_radii(dvp_scores, weight_spectrum)
        valid_radii = [
            radius
            for radius, is_valid in zip(radii, valid_classes)
            if is_valid
        ]
        if not valid_radii:
            return

        dvp_radius = max(valid_radii)
        next_side_length = 2 * (dvp_radius + self.increase_radius_step)
        next_side_length = max(2, min(self.image_size, int(next_side_length)))
        if next_side_length % 2:
            next_side_length -= 1

        metrics = self.state.heterorefine.metrics
        metrics.dvp_crossing_radius = radii
        metrics.dvp_resolution_per_volume = [
            None if not is_valid else self.image_size * self.angpix / max(radius, 1)
            for radius, is_valid in zip(radii, valid_classes)
        ]
        metrics.dvp_radius = float(dvp_radius)
        metrics.dvp_resolution = self.image_size * self.angpix / max(dvp_radius, 1)
        self.state.schedule.side_length = int(next_side_length)
        metrics.side_length_resolution = (
            2.0 * self.image_size * self.angpix / float(next_side_length)
        )