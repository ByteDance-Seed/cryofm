import math

import torch

from cryoseed.config import MainConfig
from cryoseed.state import OptimState, parse_pose_search_criterion
from cryoseed.utils.torch_utils import _norm_device


class AbInitioScheduler:
    """Checkpoint-driven scheduler for ab initio reconstruction."""

    def __init__(
        self,
        state: OptimState,
        *,
        device=None,
        image_size=None,
        angpix=None,
        particle_diameter=None,
        trans_grid_samples=5,
        confidence_threshold=0.5,
        convergence_patience=3,
        pose_rotation_stability_factor=1.0,
        pose_translation_stability_factor=0.5,
        trans_extent_scale=3.0,
        increase_radius_step=1,
        auto_local_healpix_order=4,
        auto_local_assignment_change_threshold=1.0,
        target_side_length_resolution=10.0,
        target_healpix_order=None,
        full_backprojection=False,
    ):
        self.state = state
        self.device = _norm_device(device)
        self.image_size = image_size
        self.angpix = angpix
        self.particle_diameter = particle_diameter
        self.trans_grid_samples = int(trans_grid_samples)
        self.confidence_threshold = float(confidence_threshold)
        self.convergence_patience = int(convergence_patience)
        self.pose_rotation_stability_factor = float(pose_rotation_stability_factor)
        self.pose_translation_stability_factor = float(pose_translation_stability_factor)
        self.trans_extent_scale = float(trans_extent_scale)
        self.increase_radius_step = int(increase_radius_step)
        self.auto_local_healpix_order = int(auto_local_healpix_order)
        self.auto_local_assignment_change_threshold = float(
            auto_local_assignment_change_threshold
        )
        self.target_side_length_resolution = float(target_side_length_resolution)
        self.target_healpix_order = (
            None if target_healpix_order is None else int(target_healpix_order)
        )
        self.default_full_backprojection = bool(full_backprojection)
        self.local_entry_blocked_last_step = False
        orders = torch.arange(0, 17, device=self.device, dtype=torch.float32)
        # Approximate the angular step from the square root of each HEALPix cell area.
        self.solid_angles_list = torch.rad2deg(
            torch.sqrt(4 * torch.pi / (12 * (4.0 ** orders)))
        )

    def from_config(self, config: MainConfig):
        self.image_size = int(config.data.image_size)
        self.angpix = float(config.data.angpix)
        self.particle_diameter = config.data.particle_diameter
        self.trans_grid_samples = int(config.modules.search.trans_grid_samples)
        self.confidence_threshold = float(config.abinitio.scheduler.confidence_threshold)
        self.convergence_patience = int(config.abinitio.scheduler.convergence_patience)
        self.pose_rotation_stability_factor = float(
            config.abinitio.scheduler.pose_rotation_stability_factor
        )
        self.pose_translation_stability_factor = float(
            config.abinitio.scheduler.pose_translation_stability_factor
        )
        self.trans_extent_scale = float(config.abinitio.scheduler.trans_extent_scale)
        self.increase_radius_step = int(config.abinitio.scheduler.increase_radius_step)
        self.auto_local_healpix_order = int(
            config.abinitio.scheduler.auto_local_healpix_order
        )
        self.auto_local_assignment_change_threshold = float(
            config.abinitio.scheduler.auto_local_assignment_change_threshold
        )
        self.target_side_length_resolution = float(
            config.abinitio.scheduler.target_side_length_resolution
        )
        target_healpix_order = config.abinitio.scheduler.target_healpix_order
        self.target_healpix_order = (
            None if target_healpix_order is None else int(target_healpix_order)
        )
        self.default_full_backprojection = bool(
            config.modules.volume.full_backprojection
        )
        derived_target_healpix_order = self._required_healpix_order_for_resolution(
            self.target_side_length_resolution
        )
        if self.target_healpix_order is None:
            self.target_healpix_order = derived_target_healpix_order
        else:
            self.target_healpix_order = int(self.target_healpix_order)
        self._apply_execution_flags()
        return self

    def _resolve_pose_translation_center(self) -> None:
        mode = self.state.schedule.pose_translation_center_mode
        if mode == "always":
            self.state.schedule.use_pose_translation_as_center = True
        elif mode == "never":
            self.state.schedule.use_pose_translation_as_center = False

    def _apply_execution_flags(self) -> None:
        if self.auto_local_healpix_order < 2:
            raise ValueError("abinitio.scheduler.auto_local_healpix_order must be >= 2")
        if int(self.state.schedule.healpix_order) >= int(self.auto_local_healpix_order):
            self.state.schedule.pose_search_scope = "local"
            self.state.schedule.pose_search_strategy = "euler"
        else:
            self.state.schedule.pose_search_scope = "global"
            self.state.schedule.pose_search_strategy = "healpix"
        self.state.schedule.pose_search_criterion = parse_pose_search_criterion(
            "posterior"
        )
        self.state.schedule.oversampling = 0
        self._resolve_pose_translation_center()
        self.state.schedule.proj_cache_backend = "none"
        self.state.schedule.full_backprojection = (
            self.default_full_backprojection
        )
        self.state.abinitio.engine.skip_external_reconstruct = bool(
            self.state.abinitio.engine.is_final_epoch
        )

    def _sync_learning_rate_decay_flag(self) -> None:
        self.state.abinitio.solver.activate_learning_rate_decay = bool(
            self.state.abinitio.metrics.avg_confidence >= self.confidence_threshold
        )

    def _sync_search_grad_mode(self) -> None:
        self.state.schedule.search_grad_mode = (
            "selected"
            if self.state.abinitio.metrics.avg_confidence > self.confidence_threshold
            else "full"
        )

    def _side_length_to_resolution(self, side_length: int) -> float:
        if self.image_size is None or int(self.image_size) <= 0:
            raise ValueError("data.image_size must be set to a positive value")
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")
        if int(side_length) <= 0:
            raise ValueError("side_length must be > 0")
        return 2.0 * float(self.image_size) * float(self.angpix) / float(side_length)

    def _required_healpix_order_for_resolution(self, resolution: float) -> int:
        if self.particle_diameter is None or float(self.particle_diameter) <= 0:
            raise ValueError(
                "data.particle_diameter must be set to a positive value (in Angstrom)"
            )
        angle_res = 360.0 * float(resolution) / (
            float(self.particle_diameter) * float(torch.pi)
        )
        if self.solid_angles_list.device != self.device:
            self.solid_angles_list = self.solid_angles_list.to(self.device)
        idx = torch.nonzero(self.solid_angles_list > angle_res, as_tuple=False)
        if idx.numel() == 0:
            return 0
        return min(
            int(idx[-1].item()) + 1,
            int(self.solid_angles_list.numel()) - 1,
        )

    def _required_healpix_order_for_side_length(self, side_length: int) -> int:
        return self._required_healpix_order_for_resolution(
            self._side_length_to_resolution(side_length)
        )

    def _target_side_length(self) -> int:
        if self.image_size is None or int(self.image_size) <= 0:
            raise ValueError("data.image_size must be set to a positive value")
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")
        if self.target_side_length_resolution <= 0:
            raise ValueError(
                "abinitio.scheduler.target_side_length_resolution must be > 0"
            )

        target_side_length = int(
            math.ceil(
                2.0
                * float(self.image_size)
                * float(self.angpix)
                / float(self.target_side_length_resolution)
            )
        )
        if target_side_length % 2 != 0:
            target_side_length += 1
        target_side_length = min(int(self.image_size), target_side_length)
        if target_side_length % 2 != 0:
            target_side_length -= 1
        return max(2, int(target_side_length))

    def _required_trans_grid_samples_for_side_length(
        self, side_length: int, *, trans_grid_extent: float | None = None
    ) -> int:
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")
        if trans_grid_extent is None:
            trans_grid_extent = float(self.state.schedule.trans_grid_extent)
        if trans_grid_extent < 0:
            raise ValueError("trans_grid_extent must be >= 0")

        derived_translation_step_px = (
            0.5 * self._side_length_to_resolution(side_length) / float(self.angpix)
        )
        derived_samples = int(
            math.ceil((2.0 * float(trans_grid_extent)) / derived_translation_step_px)
        )
        if derived_samples % 2 == 0:
            derived_samples += 1
        return max(int(self.trans_grid_samples), derived_samples)

    def _sync_trans_grid_samples_for_current_schedule(self) -> int:
        required_trans_grid_samples = self._required_trans_grid_samples_for_side_length(
            int(self.state.schedule.side_length),
            trans_grid_extent=float(self.state.schedule.trans_grid_extent),
        )
        activated_pose_center = self._maybe_activate_pose_translation_center(
            required_trans_grid_samples
        )
        if bool(self.state.schedule.use_pose_translation_as_center):
            self._sync_pose_centered_translation_grid(
                first_activation=activated_pose_center
            )
        else:
            self.state.schedule.trans_grid_samples = int(required_trans_grid_samples)
        return int(self.state.schedule.trans_grid_samples)

    def _maybe_activate_pose_translation_center(
        self, required_trans_grid_samples: int
    ) -> bool:
        if self.state.schedule.pose_translation_center_mode != "auto":
            return False
        if bool(self.state.schedule.use_pose_translation_as_center):
            return False
        if int(required_trans_grid_samples) > 9:
            self.state.schedule.use_pose_translation_as_center = True
            return True
        return False

    def _sync_pose_centered_translation_grid(self, *, first_activation: bool) -> None:
        if self.trans_grid_samples <= 0:
            raise ValueError("modules.search.trans_grid_samples must be > 0")
        if self.trans_extent_scale < 0:
            raise ValueError("abinitio.scheduler.trans_extent_scale must be >= 0")
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")

        target_trans_grid_extent = self.trans_extent_scale * (
            0.5
            * self._side_length_to_resolution(int(self.state.schedule.side_length))
            / float(self.angpix)
        )
        if first_activation:
            self.state.schedule.trans_grid_extent = float(target_trans_grid_extent)
        else:
            prev_trans_grid_extent = float(self.state.schedule.trans_grid_extent)
            min_trans_grid_extent = 0.5 * prev_trans_grid_extent
            max_trans_grid_extent = 2.0 * prev_trans_grid_extent
            self.state.schedule.trans_grid_extent = min(
                max(target_trans_grid_extent, min_trans_grid_extent),
                max_trans_grid_extent,
            )
        self.state.schedule.trans_grid_samples = int(self.trans_grid_samples)

    def _rotation_stability_threshold(self, healpix_order: int | None = None) -> float:
        if healpix_order is None:
            healpix_order = int(self.state.schedule.healpix_order)
        healpix_order = max(0, int(healpix_order))
        index = min(healpix_order, int(self.solid_angles_list.numel()) - 1)
        angle_step_deg = float(self.solid_angles_list[index].item())
        return float(
            torch.deg2rad(
                torch.tensor(
                    self.pose_rotation_stability_factor * angle_step_deg,
                    device=self.device,
                    dtype=torch.float32,
                )
            ).item()
        )

    def _rotation_stability_threshold_for_side_length(
        self, side_length: int | None = None
    ) -> float:
        if side_length is None:
            side_length = int(self.state.schedule.side_length)
        if self.particle_diameter is None or float(self.particle_diameter) <= 0:
            raise ValueError(
                "data.particle_diameter must be set to a positive value (in Angstrom)"
            )
        resolution = self._side_length_to_resolution(int(side_length))
        angle_step_deg = 360.0 * float(resolution) / (
            float(self.particle_diameter) * float(torch.pi)
        )
        return float(
            torch.deg2rad(
                torch.tensor(
                    self.pose_rotation_stability_factor * angle_step_deg,
                    device=self.device,
                    dtype=torch.float32,
                )
            ).item()
        )

    def _translation_stability_threshold_for_side_length(
        self, side_length: int | None = None
    ) -> float:
        if side_length is None:
            side_length = int(self.state.schedule.side_length)
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")
        derived_translation_step_px = (
            0.5 * self._side_length_to_resolution(int(side_length)) / float(self.angpix)
        )
        return self.pose_translation_stability_factor * derived_translation_step_px

    def _min_side_length_for_healpix_order(self, healpix_order: int) -> int:
        target_side_length = self._target_side_length()
        desired_order = max(0, int(healpix_order))
        for side_length in range(2, int(target_side_length) + 1, 2):
            if self._required_healpix_order_for_side_length(side_length) >= desired_order:
                return int(side_length)
        return int(target_side_length)

    def _next_side_length(self) -> int:
        if self.image_size is None or int(self.image_size) <= 0:
            raise ValueError("data.image_size must be set to a positive value")
        target_side_length = self._target_side_length()
        current_side_length = int(self.state.schedule.side_length)
        current_radius = max(1, current_side_length // 2)
        if self.state.abinitio.metrics.avg_confidence > self.confidence_threshold:
            proposed_next_side_length = self._min_side_length_for_healpix_order(
                int(self.state.schedule.healpix_order) + 1
            )
        else:
            current_radius += int(self.increase_radius_step)
            proposed_next_side_length = int(current_radius * 2)
        next_side_length = min(
            int(self.image_size),
            int(target_side_length),
            int(proposed_next_side_length),
        )
        if next_side_length % 2 != 0:
            next_side_length -= 1
        next_side_length = max(2, next_side_length)
        if (
            next_side_length <= current_side_length
            and current_side_length < int(target_side_length)
        ):
            next_side_length = min(
                int(self.image_size),
                int(target_side_length),
                current_side_length + 2,
            )
        return int(next_side_length)

    def _reset_stability_counts(self) -> None:
        self.state.abinitio.scheduler.num_checks_with_stable_side_length = 0
        self.state.abinitio.scheduler.num_checks_with_stable_pose = 0
        self.state.abinitio.scheduler.num_checks_ready_to_stop = 0
        self.state.abinitio.scheduler.has_converged = False

    def local_volume_class_change_rate(self) -> float:
        volume_class_change_rate = self.state.abinitio.metrics.ema_volume_class_change_rate
        if volume_class_change_rate is None:
            volume_class_change_rate = self.state.abinitio.metrics.volume_class_change_rate
        return float(volume_class_change_rate)

    def local_entry_blocked(self) -> bool:
        return bool(self.local_entry_blocked_last_step)

    def _is_local_entry_transition(self, next_healpix_order: int) -> bool:
        current_healpix_order = int(self.state.schedule.healpix_order)
        return (
            current_healpix_order < int(self.auto_local_healpix_order)
            and int(next_healpix_order) >= int(self.auto_local_healpix_order)
        )

    def _advance_healpix_order(self) -> bool:
        current_healpix_order = int(self.state.schedule.healpix_order)
        next_healpix_order = current_healpix_order + 1
        if self.target_healpix_order is not None:
            next_healpix_order = min(
                next_healpix_order,
                int(self.target_healpix_order),
            )
        if next_healpix_order > current_healpix_order:
            if self.auto_local_assignment_change_threshold < 0:
                raise ValueError(
                    "abinitio.scheduler.auto_local_assignment_change_threshold must be >= 0"
                )
            if (
                self._is_local_entry_transition(next_healpix_order)
                and self.local_volume_class_change_rate()
                > self.auto_local_assignment_change_threshold
            ):
                self.local_entry_blocked_last_step = True
                return False
            self.state.schedule.healpix_order = next_healpix_order
            self._reset_stability_counts()
            return True
        if self.target_healpix_order is not None:
            self.state.abinitio.scheduler.healpix_terminal_reached = True
        return False

    def _advance_side_length(self) -> bool:
        next_side_length = self._next_side_length()
        if next_side_length <= int(self.state.schedule.side_length):
            return False
        self.state.schedule.side_length = int(next_side_length)
        self.state.abinitio.scheduler.initial_healpix_alignment_done = True
        self._sync_trans_grid_samples_for_current_schedule()
        self.state.abinitio.metrics.side_length_resolution = self._side_length_to_resolution(
            int(self.state.schedule.side_length)
        )
        self._reset_stability_counts()
        return True

    def _update_stability_state(self) -> tuple[bool, bool, bool, bool]:
        if self.convergence_patience < 1:
            raise ValueError("abinitio.scheduler.convergence_patience must be >= 1")
        if self.pose_rotation_stability_factor < 0:
            raise ValueError(
                "abinitio.scheduler.pose_rotation_stability_factor must be >= 0"
            )
        if self.pose_translation_stability_factor < 0:
            raise ValueError(
                "abinitio.scheduler.pose_translation_stability_factor must be >= 0"
            )

        ema_rot_update_rms = self.state.abinitio.metrics.ema_rot_update_rms
        ema_trans_update_rms = self.state.abinitio.metrics.ema_trans_update_rms
        rotation_stable = (
            ema_rot_update_rms is not None
            and float(ema_rot_update_rms) <= self._rotation_stability_threshold()
        )
        translation_stable = (
            ema_trans_update_rms is not None
            and float(ema_trans_update_rms)
            <= self._translation_stability_threshold_for_side_length()
        )
        side_length_rotation_stable = (
            ema_rot_update_rms is not None
            and float(ema_rot_update_rms)
            <= self._rotation_stability_threshold_for_side_length()
        )
        side_length_stable = side_length_rotation_stable and translation_stable
        if side_length_stable:
            self.state.abinitio.scheduler.num_checks_with_stable_side_length += 1
        else:
            self.state.abinitio.scheduler.num_checks_with_stable_side_length = 0

        pose_stable = side_length_stable
        if pose_stable:
            self.state.abinitio.scheduler.num_checks_with_stable_pose += 1
        else:
            self.state.abinitio.scheduler.num_checks_with_stable_pose = 0

        side_length_target_reached = (
            self.state.abinitio.metrics.side_length_resolution is not None
            and float(self.state.abinitio.metrics.side_length_resolution)
            <= self.target_side_length_resolution
        )
        healpix_terminal_reached = bool(
            self.state.abinitio.scheduler.healpix_terminal_reached
        )
        healpix_ready_for_side_length = (
            int(self.state.schedule.healpix_order)
            >= self._required_healpix_order_for_side_length(
                int(self.state.schedule.side_length)
            )
        )
        final_stable = (
            (
                (side_length_target_reached and healpix_ready_for_side_length)
                or healpix_terminal_reached
            )
            and side_length_stable
            and pose_stable
        )
        if final_stable:
            self.state.abinitio.scheduler.num_checks_ready_to_stop += 1
        else:
            self.state.abinitio.scheduler.num_checks_ready_to_stop = 0
        self.state.abinitio.scheduler.has_converged = (
            self.state.abinitio.scheduler.num_checks_ready_to_stop
            >= self.convergence_patience
        )
        return side_length_stable, rotation_stable, pose_stable, final_stable

    def step(self):
        """Advance the ab initio scheduling state at a checkpoint."""
        self.local_entry_blocked_last_step = False
        self._sync_search_grad_mode()
        self._sync_learning_rate_decay_flag()
        self.state.abinitio.metrics.side_length_resolution = self._side_length_to_resolution(
            int(self.state.schedule.side_length)
        )
        self._sync_trans_grid_samples_for_current_schedule()
        required_healpix_order_for_side_length = self._required_healpix_order_for_side_length(
            int(self.state.schedule.side_length)
        )
        if not bool(self.state.abinitio.scheduler.initial_healpix_alignment_done):
            if int(self.state.schedule.healpix_order) < required_healpix_order_for_side_length:
                if self._advance_healpix_order():
                    self._apply_execution_flags()
                    return
            self.state.abinitio.scheduler.initial_healpix_alignment_done = True

        _, rotation_stable, _, _ = self._update_stability_state()
        side_length_ready = (
            self.state.abinitio.scheduler.num_checks_with_stable_side_length
            >= self.convergence_patience
        )
        healpix_ready_for_side_length = (
            int(self.state.schedule.healpix_order) >= required_healpix_order_for_side_length
        )
        side_length_target_reached = (
            self.state.abinitio.metrics.side_length_resolution is not None
            and float(self.state.abinitio.metrics.side_length_resolution)
            <= self.target_side_length_resolution
        )

        if bool(self.state.abinitio.scheduler.has_converged) and not bool(
            self.state.abinitio.engine.is_final_epoch
        ):
            self.state.abinitio.engine.is_final_epoch = True
        elif not healpix_ready_for_side_length:
            if rotation_stable:
                self._advance_healpix_order()
        elif side_length_ready:
            if side_length_target_reached:
                if (
                    not bool(self.state.abinitio.scheduler.healpix_terminal_reached)
                    and rotation_stable
                ):
                    self._advance_healpix_order()
            else:
                self._advance_side_length()
        self._apply_execution_flags()