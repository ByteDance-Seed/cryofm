import math

import torch
from cryoseed.config import MainConfig
from cryoseed.state import OptimState, parse_pose_search_criterion
from cryoseed.utils import particle_mask as particle_mask_utils
from cryoseed.utils.torch_utils import _norm_device

class HomoRefineScheduler:
    def __init__(
        self,
        state: OptimState,
        *,
        device=None,
        image_size=None,
        angpix=None,
        particle_diameter=None,
        trans_grid_samples=5,
        confidence_threshold=0.1,
        convergence_patience=3,
        fsc_resolution_improvement_threshold=5e-3,
        fsc_resolution_rebound_threshold=1e-2,
        trans_update_rms_threshold=0.5,
        increase_radius_step=10,
        increase_radius_aggressive_factor=0.25,
        increase_radius_aggressive_fsc_threshold=0.2,
        base_healpix_order=3,
        auto_local_healpix_order=4,
        use_cache=False,
        cache_max_healpix_order=4,
        ssd_cache_min_side_length=150,
        trans_extent_scale=3.0,
        full_backprojection=False,
        num_epochs=None,
    ):
        self.state = state
        self.device = _norm_device(device)
        self.image_size = image_size
        self.angpix = angpix
        self.particle_diameter = particle_diameter
        self.trans_grid_samples = int(trans_grid_samples)
        self.confidence_threshold = confidence_threshold
        self.convergence_patience = int(convergence_patience)
        self.fsc_resolution_improvement_threshold = float(
            fsc_resolution_improvement_threshold
        )
        self.fsc_resolution_rebound_threshold = float(
            fsc_resolution_rebound_threshold
        )
        self.trans_update_rms_threshold = float(trans_update_rms_threshold)
        self.increase_radius_step = increase_radius_step
        self.increase_radius_aggressive_factor = increase_radius_aggressive_factor
        self.increase_radius_aggressive_fsc_threshold = float(
            increase_radius_aggressive_fsc_threshold
        )
        self.base_healpix_order = base_healpix_order
        self.auto_local_healpix_order = auto_local_healpix_order
        self.use_cache = bool(use_cache)
        self.cache_max_healpix_order = cache_max_healpix_order
        self.ssd_cache_min_side_length = ssd_cache_min_side_length
        self.trans_extent_scale = float(trans_extent_scale)
        self.default_full_backprojection = bool(full_backprojection)
        self.num_epochs = None if num_epochs is None else int(num_epochs)
        self.first_epoch_ncc = True
        self.particle_mask_enabled = False
        self.particle_mask_protection_disable_epochs = 5
        self.particle_mask_protection_radius_factor = (
            particle_mask_utils.DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR
        )

        orders = torch.arange(1, 11, device=self.device, dtype=torch.float32)
        # Approximate the angular step from the square root of each HEALPix cell area.
        self.solid_angles_list = torch.rad2deg(
            torch.sqrt(4 * torch.pi / (12 * (4.0 ** orders)))
        )

    def from_config(self, config: MainConfig):
        self.image_size = config.data.image_size
        self.angpix = float(config.data.angpix)
        self.particle_diameter = config.data.particle_diameter
        self.trans_grid_samples = int(config.modules.search.trans_grid_samples)
        self.particle_mask_enabled = bool(config.modules.search.particle_mask.enabled)
        self.particle_mask_protection_disable_epochs = int(
            config.modules.search.particle_mask.protection_disable_epochs
        )
        self.particle_mask_protection_radius_factor = float(
            config.modules.search.particle_mask.protection_radius_factor
        )
        self.confidence_threshold = config.homorefine.scheduler.confidence_threshold
        self.convergence_patience = int(
            config.homorefine.scheduler.convergence_patience
        )
        self.fsc_resolution_improvement_threshold = float(
            config.homorefine.scheduler.fsc_resolution_improvement_threshold
        )
        self.fsc_resolution_rebound_threshold = float(
            config.homorefine.scheduler.fsc_resolution_rebound_threshold
        )
        self.trans_update_rms_threshold = float(
            config.homorefine.scheduler.trans_update_rms_threshold
        )
        self.increase_radius_step = config.homorefine.scheduler.increase_radius_step
        self.increase_radius_aggressive_factor = (
            config.homorefine.scheduler.increase_radius_aggressive_factor
        )
        self.increase_radius_aggressive_fsc_threshold = float(
            config.homorefine.scheduler.increase_radius_aggressive_fsc_threshold
        )
        self.base_healpix_order = config.homorefine.scheduler.base_healpix_order
        self.auto_local_healpix_order = (
            config.homorefine.scheduler.auto_local_healpix_order
        )
        self.use_cache = bool(config.homorefine.scheduler.use_cache)
        self.cache_max_healpix_order = config.homorefine.scheduler.cache_max_healpix_order
        self.ssd_cache_min_side_length = config.homorefine.scheduler.ssd_cache_min_side_length
        self.trans_extent_scale = float(config.homorefine.scheduler.trans_extent_scale)
        self.default_full_backprojection = bool(
            config.modules.volume.voxel.full_backprojection
        )
        self.num_epochs = int(config.homorefine.engine.num_epochs)
        self.first_epoch_ncc = bool(config.homorefine.scheduler.first_epoch_ncc)
        self._apply_execution_flags(epoch=int(self.state.progress.epoch))
        self._update_particle_mask_state(epoch=int(self.state.progress.epoch))

        return self

    def _is_last_configured_epoch(self, epoch: int) -> bool:
        return self.num_epochs is not None and epoch >= self.num_epochs - 1

    def _resolve_pose_translation_center(self, *, auto_value: bool) -> None:
        mode = self.state.schedule.pose_translation_center_mode
        if mode == "auto":
            self.state.schedule.use_pose_translation_as_center = bool(auto_value)
        elif mode == "always":
            self.state.schedule.use_pose_translation_as_center = True
        else:
            self.state.schedule.use_pose_translation_as_center = False

    def _apply_execution_flags(self, *, epoch: int) -> None:
        self._resolve_pose_translation_center(auto_value=True)
        self.state.schedule.full_backprojection = (
            self.default_full_backprojection
            or bool(self.state.homorefine.engine.is_final_epoch)
        )
        self.state.homorefine.engine.skip_external_reconstruct = bool(
            self.state.homorefine.engine.is_final_epoch
        )
        # The first epoch may use a correlation criterion (implemented via NCC) to
        # stabilize coarse global search before switching to the posterior route.
        self.state.schedule.pose_search_criterion = parse_pose_search_criterion(
            "correlation" if self.first_epoch_ncc and int(epoch) == 0 else "posterior"
        )

    def _update_particle_mask_state(self, *, epoch: int) -> None:
        if self.particle_mask_protection_disable_epochs < 0:
            raise ValueError(
                "modules.search.particle_mask.protection_disable_epochs must be >= 0"
            )
        if self.particle_mask_protection_radius_factor < 0:
            raise ValueError(
                "modules.search.particle_mask.protection_radius_factor must be >= 0"
            )
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")

        use_particle_mask = bool(self.particle_mask_enabled) and int(epoch) >= int(
            self.particle_mask_protection_disable_epochs
        )
        self.state.schedule.use_particle_mask = use_particle_mask

        if not use_particle_mask or self.particle_mask_protection_radius_factor == 0:
            self.state.schedule.particle_mask_extra_diameter_angstrom = 0.0
            return

        extra_radius_px = (
            self.particle_mask_protection_radius_factor
            * float(self.state.homorefine.metrics.trans_update_rms)
        )
        self.state.schedule.particle_mask_extra_diameter_angstrom = max(
            0.0,
            2.0 * float(self.angpix) * extra_radius_px,
        )

    def _required_trans_grid_samples_for_fsc_resolution(
        self, *, trans_grid_extent: float | None = None
    ) -> int:
        if self.trans_grid_samples <= 0:
            raise ValueError("modules.search.trans_grid_samples must be > 0")
        if self.angpix is None or float(self.angpix) <= 0:
            raise ValueError("data.angpix must be set to a positive value")
        if trans_grid_extent is None:
            trans_grid_extent = float(self.state.schedule.trans_grid_extent)
        if trans_grid_extent < 0:
            raise ValueError("trans_grid_extent must be >= 0")

        fsc_resolution = self.state.homorefine.metrics.fsc_resolution
        if fsc_resolution is None or float(fsc_resolution) <= 0:
            return int(self.trans_grid_samples)

        derived_translation_step_px = float(fsc_resolution) / float(self.angpix)
        derived_samples = int(
            math.ceil((2.0 * float(trans_grid_extent)) / derived_translation_step_px)
        )
        if derived_samples % 2 == 0:
            derived_samples += 1
        return max(int(self.trans_grid_samples), derived_samples)

    def _update_convergence_state(self) -> None:
        if self.convergence_patience < 1:
            raise ValueError("homorefine.scheduler.convergence_patience must be >= 1")
        if self.fsc_resolution_improvement_threshold < 0:
            raise ValueError(
                "homorefine.scheduler.fsc_resolution_improvement_threshold must be >= 0"
            )
        if self.fsc_resolution_rebound_threshold < 0:
            raise ValueError(
                "homorefine.scheduler.fsc_resolution_rebound_threshold must be >= 0"
            )
        if self.trans_update_rms_threshold < 0:
            raise ValueError("homorefine.scheduler.trans_update_rms_threshold must be >= 0")

        resolution_change = self.state.homorefine.metrics.fsc_resolution_change
        if resolution_change is None:
            self.state.homorefine.scheduler.num_epochs_without_resolution_gain = 0
        elif resolution_change < -self.fsc_resolution_improvement_threshold:
            self.state.homorefine.scheduler.num_epochs_without_resolution_gain = 0
        elif resolution_change > self.fsc_resolution_rebound_threshold:
            self.state.homorefine.scheduler.num_epochs_without_resolution_gain = 0
        else:
            self.state.homorefine.scheduler.num_epochs_without_resolution_gain += 1

        if (
            float(self.state.homorefine.metrics.trans_update_rms)
            <= self.trans_update_rms_threshold
        ):
            self.state.homorefine.scheduler.num_epochs_with_small_trans_update += 1
        else:
            self.state.homorefine.scheduler.num_epochs_with_small_trans_update = 0

        self.state.homorefine.scheduler.has_converged = (
            self.state.homorefine.scheduler.num_epochs_without_resolution_gain
            >= self.convergence_patience
            and self.state.homorefine.scheduler.num_epochs_with_small_trans_update
            >= self.convergence_patience
        )

    def step(self):
        """Frequency marching.

        Determine the next iteration's side length ``L`` and pose search HEALPix order
        based on the half-map FSC.

        Reference: RELION's MlOptimiser::updateImageSizeAndResolutionPointers.
        """
        output_healpix_order = self.state.schedule.healpix_order + self.state.schedule.oversampling

        if self.particle_diameter is None or float(self.particle_diameter) <= 0:
            raise ValueError("particle_diameter must be set to a positive value (in Angstrom)")

        fsc_resolution = float(self.state.homorefine.metrics.fsc_resolution)

        angle_res = 360.0 * fsc_resolution / (float(self.particle_diameter) * float(torch.pi))

        if self.solid_angles_list.device != self.device:
            self.solid_angles_list = self.solid_angles_list.to(self.device)

        idx = torch.nonzero(self.solid_angles_list > angle_res, as_tuple=False)
        if idx.numel() == 0:
            healpix_order_from_res = 1
        else:
            healpix_order_from_res = int(idx[-1].item()) + 1

        if int(self.base_healpix_order) < 2:
            raise ValueError("homorefine.scheduler.base_healpix_order must be >= 2")
        if int(self.auto_local_healpix_order) < 2:
            raise ValueError("homorefine.scheduler.auto_local_healpix_order must be >= 2")
        if self.trans_extent_scale < 0:
            raise ValueError("homorefine.scheduler.trans_extent_scale must be >= 0")

        base_healpix_order = min(
            int(self.base_healpix_order),
            healpix_order_from_res,
        )
        auto_local_healpix_order = int(self.auto_local_healpix_order)

        # We want the output healpix order to be ``healpix_order_from_res + 1``.
        if healpix_order_from_res < auto_local_healpix_order:
            self.state.schedule.pose_search_scope = "global" # global pose search
            self.state.schedule.pose_search_strategy = "healpix" # HEALPix pose search

            # Run global pose search from the configured base HEALPix order,
            # then oversample until ``healpix_order_from_res + 1``.
            # e.g. if ``auto_local_healpix_order = 4`` and ``healpix_order_from_res = 3``:
            #   - base order = 2 -> search at healpix orders 2, 3, and 4
            #   - base order = 3 -> search at healpix orders 3 and 4
            self.state.schedule.healpix_order = base_healpix_order
            self.state.schedule.oversampling = (
               healpix_order_from_res - base_healpix_order + 1
            )

        elif healpix_order_from_res == auto_local_healpix_order:
            if output_healpix_order == healpix_order_from_res or output_healpix_order == healpix_order_from_res + 1:
                self.state.schedule.pose_search_scope = "local" # local pose search
                self.state.schedule.pose_search_strategy = "euler" # Euler pose search

                # Run local pose search at ``healpix_order_from_res + 1`` once.
                # e.g. if ``auto_local_healpix_order = 4``, ``output_healpix_order = 4``,
                # and ``healpix_order_from_res = 4``, then search at healpix order 5.
                self.state.schedule.oversampling = 0
                self.state.schedule.healpix_order = healpix_order_from_res + 1
            else:
                self.state.schedule.pose_search_scope = "global" # global pose search
                self.state.schedule.pose_search_strategy = "healpix" # HEALPix pose search

                # Run global HEALPix search from the configured base order, then
                # oversample until ``healpix_order_from_res + 1``.
                # e.g. if ``auto_local_healpix_order = 4`` and ``healpix_order_from_res = 4``:
                #   - base order = 2 -> search at healpix orders 2, 3, 4, and 5
                #   - base order = 3 -> search at healpix orders 3, 4, and 5
                self.state.schedule.healpix_order = base_healpix_order
                self.state.schedule.oversampling = (
                    healpix_order_from_res - base_healpix_order + 1
                )
                
        else:
            if output_healpix_order >= auto_local_healpix_order:
                self.state.schedule.pose_search_scope = "local" # local pose search
                self.state.schedule.pose_search_strategy = "euler" # Euler pose search

                # Run local pose search at ``healpix_order_from_res + 1`` once.
                # e.g. if ``auto_local_healpix_order = 4``, ``healpix_order_from_res = 5``,
                # and ``output_healpix_order = 4``, then search at healpix order 6.
                self.state.schedule.oversampling = 0
                self.state.schedule.healpix_order = healpix_order_from_res + 1
            else:
                self.state.schedule.pose_search_scope = "global" # global pose search
                self.state.schedule.pose_search_strategy = "healpix" # HEALPix pose search

                # Run global HEALPix search from the configured base order, then
                # oversample until ``healpix_order_from_res + 1``.
                # e.g. if ``auto_local_healpix_order = 4``, ``healpix_order_from_res = 5``,
                # and ``output_healpix_order = 3``:
                #   - base order = 2 -> search at healpix orders 2, 3, 4, 5, and 6
                #   - base order = 3 -> search at healpix orders 3, 4, 5, and 6
                self.state.schedule.healpix_order = base_healpix_order
                self.state.schedule.oversampling = (
                    healpix_order_from_res - base_healpix_order + 1
                )


        fsc_scores = self.state.homorefine.metrics.fsc_scores

        below = fsc_scores < 0.143
        if bool(torch.any(below)):
            cross_index = int(torch.nonzero(below, as_tuple=False)[0].item())
        else:
            cross_index = int(fsc_scores.numel() - 1)

        current_radius = cross_index - 1
        if current_radius < 1:
            current_radius = 1
        # A large ``avg_confidence`` means pose candidates are well-separated
        # (closer to convergence), so we can increase ``L`` more aggressively.
        # A small ``avg_confidence`` means candidates are similar, so we increase
        # ``L`` more conservatively.
        # ``calc_fsc()`` returns ``nx // 2`` shells indexed from 0, while the
        # frequency-window radius is expressed in shell units starting at 1.
        # Convert radius -> 0-based FSC index and clamp at Nyquist so full-size
        # windows do not probe one element past the curve.
        side_length_limit_index = min(
            max(self.state.schedule.side_length // 2 - 1, 0),
            int(fsc_scores.numel()) - 1,
        )
        fsc_at_side_length_limit = fsc_scores[side_length_limit_index]
        if (
            self.state.homorefine.metrics.avg_confidence > self.confidence_threshold
            and fsc_at_side_length_limit > self.increase_radius_aggressive_fsc_threshold
        ):
            current_radius += self.increase_radius_aggressive_factor * self.image_size // 2
        else:
            current_radius += self.increase_radius_step

        self.state.schedule.side_length = int(current_radius * 2)             
        self.state.schedule.side_length = min(self.state.schedule.side_length, self.image_size)
        
        # Keep L even.
        if self.state.schedule.side_length % 2 != 0:
            self.state.schedule.side_length -= 1

        if self.state.schedule.side_length > self.image_size:
            self.state.schedule.side_length = self.image_size

        # Update translation grid extent
        prev_trans_grid_extent = float(self.state.schedule.trans_grid_extent)
        target_trans_grid_extent = (
            self.trans_extent_scale
            * float(self.state.homorefine.metrics.trans_update_rms)
        )
        # Keep translation-range updates within a fixed multiplicative window based on
        # the previous extent.
        min_trans_grid_extent = 0.5 * prev_trans_grid_extent
        max_trans_grid_extent = 2.0 * prev_trans_grid_extent
        self.state.schedule.trans_grid_extent = min(
            max(target_trans_grid_extent, min_trans_grid_extent),
            max_trans_grid_extent,
        )
        self.state.schedule.trans_grid_samples = (
            self._required_trans_grid_samples_for_fsc_resolution(
                trans_grid_extent=float(self.state.schedule.trans_grid_extent)
            )
        )

        # Cache configuration
        if not self.use_cache:
            self.state.schedule.proj_cache_backend = "none"
        elif self.state.schedule.healpix_order <= self.cache_max_healpix_order:
            if self.state.schedule.side_length >= self.ssd_cache_min_side_length:
                self.state.schedule.proj_cache_backend = "ssd"
            else:
                self.state.schedule.proj_cache_backend = "memory"
        else:
            self.state.schedule.proj_cache_backend = "none"

        # Update convergence statistics from the current epoch before planning
        # execution flags for the next epoch.
        self._update_convergence_state()

        current_epoch = int(self.state.progress.epoch)
        current_is_final_epoch = bool(self.state.homorefine.engine.is_final_epoch)
        next_epoch = current_epoch + 1
        self.state.homorefine.engine.is_final_epoch = (
            (
                bool(self.state.homorefine.scheduler.has_converged)
                and not current_is_final_epoch
            )
            or self._is_last_configured_epoch(next_epoch)
        )
        self._apply_execution_flags(epoch=next_epoch)
        self._update_particle_mask_state(epoch=next_epoch)