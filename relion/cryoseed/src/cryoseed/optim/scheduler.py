import torch
from cryoseed.config import MainConfig
from cryoseed.state import OptimState
from cryoseed.utils.torch_utils import _norm_device

class FrequencyMarchingScheduler:
    def __init__(
        self,
        state: OptimState,
        *,
        device=None,
        image_size=None,
        particle_diameter=None,
        confidence_threshold=0.1,
        increase_radius_step=10,
        increase_radius_aggressive_factor=0.25,
        base_healpix_order=2,
        auto_local_healpix_order=4,
        use_cache=False,
        cache_max_healpix_order=4,
        ssd_cache_min_side_length=150,
    ):
        self.state = state
        self.device = _norm_device(device)
        self.image_size = image_size
        self.particle_diameter = particle_diameter
        self.confidence_threshold = confidence_threshold
        self.increase_radius_step = increase_radius_step
        self.increase_radius_aggressive_factor = increase_radius_aggressive_factor
        self.base_healpix_order = base_healpix_order
        self.auto_local_healpix_order = auto_local_healpix_order
        self.use_cache = bool(use_cache)
        self.cache_max_healpix_order = cache_max_healpix_order
        self.ssd_cache_min_side_length = ssd_cache_min_side_length

        orders = torch.arange(1, 11, device=self.device, dtype=torch.float32)
        self.solid_angles_list = torch.rad2deg(4 * torch.pi / (12 * (2.0 ** orders)))

    def from_config(self, config: MainConfig):
        self.image_size = config.data.image_size
        self.particle_diameter = config.data.particle_diameter
        self.confidence_threshold = config.scheduler.confidence_threshold
        self.increase_radius_step = config.scheduler.increase_radius_step
        self.increase_radius_aggressive_factor = config.scheduler.increase_radius_aggressive_factor
        self.base_healpix_order = config.scheduler.base_healpix_order
        self.auto_local_healpix_order = config.scheduler.auto_local_healpix_order
        self.use_cache = bool(config.scheduler.use_cache)
        self.cache_max_healpix_order = config.scheduler.cache_max_healpix_order
        self.ssd_cache_min_side_length = config.scheduler.ssd_cache_min_side_length

        return self
    
    def step(self):
        """Frequency marching.

        Determine the next iteration's side length ``L`` and pose search HEALPix order
        based on the half-map FSC.

        Reference: RELION's MlOptimiser::updateImageSizeAndResolutionPointers.
        """
        output_healpix_order = self.state.schedule.healpix_order + self.state.schedule.oversampling

        if self.particle_diameter is None or float(self.particle_diameter) <= 0:
            raise ValueError("particle_diameter must be set to a positive value (in Angstrom)")

        fsc_resolution = float(self.state.metrics.fsc_resolution)

        angle_res = 360.0 * fsc_resolution / (float(self.particle_diameter) * float(torch.pi))

        if self.solid_angles_list.device != self.device:
            self.solid_angles_list = self.solid_angles_list.to(self.device)

        idx = torch.nonzero(self.solid_angles_list > angle_res, as_tuple=False)
        if idx.numel() == 0:
            healpix_order_from_res = 1
        else:
            healpix_order_from_res = int(idx[-1].item()) + 1

        self.state.metrics.healpix_order_from_resolution = healpix_order_from_res

        if int(self.base_healpix_order) < 2:
            raise ValueError("scheduler.base_healpix_order must be >= 2")
        if int(self.auto_local_healpix_order) < 2:
            raise ValueError("scheduler.auto_local_healpix_order must be >= 2")

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


        fsc_scores = self.state.metrics.fsc_scores

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
        if self.state.metrics.avg_confidence > self.confidence_threshold:
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

        # Reset confidence sum and count for next iteration
        self.state.metrics.confidence_sum = 0.0
        self.state.metrics.confidence_count = 0