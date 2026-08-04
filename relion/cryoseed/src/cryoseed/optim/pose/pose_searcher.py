import torch

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import masked_lerp, particle_mask
from cryoseed.fft.fft_torch import fourier_to_primal_2d, primal_to_fourier_2d
from cryoseed.utils.reproducibility import derive_seed
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics.noise import NoiseVariance
from cryoseed.modules.volume import Volume
from cryoseed.state import OptimState

from .healpix_searcher import HEALPixPoseSearcher
from .euler_searcher import EulerPoseSearcher


# Search-space code follows img -> vol -> rot -> trans whenever multiple
# candidate axes appear together. Keep locals, indexing, comments, and return
# values in that order unless the primary semantics have clearly shifted.
class PoseSearcher(torch.nn.Module):
    def __init__(
        self,
        state: OptimState,
        volume: Volume,
        pose: Pose | None,
        *,
        config: MainConfig,
        noise: NoiseVariance | None = None,
        device: torch.device | str | None = None,
        device_mesh=None,
    ):
        super().__init__()

        self.config = config
        self.state = state
        self.volume = volume
        self.pose = pose
        self.noise = noise

        dev = _norm_device(device)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=dev),
            persistent=False,
        )
        self.device_mesh = device_mesh

        self.pose_search_strategy: str | None = None
        self.pose_searcher: torch.nn.Module | None = None

        self.refresh()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _build_pose_searcher(self, strategy: str) -> torch.nn.Module:
        searcher_cls = {
            "healpix": HEALPixPoseSearcher,
            "euler": EulerPoseSearcher,
        }.get(strategy)

        if searcher_cls is None:
            raise ValueError(f"Unknown pose search strategy: {strategy}")
        if strategy == "euler" and self.pose is None:
            raise ValueError("Euler pose search requires pose to be available")

        return searcher_cls.from_config(
            config=self.config,
            state=self.state,
            volume=self.volume,
            pose=self.pose,
            noise=self.noise,
            device=self.device,
            device_mesh=self.device_mesh,
        )

    def _reset_pose_searcher(self) -> None:
        strategy = self.state.schedule.pose_search_strategy
        self.pose_searcher = None
        self.pose_searcher = self._build_pose_searcher(strategy)
        self.pose_search_strategy = strategy

    def refresh(self) -> None:
        strategy = self.state.schedule.pose_search_strategy

        if self.pose_searcher is None or self.pose_search_strategy != strategy:
            self._reset_pose_searcher()
        else:
            self.pose_searcher.refresh()

    def clear_memory_cache(self) -> None:
        if self.pose_searcher is None:
            return
        clear_fn = getattr(self.pose_searcher, "clear_memory_cache", None)
        if callable(clear_fn):
            clear_fn()

    def _normalize_batch_tensor(
        self,
        value,
        *,
        batch_size: int,
        width: int,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)

        if value.ndim == 1:
            if int(value.shape[0]) != width:
                raise ValueError(
                    f"{name} must have shape ({width},), (1, {width}), or (B, {width}); "
                    f"got {tuple(value.shape)}"
                )
            value = value.unsqueeze(0)
        elif value.ndim != 2 or int(value.shape[1]) != width:
            raise ValueError(
                f"{name} must have shape ({width},), (1, {width}), or (B, {width}); "
                f"got {tuple(value.shape)}"
            )

        value_batch = int(value.shape[0])
        if value_batch == 1:
            value = value.expand(batch_size, -1)
        elif value_batch != batch_size:
            raise ValueError(
                f"{name} batch must be 1 or match image batch B={batch_size}; "
                f"got {value_batch} with shape {tuple(value.shape)}"
            )

        return value.to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def _preprocess_image_with_center(
        self,
        image,
        *,
        trans_center: torch.Tensor,
        seed_index: torch.LongTensor | None = None,
    ):
        if (
            not self.config.modules.search.particle_mask.enabled
            or not self.state.schedule.use_particle_mask
        ):
            return image
        image_real = fourier_to_primal_2d(image).real

        particle_diameter = self.config.data.particle_diameter
        if particle_diameter is None:
            raise ValueError(
                "data.particle_diameter must be set when particle masking is enabled"
            )

        background_noise = None
        if not self.config.modules.search.particle_mask.zero_mask:
            if self.noise is None:
                raise ValueError(
                    "modules.statistics.noise.enabled must be enabled when "
                    "modules.search.particle_mask.zero_mask is disabled "
                    "(for example, when using `--no-zero-particle-mask`)"
                )
            particle_seed_index = seed_index
            if particle_seed_index is None:
                particle_seed_index = torch.arange(
                    int(image_real.shape[0]),
                    device=image_real.device,
                    dtype=torch.long,
                )
            background_noise = self.noise.sample_like(
                image_real,
                seed=derive_seed(
                    int(self.config.reproduce.seed),
                    "particle_mask_noise",
                    particle_seed_index,
                ),
            )

        trans_center = trans_center.to(
            device=image_real.device,
            dtype=image_real.real.dtype,
        )
        h = int(image_real.shape[-2])
        w = int(image_real.shape[-1])
        default_center = trans_center.new_tensor((w // 2, h // 2))
        # pose.translation stores the image shift in (x, y) pixel units, so the
        # real-space particle center is obtained by undoing that shift from the
        # centered FFT-grid origin.
        #
        # The particle mask follows the actual translation-search center:
        # local search always uses the stored pose translation, while global
        # search may fall back to a zero-centered translation origin.
        mask_centers = default_center.view(1, 2) - trans_center
        mask = particle_mask(
            h,
            w,
            particle_diameter=(
                float(particle_diameter)
                + float(self.state.schedule.particle_mask_extra_diameter_angstrom)
            ),
            angpix=float(self.config.data.angpix),
            center=mask_centers,
            soft_edge_pixels=float(
                self.config.modules.search.particle_mask.soft_edge_pixels
            ),
            device=image_real.device,
            dtype=image_real.real.dtype,
        )

        background = (
            image_real.new_zeros(())
            if background_noise is None
            else background_noise.to(
                device=image_real.device,
                dtype=image_real.dtype,
            )
        )
        image_masked_real = masked_lerp(image_real, mask, background)
        return primal_to_fourier_2d(image_masked_real)

    @torch.no_grad()
    def preprocess_image(
        self,
        image,
        *,
        particle_index: torch.LongTensor | None = None,
    ):
        if self.state.schedule.pose_search_scope == "local":
            if self.pose is None:
                raise ValueError("Local pose search requires pose to be available")
            if particle_index is None:
                raise ValueError("particle_index is required when pose is available")
            particle_index = particle_index.to(device=self.pose.device, dtype=torch.long)
            trans_center = self.pose.translation(particle_index).detach()
        elif self.pose is not None and bool(self.state.schedule.use_pose_translation_as_center):
            if particle_index is None:
                raise ValueError("particle_index is required when pose is available")
            particle_index = particle_index.to(device=self.pose.device, dtype=torch.long)
            trans_center = self.pose.translation(particle_index).detach()
        else:
            if self.pose is None and particle_index is not None:
                raise ValueError("particle_index must be None when pose is not available")
            trans_center = torch.zeros(
                (int(image.shape[0]), 2),
                device=self.device,
                dtype=torch.float32,
            )

        return self._preprocess_image_with_center(
            image,
            trans_center=trans_center,
            seed_index=particle_index,
        )

    @torch.no_grad()
    def search_no_grad(
        self,
        image,
        *,
        particle_index: torch.LongTensor | None = None,
        ctf=None,
        fixed_volume_index: torch.LongTensor | None = None,
    ):
        """Preprocess images and run the no-grad pose-search route.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Optional particle indices of shape ``(B,)``. Required when
                the active pose-search strategy needs pose-bound anchors.
            ctf: Optional per-image CTF tensor passed through to the active searcher.

        Returns:
            The return value of the strategy-specific ``search_no_grad()`` implementation.
        """
        if self.pose_searcher is None:
            raise RuntimeError("pose_searcher is not initialized. Call refresh() first.")
        image = self.preprocess_image(
            image,
            particle_index=particle_index,
        )
        search_no_grad = getattr(self.pose_searcher, "search_no_grad", None)
        if callable(search_no_grad):
            return search_no_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
                fixed_volume_index=fixed_volume_index,
            )
        return self.pose_searcher.search(
            image,
            particle_index=particle_index,
            ctf=ctf,
            fixed_volume_index=fixed_volume_index,
        )

    def search_grad(
        self,
        image,
        *,
        particle_index: torch.LongTensor | None = None,
        ctf=None,
        search_grad_mode: str | None = None,
        fixed_volume_index: torch.LongTensor | None = None,
    ):
        """Preprocess images and run the differentiable pose-search route.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Optional particle indices of shape ``(B,)``. Required when
                the active pose-search strategy needs pose-bound anchors.
            ctf: Optional per-image CTF tensor passed through to the active searcher.
            search_grad_mode: Optional differentiable search route override. ``"full"``
                runs the mathematically faithful full-NLL route, ``"selected"``
                runs the selected-only reprojection route, and ``None`` uses the
                current ``state.schedule.search_grad_mode``.

        Returns:
            The return value of the strategy-specific ``search_grad()`` implementation.
        """
        if self.pose_searcher is None:
            raise RuntimeError("pose_searcher is not initialized. Call refresh() first.")
        image = self.preprocess_image(
            image,
            particle_index=particle_index,
        )
        search_grad = getattr(self.pose_searcher, "search_grad", None)
        if not callable(search_grad):
            raise ValueError(
                f"pose_search_strategy {self.state.schedule.pose_search_strategy} "
                "does not support search_grad"
            )
        return search_grad(
            image,
            particle_index=particle_index,
            ctf=ctf,
            search_grad_mode=search_grad_mode,
            fixed_volume_index=fixed_volume_index,
        )

    def search(
        self,
        image,
        *,
        particle_index: torch.LongTensor | None = None,
        ctf=None,
        mode: str = "auto",
        search_grad_mode: str | None = None,
        fixed_volume_index: torch.LongTensor | None = None,
    ):
        """Preprocess images and dispatch to the requested pose-search route.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Optional particle indices of shape ``(B,)``. Required when
                the active pose-search strategy needs pose-bound anchors.
            ctf: Optional per-image CTF tensor passed through to the active searcher.
            mode: Search execution mode. ``"grad"`` dispatches to :meth:`search_grad`,
                ``"no_grad"`` dispatches to :meth:`search_no_grad`, and ``"auto"``
                dispatches to :meth:`search_grad` when autograd is enabled and the
                volume requires gradients.
            search_grad_mode: Optional differentiable search route override passed
                through when ``mode`` resolves to ``"grad"``.

        Returns:
            The return value of the selected search route.
        """
        if mode == "grad":
            return self.search_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
                search_grad_mode=search_grad_mode,
                fixed_volume_index=fixed_volume_index,
            )
        if mode == "auto":
            if torch.is_grad_enabled() and bool(getattr(self.volume, "requires_grad", False)):
                return self.search_grad(
                    image,
                    particle_index=particle_index,
                    ctf=ctf,
                    search_grad_mode=search_grad_mode,
                    fixed_volume_index=fixed_volume_index,
                )
            return self.search_no_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
                fixed_volume_index=fixed_volume_index,
            )
        if mode == "no_grad":
            return self.search_no_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
                fixed_volume_index=fixed_volume_index,
            )
        raise ValueError(f"Unsupported search mode: {mode!r}")

    @torch.no_grad()
    def search_from_anchor(
        self,
        image,
        *,
        quaternion,
        translation,
        ctf=None,
    ):
        """Run pose search from explicit anchors instead of dataset-bound pose indices.

        This is a thin top-level wrapper around the strategy-specific
        ``search_from_anchor()`` implementation exposed by the active pose
        searcher. The wrapper is responsible for two things only:

        1. normalizing the translation anchor batch shape for preprocessing; and
        2. applying the same particle-mask preprocessing path as regular
           :meth:`search`, but centered on the explicit translation anchor.

        Batch broadcast is supported for the explicit anchors:

        - ``quaternion`` may have shape ``(4,)``, ``(1, 4)``, or ``(B, 4)``
        - ``translation`` may have shape ``(2,)``, ``(1, 2)``, or ``(B, 2)``

        The strategy-specific searcher receives the normalized translation
        tensor after preprocessing, while quaternion broadcast is handled by the
        strategy-specific implementation itself.

        Current support is intentionally narrow: this entry is primarily meant
        for local Euler search from an explicit anchor. Global HEALPix search is
        not required to expose a matching anchor API.
        """
        if self.pose_searcher is None:
            raise RuntimeError("pose_searcher is not initialized. Call refresh() first.")

        translation = self._normalize_batch_tensor(
            translation,
            batch_size=int(image.shape[0]),
            width=2,
            name="translation",
            dtype=torch.float32,
        )
        image = self._preprocess_image_with_center(
            image,
            trans_center=translation,
        )

        search_from_anchor = getattr(self.pose_searcher, "search_from_anchor", None)
        if not callable(search_from_anchor):
            raise ValueError(
                f"pose_search_strategy {self.state.schedule.pose_search_strategy} "
                "does not support search_from_anchor"
            )

        return search_from_anchor(
            image,
            quaternion=quaternion,
            translation=translation,
            ctf=ctf,
        )