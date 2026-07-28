from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import circular_mask
from cryoseed.fft.fft_torch import fourier_to_primal_2d, primal_to_fourier_2d
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.ops.radial import radial_average, radial_broadcast, radial_residual_power as radial_residual_power_op
from cryoseed.ops.transforms import downsample2d, translate_image
from cryoseed.modules.volume import Volume

__all__ = [
    "NoiseVariance",
]


class NoiseVariance(nn.Module):
    """Maintain a 1D radial noise power spectrum (variance) estimate.

    The spectrum is stored in :attr:`variance` with shape ``(R,)`` where
    ``R = image_size // 2 + 1`` corresponds to integer Fourier radii
    ``0..image_size//2`` on an fftshifted grid.

    This module supports two update modes:

    - Full recomputation from spatial-domain data via :meth:`from_data`.
    - Incremental estimation via :meth:`accumulate` + :meth:`update`.

    Notes:
        - All spectra are stored as real-valued tensors.
        - Precision is defined as ``1 / clamp(variance, min=precision_eps)``.
    """

    @classmethod
    def from_config(
        cls,
        config: MainConfig,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        *,
        requires_accum: bool = True,
    ) -> NoiseVariance | None:
        if not config.statistics.use_noise:
            return None
        return cls(
            image_size=int(config.data.image_size),
            device=device,
            device_mesh=device_mesh,
            requires_accum=requires_accum,
            accumulate_chunk=int(config.reconstruction.accumulate_chunk),
            init_variance=float(config.statistics.init_variance),
            precision_eps=float(config.statistics.precision_eps),
            ema_decay=float(config.statistics.noise_ema_decay),
            prior_weight=float(config.statistics.noise_prior_weight),
            inflated_weight=float(config.statistics.noise_inflated_weight),
            inflated_decay=(
                None
                if config.statistics.noise_inflated_decay is None
                else float(config.statistics.noise_inflated_decay)
            ),
            inflated_scale=float(config.statistics.noise_inflated_scale),
        )

    def __init__(
        self,
        image_size: int,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        requires_accum: bool = True,
        accumulate_chunk: int = 65536,
        init_variance: float = 1.0,
        precision_eps: float = 1e-6,
        ema_decay: float = 0.0,
        prior_weight: float = 0.0,
        inflated_weight: float = 0.0,
        inflated_decay: float | None = None,
        inflated_scale: float = 8.0,
    ):
        """Create a :class:`NoiseVariance`.

        Args:
            image_size: Fourier image side length ``D``.
            device: Device for internal buffers.
            device_mesh: Optional distributed mesh used by :meth:`update` for
                data-parallel synchronization.
            requires_accum: If ``False``, :meth:`accumulate` and :meth:`update` are no-ops.
            accumulate_chunk: Chunk size along the pose dimension for :meth:`accumulate`.
            init_variance: Initial constant variance value.
            precision_eps: Clamp used when forming precision ``1 / variance``.
            ema_decay: Decay used by running-average statistics.
            prior_weight: Weight of the fixed prior variance.
            inflated_weight: Initial weight of the inflated prior variance.
            inflated_decay: Optional decay used by the inflated prior weight.
                Defaults to ``ema_decay`` when ``None``.
            inflated_scale: Multiplicative scale applied to the inflated prior variance.
        """
        super().__init__()

        self.image_size = int(image_size)
        self.num_radial_bins = self.image_size // 2 + 1

        dev = _norm_device(device)
        self.register_buffer("_device_anchor", torch.empty(0, device=dev), persistent=False)
        self.device_mesh = device_mesh
        self.requires_accum = bool(requires_accum)
        self.accumulate_chunk = int(accumulate_chunk)
        self.init_variance = float(init_variance)
        self.precision_eps = float(precision_eps)

        self.ema_decay = float(ema_decay)
        self.inflated_decay = (
            self.ema_decay if inflated_decay is None else float(inflated_decay)
        )

        self.prior_weight = float(prior_weight)
        self.inflated_weight = float(inflated_weight)
        self.inflated_scale = float(inflated_scale)
        self.use_regularization = self.prior_weight > 0 or self.inflated_weight > 0

        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if self.accumulate_chunk <= 0:
            raise ValueError(f"accumulate_chunk must be > 0, got {self.accumulate_chunk}")
        if self.init_variance <= 0:
            raise ValueError(f"init_variance must be > 0, got {self.init_variance}")
        if self.precision_eps <= 0:
            raise ValueError(f"precision_eps must be > 0, got {self.precision_eps}")
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")
        if not (0.0 <= self.inflated_decay <= 1.0):
            raise ValueError(f"inflated_decay must be in [0, 1], got {self.inflated_decay}")
        if self.prior_weight < 0:
            raise ValueError(f"prior_weight must be >= 0, got {self.prior_weight}")
        if self.inflated_weight < 0:
            raise ValueError(f"inflated_weight must be >= 0, got {self.inflated_weight}")
        if self.inflated_scale <= 0:
            raise ValueError(f"inflated_scale must be > 0, got {self.inflated_scale}")

        self.register_buffer(
            "variance",
            torch.full(
                (self.num_radial_bins,),
                self.init_variance,
                dtype=torch.float32,
                device=dev,
            ),
            persistent=True,
        )

        self.register_buffer(
            "accum_numer",
            torch.zeros_like(self.variance) if requires_accum else None,
            persistent=False,
        )
        # Per-bin denominators let low-resolution updates touch only a prefix of bins
        # without zeroing higher-frequency variance on update().
        self.register_buffer(
            "accum_denom",
            torch.zeros_like(self.variance) if requires_accum else None,
            persistent=False,
        )

        if self.ema_decay > 0:
            self.register_buffer(
                "ema_numer",
                torch.zeros(
                    (self.num_radial_bins,),
                    dtype=torch.float32,
                    device=dev,
                ),
                persistent=True,
            )
            self.register_buffer(
                "ema_denom",
                torch.zeros(
                    (self.num_radial_bins,),
                    dtype=torch.float32,
                    device=dev,
                ),
                persistent=True,
            )
        else:
            self.ema_numer = None
            self.ema_denom = None

        if self.use_regularization:
            self.register_buffer(
                "accum_weight",
                torch.zeros_like(self.variance) if requires_accum else None,
                persistent=False,
            )
            self.register_buffer(
                "prior_variance",
                torch.tensor(self.init_variance, dtype=torch.float32, device=dev),
                persistent=False,
            )
            self.register_buffer(
                "inflated_variance",
                torch.tensor(self.init_variance * self.inflated_scale, dtype=torch.float32, device=dev),
                persistent=False,
            )
            self.register_buffer(
                "inflated_weight_eff",
                torch.tensor(self.inflated_weight, dtype=torch.float32, device=dev),
                persistent=True,
            )
            if self.ema_decay > 0:
                self.register_buffer(
                    "ema_weight",
                    torch.zeros_like(self.variance),
                    persistent=True,
                )
            else:
                self.ema_weight = None
        else:
            self.accum_weight = None
            self.prior_variance = None
            self.inflated_variance = None
            self.inflated_weight_eff = None
            self.ema_weight = None




    @torch.no_grad()
    def fill_(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"value must be > 0, got {value}")
        self.variance.fill_(value)
        if self.inflated_weight_eff is not None:
            self.inflated_weight_eff.fill_(self.inflated_weight)
        self._zero_ema()
        self.zero_accum()

    @torch.no_grad()
    def from_data(self, dataloaders: list[DataLoader]) -> None:
        """Recompute the noise variance spectrum from data.

        In the current pipeline, dataloaders yield :class:`~cryoseed.data.DataBatch`.
        We prefer using ``batch.image`` directly (Fourier domain, fftshift convention)
        to avoid recomputing FFTs.

        The per-frequency variance estimate first computes the complex residual
        power

        ``E[|F|^2] - |E[F]|^2 = Var(Re F) + Var(Im F)``

        and then converts it to the per-real-channel variance used by the
        likelihood and residual accumulation paths by dividing by ``2`` under a
        circular complex Gaussian assumption.

        The result is followed by a 2D radial average.

        Args:
            dataloaders: A list of dataloaders yielding :class:`~cryoseed.data.DataBatch`.
        """
        D = int(self.image_size)
        power_sum = torch.zeros((D, D), dtype=torch.float32, device=self.device)
        image_sum = torch.zeros((D, D), dtype=torch.complex64, device=self.device)
        num_images = 0
        if self.use_regularization:
            corner_mask = ~circular_mask(
                D,
                D,
                center=(D // 2, D // 2),
                radius=float(D // 2),
                device=self.device,
            )
            corner_pixels = int(corner_mask.sum().item())
            corner_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            corner_sumsq = torch.zeros((), dtype=torch.float64, device=self.device)
            corner_count = 0

        for dataloader in dataloaders:
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)
                image = getattr(batch, "image", None)

                if image.dtype != torch.complex64:
                    raise ValueError(f"batch.image must be complex64, got {image.dtype}")
                if image.ndim != 3 or int(image.shape[-1]) != int(image.shape[-2]):
                    raise ValueError(f"batch.image must be (B,D,D), got {tuple(image.shape)}")
                if int(image.shape[-1]) != D:
                    raise ValueError(f"batch.image side length {int(image.shape[-1])} != image_size {D}")

                power_sum += (image.abs() ** 2).sum(dim=0)
                image_sum += image.sum(dim=0)
                num_images += int(image.shape[0])
                if self.use_regularization and corner_pixels > 0:
                    image_real = fourier_to_primal_2d(image).real
                    corner_values = image_real[:, corner_mask]
                    corner_sum += corner_values.sum(dtype=torch.float64)
                    corner_sumsq += corner_values.square().sum(dtype=torch.float64)
                    corner_count += int(image.shape[0]) * corner_pixels

        self.variance.zero_()
        if self.use_regularization:
            self.prior_variance.zero_()
            self.inflated_variance.zero_()
            self.inflated_weight_eff.fill_(self.inflated_weight)
        self._zero_ema()
        self.zero_accum()
        if num_images <= 0:
            return

        mean_power = power_sum / float(num_images)
        mean_image = image_sum / float(num_images)
        var2d = 0.5 * (mean_power - mean_image.abs().square())
        var2d = var2d.clamp_min(0.0)
        var1d = radial_average(var2d.unsqueeze(0), max_radius=D // 2, ndim=2, use_cache=True)
        self.variance.copy_(var1d.squeeze(0))
        if self.use_regularization and corner_count > 0:
            corner_mean = corner_sum / float(corner_count)
            prior_variance_real = corner_sumsq / float(corner_count) - corner_mean.square()
            prior_variance_fourier = prior_variance_real.clamp_min(0.0).to(dtype=torch.float32)
            prior_variance_fourier *= float(D * D) / 2.0
            self.prior_variance.copy_(prior_variance_fourier)
            self.inflated_variance.copy_(prior_variance_fourier * self.inflated_scale)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    @property
    def precision(self) -> Tensor:
        """Return the radial precision spectrum ``1 / variance``."""
        return 1.0 / self.variance.clamp_min(self.precision_eps)

    def variance_spectrum(
        self,
        ndim: int,
        side_length: int | None = None,
        max_radius: int | None = None,
        padding_mode: str = "border",
    ) -> Tensor:
        """Broadcast the stored 1D variance profile onto an ``ndim``-D FFT grid.

        Args:
            ndim: Target spatial dimensionality (2 or 3).
            side_length: Output side length. Defaults to :attr:`image_size`.
            max_radius: Maximum Fourier radius to use (inclusive). Defaults to
                ``image_size // 2``.
            padding_mode: Passed to :func:`cryoseed.ops.radial.radial_broadcast`.

        Returns:
            Real tensor of shape ``(side_length,) * ndim``.
        """
        if side_length is None:
            side_length = self.image_size
        if max_radius is None:
            max_radius = self.image_size // 2
        max_radius = int(max_radius)
        if not (0 <= max_radius <= self.num_radial_bins - 1):
            raise ValueError(
                f"max_radius must be in [0, {self.num_radial_bins - 1}], got {max_radius}"
            )
        return radial_broadcast(
            self.variance[: max_radius + 1],
            ndim,
            out_len=side_length,
            padding_mode=padding_mode,
        )

    def precision_spectrum(
        self,
        ndim: int,
        side_length: int | None = None,
        max_radius: int | None = None,
        padding_mode: str = "border",
    ) -> Tensor:
        """Broadcast the precision profile onto an ``ndim``-D FFT grid."""
        if side_length is None:
            side_length = self.image_size
        if max_radius is None:
            max_radius = self.image_size // 2
        max_radius = int(max_radius)
        if not (0 <= max_radius <= self.num_radial_bins - 1):
            raise ValueError(
                f"max_radius must be in [0, {self.num_radial_bins - 1}], got {max_radius}"
            )
        return radial_broadcast(
            self.precision[: max_radius + 1],
            ndim,
            out_len=side_length,
            padding_mode=padding_mode,
        )

    @torch.no_grad()
    def sample_like(
        self,
        image_real: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        """Sample real-space noise with the same batch shape as ``image_real``.

        The input ``seed`` is interpreted as a batch-level seed: this method
        uses a single ``torch.Generator`` and one batched ``randn`` draw for
        lower Python overhead. As a result, reproducibility is guaranteed for a
        fixed seed together with a fixed batch shape/order, but individual
        samples are not guaranteed to remain identical if batch composition or
        ordering changes.
        """
        if image_real.ndim != 3 or image_real.shape[-2] != image_real.shape[-1]:
            raise ValueError(
                f"image_real must have shape (B,D,D), got {tuple(image_real.shape)}"
            )
        _, side_length, _ = image_real.shape
        if int(side_length) != self.image_size:
            raise ValueError(
                f"image_real side length {int(side_length)} != image_size {self.image_size}"
            )

        max_seed = (1 << 63) - 1
        generator = torch.Generator(device=image_real.device)
        generator.manual_seed(int(seed) % max_seed)
        white_noise = torch.randn(
            image_real.shape,
            dtype=torch.float32,
            device=image_real.device,
            generator=generator,
        )

        noise_power = self.variance_spectrum(
            ndim=2,
            side_length=int(side_length),
            padding_mode="zeros",
        ).to(device=image_real.device, dtype=torch.float32)
        noise_fourier = primal_to_fourier_2d(white_noise)
        noise_fourier *= noise_power.clamp_min(0.0).sqrt().unsqueeze(0) / float(
            side_length
        )
        noise_real = fourier_to_primal_2d(noise_fourier).real
        return noise_real.to(dtype=image_real.dtype)

    @torch.no_grad()
    def _accumulate_from_precomputed_radial_residual_power(
        self,
        probability: Tensor,
        image_index: torch.LongTensor,
        radial_residual_power: Tensor,
        *,
        num_images: int,
    ) -> None:
        if not self.requires_accum:
            return

        if radial_residual_power.dim() != 2:
            raise ValueError(
                f"radial_residual_power must be 2D with shape (N,R), got {tuple(radial_residual_power.shape)}"
            )

        N = int(image_index.numel())
        if probability.shape != (N,):
            raise ValueError(f"probability must be (N,)=({N},), got {tuple(probability.shape)}")
        if radial_residual_power.shape[0] != N:
            raise ValueError(
                "radial_residual_power must have the same leading dimension as image_index"
            )
        if num_images <= 0:
            raise ValueError(f"num_images must be > 0, got {num_images}")

        num_bins = int(radial_residual_power.shape[1])
        if not (0 < num_bins <= self.num_radial_bins):
            raise ValueError(
                f"radial_residual_power.shape[1] must be in [1, {self.num_radial_bins}], got {num_bins}"
            )

        accum_chunk = int(self.accumulate_chunk)
        for chunk_start in range(0, N, accum_chunk):
            chunk_end = min(chunk_start + accum_chunk, N)
            prob = probability[chunk_start:chunk_end].to(device=self.device, dtype=torch.float32)
            residual = radial_residual_power[chunk_start:chunk_end].to(
                device=self.device,
                dtype=torch.float32,
            )
            noise_var_per_pose = 0.5 * prob.view(-1, 1) * residual
            self.accum_numer[:num_bins] += noise_var_per_pose.sum(dim=0).to(
                self.accum_numer.dtype
            )

        self.accum_denom[:num_bins] += float(num_images)

    @torch.no_grad()
    def _accumulate_from_observations(
        self,
        image: Tensor,
        ctf: Tensor | None,
        volume: Volume,
        probability: Tensor,
        image_index: torch.LongTensor,
        volume_index: torch.LongTensor,
        rotation: Tensor,
        translation: Tensor,
        *,
        side_length: int | None,
    ) -> None:
        if image.dim() != 3 or image.shape[1] != image.shape[2]:
            raise ValueError(f"image must be (B,D,D), got {tuple(image.shape)}")

        B = int(image.shape[0])
        D = int(image.shape[1])
        L = D if side_length is None else int(side_length)
        if D != int(self.image_size):
            raise ValueError(f"image side length {D} does not match image_size {int(self.image_size)}")
        if not (0 < L <= D):
            raise ValueError(f"side_length must be in [1, {D}], got {L}")

        N = int(image_index.numel())
        if volume_index.numel() != N:
            raise ValueError("volume_index must have the same number of elements as image_index")
        if rotation.shape[0] != N or rotation.shape[-2:] != (3, 3):
            raise ValueError(f"rotation must be (N,3,3) with N={N}, got {tuple(rotation.shape)}")
        if translation.shape != (N, 2):
            raise ValueError(f"translation must be (N,2)=({N},2), got {tuple(translation.shape)}")
        if probability.shape != (N,):
            raise ValueError(f"probability must be (N,)=({N},), got {tuple(probability.shape)}")

        if ctf is not None:
            if ctf.ndim != 3:
                raise ValueError(f"ctf must be 3D with shape (B,D,D) or (B,L,L), got {tuple(ctf.shape)}")
            if int(ctf.shape[0]) != B:
                raise ValueError(
                    f"ctf batch must match image batch: expected B={B}, got ctf.shape[0]={int(ctf.shape[0])} "
                    f"with full shape {tuple(ctf.shape)}"
                )
            ctf_side = int(ctf.shape[-1])
            if int(ctf.shape[-2]) != ctf_side or ctf_side not in (D, L):
                raise ValueError(
                    f"ctf must have shape (B,{D},{D}) or (B,{L},{L}), got {tuple(ctf.shape)}"
                )

        K = int(getattr(volume, "num_volumes", 1))
        device = self.device
        accum_chunk = int(self.accumulate_chunk)

        for chunk_start in range(0, N, accum_chunk):
            chunk_end = min(chunk_start + accum_chunk, N)
            chunk_size = int(chunk_end - chunk_start)

            prob = probability[chunk_start:chunk_end].to(device=device, dtype=torch.float32)
            img_idx = image_index[chunk_start:chunk_end].to(device=device, dtype=torch.long)
            vol_idx = volume_index[chunk_start:chunk_end].to(device=device, dtype=torch.long)
            rot = rotation[chunk_start:chunk_end].to(device=device)
            trans = translation[chunk_start:chunk_end].to(device=device, dtype=torch.float32)

            img_pose = image[img_idx]
            trans_image = translate_image(img_pose, trans)
            if L != D:
                trans_image = downsample2d(trans_image, L)

            rot_kq = rot.view(1, chunk_size, 3, 3).expand(K, -1, -1, -1).contiguous()
            proj_all = volume.project(rot_kq, side_length=L)
            proj_image = proj_all[vol_idx, torch.arange(chunk_size, device=device)]

            if ctf is not None:
                ctf_pose = ctf[img_idx]
                if int(ctf_pose.shape[-1]) != L:
                    ctf_pose = downsample2d(ctf_pose, L)
                proj_image = proj_image * ctf_pose

            pair_idx = torch.arange(chunk_size, device=device, dtype=torch.long)
            radial_residual_power = radial_residual_power_op(
                proj_image,
                trans_image,
                input_indices=pair_idx,
                target_indices=pair_idx,
                side_length=L,
                max_radius=L // 2,
                ndim=2,
                use_cache=True,
            )
            noise_var_per_pose = 0.5 * prob.view(-1, 1) * radial_residual_power
            self.accum_numer[: L // 2 + 1] += noise_var_per_pose.sum(dim=0).to(
                self.accum_numer.dtype
            )

        self.accum_denom[: L // 2 + 1] += float(B)

    @torch.no_grad()
    def _accumulate_regularization_weight(
        self,
        ctf: Tensor | None,
        *,
        batch_size: int,
        side_length: int,
    ) -> None:
        if not self.use_regularization:
            return

        num_bins = int(side_length) // 2 + 1
        if ctf is None:
            self.accum_weight[:num_bins] += float(batch_size)
            return

        ctf_weight = ctf.to(device=self.device, dtype=torch.float32)
        if int(ctf_weight.shape[-1]) != int(side_length):
            ctf_weight = downsample2d(ctf_weight, int(side_length))
        weight = radial_average(
            ctf_weight.square().sum(dim=0),
            max_radius=int(side_length) // 2,
            ndim=2,
            use_cache=True,
        )
        self.accum_weight[:num_bins] += weight.to(self.accum_weight.dtype)

    @torch.no_grad()
    def accumulate(
        self,
        image: Tensor | None = None,
        ctf: Tensor | None = None,
        volume: Volume | None = None,
        probability: Tensor | None = None,
        image_index: torch.LongTensor | None = None,
        volume_index: torch.LongTensor | None = None,
        rotation: Tensor | None = None,
        translation: Tensor | None = None,
        *,
        radial_residual_power: Tensor | None = None,
        num_images: int | None = None,
        side_length: int | None = None,
    ) -> None:
        """Accumulate a weighted residual-based estimate of the noise variance.

        Two input modes are supported:

        - Self-contained mode: compute residuals from ``image/ctf/volume/rotation/translation`` via
          ``|proj(volume, R; L) - translate(image, t; L)|^2`` and then take a radial average on
          radii ``0..L//2``.
        - Precomputed mode: consume ``radial_residual_power`` directly.

        In both cases this accumulates a probability-weighted radial average into internal
        numerator/denominator buffers for later normalization by :meth:`update`.

        Args:
            image: Fourier-domain images with shape ``(B, D, D)``.
            ctf: Optional CTF with shape ``(B, D, D)`` or ``(B, L, L)``.
            volume: Volume module exposing ``project(rotation, side_length=...)``.
            image_index: Pose-to-image mapping with shape ``(N,)``.
            volume_index: Pose-to-volume mapping with shape ``(N,)``.
            rotation: Rotation matrices with shape ``(N, 3, 3)``.
            translation: Translations with shape ``(N, 2)`` in pixels.
            probability: Per-pose weights with shape ``(N,)``.
            radial_residual_power: Optional precomputed radial residual power with
                shape ``(N, R)`` where ``R <= image_size // 2 + 1``.
            num_images: Required when ``radial_residual_power`` is provided.
            side_length: Frequency window side length ``L`` used by the self-contained mode.
                Defaults to ``image.shape[-1]``.
        """
        if not self.requires_accum:
            return

        if radial_residual_power is not None:
            if probability is None or image_index is None or num_images is None:
                raise ValueError(
                    "probability, image_index, and num_images are required when radial_residual_power is provided"
                )
            self._accumulate_from_precomputed_radial_residual_power(
                probability,
                image_index,
                radial_residual_power,
                num_images=int(num_images),
            )
            if self.use_regularization:
                self._accumulate_regularization_weight(
                    ctf,
                    batch_size=int(num_images),
                    side_length=int(self.image_size) if side_length is None else int(side_length),
                )
            return

        if image is None or volume is None or probability is None or image_index is None:
            raise ValueError(
                "image, volume, probability, and image_index are required when radial_residual_power is not provided"
            )
        if volume_index is None or rotation is None or translation is None:
            raise ValueError(
                "volume_index, rotation, and translation are required when radial_residual_power is not provided"
            )
        self._accumulate_from_observations(
            image,
            ctf,
            volume,
            probability,
            image_index,
            volume_index,
            rotation,
            translation,
            side_length=side_length,
        )

        if self.use_regularization:
            D = int(image.shape[-1])
            L = D if side_length is None else int(side_length)
            self._accumulate_regularization_weight(
                ctf,
                batch_size=int(image.shape[0]),
                side_length=L,
            )

    @torch.no_grad()
    def update(self) -> None:
        """Finalize :attr:`variance` from accumulated numerator/denominator.

        If ``torch.distributed`` is initialized, this synchronizes the accumulators
        across the data-parallel group (``device_mesh.get_group(0)`` if available,
        otherwise ``WORLD``).
        """
        if (not self.requires_accum) or self.accum_numer is None or self.accum_denom is None:
            return
        if self.use_regularization and self.accum_weight is None:
            return

        numer = self.accum_numer
        denom = self.accum_denom
        weight = self.accum_weight if self.use_regularization else None

        dist = getattr(torch, "distributed", None)

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
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
                numer = numer.clone()
                denom = denom.clone()
                dist.all_reduce(
                    numer,
                    op=dist.ReduceOp.SUM,
                    group=group,
                )
                dist.all_reduce(
                    denom,
                    op=dist.ReduceOp.SUM,
                    group=group,
                )

                if self.use_regularization:
                    weight = weight.clone()
                    dist.all_reduce(
                        weight,
                        op=dist.ReduceOp.SUM,
                        group=group,
                    )

        numer_eff = numer
        denom_eff = denom
        weight_eff = weight
        if self.ema_decay > 0:
            self.ema_numer.mul_(self.ema_decay).add_(numer)
            self.ema_denom.mul_(self.ema_decay).add_(denom)
            numer_eff = self.ema_numer
            denom_eff = self.ema_denom
            if self.use_regularization:
                self.ema_weight.mul_(self.ema_decay).add_(weight)
                weight_eff = self.ema_weight

        new_variance = self.variance.clone()
        valid = denom_eff > 0
        if valid.any():
            new_variance[valid] = numer_eff[valid] / denom_eff[valid]

        if self.use_regularization:
            # The inflated prior follows \hat{w}_k = \hat{w}_0 * gamma^k,
            # so decay the running weight before using it at the current step.
            self.inflated_weight_eff.mul_(self.inflated_decay)
            inflated_weight_eff = float(self.inflated_weight_eff.item())
            total_weight = weight_eff + self.prior_weight + inflated_weight_eff
            reg_valid = total_weight > 0
            if reg_valid.any():
                new_variance[reg_valid] = (
                    weight_eff[reg_valid] * new_variance[reg_valid]
                    + self.prior_weight * self.prior_variance
                    + inflated_weight_eff * self.inflated_variance
                ) / total_weight[reg_valid]

        self.variance.copy_(new_variance)

    @torch.no_grad()
    def zero_accum(self, *, set_to_none: bool = False) -> None:
        """Reset accumulation buffers.

        Args:
            set_to_none: If ``True``, set accumulator buffers to ``None``.
                Otherwise, allocate (if needed) and fill with zeros.
        """
        if set_to_none:
            self.accum_numer = None
            self.accum_denom = None
            if self.use_regularization:
                self.accum_weight = None
            return

        if not self.requires_accum:
            return

        if self.accum_numer is None or self.accum_numer.shape != self.variance.shape:
            self.accum_numer = torch.zeros_like(self.variance, dtype=torch.float32)
        if self.accum_denom is None or self.accum_denom.shape != self.variance.shape:
            self.accum_denom = torch.zeros_like(self.variance, dtype=torch.float32)
        if (
            self.use_regularization
            and (self.accum_weight is None or self.accum_weight.shape != self.variance.shape)
        ):
            self.accum_weight = torch.zeros_like(self.variance, dtype=torch.float32)

        self.accum_numer.zero_()
        self.accum_denom.zero_()
        if self.use_regularization:
            self.accum_weight.zero_()

    @torch.no_grad()
    def _zero_ema(self) -> None:
        if self.ema_numer is not None:
            self.ema_numer.zero_()
        if self.ema_denom is not None:
            self.ema_denom.zero_()
        if self.ema_weight is not None:
            self.ema_weight.zero_()