from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from cryoseed.config import MainConfig
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.fft.fft_torch import primal_to_fourier_2d
from cryoseed.ops.radial import radial_average, radial_broadcast
from cryoseed.ops.transforms import translate_image
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
    ) -> NoiseVariance | None:
        if not config.statistics.use_noise:
            return None
        return cls(
            image_size=int(config.data.image_size),
            device=device,
            device_mesh=device_mesh,
            requires_accum=bool(config.reconstruction.requires_accum),
            accumulate_chunk=int(config.reconstruction.accumulate_chunk),
            init_variance=float(config.statistics.init_variance),
            precision_eps=float(config.statistics.precision_eps),
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

        if self.accumulate_chunk <= 0:
            raise ValueError(f"accumulate_chunk must be > 0, got {self.accumulate_chunk}")
        if self.init_variance <= 0:
            raise ValueError(f"init_variance must be > 0, got {self.init_variance}")
        if self.precision_eps <= 0:
            raise ValueError(f"precision_eps must be > 0, got {self.precision_eps}")

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
            torch.zeros_like(self.variance),
            persistent=False,
        )
        self.register_buffer(
            "accum_denom",
            torch.tensor([0.0], dtype=torch.float32, device=dev),
            persistent=False,
        )

    @torch.no_grad()
    def fill_(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"value must be > 0, got {value}")
        self.variance.fill_(value)
        self.zero_accum()

    @torch.no_grad()
    def from_data(self, dataloaders: list[DataLoader]) -> None:
        """Recompute the noise variance spectrum from data.

        In the current pipeline, dataloaders yield :class:`~cryoseed.data.DataBatch`.
        We prefer using ``batch.image`` directly (Fourier domain, fftshift convention)
        to avoid recomputing FFTs.

        The per-frequency variance estimate is:

        ``Var[F] = E[|F|^2] - |E[F]|^2``

        followed by a 2D radial average.

        Args:
            dataloaders: A list of dataloaders yielding :class:`~cryoseed.data.DataBatch`.
        """
        D = int(self.image_size)
        power_sum = torch.zeros((D, D), dtype=torch.float32, device=self.device)
        image_sum = torch.zeros((D, D), dtype=torch.complex64, device=self.device)
        num_images = 0

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

        self.variance.zero_()
        if num_images <= 0:
            return

        mean_power = power_sum / float(num_images)
        mean_image = image_sum / float(num_images)
        var2d = mean_power - mean_image.abs().square()
        var2d = var2d.clamp_min(0.0)
        var1d = radial_average(var2d.unsqueeze(0), max_radius=D // 2, ndim=2, use_cache=True)
        self.variance.copy_(var1d.squeeze(0))

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
    def accumulate(
        self,
        image: Tensor,
        ctf: Tensor | None,
        volume: Volume,
        probability: Tensor,
        image_index: torch.LongTensor,
        volume_index: torch.LongTensor,
        rotation: Tensor,
        translation: Tensor,
    ) -> None:
        """Accumulate a weighted residual-based estimate of the noise variance.

        This computes (per pose hypothesis) the squared residual in Fourier space:

        ``|proj(volume, R) - translate(image, t)|^2``

        and accumulates a probability-weighted radial average into internal
        numerator/denominator buffers for later normalization by :meth:`update`.

        Args:
            image: Fourier-domain images with shape ``(B, D, D)``.
            ctf: Optional CTF with shape ``(B, D, D)``.
            volume: Volume module exposing ``project(rotation, side_length=...)``.
            image_index: Pose-to-image mapping with shape ``(N,)``.
            volume_index: Pose-to-volume mapping with shape ``(N,)``.
            rotation: Rotation matrices with shape ``(N, 3, 3)``.
            translation: Translations with shape ``(N, 2)`` in pixels.
            probability: Per-pose weights with shape ``(N,)``.
        """
        if not self.requires_accum:
            return

        if image.dim() != 3 or image.shape[1] != image.shape[2]:
            raise ValueError(f"image must be (B,D,D), got {tuple(image.shape)}")

        B = int(image.shape[0])
        D = int(image.shape[1])
        if D != int(self.image_size):
            raise ValueError(f"image side length {D} does not match image_size {int(self.image_size)}")

        N = int(image_index.numel())
        if volume_index.numel() != N:
            raise ValueError("volume_index must have the same number of elements as image_index")
        if rotation.shape[0] != N or rotation.shape[-2:] != (3, 3):
            raise ValueError(f"rotation must be (N,3,3) with N={N}, got {tuple(rotation.shape)}")
        if translation.shape != (N, 2):
            raise ValueError(f"translation must be (N,2)=({N},2), got {tuple(translation.shape)}")
        if probability.shape != (N,):
            raise ValueError(f"probability must be (N,)=({N},), got {tuple(probability.shape)}")

        if ctf is not None and ctf.shape != image.shape:
            raise ValueError(f"ctf must have the same shape as image (B,D,D), got {tuple(ctf.shape)}")

        noise_var_per_img = torch.zeros(
            (B, self.num_radial_bins),
            device=self.device,
            dtype=torch.float32,
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

            rot_kq = rot.view(1, chunk_size, 3, 3).expand(K, -1, -1, -1).contiguous()
            proj_all = volume.project(rot_kq, side_length=D)
            proj_image = proj_all[vol_idx, torch.arange(chunk_size, device=device)]

            if ctf is not None:
                proj_image = proj_image * ctf[img_idx]

            err2 = (proj_image - trans_image).abs().square()
            err2_1d = radial_average(err2, max_radius=D // 2, ndim=2, use_cache=True)
            noise_var_per_pose = 0.5 * prob.view(-1, 1) * err2_1d

            noise_var_per_img.scatter_add_(
                0,
                img_idx.view(chunk_size, 1).expand(-1, self.num_radial_bins),
                noise_var_per_pose,
            )

        self.accum_numer += noise_var_per_img.sum(dim=0).to(self.accum_numer.dtype)
        self.accum_denom += B

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
            return

        if not self.requires_accum:
            return

        if self.accum_numer is None or self.accum_numer.shape != self.variance.shape:
            self.accum_numer = torch.zeros_like(self.variance, dtype=torch.float32)
        if self.accum_denom is None or self.accum_denom.numel() != 1:
            self.accum_denom = torch.tensor([0.0], dtype=torch.float32, device=self.device)

        self.accum_numer.zero_()
        self.accum_denom.zero_()

    @torch.no_grad()
    def update(self) -> None:
        """Finalize :attr:`variance` from accumulated numerator/denominator.

        If ``torch.distributed`` is initialized, this synchronizes the accumulators
        across the data-parallel group (``device_mesh.get_group(0)`` if available,
        otherwise ``WORLD``).
        """
        if (not self.requires_accum) or self.accum_numer is None or self.accum_denom is None:
            return

        numer = self.accum_numer
        denom = self.accum_denom

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

        self.variance.zero_()
        if denom.item() > 0:
            torch.div(numer, denom, out=self.variance)