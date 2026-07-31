from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cryoseed.config import MainConfig

from cryoseed.fft.coords import fftindex_coords3d
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.ops.radial import radial_average, radial_broadcast

__all__ = [
    "PriorVariance",
]


class PriorVariance(nn.Module):
    """Maintain a 1D radial prior power spectrum (variance) estimate for a 3D volume.

    The spectrum is stored in :attr:`variance` with shape ``(R,)`` where
    ``R = image_size // 2 + 1`` corresponds to integer Fourier radii
    ``0..image_size//2`` on an fftshifted grid.

    Notes:
        - All spectra are stored as real-valued tensors.
        - Precision is defined as ``1 / clamp(variance, min=precision_eps)``.
    """

    @classmethod
    def from_config(
        cls,
        config: MainConfig,
        device: torch.device | str | None = None,
    ) -> PriorVariance | None:
        if not config.modules.statistics.prior.enabled:
            return None
        return cls(
            image_size=int(config.data.image_size),
            init_volume=None,
            init_lowpass_cutoff=config.modules.statistics.prior.init_lowpass_cutoff,
            device=device,
            init_variance=float(config.modules.statistics.prior.init_variance),
            tail_floor=float(config.modules.statistics.prior.tail_floor),
            precision_eps=float(config.modules.statistics.prior.precision_eps),
        )

    def __init__(
        self,
        image_size: int,
        *,
        init_volume: Tensor | None = None,
        init_lowpass_cutoff: int | None = None,
        device: torch.device | str | None = None,
        init_variance: float = 1.0,
        tail_floor: float = 1e-5,
        precision_eps: float = 1e-6,
    ):
        """Create a :class:`PriorVariance`.

        Args:
            image_size: Side length ``D`` of the cubic Fourier grid.
            init_volume: Optional initial Fourier volume used to initialize
                :attr:`variance`.
            init_lowpass_cutoff: Optional low-pass size ``L`` used for the
                exponential-tail initialization. The tail is applied outside
                radius ``L // 2``.
            device: Device used to initialize buffers.
            init_variance: Initial constant variance value.
            tail_floor: Exponential tail value at the maximum Fourier radius.
            precision_eps: Clamp used when forming precision ``1 / variance``.
        """
        super().__init__()
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")

        self.init_lowpass_cutoff = None if init_lowpass_cutoff is None else int(init_lowpass_cutoff)
        if self.init_lowpass_cutoff is not None:
            if self.init_lowpass_cutoff <= 0:
                raise ValueError(
                    f"init_lowpass_cutoff must be > 0, got {self.init_lowpass_cutoff}"
                )
            if self.init_lowpass_cutoff > self.image_size:
                raise ValueError(
                    "init_lowpass_cutoff must be <= image_size, got "
                    f"{self.init_lowpass_cutoff} vs {self.image_size}"
                )

        self.tail_floor = float(tail_floor)

        dev = _norm_device(device)
        self.register_buffer("_device_anchor", torch.empty(0, device=dev), persistent=False)

        self.init_variance = float(init_variance)
        self.precision_eps = float(precision_eps)

        if self.init_variance <= 0:
            raise ValueError(f"init_variance must be > 0, got {self.init_variance}")
        if self.precision_eps <= 0:
            raise ValueError(f"precision_eps must be > 0, got {self.precision_eps}")

        self.num_radial_bins = self.image_size // 2 + 1

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

        if init_volume is not None:
            self.from_volume(init_volume)

    @torch.no_grad()
    def from_volume(self, volume: Tensor) -> None:
        """Initialize :attr:`variance` from a Fourier-space volume.

        Computes ``0.5 * |V|^2`` and radially averages it over a centered 3D FFT grid.
        If :attr:`init_lowpass_cutoff` is provided, values outside radius
        ``init_lowpass_cutoff // 2`` are replaced by an exponential tail.

        Args:
            volume: Complex tensor with shape ``(D, D, D)`` or ``(K, D, D, D)`` where
                ``D == image_size`` and (if present) ``K == 1``.
        """
        D = int(self.image_size)

        if volume.ndim == 4:
            K = int(volume.shape[0])
            if K != 1:
                raise ValueError(f"volume must have K==1 when given as (K,D,D,D), got K={K}")
            volume = volume[0]
        elif volume.ndim != 3:
            raise ValueError(
                f"volume must be (D,D,D) or (K,D,D,D) with K==1, got {tuple(volume.shape)}"
            )

        if tuple(volume.shape) != (D, D, D):
            raise ValueError(f"volume must have shape (D,D,D)=({D},{D},{D}), got {tuple(volume.shape)}")
        if volume.dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"volume must be complex, got {volume.dtype}")

        volume = volume.detach()
        if volume.device != self.device:
            volume = volume.to(self.device)
        volume = volume.clone()

        power = 0.5 * volume.abs().square()

        if self.init_lowpass_cutoff is not None:
            radius = int(self.init_lowpass_cutoff) // 2
            coords = fftindex_coords3d(D, device=self.device)
            x = coords[:, 0].to(dtype=torch.float32)
            y = coords[:, 1].to(dtype=torch.float32)
            z = coords[:, 2].to(dtype=torch.float32)
            r = torch.sqrt((x * x) + (y * y) + (z * z)).view(D, D, D)

            alpha = -math.log(self.tail_floor) / (math.sqrt(3.0) * (D // 2))
            tail = torch.exp(-alpha * r).to(dtype=power.dtype)
            mask = r > float(radius)

            tail = tail.view(*([1] * (power.ndim - 3)), D, D, D)
            mask = mask.view(*([1] * (power.ndim - 3)), D, D, D)
            power = torch.where(mask, tail, power)

        radial = radial_average(power, max_radius=D // 2, ndim=3, use_cache=True)
        if radial.ndim > 1:
            radial = radial.reshape(-1, radial.shape[-1]).mean(dim=0)
        self.variance.copy_(radial.to(dtype=self.variance.dtype))

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    @torch.no_grad()
    def fill_(self, value: float) -> None:
        """Fill the stored variance spectrum with a constant value."""
        value = float(value)
        if value <= 0:
            raise ValueError(f"value must be > 0, got {value}")
        self.variance.fill_(value)

    @property
    def precision(self) -> Tensor:
        """Return the radial precision spectrum ``1 / variance``."""
        return 1.0 / self.variance.clamp_min(self.precision_eps)

    def variance_spectrum(
        self,
        ndim: int,
        side_length: int | None = None,
        max_radius: int | None = None,
        padding_mode: str = "zeros",
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
        padding_mode: str = "zeros",
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
    def update(self, fsc_values: Tensor, weight: Tensor) -> None:
        """Update :attr:`variance` from FSC and backprojection weights.

        The update uses the spectral SNR estimate ``SSNR = FSC / (1 - FSC)`` and
        sets:

        ``variance = SSNR / (weight + precision_eps)``.

        Args:
            fsc_values: 1D tensor of FSC values with shape ``(R,)`` or ``(R-1,)``.
                If ``(R-1,)`` is provided, a fixed DC value is prepended.
            weight: 1D tensor of radially-averaged backprojection weights with shape
                ``(R,)``.
        """
        fsc = torch.as_tensor(fsc_values, device=self.device, dtype=torch.float32)
        w = torch.as_tensor(weight, device=self.device, dtype=torch.float32)

        if torch.any(w < -1e-9):
            raise ValueError("weight must be nonnegative")
        w = w.clamp_min(0.0)

        if fsc.ndim != 1 or w.ndim != 1:
            raise ValueError("fsc_values and weight must be 1D tensors")

        R = int(self.variance.numel())
        if int(fsc.numel()) == R - 1:
            dc_fsc = 0.999
            fsc = torch.cat((fsc.new_tensor([dc_fsc]), fsc), dim=0)

        if int(fsc.numel()) != R:
            raise ValueError(f"fsc_values must have length {R} (or {R-1}), got {int(fsc.numel())}")
        if int(w.numel()) != R:
            raise ValueError(f"weight must have length {R}, got {int(w.numel())}")

        fsc = fsc.clamp(min=0.001, max=0.999)
        ssnr = fsc / (1.0 - fsc)
        new_var = ssnr / (w + self.precision_eps)
        self.variance.copy_(new_var.to(dtype=self.variance.dtype))