from dataclasses import dataclass
from typing import Union, List

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
import torch


from cryoseed.fft.fft_numpy import fft3_freq, np_ft_to_real, np_real_to_ft
from cryoseed.fft.fft_torch import primal_to_fourier_3d, fourier_to_primal_3d
from cryoseed.ops.radial import radial_broadcast

__all__ = [
    "calc_cc",
    "calc_cc_numpy",
    "calc_corrected_fsc",
    "calc_corrected_fsc_numpy",
    "calc_fsc",
    "calc_fsc_numpy",
    "calc_solvent_corrected_fsc",
    "calc_solvent_corrected_fsc_numpy",
    "fsc_to_resolution",
    "fsc_to_resolution_numpy",
    "fsc_weight_1d",
    "apply_fsc_weighting_3d",
    "get_fsc_map",
    "get_fsc_map_numpy",
    "plot_fsc",
    "plot_fsc_multiple",
    "save_fsc_npz",
    "save_fsc_txt",
    "load_fsc_txt",
    "randomize_phases_beyond",
    "randomize_phases_beyond_numpy",
    "SolventCorrectedFSC",
]


# =============================================================================
# Shared Types And Helpers
# =============================================================================

@dataclass(frozen=True)
class SolventCorrectedFSC:
    fsc_freqs: torch.Tensor | np.ndarray
    corrected: torch.Tensor | np.ndarray
    unmasked: torch.Tensor | np.ndarray
    masked: torch.Tensor | np.ndarray
    randomized_masked: torch.Tensor | np.ndarray | None
    phase_randomization_frequency: float | None

def _in_mask_ndi_sum(data, mask=None, labels=None, index=None):
    if mask is None:
        data_masked = data
    else:
        data_masked = np.where(mask, data, 0)
    return ndimage.sum(data_masked, labels=labels, index=index)


def _to_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coerce_real_volume(
    x,
    *,
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if device is not None or dtype is not None:
            x = x.to(
                device=device if device is not None else x.device,
                dtype=dtype if dtype is not None else x.dtype,
            )
    else:
        x = torch.as_tensor(x, device=device, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float32 if dtype is None else dtype)
    return x

def _torch_fft3_radii(shape, *, device, dtype):
    nz, ny, nx = (int(s) for s in shape)
    kz = torch.fft.fftshift(torch.fft.fftfreq(nz, device=device, dtype=dtype))
    ky = torch.fft.fftshift(torch.fft.fftfreq(ny, device=device, dtype=dtype))
    kx = torch.fft.fftshift(torch.fft.fftfreq(nx, device=device, dtype=dtype))
    zz, yy, xx = torch.meshgrid(kz, ky, kx, indexing="ij")
    radii = torch.sqrt(zz.square() + yy.square() + xx.square())
    return kz, ky, kx, radii


_FSC_XTICK_CANDIDATES = (
    (0.01, "1/100"),
    (0.05, "1/20"),
    (0.0833, "1/12"),
    (0.125, "1/8"),
    (0.1667, "1/6"),
    (0.2, "1/5"),
    (0.25, "1/4"),
    (0.3333, "1/3"),
    (0.4, "1/2.5"),
    (0.5, "1/2"),
)
_FSC_YTICK_LOCS = (0.0, 0.125, 0.143, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
_FSC_YTICK_LABELS = ("0", " ", "0.143", "0.25", " ", "0.5", " ", "0.75", " ", "1.0")


def _normalize_thresholds(threshold: Union[List, float]) -> list[float]:
    if isinstance(threshold, float):
        return [threshold]
    return [float(t) for t in threshold]


def _fsc_axis_ticks(fsc_freqs) -> tuple[list[float], list[str]]:
    max_freq = float(np.max(fsc_freqs))
    xtick_locs = [freq for freq, _ in _FSC_XTICK_CANDIDATES if freq <= max_freq]
    xtick_labels = [label for freq, label in _FSC_XTICK_CANDIDATES if freq <= max_freq]
    return xtick_locs, xtick_labels


def _apply_fsc_axis_style(fsc_freqs) -> None:
    xtick_locs, xtick_labels = _fsc_axis_ticks(fsc_freqs)
    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("FSC")
    plt.xticks(xtick_locs, xtick_labels)
    plt.yticks(_FSC_YTICK_LOCS, _FSC_YTICK_LABELS)


# =============================================================================
# Torch Primary Implementations
# =============================================================================

def calc_fsc(vol0, vol1, fourier_mask=None, num_shells=None):
    """Calculate the FSC curve between two 3D torch half-maps."""
    if not isinstance(vol0, torch.Tensor):
        vol0 = torch.as_tensor(vol0)
    if not isinstance(vol1, torch.Tensor):
        vol1 = torch.as_tensor(vol1)

    if tuple(vol0.shape) != tuple(vol1.shape):
        raise AssertionError(f"Shape mis-match {tuple(vol0.shape)} vs {tuple(vol1.shape)}")
    if vol0.ndim != 3:
        raise ValueError(f"calc_fsc expects 3D volumes, got shape={tuple(vol0.shape)}")

    device = vol0.device
    real_dtype = torch.float64
    vol0 = vol0.to(dtype=real_dtype)
    vol1 = vol1.to(device=device, dtype=real_dtype)

    nx = int(vol0.shape[-1])
    if num_shells is None:
        num_shells = nx // 2 + 1
    num_shells = int(num_shells)

    vol0_ft = primal_to_fourier_3d(vol0)
    vol1_ft = primal_to_fourier_3d(vol1)

    kz, ky, kx, radii = _torch_fft3_radii(vol0.shape, device=device, dtype=real_dtype)
    max_radii = torch.stack((kz.abs().amax(), ky.abs().amax(), kx.abs().amax())).amin()
    shell_radii = torch.linspace(0, max_radii, num_shells, device=device, dtype=real_dtype)

    labels = torch.bucketize(radii.reshape(-1), shell_radii, right=False)
    cross = (vol0_ft * vol1_ft.conj()).real.reshape(-1)
    power0 = vol0_ft.abs().square().reshape(-1)
    power1 = vol1_ft.abs().square().reshape(-1)

    selection = labels < num_shells
    if fourier_mask is not None:
        selection = selection & torch.as_tensor(
            fourier_mask,
            device=device,
            dtype=torch.bool,
        ).reshape(-1)

    labels = labels[selection]
    cross = cross[selection]
    power0 = power0[selection]
    power1 = power1[selection]

    numerator = torch.bincount(labels, weights=cross, minlength=num_shells)[1:]
    denom0 = torch.bincount(labels, weights=power0, minlength=num_shells)[1:]
    denom1 = torch.bincount(labels, weights=power1, minlength=num_shells)[1:]
    denominator = torch.sqrt(denom0 * denom1)

    fsc_scores = torch.zeros(num_shells - 1, device=device, dtype=real_dtype)
    valid = denominator > 1e-3
    fsc_scores[valid] = numerator[valid] / denominator[valid]
    fsc_freqs = shell_radii[1:]
    return fsc_scores, fsc_freqs


def randomize_phases_beyond(
    volume: torch.Tensor,
    frequency: float,
    *,
    seed: int,
) -> torch.Tensor:
    volume = _coerce_real_volume(volume, name="volume")
    frequency = float(frequency)
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    real_dtype = torch.float64 if volume.dtype == torch.float64 else torch.float32
    volume = volume.to(dtype=real_dtype)
    fourier = primal_to_fourier_3d(volume)
    _, _, _, radii = _torch_fft3_radii(
        volume.shape,
        device=volume.device,
        dtype=real_dtype,
    )

    generator = torch.Generator(device=volume.device)
    generator.manual_seed(int(seed))
    random_field = torch.randn(
        volume.shape,
        generator=generator,
        device=volume.device,
        dtype=real_dtype,
    )
    random_fourier = primal_to_fourier_3d(random_field)
    random_phase = random_fourier / random_fourier.abs().clamp_min(
        torch.finfo(real_dtype).eps
    )

    randomized_fourier = fourier.clone()
    high_frequency = radii >= frequency
    randomized_fourier[high_frequency] = (
        fourier.abs()[high_frequency] * random_phase[high_frequency]
    )
    randomized = fourier_to_primal_3d(randomized_fourier).real
    return randomized.to(dtype=volume.dtype)


def calc_corrected_fsc(
    masked_fsc,
    randomized_masked_fsc,
    *,
    correction_start_index: int,
):
    """Apply phase-randomized FSC correction to a masked FSC curve."""
    if not isinstance(masked_fsc, torch.Tensor):
        masked_fsc = torch.as_tensor(masked_fsc)
    real_dtype = torch.float64 if masked_fsc.dtype == torch.float64 else torch.float32
    masked_fsc = masked_fsc.to(dtype=real_dtype).reshape(-1)
    randomized_masked_fsc = torch.as_tensor(
        randomized_masked_fsc,
        device=masked_fsc.device,
        dtype=real_dtype,
    ).reshape(-1)
    if int(masked_fsc.numel()) != int(randomized_masked_fsc.numel()):
        raise ValueError("masked_fsc and randomized_masked_fsc must have the same length")

    correction_start = min(int(masked_fsc.numel()), max(0, int(correction_start_index)))
    corrected_fsc = masked_fsc.clone()
    numerator = masked_fsc[correction_start:] - randomized_masked_fsc[correction_start:]
    denominator = 1.0 - randomized_masked_fsc[correction_start:]
    valid = (
        (denominator > 1e-8)
        & (randomized_masked_fsc[correction_start:] <= masked_fsc[correction_start:])
    )
    corrected_tail = torch.zeros_like(numerator)
    corrected_tail[valid] = numerator[valid] / denominator[valid]
    corrected_fsc[correction_start:] = corrected_tail
    return corrected_fsc


def calc_solvent_corrected_fsc(
    unmasked_vol0: torch.Tensor,
    unmasked_vol1: torch.Tensor,
    mask: torch.Tensor,
    *,
    randomize_threshold: float = 0.8,
    seed: int = 0,
    num_shells: int | None = None,
) -> SolventCorrectedFSC:
    device = unmasked_vol0.device
    unmasked_vol0 = _coerce_real_volume(
        unmasked_vol0,
        name="unmasked_vol0",
        device=device,
    )
    real_dtype = (
        torch.float64 if unmasked_vol0.dtype == torch.float64 else torch.float32
    )
    unmasked_vol0 = unmasked_vol0.to(dtype=real_dtype)
    unmasked_vol1 = _coerce_real_volume(
        unmasked_vol1,
        name="unmasked_vol1",
        device=unmasked_vol0.device,
        dtype=real_dtype,
    )
    mask = _coerce_real_volume(
        mask,
        name="mask",
        device=unmasked_vol0.device,
        dtype=real_dtype,
    )
    if (
        unmasked_vol0.shape != unmasked_vol1.shape
        or unmasked_vol0.shape != mask.shape
    ):
        raise ValueError(
            "unmasked_vol0, unmasked_vol1, and mask must have identical shapes, got "
            f"{tuple(unmasked_vol0.shape)}, {tuple(unmasked_vol1.shape)}, and {tuple(mask.shape)}"
        )
    if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
        raise ValueError("mask values must be finite and within [0, 1]")

    unmasked_fsc, fsc_freqs = calc_fsc(
        unmasked_vol0,
        unmasked_vol1,
        num_shells=num_shells,
    )
    masked_fsc, masked_fsc_freqs = calc_fsc(
        unmasked_vol0 * mask,
        unmasked_vol1 * mask,
        num_shells=num_shells,
    )
    if not torch.equal(fsc_freqs, masked_fsc_freqs):
        raise RuntimeError("masked and unmasked FSC frequencies do not match")

    below = torch.nonzero(unmasked_fsc < float(randomize_threshold), as_tuple=False)
    if int(below.numel()) == 0:
        return SolventCorrectedFSC(
            fsc_freqs=fsc_freqs,
            corrected=unmasked_fsc.clone(),
            unmasked=unmasked_fsc,
            masked=masked_fsc,
            randomized_masked=None,
            phase_randomization_frequency=None,
        )

    phase_randomization_index = int(below[0, 0].item())
    phase_randomization_frequency = float(fsc_freqs[phase_randomization_index].item())
    randomized_vol0 = randomize_phases_beyond(
        unmasked_vol0,
        phase_randomization_frequency,
        seed=int(seed),
    )
    randomized_vol1 = randomize_phases_beyond(
        unmasked_vol1,
        phase_randomization_frequency,
        seed=int(seed) + 1,
    )
    randomized_masked_fsc, randomized_fsc_freqs = calc_fsc(
        randomized_vol0 * mask,
        randomized_vol1 * mask,
        num_shells=num_shells,
    )
    if not torch.equal(fsc_freqs, randomized_fsc_freqs):
        raise RuntimeError("randomized and unmasked FSC frequencies do not match")

    corrected_fsc = calc_corrected_fsc(
        masked_fsc,
        randomized_masked_fsc,
        correction_start_index=phase_randomization_index + 2,
    )

    return SolventCorrectedFSC(
        fsc_freqs=fsc_freqs,
        corrected=corrected_fsc,
        unmasked=unmasked_fsc,
        masked=masked_fsc,
        randomized_masked=randomized_masked_fsc,
        phase_randomization_frequency=phase_randomization_frequency,
    )


# if all fsc is small than threshold, this function will output the resolution defined by the minimum frequency
def fsc_to_resolution(fsc_scores, fsc_freqs=None, threshold=0.143, angpix=1.0):
    if not isinstance(fsc_scores, torch.Tensor):
        fsc_scores = torch.as_tensor(fsc_scores, dtype=torch.float32)
    fsc_scores = fsc_scores.reshape(-1)

    has_fsc_freqs = fsc_freqs is not None
    if fsc_freqs is None:
        fsc_freqs = torch.arange(
            int(fsc_scores.numel()),
            device=fsc_scores.device,
            dtype=fsc_scores.dtype,
        )
    elif not isinstance(fsc_freqs, torch.Tensor):
        fsc_freqs = torch.as_tensor(
            fsc_freqs,
            device=fsc_scores.device,
            dtype=fsc_scores.dtype,
        )
    else:
        fsc_freqs = fsc_freqs.to(device=fsc_scores.device, dtype=fsc_scores.dtype)
    fsc_freqs = fsc_freqs.reshape(-1)

    if int(fsc_scores.numel()) != int(fsc_freqs.numel()):
        raise AssertionError("fsc_scores and fsc_freqs must have the same length")

    below_threshold = fsc_scores <= float(threshold)
    below_idx = below_threshold.nonzero(as_tuple=False)
    cross_index = (
        int(below_idx[0, 0].item())
        if int(below_idx.numel()) > 0
        else int(fsc_scores.numel()) - 1
    )

    if not has_fsc_freqs:
        return cross_index

    freq = fsc_freqs[cross_index]
    if (
        cross_index > 0
        and float(fsc_scores[cross_index - 1].item()) > float(threshold)
        and float(fsc_scores[cross_index].item()) < float(threshold)
    ):
        frac = (float(threshold) - float(fsc_scores[cross_index].item())) / (
            float(fsc_scores[cross_index - 1].item())
            - float(fsc_scores[cross_index].item())
        )
        freq = fsc_freqs[cross_index] * (1.0 - frac) + fsc_freqs[cross_index - 1] * frac

    return float(angpix) / float(freq.item())


def fsc_weight_1d(fsc_scores, num_shells: int, eps: float = 1e-6):
    fsc = torch.as_tensor(fsc_scores, dtype=torch.float32).reshape(-1)
    num_shells = int(num_shells)
    if num_shells <= 0:
        raise ValueError(f"num_shells must be > 0, got {num_shells}")

    if int(fsc.numel()) == num_shells - 1:
        fsc = torch.cat((fsc.new_tensor([1.0]), fsc), dim=0)
    elif int(fsc.numel()) != num_shells:
        raise ValueError(
            f"fsc_scores must have length {num_shells} or {num_shells - 1}, got {int(fsc.numel())}"
        )

    fsc = fsc.clamp(min=0.0, max=1.0 - float(eps))
    weight = torch.sqrt((2.0 * fsc) / (1.0 + fsc).clamp_min(float(eps)))
    weight[0] = 1.0
    return weight


def apply_fsc_weighting_3d(volume_real, fsc_scores, eps: float = 1e-6):
    volume_real = torch.as_tensor(volume_real)
    if volume_real.ndim != 3:
        raise ValueError(f"volume_real must be 3D, got shape={tuple(volume_real.shape)}")
    if not (
        int(volume_real.shape[0]) == int(volume_real.shape[1]) == int(volume_real.shape[2])
    ):
        raise ValueError(f"volume_real must be cubic, got shape={tuple(volume_real.shape)}")

    side_length = int(volume_real.shape[-1])
    num_shells = side_length // 2 + 1
    weight_1d = fsc_weight_1d(fsc_scores, num_shells=num_shells, eps=eps).to(
        device=volume_real.device,
        dtype=volume_real.dtype,
    )
    weight_3d = radial_broadcast(
        weight_1d,
        ndim=3,
        out_len=side_length,
        padding_mode="border",
    )
    volume = primal_to_fourier_3d(volume_real)
    return fourier_to_primal_3d(volume * weight_3d).real


def calc_cc(vol0, vol1, eps=1e-7):
    if not isinstance(vol0, torch.Tensor):
        vol0 = torch.as_tensor(vol0)
    if not isinstance(vol1, torch.Tensor):
        vol1 = torch.as_tensor(vol1)
    if tuple(vol0.shape) != tuple(vol1.shape):
        raise AssertionError(f"Shape mis-match {tuple(vol0.shape)} vs {tuple(vol1.shape)}")

    device = vol0.device
    real_dtype = (
        torch.float64
        if vol0.dtype == torch.float64 or vol1.dtype == torch.float64
        else torch.float32
    )
    vol0 = vol0.to(dtype=real_dtype).reshape(-1)
    vol1 = vol1.to(device=device, dtype=real_dtype).reshape(-1)

    norm0 = torch.linalg.vector_norm(vol0, ord=2)
    norm1 = torch.linalg.vector_norm(vol1, ord=2)
    return torch.sum(vol0 * vol1) / (norm0 * norm1 + float(eps))


def get_fsc_map(vol0, vol1, mask):
    """Adapted from spisonet."""
    if not isinstance(vol0, torch.Tensor):
        vol0 = torch.as_tensor(vol0)
    if not isinstance(vol1, torch.Tensor):
        vol1 = torch.as_tensor(vol1)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if tuple(vol0.shape) != tuple(vol1.shape) or tuple(vol0.shape) != tuple(mask.shape):
        raise ValueError(
            "vol0, vol1, and mask must have identical shapes, got "
            f"{tuple(vol0.shape)}, {tuple(vol1.shape)}, and {tuple(mask.shape)}"
        )

    device = vol0.device
    real_dtype = (
        torch.float64
        if vol0.dtype == torch.float64 or vol1.dtype == torch.float64 or mask.dtype == torch.float64
        else torch.float32
    )
    masked_vol0 = vol0.to(dtype=real_dtype) * mask.to(device=device, dtype=real_dtype)
    masked_vol1 = vol1.to(device=device, dtype=real_dtype) * mask.to(device=device, dtype=real_dtype)
    fourier0 = primal_to_fourier_3d(masked_vol0)
    fourier1 = primal_to_fourier_3d(masked_vol1)
    numerator = (fourier0 * fourier1.conj()).real
    power0 = (fourier0 * fourier0.conj()).real
    power1 = (fourier1 * fourier1.conj()).real
    return numerator / torch.sqrt(power0 * power1)


# =============================================================================
# NumPy Implementations
# =============================================================================

def calc_fsc_numpy(vol0, vol1, fourier_mask=None, num_shells=None):
    """Calculate the FSC curve between two 3D NumPy half-maps."""
    assert vol0.shape == vol1.shape, f"Shape mis-match {vol0.shape} vs {vol1.shape}"

    nz, ny, nx = vol0.shape
    if num_shells is None:
        num_shells = nx // 2 + 1

    fourier0 = np_real_to_ft(vol0)
    fourier1 = np_real_to_ft(vol1)

    kz, ky, kx, radii = fft3_freq((nz, ny, nx))
    max_radii = min(np.abs(kz).max(), np.abs(ky).max(), np.abs(kx).max())
    shell_radii = np.linspace(0, max_radii, num_shells, endpoint=True)

    labels = np.searchsorted(shell_radii, radii, side="left")
    index = np.arange(1, len(shell_radii))
    shell_count = len(index)

    numerator = _in_mask_ndi_sum(
        np.real(fourier0 * np.conj(fourier1)),
        mask=fourier_mask,
        labels=labels,
        index=index,
    )
    denominator = np.sqrt(
        _in_mask_ndi_sum(
            np.abs(fourier0) ** 2,
            mask=fourier_mask,
            labels=labels,
            index=index,
        )
        * _in_mask_ndi_sum(
            np.abs(fourier1) ** 2,
            mask=fourier_mask,
            labels=labels,
            index=index,
        )
    )

    fsc_scores = np.zeros(shell_count)
    for i in range(shell_count):
        if denominator[i] > 1e-3:
            fsc_scores[i] = numerator[i] / denominator[i]

    fsc_freqs = shell_radii[1:]
    return fsc_scores, fsc_freqs


def randomize_phases_beyond_numpy(
    volume: np.ndarray,
    frequency: float,
    *,
    seed: int,
) -> np.ndarray:
    frequency = float(frequency)
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {tuple(volume.shape)}")

    real_dtype = np.float64 if volume.dtype == np.float64 else np.float32
    volume_np = np.asarray(volume, dtype=real_dtype)
    fourier = np_real_to_ft(volume_np)
    _, _, _, radii = fft3_freq(volume_np.shape)

    rng = np.random.default_rng(int(seed))
    random_field = rng.standard_normal(volume_np.shape).astype(real_dtype, copy=False)
    random_fourier = np_real_to_ft(random_field)
    eps = np.finfo(real_dtype).eps
    random_phase = random_fourier / np.clip(np.abs(random_fourier), eps, None)

    randomized_fourier = fourier.copy()
    high_frequency = radii >= frequency
    randomized_fourier[high_frequency] = (
        np.abs(fourier)[high_frequency] * random_phase[high_frequency]
    )
    randomized = np_ft_to_real(randomized_fourier).real
    return randomized.astype(volume.dtype, copy=False)


def calc_corrected_fsc_numpy(
    masked_fsc,
    randomized_masked_fsc,
    *,
    correction_start_index: int,
) -> np.ndarray:
    """Apply phase-randomized FSC correction to a masked NumPy FSC curve."""
    masked_fsc = np.asarray(masked_fsc)
    randomized_masked_fsc = np.asarray(randomized_masked_fsc, dtype=masked_fsc.dtype)
    if masked_fsc.ndim != 1 or randomized_masked_fsc.ndim != 1:
        raise ValueError("masked_fsc and randomized_masked_fsc must be 1D")
    if masked_fsc.shape != randomized_masked_fsc.shape:
        raise ValueError("masked_fsc and randomized_masked_fsc must have the same shape")

    correction_start = min(masked_fsc.size, max(0, int(correction_start_index)))
    corrected_fsc = masked_fsc.copy()
    numerator = masked_fsc[correction_start:] - randomized_masked_fsc[correction_start:]
    denominator = 1.0 - randomized_masked_fsc[correction_start:]
    valid = (
        (denominator > 1e-8)
        & (randomized_masked_fsc[correction_start:] <= masked_fsc[correction_start:])
    )
    corrected_tail = np.zeros_like(numerator)
    corrected_tail[valid] = numerator[valid] / denominator[valid]
    corrected_fsc[correction_start:] = corrected_tail
    return corrected_fsc


def calc_solvent_corrected_fsc_numpy(
    unmasked_vol0: np.ndarray,
    unmasked_vol1: np.ndarray,
    mask: np.ndarray,
    *,
    randomize_threshold: float = 0.8,
    seed: int = 0,
    num_shells: int | None = None,
) -> SolventCorrectedFSC:
    unmasked_vol0 = np.asarray(unmasked_vol0)
    unmasked_vol1 = np.asarray(unmasked_vol1)
    mask = np.asarray(mask)
    if (
        unmasked_vol0.shape != unmasked_vol1.shape
        or unmasked_vol0.shape != mask.shape
    ):
        raise ValueError(
            "unmasked_vol0, unmasked_vol1, and mask must have identical shapes, got "
            f"{tuple(unmasked_vol0.shape)}, {tuple(unmasked_vol1.shape)}, and {tuple(mask.shape)}"
        )
    if unmasked_vol0.ndim != 3:
        raise ValueError(
            f"unmasked volumes must be 3D, got shape {tuple(unmasked_vol0.shape)}"
        )
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("mask values must be finite and within [0, 1]")

    unmasked_fsc, fsc_freqs = calc_fsc_numpy(
        unmasked_vol0,
        unmasked_vol1,
        num_shells=num_shells,
    )
    masked_fsc, masked_fsc_freqs = calc_fsc_numpy(
        unmasked_vol0 * mask,
        unmasked_vol1 * mask,
        num_shells=num_shells,
    )
    if not np.array_equal(fsc_freqs, masked_fsc_freqs):
        raise RuntimeError("masked and unmasked FSC frequencies do not match")

    below = np.flatnonzero(unmasked_fsc < float(randomize_threshold))
    if below.size == 0:
        return SolventCorrectedFSC(
            fsc_freqs=fsc_freqs,
            corrected=unmasked_fsc.copy(),
            unmasked=unmasked_fsc,
            masked=masked_fsc,
            randomized_masked=None,
            phase_randomization_frequency=None,
        )

    phase_randomization_index = int(below[0])
    phase_randomization_frequency = float(fsc_freqs[phase_randomization_index])
    randomized_vol0 = randomize_phases_beyond_numpy(
        unmasked_vol0,
        phase_randomization_frequency,
        seed=int(seed),
    )
    randomized_vol1 = randomize_phases_beyond_numpy(
        unmasked_vol1,
        phase_randomization_frequency,
        seed=int(seed) + 1,
    )
    randomized_masked_fsc, randomized_fsc_freqs = calc_fsc_numpy(
        randomized_vol0 * mask,
        randomized_vol1 * mask,
        num_shells=num_shells,
    )
    if not np.array_equal(fsc_freqs, randomized_fsc_freqs):
        raise RuntimeError("randomized and unmasked FSC frequencies do not match")

    corrected_fsc = calc_corrected_fsc_numpy(
        masked_fsc,
        randomized_masked_fsc,
        correction_start_index=phase_randomization_index + 2,
    )

    return SolventCorrectedFSC(
        fsc_freqs=fsc_freqs,
        corrected=corrected_fsc,
        unmasked=unmasked_fsc,
        masked=masked_fsc,
        randomized_masked=randomized_masked_fsc,
        phase_randomization_frequency=phase_randomization_frequency,
    )


# if all fsc is small than threshold, this function will output the resolution defined by the minimum frequency
def fsc_to_resolution_numpy(fsc_scores, fsc_freqs=None, threshold=0.143, angpix=1.0):
    if fsc_freqs is not None:
        assert len(fsc_scores) == len(fsc_freqs)

    below_threshold = fsc_scores <= threshold
    cross_index = (
        np.where(below_threshold)[0][0]
        if np.any(below_threshold)
        else len(fsc_scores) - 1
    )

    if fsc_freqs is None:
        return cross_index

    freq = fsc_freqs[cross_index]
    if (
        cross_index > 0
        and fsc_scores[cross_index - 1] > threshold > fsc_scores[cross_index]
    ):
        frac = (threshold - fsc_scores[cross_index]) / (
            fsc_scores[cross_index - 1] - fsc_scores[cross_index]
        )
        freq = (
            fsc_freqs[cross_index] * (1.0 - frac)
            + fsc_freqs[cross_index - 1] * frac
        )

    return angpix / freq


def calc_cc_numpy(vol0, vol1, eps=1e-7):
    assert vol0.shape == vol1.shape, f"Shape mis-match {vol0.shape} vs {vol1.shape}"

    vol0 = vol0.ravel()
    vol1 = vol1.ravel()

    norm0 = np.linalg.norm(vol0, 2)
    norm1 = np.linalg.norm(vol1, 2)

    return np.sum(vol0 * vol1) / (norm0 * norm1 + eps)


def get_fsc_map_numpy(vol0, vol1, mask):
    """Adapted from spisonet."""
    masked_vol0 = vol0 * mask
    masked_vol1 = vol1 * mask
    fourier0 = np_real_to_ft(masked_vol0)
    fourier1 = np_real_to_ft(masked_vol1)
    numerator = np.real(np.multiply(fourier0, np.conj(fourier1)))
    power0 = np.real(np.multiply(fourier0, np.conj(fourier0)))
    power1 = np.real(np.multiply(fourier1, np.conj(fourier1)))
    return numerator / np.sqrt(power0 * power1)


# =============================================================================
# Visualization
# =============================================================================

def plot_fsc(
    fsc_scores,
    fsc_freqs,
    threshold: Union[List, float] = 0.143,
    angpix=1.0,
    save_path=None,
    fontname=None,
    color=None,
):
    """Plot a single FSC curve and annotate threshold-based resolutions."""
    fsc_scores = _to_numpy_array(fsc_scores)
    fsc_freqs = _to_numpy_array(fsc_freqs)

    if fontname is not None:
        plt.rcParams["font.family"] = fontname

    thresholds = _normalize_thresholds(threshold)

    res_list = []
    for t in thresholds:
        res = fsc_to_resolution(fsc_scores, fsc_freqs, t, angpix)
        res_list.append(res)
        freq = angpix / res
        plt.annotate(
            r"{:1.2f} $\AA$".format(res),
            xy=(freq, t),
            xytext=(freq, t + 0.05),
            fontsize=12.0,
            # arrowprops={"width": 1, "shrink": .05}
        )
        plt.axhline(t, color="k", linestyle=":")

    plt.plot(fsc_freqs, fsc_scores, linewidth=1, color=color)
    _apply_fsc_axis_style(fsc_freqs)
    plt.title("FSC curves", fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
    plt.cla()
    plt.close()
    return res_list


def plot_fsc_multiple(
    fsc_list,
    fsc_freqs,
    labels=None,
    threshold: Union[List, float] = 0.143,
    angpix=1.0,
    save_path=None,
    fontname=None,
    colors=None,
):
    """Plot multiple FSC curves on the same axes."""
    fsc_list = [_to_numpy_array(fsc) for fsc in fsc_list]
    fsc_freqs = _to_numpy_array(fsc_freqs)

    if fontname is not None:
        plt.rcParams["font.family"] = fontname

    thresholds = _normalize_thresholds(threshold)
    if labels is not None and len(labels) != len(fsc_list):
        raise ValueError("labels and fsc_list must have the same length")
    if colors is not None and len(colors) != len(fsc_list):
        raise ValueError("colors and fsc_list must have the same length")

    for t in thresholds:
        plt.axhline(t, color="k", linestyle=":")

    for i, fsc in enumerate(fsc_list):
        if labels is None:
            base_label = f"Curve {i}"
        else:
            base_label = labels[i]
        resolution = fsc_to_resolution(fsc, fsc_freqs, thresholds[0], angpix)
        label = f"{base_label} ({resolution:.2f} A)"
        color = None if colors is None else colors[i]
        plt.plot(fsc_freqs, fsc, linewidth=1.2, label=label, color=color)

    _apply_fsc_axis_style(fsc_freqs)
    plt.title("FSC Curves")
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.cla()
    plt.close()

# =============================================================================
# Serialization And File I/O
# =============================================================================

def save_fsc_npz(
    path: str | Path,
    fsc_freqs,
    fsc,
    *,
    epoch=None,
    resolution=None,
    resolution_05=None,
    fsc_unmasked=None,
    fsc_masked=None,
    fsc_randomized_masked=None,
    fsc_corrected=None,
    phase_randomization_frequency=None,
    fsc_gt=None,
):
    """Save FSC data and optional metadata to an NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "fsc_freqs": _to_numpy_array(fsc_freqs).astype(np.float32, copy=False),
        "fsc": _to_numpy_array(fsc).astype(np.float32, copy=False),
    }

    if epoch is not None:
        data["epoch"] = int(epoch)
    if resolution is not None:
        data["resolution_0143"] = float(resolution)
    if resolution_05 is not None:
        data["resolution_05"] = float(resolution_05)
    if fsc_unmasked is not None:
        data["fsc_unmasked"] = _to_numpy_array(fsc_unmasked).astype(np.float32, copy=False)
    if fsc_masked is not None:
        data["fsc_masked"] = _to_numpy_array(fsc_masked).astype(np.float32, copy=False)
    if fsc_randomized_masked is not None:
        data["fsc_randomized_masked"] = _to_numpy_array(fsc_randomized_masked).astype(
            np.float32,
            copy=False,
        )
    if fsc_corrected is not None:
        data["fsc_corrected"] = _to_numpy_array(fsc_corrected).astype(np.float32, copy=False)
    if phase_randomization_frequency is not None:
        data["phase_randomization_frequency"] = float(phase_randomization_frequency)
    if fsc_gt is not None:
        data["fsc_unmasked"] = _to_numpy_array(fsc_gt).astype(np.float32, copy=False)

    np.savez(path, **data)


def load_fsc_npz(path: str | Path):
    """Load FSC data and metadata from an NPZ file."""
    path = Path(path)

    data = np.load(path)

    fsc_freqs = (
        data["fsc_freqs"]
        if "fsc_freqs" in data
        else data["frequency"] if "frequency" in data else data["freq"]
    )
    fsc = data["fsc"]

    result = {
        "fsc_freqs": fsc_freqs,
        "fsc": fsc,
    }

    # optional fields
    if "epoch" in data:
        result["epoch"] = int(data["epoch"])
    elif "iter" in data:
        result["epoch"] = int(data["iter"])
    if "resolution_0143" in data:
        result["resolution"] = float(data["resolution_0143"])
    if "resolution_05" in data:
        result["resolution_05"] = float(data["resolution_05"])
    if "fsc_corrected" in data:
        result["fsc_corrected"] = data["fsc_corrected"]
    if "fsc_unmasked" in data:
        result["fsc_unmasked"] = data["fsc_unmasked"]
    elif "fsc_gt" in data:
        result["fsc_unmasked"] = data["fsc_gt"]
    if "fsc_masked" in data:
        result["fsc_masked"] = data["fsc_masked"]
    if "fsc_randomized_masked" in data:
        result["fsc_randomized_masked"] = data["fsc_randomized_masked"]
    if "phase_randomization_frequency" in data:
        result["phase_randomization_frequency"] = float(data["phase_randomization_frequency"])

    return result


def save_fsc_txt(
    path: str | Path,
    fsc_freqs,
    fsc_scores,
    *,
    epoch=None,
    resolution=None,
    fsc_unmasked=None,
    fsc_masked=None,
    fsc_randomized_masked=None,
    fsc_corrected=None,
    phase_randomization_frequency: float | None = None,
):
    """Save FSC data to a plain-text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fsc_freqs = _to_numpy_array(fsc_freqs).astype(np.float32, copy=False).reshape(-1)
    fsc_scores = _to_numpy_array(fsc_scores).astype(np.float32, copy=False).reshape(-1)

    if len(fsc_freqs) != len(fsc_scores):
        raise ValueError("fsc_freqs and fsc_scores must have the same length")

    columns = {
        "fsc_freqs": fsc_freqs,
        "fsc_scores": fsc_scores,
    }
    optional_columns = (
        ("fsc_unmasked", fsc_unmasked),
        ("fsc_masked", fsc_masked),
        ("fsc_randomized_masked", fsc_randomized_masked),
        ("fsc_corrected", fsc_corrected),
    )
    for name, values in optional_columns:
        if values is None:
            continue
        column = _to_numpy_array(values).astype(np.float32, copy=False).reshape(-1)
        if len(column) != len(fsc_freqs):
            raise ValueError(f"{name} must have the same length as fsc_freqs")
        columns[name] = column

    with open(path, "w") as f:
        # ===== header =====
        f.write("# CryoSeed FSC\n")
        if epoch is not None:
            f.write(f"# epoch: {epoch}\n")
        if resolution is not None:
            f.write(f"# resolution(0.143): {resolution:.2f} Angstrom\n")
        if phase_randomization_frequency is not None:
            f.write(f"# phase_randomization_frequency: {phase_randomization_frequency:.6f}\n")

        column_names = list(columns.keys())
        f.write(f"# columns: {' '.join(column_names)}\n")

        # ===== data =====
        row_count = len(fsc_freqs)
        for idx in range(row_count):
            row = " ".join(f"{columns[name][idx]:10.6f}" for name in column_names)
            f.write(f"{row}\n")


def load_fsc_txt(path: str | Path):
    """Load FSC data and metadata from a plain-text file."""
    path = Path(path)

    columns = {"fsc_freqs": [], "fsc": []}

    meta = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # ===== header =====
            if line.startswith("#"):
                if "epoch:" in line:
                    meta["epoch"] = int(line.split(":", 1)[1].strip())
                elif "iter:" in line:
                    meta["epoch"] = int(line.split(":", 1)[1].strip())
                elif "resolution" in line:
                    val = line.split(":")[1].replace("Angstrom", "").strip()
                    meta["resolution"] = float(val)
                elif "fsc_label:" in line:
                    meta["fsc_label"] = line.split(":", 1)[1].strip()
                elif "phase_randomization_frequency:" in line:
                    meta["phase_randomization_frequency"] = float(
                        line.split(":", 1)[1].strip()
                    )
                elif "columns:" in line:
                    column_names = line.split(":", 1)[1].strip().split()
                    columns = {name: [] for name in column_names}
                continue

            # ===== data =====
            parts = line.split()
            if len(parts) >= len(columns):
                for name, value in zip(columns.keys(), parts):
                    columns[name].append(float(value))

    result = {
        name: np.array(values, dtype=np.float32)
        for name, values in columns.items()
    }
    if "fsc" in result and "fsc_scores" not in result:
        result["fsc_scores"] = result["fsc"]
    if "fsc_scores" in result and "fsc" not in result:
        result["fsc"] = result["fsc_scores"]
    return {**result, **meta}