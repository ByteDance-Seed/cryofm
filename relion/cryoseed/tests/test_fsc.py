from pathlib import Path

import numpy as np
import pytest
import torch

from cryoseed.config import MainConfig
from cryoseed.fft.fft_torch import primal_to_fourier_3d
from cryoseed.metrics.fsc import (
    calc_cc,
    calc_cc_numpy,
    calc_fsc,
    calc_fsc_numpy,
    calc_solvent_corrected_fsc,
    calc_solvent_corrected_fsc_numpy,
    fsc_to_resolution,
    fsc_to_resolution_numpy,
    get_fsc_map,
    get_fsc_map_numpy,
    load_fsc_npz,
    load_fsc_txt,
    randomize_phases_beyond,
    randomize_phases_beyond_numpy,
    save_fsc_npz,
    save_fsc_txt,
)


def _make_correlated_volumes(shape, seed=0):
    rng = np.random.default_rng(seed)
    vol0 = rng.standard_normal(shape, dtype=np.float32)
    vol1 = vol0 + 0.1 * rng.standard_normal(shape, dtype=np.float32)
    return vol0, vol1


def _make_fourier_mask(shape, cutoff=0.2):
    kz = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(shape[2]))
    zz, yy, xx = np.meshgrid(kz, ky, kx, indexing="ij")
    radii = np.sqrt(zz**2 + yy**2 + xx**2)
    return radii <= cutoff


def _calc_fsc_numpy_reference(vol1, vol2, num_shells=None):
    vol1 = np.asarray(vol1, dtype=np.float64)
    vol2 = np.asarray(vol2, dtype=np.float64)
    if num_shells is None:
        num_shells = vol1.shape[-1] // 2 + 1

    f1 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol1)))
    f2 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol2)))

    kz = np.fft.fftshift(np.fft.fftfreq(vol1.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(vol1.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(vol1.shape[2]))
    zz, yy, xx = np.meshgrid(kz, ky, kx, indexing="ij")
    radii = np.sqrt(zz**2 + yy**2 + xx**2)
    max_radii = min(np.abs(kz).max(), np.abs(ky).max(), np.abs(kx).max())
    shell_radii = np.linspace(0.0, max_radii, num_shells, endpoint=True)
    labels = np.searchsorted(shell_radii, radii, side="left")
    valid = labels < num_shells
    labels = labels[valid]

    cross = np.real(f1 * np.conj(f2))[valid]
    power1 = (np.abs(f1) ** 2)[valid]
    power2 = (np.abs(f2) ** 2)[valid]

    numerator = np.bincount(labels, weights=cross, minlength=num_shells)[1:]
    denom1 = np.bincount(labels, weights=power1, minlength=num_shells)[1:]
    denom2 = np.bincount(labels, weights=power2, minlength=num_shells)[1:]
    denominator = np.sqrt(denom1 * denom2)

    fsc = np.zeros(num_shells - 1, dtype=np.float64)
    stable = denominator > 1e-3
    fsc[stable] = numerator[stable] / denominator[stable]
    return fsc, shell_radii[1:]


def _randomize_phases_beyond_reference(volume, frequency, *, seed):
    volume = np.asarray(volume, dtype=np.float64)
    fourier = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
    kz = np.fft.fftshift(np.fft.fftfreq(volume.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(volume.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(volume.shape[2]))
    zz, yy, xx = np.meshgrid(kz, ky, kx, indexing="ij")
    radii = np.sqrt(zz**2 + yy**2 + xx**2)

    rng = np.random.default_rng(int(seed))
    random_field = rng.standard_normal(volume.shape)
    random_fourier = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(random_field)))
    random_phase = np.exp(1j * np.angle(random_fourier))

    randomized_fourier = fourier.copy()
    high_frequency = radii >= float(frequency)
    randomized_fourier[high_frequency] = (
        np.abs(fourier[high_frequency]) * random_phase[high_frequency]
    )
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(randomized_fourier))
    ).real.astype(volume.dtype, copy=False)


def _calc_solvent_corrected_fsc_reference(vol1, vol2, mask, *, randomize_threshold=0.8, seed=0):
    vol1 = np.asarray(vol1, dtype=np.float64)
    vol2 = np.asarray(vol2, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    unmasked, fsc_freqs = _calc_fsc_numpy_reference(vol1, vol2)
    masked, _ = _calc_fsc_numpy_reference(vol1 * mask, vol2 * mask)

    below = np.flatnonzero(unmasked < float(randomize_threshold))
    if below.size == 0:
        return {
            "fsc_freqs": fsc_freqs,
            "corrected": unmasked.copy(),
            "unmasked": unmasked,
            "masked": masked,
            "randomized_masked": None,
            "phase_randomization_frequency": None,
        }

    phase_randomization_index = int(below[0])
    phase_randomization_frequency = float(fsc_freqs[phase_randomization_index])
    randomized1 = _randomize_phases_beyond_reference(
        vol1,
        phase_randomization_frequency,
        seed=int(seed),
    )
    randomized2 = _randomize_phases_beyond_reference(
        vol2,
        phase_randomization_frequency,
        seed=int(seed) + 1,
    )
    randomized_masked, _ = _calc_fsc_numpy_reference(randomized1 * mask, randomized2 * mask)

    corrected = masked.copy()
    correction_start = min(corrected.size, phase_randomization_index + 2)
    numerator = masked[correction_start:] - randomized_masked[correction_start:]
    denominator = 1.0 - randomized_masked[correction_start:]
    valid = (denominator > 1e-8) & (randomized_masked[correction_start:] <= masked[correction_start:])
    corrected_tail = np.zeros_like(numerator)
    corrected_tail[valid] = numerator[valid] / denominator[valid]
    corrected[correction_start:] = corrected_tail

    return {
        "fsc_freqs": fsc_freqs,
        "corrected": corrected,
        "unmasked": unmasked,
        "masked": masked,
        "randomized_masked": randomized_masked,
        "phase_randomization_frequency": phase_randomization_frequency,
    }


@pytest.mark.parametrize(
    ("shape", "use_mask"),
    [
        ((16, 16, 16), False),
        ((16, 16, 16), True),
        ((17, 17, 17), False),
        ((17, 17, 17), True),
    ],
)
def test_calc_fsc_torch_matches_numpy(shape, use_mask):
    vol0_np, vol1_np = _make_correlated_volumes(shape, seed=sum(shape) + int(use_mask))
    fourier_mask = _make_fourier_mask(shape) if use_mask else None

    fsc_np, fsc_freqs_np = calc_fsc_numpy(vol0_np, vol1_np, fourier_mask=fourier_mask)
    fsc_torch, fsc_freqs_torch = calc_fsc(
        torch.from_numpy(vol0_np),
        torch.from_numpy(vol1_np),
        fourier_mask=fourier_mask,
    )

    np.testing.assert_allclose(
        fsc_freqs_torch.detach().cpu().numpy(),
        fsc_freqs_np,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fsc_torch.detach().cpu().numpy(),
        fsc_np,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("shape", "threshold", "angpix"),
    [
        ((16, 16, 16), 0.143, 1.5),
        ((17, 17, 17), 0.5, 0.83),
    ],
)
def test_fsc_to_resolution_torch_matches_numpy(shape, threshold, angpix):
    vol0_np, vol1_np = _make_correlated_volumes(shape, seed=123 + sum(shape))
    fsc_np, fsc_freqs_np = calc_fsc_numpy(vol0_np, vol1_np)
    fsc_torch, fsc_freqs_torch = calc_fsc(
        torch.from_numpy(vol0_np),
        torch.from_numpy(vol1_np),
    )

    res_np = fsc_to_resolution_numpy(
        fsc_np,
        fsc_freqs_np,
        threshold=threshold,
        angpix=angpix,
    )
    res_torch = fsc_to_resolution(
        fsc_torch,
        fsc_freqs_torch,
        threshold=threshold,
        angpix=angpix,
    )

    assert res_torch == pytest.approx(res_np, rel=1e-6, abs=1e-6)


def test_fsc_to_resolution_explicit_fsc_freqs_matches_numpy():
    fsc_np = np.array([0.9, 0.6, 0.2, 0.1, 0.05], dtype=np.float32)
    fsc_freqs_np = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)
    fsc_torch = torch.from_numpy(fsc_np)
    fsc_freqs_torch = torch.from_numpy(fsc_freqs_np)

    res_np = fsc_to_resolution_numpy(fsc_np, fsc_freqs=fsc_freqs_np, threshold=0.143, angpix=1.0)
    res_torch = fsc_to_resolution(fsc_torch, fsc_freqs=fsc_freqs_torch, threshold=0.143, angpix=1.0)

    assert res_torch == pytest.approx(res_np, rel=1e-6, abs=1e-6)


def test_primal_to_fourier_3d_preserves_float64_precision():
    volume = torch.randn(9, 9, 9, dtype=torch.float64)
    ft = primal_to_fourier_3d(volume)

    assert ft.dtype == torch.complex128


def test_solvent_corrected_fsc_torch_matches_reference_numpy_logic():
    vol0_np, vol1_np = _make_correlated_volumes((15, 15, 15), seed=77)
    mask = np.zeros((15, 15, 15), dtype=np.float32)
    mask[3:12, 3:12, 3:12] = 1.0

    expected = _calc_solvent_corrected_fsc_reference(
        vol0_np,
        vol1_np,
        mask,
        randomize_threshold=0.95,
        seed=19,
    )
    actual = calc_solvent_corrected_fsc(
        torch.from_numpy(vol0_np),
        torch.from_numpy(vol1_np),
        torch.from_numpy(mask),
        randomize_threshold=0.95,
        seed=19,
    )
    actual_numpy = calc_solvent_corrected_fsc_numpy(
        vol0_np,
        vol1_np,
        mask,
        randomize_threshold=0.95,
        seed=19,
    )

    np.testing.assert_allclose(actual.fsc_freqs.detach().cpu().numpy(), expected["fsc_freqs"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.unmasked.detach().cpu().numpy(), expected["unmasked"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual.masked.detach().cpu().numpy(), expected["masked"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual.corrected.detach().cpu().numpy(), expected["corrected"], rtol=1e-5, atol=1e-5)
    assert actual.randomized_masked is not None
    np.testing.assert_allclose(
        actual.randomized_masked.detach().cpu().numpy(),
        expected["randomized_masked"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert actual.phase_randomization_frequency == pytest.approx(
        expected["phase_randomization_frequency"],
        rel=1e-6,
        abs=1e-6,
    )
    np.testing.assert_allclose(actual_numpy.fsc_freqs, expected["fsc_freqs"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_numpy.corrected, expected["corrected"], rtol=1e-5, atol=1e-5)


def test_save_fsc_txt_keeps_fsc_scores_as_primary_curve_and_writes_fsc_corrected(tmp_path):
    path = tmp_path / "fsc.txt"
    fsc_freqs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fsc_scores = np.array([0.9, 0.4, 0.2], dtype=np.float32)
    fsc_unmasked = np.array([0.95, 0.5, 0.25], dtype=np.float32)
    fsc_masked = np.array([0.92, 0.45, 0.22], dtype=np.float32)
    fsc_randomized_masked = np.array([0.0, 0.05, 0.1], dtype=np.float32)
    fsc_corrected = np.array([0.9, 0.42, 0.13], dtype=np.float32)

    save_fsc_txt(
        path,
        fsc_freqs,
        fsc_scores,
        epoch=7,
        resolution=3.25,
        fsc_unmasked=fsc_unmasked,
        fsc_masked=fsc_masked,
        fsc_randomized_masked=fsc_randomized_masked,
        fsc_corrected=fsc_corrected,
        phase_randomization_frequency=0.2,
    )

    text = path.read_text()
    assert "# fsc_label:" not in text
    assert (
        "# columns: fsc_freqs fsc_scores fsc_unmasked fsc_masked "
        "fsc_randomized_masked fsc_corrected"
    ) in text

    loaded = load_fsc_txt(path)
    np.testing.assert_allclose(loaded["fsc_freqs"], fsc_freqs)
    np.testing.assert_allclose(loaded["fsc_scores"], fsc_scores)
    np.testing.assert_allclose(loaded["fsc"], fsc_scores)
    np.testing.assert_allclose(loaded["fsc_unmasked"], fsc_unmasked)
    np.testing.assert_allclose(loaded["fsc_masked"], fsc_masked)
    np.testing.assert_allclose(loaded["fsc_randomized_masked"], fsc_randomized_masked)
    np.testing.assert_allclose(loaded["fsc_corrected"], fsc_corrected)
    assert loaded["epoch"] == 7
    assert loaded["resolution"] == pytest.approx(3.25)
    assert loaded["phase_randomization_frequency"] == pytest.approx(0.2)


def test_save_fsc_npz_writes_epoch_and_round_trips_optional_curves(tmp_path):
    path = tmp_path / "fsc.npz"
    fsc_freqs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fsc_scores = np.array([0.9, 0.4, 0.2], dtype=np.float32)
    fsc_unmasked = np.array([0.95, 0.5, 0.25], dtype=np.float32)
    fsc_masked = np.array([0.92, 0.45, 0.22], dtype=np.float32)
    fsc_randomized_masked = np.array([0.0, 0.05, 0.1], dtype=np.float32)
    fsc_corrected = np.array([0.9, 0.42, 0.13], dtype=np.float32)

    save_fsc_npz(
        path,
        fsc_freqs,
        fsc_scores,
        epoch=7,
        resolution=3.25,
        fsc_unmasked=fsc_unmasked,
        fsc_masked=fsc_masked,
        fsc_randomized_masked=fsc_randomized_masked,
        fsc_corrected=fsc_corrected,
        phase_randomization_frequency=0.2,
    )

    raw = np.load(path)
    assert "epoch" in raw
    assert "iter" not in raw
    assert int(raw["epoch"]) == 7

    loaded = load_fsc_npz(path)
    np.testing.assert_allclose(loaded["fsc_freqs"], fsc_freqs)
    np.testing.assert_allclose(loaded["fsc"], fsc_scores)
    np.testing.assert_allclose(loaded["fsc_unmasked"], fsc_unmasked)
    np.testing.assert_allclose(loaded["fsc_masked"], fsc_masked)
    np.testing.assert_allclose(loaded["fsc_randomized_masked"], fsc_randomized_masked)
    np.testing.assert_allclose(loaded["fsc_corrected"], fsc_corrected)
    assert loaded["epoch"] == 7
    assert loaded["resolution"] == pytest.approx(3.25)
    assert loaded["phase_randomization_frequency"] == pytest.approx(0.2)


def test_load_fsc_npz_maps_legacy_iter_to_epoch(tmp_path):
    path = tmp_path / "legacy_fsc.npz"
    np.savez(
        path,
        fsc_freqs=np.array([0.1, 0.2], dtype=np.float32),
        fsc=np.array([0.9, 0.3], dtype=np.float32),
        iter=np.array(5, dtype=np.int32),
    )

    loaded = load_fsc_npz(path)

    assert loaded["epoch"] == 5
    np.testing.assert_allclose(loaded["fsc_freqs"], np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(loaded["fsc"], np.array([0.9, 0.3], dtype=np.float32))


def test_randomize_phases_beyond_torch_is_primary_and_numpy_wrapper_matches():
    volume_np, _ = _make_correlated_volumes((9, 9, 9), seed=5)
    randomized_torch = randomize_phases_beyond(torch.from_numpy(volume_np), 0.2, seed=3)
    randomized_np = randomize_phases_beyond_numpy(volume_np, 0.2, seed=3)

    assert isinstance(randomized_torch, torch.Tensor)
    assert isinstance(randomized_np, np.ndarray)
    np.testing.assert_allclose(randomized_torch.detach().cpu().numpy(), randomized_np, rtol=1e-5, atol=1e-5)


def test_calc_cc_torch_matches_numpy():
    vol0_np, vol1_np = _make_correlated_volumes((11, 11, 11), seed=9)

    cc_np = calc_cc_numpy(vol0_np, vol1_np)
    cc_torch = calc_cc(torch.from_numpy(vol0_np), torch.from_numpy(vol1_np))

    assert isinstance(cc_torch, torch.Tensor)
    assert cc_torch.ndim == 0
    assert float(cc_torch.item()) == pytest.approx(float(cc_np), rel=1e-6, abs=1e-6)


def test_get_fsc_map_torch_matches_numpy():
    vol0_np, vol1_np = _make_correlated_volumes((9, 9, 9), seed=21)
    mask = np.zeros((9, 9, 9), dtype=np.float32)
    mask[2:7, 2:7, 2:7] = 1.0

    fsc_map_np = get_fsc_map_numpy(vol0_np, vol1_np, mask)
    fsc_map_torch = get_fsc_map(
        torch.from_numpy(vol0_np),
        torch.from_numpy(vol1_np),
        torch.from_numpy(mask),
    )

    assert isinstance(fsc_map_torch, torch.Tensor)
    np.testing.assert_allclose(
        fsc_map_torch.detach().cpu().numpy(),
        fsc_map_np,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )


def test_full_config_loads_without_removed_reconstruction_flags():
    full_config_path = Path(__file__).resolve().parents[1] / "full_config.yaml"
    cfg = MainConfig.from_file(str(full_config_path))

    assert not hasattr(cfg.reconstruction, "requires_grad")
    assert not hasattr(cfg.reconstruction, "requires_accum")