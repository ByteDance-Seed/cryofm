from __future__ import annotations

import pytest
import torch

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import masked_lerp, particle_mask
from cryoseed.data import DataBatch
from cryoseed.fft.fft_torch import primal_to_fourier_2d
from cryoseed.optim.em import EMSolver
from cryoseed.optim.pose.pose_searcher import PoseSearcher
from cryoseed.state import OptimState


class _CaptureVolume:
    def __init__(self):
        self.backproject_image = None

    def backproject(self, image, *_args, **_kwargs):
        del _args, _kwargs
        self.backproject_image = image

    def update(self, *_args, **_kwargs):
        del _args, _kwargs
        pass

    def zero_accum(self):
        pass


class _Pose:
    def __init__(self, translation=None):
        if translation is None:
            translation = torch.zeros((1, 2), dtype=torch.float32)
        self._translation = translation

    @property
    def device(self):
        return self._translation.device

    def translation(self, index):
        return self._translation[index]

    def update(self):
        pass

    def zero_accum(self):
        pass


class _CaptureNoise:
    def __init__(self, background_noise):
        self.background_noise = background_noise
        self.sample_seed = None

    def sample_like(self, _image_real, *, seed):
        del _image_real
        self.sample_seed = seed
        return self.background_noise

    def variance_spectrum(self, *, ndim):
        assert ndim == 2
        return torch.ones(self.background_noise.shape[-2:])

    def accumulate(self, *_args, **_kwargs):
        del _args, _kwargs
        pass

    def update(self):
        pass

    def zero_accum(self):
        pass


class _CapturePoseSearcher:
    preprocess_image = PoseSearcher.preprocess_image
    search = PoseSearcher.search

    class _InnerSearcher:
        def __init__(self):
            self.search_image = None

        def search(
            self,
            image,
            *,
            _particle_index,
            _ctf=None,
        ):
            del _particle_index, _ctf
            self.search_image = image
            device = image.device
            return (
                torch.ones((1,), device=device),
                torch.zeros((1,), dtype=torch.long, device=device),
                torch.zeros((1,), dtype=torch.long, device=device),
                torch.eye(3, device=device).unsqueeze(0),
                torch.zeros((1, 2), device=device),
                None,
            )

    def __init__(self, config: MainConfig, noise=None, translation=None):
        self.config = config
        self.volume = _CaptureVolume()
        self.noise = noise
        self.pose = _Pose(translation=translation)
        self.pose_searcher = self._InnerSearcher()

    @property
    def search_image(self):
        return self.pose_searcher.search_image

    def clear_memory_cache(self):
        pass

    def refresh(self):
        pass


def _make_solver_and_batch(
    *,
    mask_enabled: bool,
    zero_mask: bool = True,
    with_noise_model: bool = True,
    translation: torch.Tensor | None = None,
):
    config = MainConfig()
    config.data.image_size = 10
    config.data.angpix = 2.0
    config.data.particle_diameter = 8.0
    config.modules.search.particle_mask.enabled = mask_enabled
    config.modules.search.particle_mask.zero_mask = zero_mask
    config.modules.search.particle_mask.soft_edge_pixels = 2.0

    torch.manual_seed(4)
    image_real = torch.randn((1, 10, 10), dtype=torch.float32)
    image = primal_to_fourier_2d(image_real)
    batch = DataBatch(
        image=image,
        image_real=image_real,
        particle_index=torch.zeros((1,), dtype=torch.long),
    )

    noise = None
    if not zero_mask and with_noise_model:
        noise = _CaptureNoise(torch.full_like(image_real, -2.0))
    searcher = _CapturePoseSearcher(config, noise=noise, translation=translation)
    solver = EMSolver(
        state=OptimState(),
        pose_searcher=searcher,
        prior=None,
    )
    return solver, searcher, batch


def test_pose_search_uses_masked_particle_and_backprojection_uses_unmasked_particle():
    solver, searcher, batch = _make_solver_and_batch(mask_enabled=True)

    result = solver.infer(batch)
    mask = particle_mask(
        batch.image_real.shape[-2],
        batch.image_real.shape[-1],
        particle_diameter=8.0,
        angpix=2.0,
        soft_edge_pixels=2.0,
        device=batch.image_real.device,
        dtype=batch.image_real.real.dtype,
    )
    expected_search_real = masked_lerp(batch.image_real, mask.unsqueeze(0))
    expected_search = primal_to_fourier_2d(expected_search_real)

    torch.testing.assert_close(searcher.search_image, expected_search)
    assert not torch.equal(searcher.search_image, batch.image)
    assert result.image is batch.image
    assert not hasattr(solver, "config")

    solver.accumulate(result)
    assert searcher.volume.backproject_image is batch.image


def test_noise_masking_requires_noise_estimation():
    solver, _, batch = _make_solver_and_batch(
        mask_enabled=True,
        zero_mask=False,
        with_noise_model=False,
    )

    with pytest.raises(ValueError, match="modules\\.statistics\\.noise\\.enabled"):
        solver.infer(batch)