from __future__ import annotations

import torch

from cryoseed.config import MainConfig
from cryoseed.optim.em import EMSolver
from cryoseed.state import OptimState


class _Volume:
    def __init__(self):
        self.backproject_kwargs = None
        self.update_count = 0

    def backproject(self, *args, **kwargs):
        self.backproject_kwargs = kwargs

    def update(self, *args, **kwargs):
        self.update_count += 1

    def zero_accum(self):
        pass


class _Noise:
    def __init__(self):
        self.accumulate_count = 0
        self.update_count = 0

    def variance_spectrum(self, *, ndim):
        assert ndim == 2
        return torch.ones((8, 8))

    def accumulate(self, *args, **kwargs):
        self.accumulate_count += 1

    def update(self):
        self.update_count += 1

    def zero_accum(self):
        pass


class _Pose:
    def update(self):
        pass

    def zero_accum(self):
        pass


class _Searcher:
    def __init__(self, config, state):
        self.config = config
        self.state = state
        self.volume = _Volume()
        self.noise = _Noise()
        self.pose = _Pose()
        self.pose_search_criterion = None

    def search(
        self,
        image,
        *,
        particle_index,
        ctf=None,
    ):
        self.pose_search_criterion = self.state.schedule.pose_search_criterion
        return (
            torch.ones(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.eye(3).unsqueeze(0),
            torch.zeros((1, 2)),
            None,
        )

    def clear_memory_cache(self):
        pass

    def refresh(self):
        pass


def _make_solver():
    config = MainConfig()
    config.homorefine.scheduler.first_epoch_ncc = True
    state = OptimState()
    state.progress.epoch = 0
    state.schedule.side_length = 8
    state.schedule.full_backprojection = True
    state.schedule.pose_search_criterion = "correlation"
    searcher = _Searcher(config, state)
    return EMSolver(state=state, pose_searcher=searcher), searcher


def test_first_epoch_ncc_controls_search_noise_and_backprojection_radius():
    solver, searcher = _make_solver()
    image = torch.ones((1, 8, 8), dtype=torch.complex64)
    particle_index = torch.zeros(1, dtype=torch.long)

    result = solver.expectation(image, particle_index=particle_index)
    assert solver.state.schedule.pose_search_criterion == "correlation"
    assert searcher.pose_search_criterion == "correlation"

    solver.maximization(
        image,
        prob=result[0],
        prob2img_idx=result[1],
        prob2vol_idx=result[2],
        rotmat=result[3],
        trans=result[4],
        radial_residual_power=result[5],
    )
    assert searcher.volume.backproject_kwargs["radius"] == 4
    assert searcher.noise.accumulate_count == 0

    solver.update()
    assert searcher.volume.update_count == 1
    assert searcher.noise.update_count == 0


def test_later_epochs_restore_likelihood_path_and_noise_updates():
    solver, searcher = _make_solver()
    solver.state.progress.epoch = 1
    solver.state.schedule.pose_search_criterion = "posterior"
    image = torch.ones((1, 8, 8), dtype=torch.complex64)
    particle_index = torch.zeros(1, dtype=torch.long)

    result = solver.expectation(image, particle_index=particle_index)
    assert solver.state.schedule.pose_search_criterion == "posterior"
    assert searcher.pose_search_criterion == "posterior"

    solver.maximization(
        image,
        prob=result[0],
        prob2img_idx=result[1],
        prob2vol_idx=result[2],
        rotmat=result[3],
        trans=result[4],
        radial_residual_power=result[5],
    )
    assert searcher.volume.backproject_kwargs["radius"] is None
    assert searcher.noise.accumulate_count == 1

    solver.update()
    assert searcher.noise.update_count == 1