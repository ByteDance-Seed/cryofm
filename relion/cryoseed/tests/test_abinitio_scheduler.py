from __future__ import annotations

from cryoseed.optim.scheduler.abinitio import AbInitioScheduler
from cryoseed.state import OptimState


def test_abinitio_convergence_switches_to_final_mode_within_epoch():
    state = OptimState()
    state.schedule.side_length = 100
    state.schedule.healpix_order = 3
    state.schedule.trans_grid_extent = 5.0
    state.abinitio.metrics.ema_rot_update_rms = 0.0
    state.abinitio.metrics.ema_trans_update_rms = 0.0

    scheduler = AbInitioScheduler(
        state,
        device="cpu",
        image_size=200,
        angpix=1.0,
        particle_diameter=200.0,
        trans_grid_samples=5,
        convergence_patience=1,
        target_side_length_resolution=10.0,
        target_healpix_order=3,
    )

    scheduler.step()

    assert state.abinitio.scheduler.has_converged is True
    assert state.abinitio.engine.is_final_epoch is True
    assert state.schedule.full_backprojection is False
    assert state.abinitio.engine.skip_external_reconstruct is True


def test_abinitio_gradually_aligns_healpix_order_to_current_side_length():
    state = OptimState()
    state.schedule.side_length = 10
    state.schedule.healpix_order = 0
    state.schedule.trans_grid_extent = 5.0

    scheduler = AbInitioScheduler(
        state,
        device="cpu",
        image_size=200,
        angpix=1.0,
        particle_diameter=200.0,
        trans_grid_samples=5,
        convergence_patience=1,
        target_side_length_resolution=10.0,
    )

    assert scheduler._required_healpix_order_for_side_length(10) == 2

    scheduler.step()
    assert state.schedule.healpix_order == 1
    assert state.abinitio.scheduler.num_checks_with_stable_side_length == 0
    assert state.abinitio.scheduler.num_checks_with_stable_pose == 0
    assert state.abinitio.scheduler.num_checks_ready_to_stop == 0


def test_abinitio_derives_translation_grid_samples_from_resolution_and_extent():
    state = OptimState()
    state.schedule.side_length = 200
    state.schedule.trans_grid_extent = 5.0
    state.schedule.trans_grid_samples = 5

    scheduler = AbInitioScheduler(
        state,
        device="cpu",
        image_size=200,
        angpix=1.0,
        particle_diameter=200.0,
        trans_grid_samples=5,
        convergence_patience=1,
        target_side_length_resolution=10.0,
    )

    assert scheduler._required_trans_grid_samples_for_side_length(200) == 11


def test_abinitio_translation_grid_samples_respect_configured_minimum():
    state = OptimState()
    state.schedule.side_length = 40
    state.schedule.trans_grid_extent = 5.0
    state.schedule.trans_grid_samples = 5

    scheduler = AbInitioScheduler(
        state,
        device="cpu",
        image_size=200,
        angpix=1.0,
        particle_diameter=200.0,
        trans_grid_samples=5,
        convergence_patience=1,
        target_side_length_resolution=10.0,
    )

    assert scheduler._required_trans_grid_samples_for_side_length(40) == 5


def test_abinitio_switches_search_grad_mode_at_confidence_threshold():
    state = OptimState()
    scheduler = AbInitioScheduler(
        state,
        device="cpu",
        image_size=200,
        angpix=1.0,
        particle_diameter=200.0,
        confidence_threshold=0.5,
    )

    state.abinitio.metrics.avg_confidence = 0.5
    scheduler.step()
    assert state.schedule.search_grad_mode == "full"

    state.abinitio.metrics.avg_confidence = 0.5001
    scheduler.step()
    assert state.schedule.search_grad_mode == "selected"