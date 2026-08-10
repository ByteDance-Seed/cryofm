from __future__ import annotations

import pytest
import torch

from cryoseed.modules.volume.voxel import VoxelGrid


def test_voxel_grid_owns_default_optimizer_and_scheduler() -> None:
    volume = VoxelGrid(
        grid_size=4,
        requires_accum=False,
        requires_grad=True,
        learning_rate=0.5,
        learning_rate_decay=0.25,
        momentum=0.0,
    )

    assert isinstance(volume.optimizer, torch.optim.SGD)
    assert isinstance(
        volume.lr_scheduler,
        torch.optim.lr_scheduler.ExponentialLR,
    )
    assert volume.optimizer.param_groups[0]["lr"] == 0.5

    volume.update_lr()
    assert volume.optimizer.param_groups[0]["lr"] == 0.125

    previous_scheduler = volume.lr_scheduler
    volume.reset_lr(0.2)
    assert volume.optimizer.param_groups[0]["lr"] == 0.2
    assert volume.lr_scheduler is not previous_scheduler

    volume.update_lr()
    assert volume.optimizer.param_groups[0]["lr"] == 0.05

    volume.reset_lr()
    assert volume.optimizer.param_groups[0]["lr"] == 0.5


def test_reset_lr_validates_per_group_rates() -> None:
    volume = VoxelGrid(
        grid_size=4,
        requires_accum=False,
        requires_grad=True,
    )

    volume.reset_lr([0.25])
    assert volume.optimizer is not None
    assert volume.optimizer.param_groups[0]["lr"] == 0.25

    with pytest.raises(ValueError, match="sequence length must match"):
        volume.reset_lr([0.1, 0.2])
    with pytest.raises(ValueError, match="must be > 0"):
        volume.reset_lr(0.0)


def test_requires_grad_only_changes_autograd_participation() -> None:
    volume = VoxelGrid(
        grid_size=4,
        requires_accum=False,
        requires_grad=True,
    )
    optimizer = volume.optimizer
    scheduler = volume.lr_scheduler

    volume.requires_grad_(False)
    assert not volume.volume.requires_grad
    assert volume.optimizer is optimizer
    assert volume.lr_scheduler is scheduler

    volume.requires_grad_(True)
    assert volume.volume.requires_grad
    assert volume.optimizer is optimizer
    assert volume.lr_scheduler is scheduler


def test_optimizer_and_scheduler_factories_override_defaults() -> None:
    volume = VoxelGrid(
        grid_size=4,
        requires_accum=False,
        requires_grad=True,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.01),
        lr_scheduler_factory=lambda optimizer: (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        ),
    )

    assert isinstance(volume.optimizer, torch.optim.Adam)
    assert isinstance(volume.lr_scheduler, torch.optim.lr_scheduler.StepLR)


def test_gradient_and_em_accumulators_are_cleared_independently() -> None:
    volume = VoxelGrid(
        grid_size=4,
        requires_accum=True,
        requires_grad=True,
    )
    volume.volume.grad = torch.ones_like(volume.volume)
    assert volume.accum_numer is not None
    volume.accum_numer.fill_(1)

    volume.zero_accum()
    assert volume.volume.grad is not None
    assert torch.count_nonzero(volume.accum_numer) == 0

    volume.accum_numer.fill_(1)
    volume.zero_grad(set_to_none=True)
    assert volume.volume.grad is None
    assert torch.count_nonzero(volume.accum_numer) > 0