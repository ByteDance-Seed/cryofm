from __future__ import annotations

import torch

from cryoseed.fft.coords import fftindex_radial2d
from cryoseed.ops.radial import radial_residual_power


def _reference_radial_residual_power(
    input: torch.Tensor,
    target: torch.Tensor,
    input_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    side_length: int,
    max_radius: int,
) -> torch.Tensor:
    radial = fftindex_radial2d(side_length, device=input.device)
    valid_mask = radial <= max_radius
    radial_indices = radial[valid_mask].to(torch.long)
    num_radial_bins = max_radius + 1
    counts = torch.bincount(radial_indices, minlength=num_radial_bins).to(torch.float32)
    radial_denom = torch.zeros((num_radial_bins,), device=input.device, dtype=torch.float32)
    radial_denom[counts > 0] = 1.0 / counts[counts > 0]
    radial_weight = radial_denom.index_select(0, radial_indices)

    input_valid = input.flatten(start_dim=1).index_select(1, valid_mask.nonzero(as_tuple=False).view(-1))
    target_valid = target.flatten(start_dim=1).index_select(1, valid_mask.nonzero(as_tuple=False).view(-1))
    proj_sel = input_valid.index_select(0, input_indices)
    target_sel = target_valid.index_select(0, target_indices)
    weighted = (proj_sel - target_sel).abs().square().to(torch.float32) * radial_weight.view(1, -1)
    out = torch.zeros((input_indices.numel(), num_radial_bins), device=input.device, dtype=torch.float32)
    out.scatter_add_(1, radial_indices.view(1, -1).expand(input_indices.numel(), -1), weighted)
    return out


def test_radial_residual_power_matches_reference_on_full_grid():
    torch.manual_seed(21)
    side_length = 6
    max_radius = side_length // 2
    input = torch.randn(3, side_length, side_length, dtype=torch.complex64)
    target = torch.randn(4, side_length, side_length, dtype=torch.complex64)
    input_indices = torch.tensor([0, 2], dtype=torch.int64)
    target_indices = torch.tensor([3, 1], dtype=torch.int64)

    out = radial_residual_power(
        input,
        target,
        input_indices=input_indices,
        target_indices=target_indices,
        side_length=side_length,
        max_radius=max_radius,
        ndim=2,
        use_cache=True,
    )
    ref = _reference_radial_residual_power(
        input,
        target,
        input_indices,
        target_indices,
        side_length=side_length,
        max_radius=max_radius,
    )

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_radial_residual_power_accepts_flattened_valid_pixels():
    torch.manual_seed(22)
    side_length = 8
    max_radius = side_length // 2
    radial = fftindex_radial2d(side_length)
    valid_idx = (radial <= max_radius).nonzero(as_tuple=False).view(-1)

    input_full = torch.randn(2, side_length, side_length, dtype=torch.complex64)
    target_full = torch.randn(2, side_length, side_length, dtype=torch.complex64)
    input_valid = input_full.flatten(start_dim=1).index_select(1, valid_idx)
    target_valid = target_full.flatten(start_dim=1).index_select(1, valid_idx)
    input_indices = torch.tensor([0, 1], dtype=torch.int64)
    target_indices = torch.tensor([1, 0], dtype=torch.int64)

    out = radial_residual_power(
        input_valid,
        target_valid,
        input_indices=input_indices,
        target_indices=target_indices,
        side_length=side_length,
        max_radius=max_radius,
        ndim=2,
        use_cache=True,
    )
    ref = _reference_radial_residual_power(
        input_full,
        target_full,
        input_indices,
        target_indices,
        side_length=side_length,
        max_radius=max_radius,
    )

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)