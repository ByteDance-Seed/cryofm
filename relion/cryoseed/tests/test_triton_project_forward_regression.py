from __future__ import annotations

import pytest
import torch

from tests._triton_test_utils import CUDA_TRITON_REQUIRED, random_rotation

from cryoseed.backends.torch.project import project as project_torch
from cryoseed.backends.triton.project import project as project_triton

pytestmark = CUDA_TRITON_REQUIRED


def _z_rotation(angle_rad: float, device: torch.device) -> torch.Tensor:
    c = torch.cos(torch.tensor(angle_rad, device=device, dtype=torch.float32))
    s = torch.sin(torch.tensor(angle_rad, device=device, dtype=torch.float32))
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("channel_last", [True, False])
@pytest.mark.parametrize("length", [5, 6])
def test_triton_project_identity_matches_direct_center_slice(length: int, channel_last: bool):
    torch.manual_seed(7)
    device = torch.device("cuda")

    if channel_last:
        volume = torch.randn(1, length, length, length, 2, device=device, dtype=torch.float32)
        expected = volume[:, length // 2, :, :, :]
    else:
        volume = torch.randn(1, 2, length, length, length, device=device, dtype=torch.float32)
        expected = volume[:, :, length // 2, :, :]

    rotation = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3)
    out = project_triton(volume, rotation, channel_last=channel_last)

    torch.testing.assert_close(out[:, 0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("channel_last", [True, False])
@pytest.mark.parametrize("length", [5, 6])
def test_triton_project_forward_matches_torch_reference(length: int, channel_last: bool):
    torch.manual_seed(8)
    device = torch.device("cuda")

    batch, poses = 1, 3
    rotation = random_rotation(batch, poses, device=device)

    if channel_last:
        volume = torch.randn(batch, length, length, length, 2, device=device, dtype=torch.float32)
    else:
        volume = torch.randn(batch, 2, length, length, length, device=device, dtype=torch.float32)

    out_triton = project_triton(volume, rotation, channel_last=channel_last)
    out_torch = project_torch(volume, rotation, channel_last=channel_last)

    torch.testing.assert_close(out_triton, out_torch, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("channel_last", [True, False])
def test_triton_project_partial_oob_matches_torch_reference(channel_last: bool):
    torch.manual_seed(9)
    device = torch.device("cuda")
    length = 6

    if channel_last:
        volume = torch.ones(1, length, length, length, 2, device=device, dtype=torch.float32)
    else:
        volume = torch.ones(1, 2, length, length, length, device=device, dtype=torch.float32)

    rotation = _z_rotation(torch.pi / 12.0, device=device).view(1, 1, 3, 3)

    out_triton = project_triton(volume, rotation, channel_last=channel_last)
    out_torch = project_torch(volume, rotation, channel_last=channel_last)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-5, atol=1e-5)

    if channel_last:
        edge_value = float(out_torch[0, 0, 0, length - 1, 0].detach().cpu())
    else:
        edge_value = float(out_torch[0, 0, 0, 0, length - 1].detach().cpu())

    assert 0.0 < edge_value < 1.0