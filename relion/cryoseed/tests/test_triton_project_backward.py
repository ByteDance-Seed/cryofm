from __future__ import annotations

import pytest
import torch

from tests._triton_test_utils import CUDA_TRITON_REQUIRED, fd_real_scalar, random_rotation

from cryoseed.backends.torch.project import project as project_torch
from cryoseed.backends.triton.project import project as project_triton

pytestmark = CUDA_TRITON_REQUIRED


@pytest.mark.parametrize("channel_last", [True, False])
def test_triton_project_backward_matches_torch(channel_last: bool):
    torch.manual_seed(0)
    device = torch.device("cuda")

    b, q, l, c = 1, 3, 6, 2
    rotation = random_rotation(b, q, device=device)

    if channel_last:
        base = torch.randn(b, l, l, l, c, device=device, dtype=torch.float32)
    else:
        base = torch.randn(b, c, l, l, l, device=device, dtype=torch.float32)

    # Only validate grad_input here. Triton rotation backward is intentionally unsupported.
    vol_triton = base.clone().detach().requires_grad_(True)
    vol_torch = base.clone().detach().requires_grad_(True)

    out_triton = project_triton(vol_triton, rotation, channel_last=channel_last)
    out_torch = project_torch(vol_torch, rotation, channel_last=channel_last)

    torch.testing.assert_close(out_triton, out_torch, rtol=2e-3, atol=2e-3)

    probe = torch.randn_like(out_torch)
    (out_triton * probe).sum().backward()
    (out_torch * probe).sum().backward()

    torch.testing.assert_close(vol_triton.grad, vol_torch.grad, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("channel_last", [True, False])
def test_triton_project_backward_matches_torch_gen_kernel(channel_last: bool):
    torch.manual_seed(11)
    device = torch.device("cuda")

    b, q, l, c = 1, 3, 6, 3
    rotation = random_rotation(b, q, device=device)

    if channel_last:
        base = torch.randn(b, l, l, l, c, device=device, dtype=torch.float32)
    else:
        base = torch.randn(b, c, l, l, l, device=device, dtype=torch.float32)

    vol_triton = base.clone().detach().requires_grad_(True)
    vol_torch = base.clone().detach().requires_grad_(True)

    out_triton = project_triton(vol_triton, rotation, channel_last=channel_last)
    out_torch = project_torch(vol_torch, rotation, channel_last=channel_last)

    torch.testing.assert_close(out_triton, out_torch, rtol=2e-3, atol=2e-3)

    probe = torch.randn_like(out_torch)
    (out_triton * probe).sum().backward()
    (out_torch * probe).sum().backward()

    torch.testing.assert_close(vol_triton.grad, vol_torch.grad, rtol=5e-3, atol=5e-3)


def test_triton_project_backward_matches_finite_difference():
    torch.manual_seed(1)
    device = torch.device("cuda")

    b, q, l, c = 1, 2, 6, 2
    rotation = random_rotation(b, q, device=device)
    volume = torch.randn(b, l, l, l, c, device=device, dtype=torch.float32, requires_grad=True)
    probe = torch.randn(b, q, l, l, c, device=device, dtype=torch.float32)

    def loss_fn(vol: torch.Tensor) -> torch.Tensor:
        return (project_triton(vol, rotation, channel_last=True) * probe).sum()

    loss = loss_fn(volume)
    loss.backward()

    check_indices = [
        (0, 1, 2, 3, 0),
        (0, 4, 1, 2, 1),
        (0, 2, 4, 1, 0),
    ]
    for idx in check_indices:
        fd = fd_real_scalar(loss_fn, volume, idx, eps=1e-3)
        analytic = float(volume.grad[idx].detach().cpu())
        assert analytic == pytest.approx(fd, rel=5e-2, abs=5e-2)


def test_triton_project_rotation_grad_is_explicitly_unimplemented():
    torch.manual_seed(2)
    device = torch.device("cuda")

    volume = torch.randn(1, 6, 6, 6, 2, device=device, dtype=torch.float32)
    rotation = random_rotation(1, 2, device=device).requires_grad_(True)

    loss = project_triton(volume, rotation, channel_last=True).sum()
    with pytest.raises(NotImplementedError, match="grad_rotation"):
        loss.backward()