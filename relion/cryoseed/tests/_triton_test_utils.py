from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

CUDA_TRITON_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton tests",
)


def random_rotation(batch: int, poses: int, device: torch.device) -> torch.Tensor:
    raw = torch.randn(batch * poses, 3, 3, device=device, dtype=torch.float32)
    u, _, vh = torch.linalg.svd(raw, full_matrices=False)
    rot = u @ vh
    det = torch.det(rot)
    if (det < 0).any():
        u = u.clone()
        u[det < 0, :, -1] *= -1
        rot = u @ vh
    return rot.view(batch, poses, 3, 3)


def fd_real_scalar(loss_fn, x: torch.Tensor, idx, eps: float = 1e-3) -> float:
    xp = x.detach().clone()
    xm = x.detach().clone()
    delta = xp.new_tensor(eps)
    xp[idx] += delta
    xm[idx] -= delta
    lp = float(loss_fn(xp).detach().cpu())
    lm = float(loss_fn(xm).detach().cpu())
    return (lp - lm) / (2.0 * eps)


def fd_complex_scalar(loss_fn, x: torch.Tensor, idx, *, imag: bool, eps: float = 1e-3) -> float:
    xp = x.detach().clone()
    xm = x.detach().clone()
    delta = xp.new_tensor(1j * eps if imag else eps)
    xp[idx] += delta
    xm[idx] -= delta
    lp = float(loss_fn(xp).detach().cpu())
    lm = float(loss_fn(xm).detach().cpu())
    return (lp - lm) / (2.0 * eps)