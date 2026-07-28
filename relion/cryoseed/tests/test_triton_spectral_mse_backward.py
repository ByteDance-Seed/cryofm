from __future__ import annotations

import pytest
import torch

from tests._triton_test_utils import CUDA_TRITON_REQUIRED, fd_complex_scalar, fd_real_scalar

from cryoseed.backends.torch.spectral_mse_loss import spectral_mse_loss as spectral_mse_loss_torch
from cryoseed.backends.triton.spectral_mse_loss import spectral_mse_loss as spectral_mse_loss_triton

pytestmark = CUDA_TRITON_REQUIRED


@pytest.mark.parametrize(("reduction", "spectral_reduction"), [("mean", "mean"), ("sum", "sum")])
def test_triton_spectral_mse_auto_matches_explicit_spectral_reduction(
    reduction: str, spectral_reduction: str
):
    torch.manual_seed(2)
    device = torch.device("cuda")

    b, ci, co, h, w = 1, 2, 2, 2, 3
    x = torch.randn(b, ci, h, w, device=device, dtype=torch.complex64)
    y = torch.randn(b, co, h, w, device=device, dtype=torch.complex64)
    weight = 0.2 + torch.rand(h, w, device=device, dtype=torch.float32)

    triton_auto = spectral_mse_loss_triton(x, y, weight=weight, reduction=reduction)
    triton_explicit = spectral_mse_loss_triton(
        x, y, weight=weight, reduction=reduction, spectral_reduction=spectral_reduction
    )
    torch_auto = spectral_mse_loss_torch(x, y, weight=weight, reduction=reduction)
    torch_explicit = spectral_mse_loss_torch(
        x, y, weight=weight, reduction=reduction, spectral_reduction=spectral_reduction
    )

    torch.testing.assert_close(triton_auto, triton_explicit, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch_auto, torch_explicit, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(triton_auto, torch_auto, rtol=2e-4, atol=2e-4)


def test_triton_spectral_mse_none_requires_explicit_spectral_reduction():
    torch.manual_seed(2)
    device = torch.device("cuda")

    b, ci, co, h, w = 1, 2, 2, 2, 3
    x = torch.randn(b, ci, h, w, device=device, dtype=torch.complex64)
    y = torch.randn(b, co, h, w, device=device, dtype=torch.complex64)
    weight = 0.2 + torch.rand(h, w, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="spectral_reduction must be explicitly set"):
        spectral_mse_loss_torch(x, y, weight=weight, reduction="none")

    with pytest.raises(ValueError, match="spectral_reduction must be explicitly set"):
        spectral_mse_loss_triton(x, y, weight=weight, reduction="none")


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_triton_spectral_mse_broadcast_backward_matches_torch(reduction: str):
    torch.manual_seed(3)
    device = torch.device("cuda")

    b, ci, co, h, w = 2, 3, 2, 3, 4
    input_base = torch.randn(b, ci, h, w, device=device, dtype=torch.complex64)
    target_base = torch.randn(b, co, h, w, device=device, dtype=torch.complex64)
    weight_base = 0.2 + torch.rand(h, w, device=device, dtype=torch.float32)

    x_triton = input_base.clone().detach().requires_grad_(True)
    y_triton = target_base.clone().detach().requires_grad_(True)
    w_triton = weight_base.clone().detach().requires_grad_(True)

    x_torch = input_base.clone().detach().requires_grad_(True)
    y_torch = target_base.clone().detach().requires_grad_(True)
    w_torch = weight_base.clone().detach().requires_grad_(True)

    loss_triton = spectral_mse_loss_triton(
        x_triton,
        y_triton,
        weight=w_triton,
        reduction=reduction,
        spectral_reduction=reduction,
    )
    loss_torch = spectral_mse_loss_torch(
        x_torch,
        y_torch,
        weight=w_torch,
        reduction=reduction,
        spectral_reduction=reduction,
    )

    torch.testing.assert_close(loss_triton, loss_torch, rtol=2e-4, atol=2e-4)

    loss_triton.backward()
    loss_torch.backward()

    torch.testing.assert_close(x_triton.grad, x_torch.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(y_triton.grad, y_torch.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(w_triton.grad, w_torch.grad, rtol=2e-3, atol=2e-3)


def test_triton_spectral_mse_broadcast_backward_matches_finite_difference():
    torch.manual_seed(4)
    device = torch.device("cuda")

    b, ci, co, h, w = 1, 2, 2, 2, 3
    x = torch.randn(b, ci, h, w, device=device, dtype=torch.complex64, requires_grad=True)
    y = torch.randn(b, co, h, w, device=device, dtype=torch.complex64, requires_grad=True)
    weight = (0.2 + torch.rand(h, w, device=device, dtype=torch.float32)).requires_grad_(True)

    def loss_fn(inp: torch.Tensor, tgt: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
        return spectral_mse_loss_triton(
            inp,
            tgt,
            weight=wt,
            reduction="sum",
            spectral_reduction="sum",
        )

    loss = loss_fn(x, y, weight)
    loss.backward()

    x_idx = (0, 1, 0, 2)
    y_idx = (0, 0, 1, 1)
    w_idx = (1, 2)

    fd_x_re = fd_complex_scalar(lambda t: loss_fn(t, y.detach(), weight.detach()), x, x_idx, imag=False)
    fd_x_im = fd_complex_scalar(lambda t: loss_fn(t, y.detach(), weight.detach()), x, x_idx, imag=True)
    fd_y_re = fd_complex_scalar(lambda t: loss_fn(x.detach(), t, weight.detach()), y, y_idx, imag=False)
    fd_y_im = fd_complex_scalar(lambda t: loss_fn(x.detach(), t, weight.detach()), y, y_idx, imag=True)
    fd_w = fd_real_scalar(lambda t: loss_fn(x.detach(), y.detach(), t), weight, w_idx)

    assert float(x.grad[x_idx].real.detach().cpu()) == pytest.approx(fd_x_re, rel=5e-2, abs=5e-2)
    assert float(x.grad[x_idx].imag.detach().cpu()) == pytest.approx(fd_x_im, rel=5e-2, abs=5e-2)
    assert float(y.grad[y_idx].real.detach().cpu()) == pytest.approx(fd_y_re, rel=5e-2, abs=5e-2)
    assert float(y.grad[y_idx].imag.detach().cpu()) == pytest.approx(fd_y_im, rel=5e-2, abs=5e-2)
    assert float(weight.grad[w_idx].detach().cpu()) == pytest.approx(fd_w, rel=5e-2, abs=5e-2)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_triton_spectral_mse_indexed_backward_matches_torch(reduction: str):
    torch.manual_seed(5)
    device = torch.device("cuda")

    n_in, n_tgt, m, h, w = 5, 6, 7, 3, 4
    input_base = torch.randn(n_in, h, w, device=device, dtype=torch.complex64)
    target_base = torch.randn(n_tgt, h, w, device=device, dtype=torch.complex64)
    weight_base = 0.2 + torch.rand(h, w, device=device, dtype=torch.float32)
    input_indices = torch.tensor([0, 2, 4, 1, 3, 2, 0], device=device, dtype=torch.int64)
    target_indices = torch.tensor([1, 5, 3, 0, 2, 4, 1], device=device, dtype=torch.int64)
    assert input_indices.numel() == m

    x_triton = input_base.clone().detach().requires_grad_(True)
    y_triton = target_base.clone().detach().requires_grad_(True)
    w_triton = weight_base.clone().detach().requires_grad_(True)

    x_torch = input_base.clone().detach().requires_grad_(True)
    y_torch = target_base.clone().detach().requires_grad_(True)
    w_torch = weight_base.clone().detach().requires_grad_(True)

    loss_triton = spectral_mse_loss_triton(
        x_triton,
        y_triton,
        weight=w_triton,
        input_indices=input_indices,
        target_indices=target_indices,
        reduction=reduction,
        spectral_reduction=reduction,
    )
    loss_torch = spectral_mse_loss_torch(
        x_torch,
        y_torch,
        weight=w_torch,
        input_indices=input_indices,
        target_indices=target_indices,
        reduction=reduction,
        spectral_reduction=reduction,
    )

    torch.testing.assert_close(loss_triton, loss_torch, rtol=2e-4, atol=2e-4)

    loss_triton.backward()
    loss_torch.backward()

    torch.testing.assert_close(x_triton.grad, x_torch.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(y_triton.grad, y_torch.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(w_triton.grad, w_torch.grad, rtol=2e-3, atol=2e-3)


def test_triton_spectral_mse_indexed_backward_matches_finite_difference():
    torch.manual_seed(6)
    device = torch.device("cuda")

    x = torch.randn(4, 2, 3, device=device, dtype=torch.complex64, requires_grad=True)
    y = torch.randn(5, 2, 3, device=device, dtype=torch.complex64, requires_grad=True)
    weight = (0.2 + torch.rand(2, 3, device=device, dtype=torch.float32)).requires_grad_(True)
    input_indices = torch.tensor([0, 1, 3, 2], device=device, dtype=torch.int64)
    target_indices = torch.tensor([4, 1, 0, 2], device=device, dtype=torch.int64)

    def loss_fn(inp: torch.Tensor, tgt: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
        return spectral_mse_loss_triton(
            inp,
            tgt,
            weight=wt,
            input_indices=input_indices,
            target_indices=target_indices,
            reduction="sum",
            spectral_reduction="sum",
        )

    loss = loss_fn(x, y, weight)
    loss.backward()

    x_idx = (3, 1, 2)
    y_idx = (4, 0, 1)
    w_idx = (0, 2)

    fd_x_re = fd_complex_scalar(lambda t: loss_fn(t, y.detach(), weight.detach()), x, x_idx, imag=False)
    fd_x_im = fd_complex_scalar(lambda t: loss_fn(t, y.detach(), weight.detach()), x, x_idx, imag=True)
    fd_y_re = fd_complex_scalar(lambda t: loss_fn(x.detach(), t, weight.detach()), y, y_idx, imag=False)
    fd_y_im = fd_complex_scalar(lambda t: loss_fn(x.detach(), t, weight.detach()), y, y_idx, imag=True)
    fd_w = fd_real_scalar(lambda t: loss_fn(x.detach(), y.detach(), t), weight, w_idx)

    assert float(x.grad[x_idx].real.detach().cpu()) == pytest.approx(fd_x_re, rel=5e-2, abs=5e-2)
    assert float(x.grad[x_idx].imag.detach().cpu()) == pytest.approx(fd_x_im, rel=5e-2, abs=5e-2)
    assert float(y.grad[y_idx].real.detach().cpu()) == pytest.approx(fd_y_re, rel=5e-2, abs=5e-2)
    assert float(y.grad[y_idx].imag.detach().cpu()) == pytest.approx(fd_y_im, rel=5e-2, abs=5e-2)
    assert float(weight.grad[w_idx].detach().cpu()) == pytest.approx(fd_w, rel=5e-2, abs=5e-2)