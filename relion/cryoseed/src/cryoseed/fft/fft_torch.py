"""
PyTorch-based implementations for conversions between real and Fourier (ft) space.
Note:
    - We use the default normalization mode "backward" used by PyTorch, i.e., normalization only
      happens in the backward direction, e.g., ifft.
"""
import torch


def _promote_real_fft_dtype(r: torch.Tensor) -> torch.Tensor:
    """Preserve float32/float64 inputs and upcast lower-precision real tensors for FFT."""
    if r.dtype in (torch.float32, torch.float64):
        return r
    return r.to(dtype=torch.float32)


def fourier_phase_shift(
    X: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    X: Tensor of shape (N, L, L), fourier images
    shift: Tensor of shape (N, 2), shift in pixels
    returns: shifted Fourier images of shape (N, L, L)
    """
    device = X.device
    N, L, _ = X.shape

    # Create frequency grid
    freqs = torch.fft.fftshift(torch.fft.fftfreq(L, d=1.0)).to(
        device
    )  # d=1.0 means pixel spacing = 1
    ky, kx = torch.meshgrid(freqs, freqs, indexing="ij")  # shape (L, L)
    kx = kx.unsqueeze(0)  # (1, L, L)
    ky = ky.unsqueeze(0)  # (1, L, L)

    # Apply phase shift: exp(-2πi (x·fx + y·fy))
    phase = torch.exp(
        -2j * torch.pi * (shift[:, 0, None, None] * kx + shift[:, 1, None, None] * ky)
    )  # shape: (N, L, L)

    # 4. Apply phase shift to Fourier image
    X_shifted = X * phase

    return X_shifted


@torch.autocast("cuda")
def primal_to_fourier_2d(r: torch.Tensor, norm="backward") -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(_promote_real_fft_dtype(r), dim=(-2, -1))
        f = torch.fft.fftshift(
            torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1), norm=norm),
            dim=(-2, -1),
        )
    return f


@torch.autocast("cuda")
def primal_to_fourier_3d(r: torch.Tensor, norm="backward") -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(_promote_real_fft_dtype(r), dim=(-3, -2, -1))
        f = torch.fft.fftshift(
            torch.fft.fftn(
                r,
                s=(r.shape[-3], r.shape[-2], r.shape[-1]),
                dim=(-3, -2, -1),
                norm=norm,
            ),
            dim=(-3, -2, -1),
        )
    return f


def fourier_to_primal_2d(f: torch.Tensor, norm="backward") -> torch.Tensor:
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1), norm=norm),
        dim=(-2, -1),
    )


def fourier_to_primal_3d(r: torch.Tensor, norm="backward") -> torch.Tensor:
    r = torch.fft.ifftshift(r, dim=(-3, -2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(
            r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1), norm=norm
        ),
        dim=(-3, -2, -1),
    )