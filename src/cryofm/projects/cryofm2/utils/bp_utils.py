# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def fourier_translation(X: torch.Tensor, x_shifts: torch.Tensor, y_shifts: torch.Tensor) -> torch.Tensor:
    """
    X: Tensor of shape (N, L, L), fourier images
    x_shifts, y_shifts: Tensors of shape (N,), shift in pixels
    returns: shifted Fourier images of shape (N, L, L)
    """
    device = X.device
    N, L, _ = X.shape

    # Create frequency grid
    freqs = torch.fft.fftshift(torch.fft.fftfreq(L, d=1.0)).to(device)  # d=1.0 means pixel spacing = 1
    ky, kx = torch.meshgrid(freqs, freqs, indexing='ij')  # shape (L, L)
    kx = kx.unsqueeze(0)  # (1, L, L)
    ky = ky.unsqueeze(0)  # (1, L, L)

    # Apply phase shift: exp(-2πi (x·fx + y·fy))
    phase = torch.exp(
        -2j * torch.pi * (
            x_shifts[:, None, None] * kx +
            y_shifts[:, None, None] * ky
        )
    )  # shape: (N, L, L)

    # 4. Apply phase shift to Fourier image
    X_shifted = X * phase

    return X_shifted


def create_so3(eulers: torch.Tensor, device=None) -> torch.Tensor:
    """
    Create a batch of Z-Y-Z rotation matrices from Euler angles.

    Parameters:
    -----------
    eulers : torch.Tensor, shape (N, 3)
        Euler angles in radians (phi, theta, psi) for each sample.
    device : optional
        Device to place the resulting tensor on.

    Returns:
    --------
    R : torch.Tensor, shape (N, 3, 3)
        Batch of rotation matrices.
    """
    phi, theta, psi = eulers[:, 0], eulers[:, 1], eulers[:, 2]

    cphi, sphi = torch.cos(phi), torch.sin(phi)
    ctheta, stheta = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    Rz_phi = torch.stack([
        torch.stack([ cphi, -sphi, torch.zeros_like(cphi)], dim=1),
        torch.stack([ sphi,  cphi, torch.zeros_like(cphi)], dim=1),
        torch.stack([ torch.zeros_like(cphi), torch.zeros_like(cphi), torch.ones_like(cphi)], dim=1),
    ], dim=1)  # (N, 3, 3)

    Ry_theta = torch.stack([
        torch.stack([ ctheta, torch.zeros_like(ctheta),  stheta], dim=1),
        torch.stack([ torch.zeros_like(ctheta), torch.ones_like(ctheta), torch.zeros_like(ctheta)], dim=1),
        torch.stack([-stheta, torch.zeros_like(ctheta),  ctheta], dim=1),
    ], dim=1)

    Rz_psi = torch.stack([
        torch.stack([ cpsi, -spsi, torch.zeros_like(cpsi)], dim=1),
        torch.stack([ spsi,  cpsi, torch.zeros_like(cpsi)], dim=1),
        torch.stack([ torch.zeros_like(cpsi), torch.zeros_like(cpsi), torch.ones_like(cpsi)], dim=1),
    ], dim=1)

    R = torch.bmm(torch.bmm(Rz_psi, Ry_theta), Rz_phi).to(device=device)
    return R
