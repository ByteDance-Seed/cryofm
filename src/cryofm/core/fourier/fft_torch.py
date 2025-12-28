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

# This file is adapted from the CryoSTAR project:
# https://github.com/bytedance/cryostar
# Original code is licensed under the Apache License, Version 2.0.
# Copyright (c) 2023 ByteDance.
import torch


def hartley_to_fourier_2d(H):
    """
        Example:
            >>> f = hartley_to_fourier_2d(torch.rand(4, 4))
            >>> torch.fft.ifft2(torch.fft.ifftshift(f)).imag  # the imag part is 0
    """
    assert len(H.shape) == 2 and H.shape[0] == H.shape[1]
    D = H.shape[0]
    dims = (0, 1)

    if D % 2 == 1:
        H_flip = torch.flip(H, dims=dims)
        F = (H + H_flip) / 2 - (H - H_flip) / 2 * 1j
        return F
    else:
        H_aug = torch.zeros((D + 1, D + 1), dtype=H.dtype, device=H.device)
        H_aug[:D, :D] = H
        H_aug[D, :D] = H[0, :D]
        H_aug[:D, D] = H[:D, 0]
        H_aug[D, D] = H[0, 0]
        H_aug_flip = torch.flip(H_aug, dims=dims)
        F_aug = (H_aug + H_aug_flip) / 2 - (H_aug - H_aug_flip) / 2 * 1j
        F = F_aug[:D, :D]
        return F


def batch_hartley_to_fourier_2d(Hs):
    """
        A batch version of `hartley_to_fourier_2d`.
        Example:
            >>> x = torch.rand(3, 4, 4)
            >>> y1 = torch.stack(list(hartley_to_fourier_2d(x[i]) for i in range(3)))
            >>> y2 = batch_hartley_to_fourier_2d(x)
            >>> (y1 - y2).abs().sum()
    """
    assert len(Hs.shape) == 3 and Hs.shape[1] == Hs.shape[2]
    bsz, D, _ = Hs.shape
    dims = (1, 2)

    if D % 2 == 1:
        Hs_flip = torch.flip(Hs, dims=dims)
        F = (Hs + Hs_flip) / 2 - (Hs - Hs_flip) / 2 * 1j
        return F
    else:
        Hs_aug = torch.zeros((bsz, D + 1, D + 1), dtype=Hs.dtype, device=Hs.device)
        Hs_aug[:, :D, :D] = Hs
        Hs_aug[:, D, :D] = Hs[:, 0, :D]
        Hs_aug[:, :D, D] = Hs[:, :D, 0]
        Hs_aug[:, D, D] = Hs[:, 0, 0]
        Hs_aug_flip = torch.flip(Hs_aug, dims=dims)
        Fs_aug = (Hs_aug + Hs_aug_flip) / 2 - (Hs_aug - Hs_aug_flip) / 2 * 1j
        Fs = Fs_aug[:, :D, :D]
        return Fs


def fourier_to_hartley_3d(F):
    return F.real - F.imag


def fourier_to_hartley_2d(F):
    return F.real - F.imag


def hartley_to_fourier_3d(H):
    """
        Example:
            >>> f = hartley_to_fourier_3d(torch.rand(4, 4, 4))
            >>> torch.fft.ifftn(torch.fft.ifftshift(f), dim=(-2, -1, 0)).imag  # the imag part is 0
    """
    assert len(H.shape) == 3 and H.shape[0] == H.shape[1] == H.shape[2]
    D = H.shape[0]
    dims = (0, 1, 2)

    if D % 2 == 1:
        H_flip = torch.flip(H, dims=dims)
        F = (H + H_flip) / 2 - (H - H_flip) / 2 * 1j
        return F
    else:
        H_aug = torch.zeros((D + 1, D + 1, D + 1), dtype=H.dtype, device=H.device)
        H_aug[:D, :D, :D] = H
        # set three faces
        H_aug[D, :D, :D] = H[0, :, :]
        H_aug[:D, D, :D] = H[:, 0, :]
        H_aug[:D, :D, D] = H[:, :, 0]
        # set three sides
        H_aug[:D, D, D] = H[:, 0, 0]
        H_aug[D, :D, D] = H[0, :, 0]
        H_aug[D, D, :D] = H[0, 0, :]
        # set vertex
        H_aug[D, D, D] = H[0, 0, 0]
        H_aug_flip = torch.flip(H_aug, dims=dims)
        F_aug = (H_aug + H_aug_flip) / 2 - (H_aug - H_aug_flip) / 2 * 1j
        F = F_aug[:D, :D, :D]
        return F


def primal_to_hartley_3d(r):
    h = primal_to_fourier_3d(r)
    return h.real - h.imag


def hartley_to_primal_3d(h):
    r = primal_to_fourier_3d(h)
    r = r / (r.shape[-1] * r.shape[-2] * r.shape[-3])
    return r.real - r.imag


@torch.autocast("cuda")
def primal_to_fourier_2d(r: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(r.float(), dim=(-2, -1))
        f = torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1))
    return f


@torch.autocast("cuda")
def primal_to_fourier_3d(r: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        r = torch.fft.ifftshift(r.float(), dim=(-3, -2, -1))
        f = torch.fft.fftshift(torch.fft.fftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                               dim=(-3, -2, -1))
    return f


def fourier_to_primal_2d(f: torch.Tensor) -> torch.Tensor:
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1))


def fourier_to_primal_3d(r: torch.Tensor) -> torch.Tensor:
    r = torch.fft.ifftshift(r, dim=(-3, -2, -1))
    return torch.fft.fftshift(torch.fft.ifftn(r, s=(r.shape[-3], r.shape[-2], r.shape[-1]), dim=(-3, -2, -1)),
                              dim=(-3, -2, -1))
