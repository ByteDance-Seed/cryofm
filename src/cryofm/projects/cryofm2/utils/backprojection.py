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
from torch import Tensor
from typing import Tuple


class Backprojector:
    '''
    Backproject a batch of 2D fourier images to 3D volume

    Relion source: backprojector.cpp/BackProjector::backproject2Dto3D

    TODO: Just use half images
    TODO: Try to do interpolation then exclude out-of-bound pixels for efficiency

    '''

    def __init__(self, L, R: Tensor,  # shape: (N, 3, 3), rotation matrix
                 max_r2: float,  # scalar: maximum radius**2 of image to backproject
                 pixel_weights: Tensor = None  # shape: (L, L), weight for each pixel ):
                 ):
        '''
        Determine Nyquist region, Rotate the grid, and determine inbound pixels
        '''

        self.device = R.device
        self.N = R.shape[0]
        self.L = L
        # DC term is loctated at f_image[L//2, L//2]
        center = L // 2

        # Note: Not clear how it is used. Probably useless
        if pixel_weights is None:
            pixel_weights = torch.ones(L, L)

        ys, xs = torch.meshgrid(torch.arange(-L // 2, L // 2, device=self.device),
                                torch.arange(-L // 2, L // 2, device=self.device), indexing='ij')

        # Nyquist mask
        # Excluded pixels from the confidence interval determind by Nyquist rate
        # Simplified from Relion, which also considers magMatrix
        r2 = xs ** 2 + ys ** 2
        self.Nyq_mask = r2 <= max_r2
        xs = xs[self.Nyq_mask]
        ys = ys[self.Nyq_mask]

        # weighting for each pixel
        self.pixel_weights = pixel_weights[self.Nyq_mask].reshape(-1)  # (P,)

        coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # [P, 2]

        xy0 = torch.cat([coords, torch.zeros((coords.shape[0], 1), device=self.device)], dim=1)
        xy0 = xy0.unsqueeze(0).expand(self.N, -1, -1)  # [N, P, 3]

        # Doing inverse rotatio to the image to get the slice coordinate
        # Note: R is the original rotation matrix when projecting the volume to the particle
        # This is equivalent to the inverse rotation since R.inv = R.T
        xyz_rot = torch.bmm(xy0, R)
        x_rot, y_rot, z_rot = xyz_rot[:, :, 0], xyz_rot[:, :, 1], xyz_rot[:, :, 2]

        x0 = torch.floor(x_rot).to(torch.int32)  # [N,P]
        y0 = torch.floor(y_rot).to(torch.int32)
        z0 = torch.floor(z_rot).to(torch.int32)
        fx = x_rot - x0.float()
        fy = y_rot - y0.float()
        fz = z_rot - z0.float()
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        self.interp_weights = torch.stack([
            (1 - fz) * (1 - fy) * (1 - fx),
            (1 - fz) * (1 - fy) * fx,
            (1 - fz) * fy * (1 - fx),
            (1 - fz) * fy * fx,
            fz * (1 - fy) * (1 - fx),
            fz * (1 - fy) * fx,
            fz * fy * (1 - fx),
            fz * fy * fx
        ]).transpose(0, 1)  # (N, 8, P)

        voxel_coords = torch.stack([
            torch.stack([z0, y0, x0], dim=-1),
            torch.stack([z0, y0, x1], dim=-1),
            torch.stack([z0, y1, x0], dim=-1),
            torch.stack([z0, y1, x1], dim=-1),
            torch.stack([z1, y0, x0], dim=-1),
            torch.stack([z1, y0, x1], dim=-1),
            torch.stack([z1, y1, x0], dim=-1),
            torch.stack([z1, y1, x1], dim=-1)
        ], dim=0).transpose(0, 1)  # [N,8,P,3]

        self.voxel_coords = voxel_coords + center

        # Only consider the pixel all its nearby voxels are in bound
        in_bounds = (self.voxel_coords >= 0) & (self.voxel_coords < L)  # (N, 8, P, 3)
        self.all_in_bounds = in_bounds.all(dim=3).all(dim=1)  # (N, P)

    def backproject2Dto3D(self,
                          f_image: Tensor,  # shape: (N, L, L), 2D image
                          ):

        L = f_image.shape[1]
        assert L == self.L, f"Image size L={L} doesn't match grid size self.L={self.L}"

        freqs = f_image[:, self.Nyq_mask].reshape(self.N, -1)  # (N, P)
        V_flat = torch.zeros(self.L * self.L * self.L, dtype=f_image.dtype, device=self.device)
        W_flat = torch.zeros(self.L * self.L * self.L, dtype=torch.float32, device=self.device)

        # For loop is necessary since for each image, #in bound pixels are different
        for i in range(self.N):
            all_in_bound = self.all_in_bounds[i, :]

            v_coord = self.voxel_coords[i, :, all_in_bound, :]
            freq = freqs[i, all_in_bound]
            pw = self.pixel_weights[all_in_bound]
            interp_w = self.interp_weights[i, :, all_in_bound]

            f_total = interp_w * freq.unsqueeze(0)  # [8,P']
            w_total = interp_w * pw.unsqueeze(0)
            lin_idx = v_coord[:, :, 0] * L * L + v_coord[:, :, 1] * L + v_coord[:, :, 2]
            # TODO: index_add_ is unstable for GPU parallel. But it is safe for now since we iterate over single image
            V_flat.index_add_(0, lin_idx.reshape(-1), f_total.reshape(-1))
            W_flat.index_add_(0, lin_idx.reshape(-1), w_total.reshape(-1))
            # V_flat.scatter_add_(0, lin_idx.reshape(-1).to(torch.int64), f_total.reshape(-1))
            # W_flat.scatter_add_(0, lin_idx.reshape(-1).to(torch.int64), w_total.reshape(-1))

        V = V_flat.view(L, L, L)
        V_idx = torch.nonzero(V, as_tuple=True)
        W = W_flat.view(L, L, L)
        W_idx = torch.nonzero(W, as_tuple=True)
        return V, W, V_idx, W_idx


# --------------------------------------Deprecated---------------------------------------------


def backproject2Dto3D(
        f_image: Tensor,  # shape: (N, L, L), 2D fourier image
        R: Tensor,  # shape: (N, 3, 3), rotation matrix
        max_r2: float,  # scalar: maximum radius**2 of image to backproject
        pixel_weights: Tensor = None,  # shape: (L, L), weight for each pixel
) -> Tuple[Tensor, Tensor]:
    '''
    Backproject a batch of 2D fourier images to 3D volume

    Relion source: backprojector.cpp/BackProjector::backproject2Dto3D

    TODO: Just use half images
    '''

    device = f_image.device
    N, L, _ = f_image.shape
    # DC term is loctated at f_image[L//2, L//2]
    center = L // 2

    # Note: Not clear how it is used. Probably useless
    if pixel_weights is None:
        pixel_weights = torch.ones(L, L)

    ys, xs = torch.meshgrid(torch.arange(-L // 2, L // 2, device=device),
                            torch.arange(-L // 2, L // 2, device=device), indexing='ij')

    # Nyquist mask
    # Excluded pixels from the confidence interval determind by Nyquist rate
    # Simplified from Relion, which also considers magMatrix
    r2 = xs ** 2 + ys ** 2
    Nyq_mask = r2 <= max_r2
    xs = xs[Nyq_mask]
    ys = ys[Nyq_mask]

    # weighting for each pixel
    pixel_weights = pixel_weights[Nyq_mask].reshape(-1)  # (P,)

    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # [P, 2]
    freqs = f_image[:, Nyq_mask].reshape(N, -1)  # (N, P)

    xy0 = torch.cat([coords, torch.zeros((coords.shape[0], 1), device=device)], dim=1)
    xy0 = xy0.unsqueeze(0).expand(N, -1, -1)  # [N, P, 3]

    # Doing inverse rotatio to the image to get the slice coordinate
    # Note: R is the original rotation matrix when projecting the volume to the particle
    # This is equivalent to the inverse rotation since R.inv = R.T
    xyz_rot = torch.bmm(xy0, R)
    x_rot, y_rot, z_rot = xyz_rot[:, :, 0], xyz_rot[:, :, 1], xyz_rot[:, :, 2]

    x0 = torch.floor(x_rot).to(torch.int32)  # [N,P]
    y0 = torch.floor(y_rot).to(torch.int32)
    z0 = torch.floor(z_rot).to(torch.int32)
    fx = x_rot - x0.float()
    fy = y_rot - y0.float()
    fz = z_rot - z0.float()
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    interp_weights = torch.stack([
        (1 - fz) * (1 - fy) * (1 - fx),
        (1 - fz) * (1 - fy) * fx,
        (1 - fz) * fy * (1 - fx),
        (1 - fz) * fy * fx,
        fz * (1 - fy) * (1 - fx),
        fz * (1 - fy) * fx,
        fz * fy * (1 - fx),
        fz * fy * fx
    ]).transpose(0, 1)  # (N, 8, P)

    voxel_coords = torch.stack([
        torch.stack([z0, y0, x0], dim=-1),
        torch.stack([z0, y0, x1], dim=-1),
        torch.stack([z0, y1, x0], dim=-1),
        torch.stack([z0, y1, x1], dim=-1),
        torch.stack([z1, y0, x0], dim=-1),
        torch.stack([z1, y0, x1], dim=-1),
        torch.stack([z1, y1, x0], dim=-1),
        torch.stack([z1, y1, x1], dim=-1)
    ], dim=0).transpose(0, 1)  # [N,8,P,3]

    voxel_coords += center

    # Only consider the pixel all its nearby voxels are in bound
    in_bounds = (voxel_coords >= 0) & (voxel_coords < L)  # (N, 8, P, 3)
    all_in_bounds = in_bounds.all(dim=3).all(dim=1)  # (N, P)

    V_flat = torch.zeros(L * L * L, dtype=f_image.dtype, device=device)
    W_flat = torch.zeros(L * L * L, dtype=torch.float32, device=device)

    # For loop is necessary since for each image, #in bound pixels are different
    for i in range(N):
        all_in_bound = all_in_bounds[i, :]

        v_coord = voxel_coords[i, :, all_in_bound, :]
        freq = freqs[i, all_in_bound]
        pw = pixel_weights[all_in_bound]
        interp_w = interp_weights[i, :, all_in_bound]

        f_total = interp_w * freq.unsqueeze(0)  # [8,P']
        w_total = interp_w * pw.unsqueeze(0)
        lin_idx = v_coord[:, :, 0] * L * L + v_coord[:, :, 1] * L + v_coord[:, :, 2]
        V_flat.index_add_(0, lin_idx.reshape(-1), f_total.reshape(-1))
        W_flat.index_add_(0, lin_idx.reshape(-1), w_total.reshape(-1))

    V = V_flat.view(L, L, L)
    W = W_flat.view(L, L, L)
    return V, W


def backproject2Dto3D_single(
        f_image: Tensor,  # shape: (L, L), 2D fourier image
        R: Tensor,  # shape: (3, 3), rotation matrix
        max_r2: float,  # scalar: maximum radius**2 of image to backproject
        pixel_weights: Tensor = None,  # shape: (L, L), weight for each pixel
) -> Tuple[Tensor, Tensor]:
    '''
    Backproject a single 2D fourier image to 3D volume

    Relion source: backprojector.cpp/BackProjector::backproject2Dto3D

    TODO: Just use half images
    '''

    device = f_image.device
    L = f_image.shape[0]
    # DC term is loctated at f_image[L//2, L//2]
    center = L // 2

    ys, xs = torch.meshgrid(torch.arange(-L // 2, L // 2, device=device),
                            torch.arange(-L // 2, L // 2, device=device), indexing='ij')

    # Nyquist mask
    # Excluded pixels from the confidence interval determind by Nyquist rate
    # Simplified from Relion, which also considers magMatrix
    r2 = xs ** 2 + ys ** 2
    Nyq_mask = r2 <= max_r2
    xs = xs[Nyq_mask]
    ys = ys[Nyq_mask]

    # weighting for each pixel
    pixel_weights = pixel_weights[Nyq_mask].reshape(-1)

    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # [P, 2]
    freqs = f_image[Nyq_mask].reshape(-1)

    xy0 = torch.cat([coords, torch.zeros((coords.shape[0], 1), device=device)], dim=1)  # [P, 3]

    # Doing inverse rotatio to the image to get the slice coordinate
    # Note: R is the original rotation matrix when projecting the volume to the particle
    # This is equivalent to the inverse rotation since R.inv = R.T
    xyz_rot = xy0 @ R  # [P, 3] x [3, 3]
    x_rot, y_rot, z_rot = xyz_rot[:, 0], xyz_rot[:, 1], xyz_rot[:, 2]

    x0 = torch.floor(x_rot).to(torch.int32)
    y0 = torch.floor(y_rot).to(torch.int32)
    z0 = torch.floor(z_rot).to(torch.int32)
    fx = x_rot - x0.float()
    fy = y_rot - y0.float()
    fz = z_rot - z0.float()
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    interp_weights = torch.stack([
        (1 - fz) * (1 - fy) * (1 - fx),
        (1 - fz) * (1 - fy) * fx,
        (1 - fz) * fy * (1 - fx),
        (1 - fz) * fy * fx,
        fz * (1 - fy) * (1 - fx),
        fz * (1 - fy) * fx,
        fz * fy * (1 - fx),
        fz * fy * fx
    ])  # (8, P)

    voxel_coords = torch.stack([
        z0, y0, x0,
        z0, y0, x1,
        z0, y1, x0,
        z0, y1, x1,
        z1, y0, x0,
        z1, y0, x1,
        z1, y1, x0,
        z1, y1, x1,
    ], dim=1).reshape(-1, 8, 3)

    voxel_coords = voxel_coords.transpose(0, 1)  # [8,P,3]

    voxel_coords += center

    # Only consider the pixel all its nearby voxels are in bound
    in_bounds = (voxel_coords >= 0) & (voxel_coords < L)
    all_in_bounds = in_bounds.all(dim=2).all(dim=0)

    voxel_coords = voxel_coords[:, all_in_bounds, :]

    f_total = interp_weights * freqs.unsqueeze(0)  # [8,P] # Multiply each col by freq
    f_total = f_total[:, all_in_bounds]

    w_total = interp_weights * pixel_weights.unsqueeze(0)  # [8,P]
    w_total = w_total[:, all_in_bounds]

    lin_idx = voxel_coords[:, :, 0] * L * L + voxel_coords[:, :, 1] * L + voxel_coords[:, :, 2]

    V_flat = torch.zeros(L * L * L, dtype=torch.complex64, device=device)
    W_flat = torch.zeros(L * L * L, dtype=torch.float32, device=device)

    f_total = f_total.to(torch.complex64)
    V_flat.index_add_(0, lin_idx.reshape(-1), f_total.reshape(-1))

    W_flat.index_add_(0, lin_idx.reshape(-1), w_total.reshape(-1))

    V = V_flat.view(L, L, L)
    W = W_flat.view(L, L, L)
    return V, W


def backproject2Dto3D_single_loop(
        f_image: Tensor,  # shape: (L, L), 2D fourier image
        R: Tensor,  # shape: (3, 3), rotation matrix
        max_r2: float,  # scalar: maximum radius**2 of image to backproject
        pixel_weights: Tensor = None,  # shape: (L, L), weight for each pixel
) -> Tuple[Tensor, Tensor]:
    '''
    Backproject a single 2D fourier image to 3D volume

    Coding just like Relion using for loop to iterate each pixel, which is much less efficient in GPU
    '''
    L = f_image.shape[0]
    weight = torch.zeros((L, L, L))
    V = torch.zeros((L, L, L), dtype=torch.complex64)

    Rinv = R.T
    coords_rot0 = []
    interp_weights0 = []
    v_ = []
    for y in range(-L // 2, L // 2):
        for x in range(-L // 2, L // 2):
            # if x==0 and y<0:
            #     continue
            # TODO: 2D weight for each pos, set as 1 for now
            # TODO: Change the name...

            # Excluded pixels from the confidence interval determind by Nyquist rate
            # Simplified from Relion, which also considers magMatrix
            if x ** 2 + y ** 2 > max_r2:
                continue
            my_weight = pixel_weights[y + L // 2, x + L // 2]

            if my_weight < 0: continue

            freq = f_image[y + L // 2, x + L // 2]
            # xu = m00 * x + m01 * y
            # yu = m10 * x + m11 * y
            xu = x
            yu = y
            z_on_ewaldp = 0

            # Coordinate after rotation
            xp = Rinv[0, 0] * xu + Rinv[0, 1] * yu + Rinv[0, 2] * z_on_ewaldp
            yp = Rinv[1, 0] * xu + Rinv[1, 1] * yu + Rinv[1, 2] * z_on_ewaldp
            zp = Rinv[2, 0] * xu + Rinv[2, 1] * yu + Rinv[2, 2] * z_on_ewaldp

            coords_rot0.append([xp.item(), yp.item(), zp.item()])
            # if xp < 0:
            #      # Get complex conjugated hermitian symmetry pair
            #      xp, yp, zp = -xp, -yp, -zp
            #      freq = torch.conj(freq)
            # Trilinear interpolation
            x0 = torch.floor(xp).to(torch.int32)
            fx = xp - x0
            x1 = x0 + 1

            y0 = torch.floor(yp).to(torch.int32)
            fy = yp - y0
            y1 = y0 + 1

            z0 = torch.floor(zp).to(torch.int32)
            fz = zp - z0
            z1 = z0 + 1

            mfx = 1.0 - fx
            mfy = 1.0 - fy
            mfz = 1.0 - fz

            dd000 = mfz * mfy * mfx
            dd001 = mfz * mfy * fx
            dd010 = mfz * fy * mfx
            dd011 = mfz * fy * fx
            dd100 = fz * mfy * mfx
            dd101 = fz * mfy * fx
            dd110 = fz * fy * mfx
            dd111 = fz * fy * fx

            interp_weights0.append(
                [dd000.item(), dd001.item(), dd010.item(), dd011.item(), dd100.item(), dd101.item(), dd110.item(),
                 dd111.item()])
            v_.append([[z0, y0, x0],
                       [z0, y0, x1],
                       [z0, y1, x0],
                       [z0, y1, x1],
                       [z1, y0, x0],
                       [z1, y0, x1],
                       [z1, y1, x0],
                       [z1, y1, x1]])
            # Excluded out-of-bound
            # Note: Relion did y0-=STARTINGY L255, probably the split index of the volume
            if (x0 + L // 2 < 0 or x1 + L // 2 >= L or
                    y0 + L // 2 < 0 or y1 + L // 2 >= L or
                    z0 + L // 2 < 0 or z1 + L // 2 >= L):
                continue

            center = L // 2

            def is_dc(z, y, x):
                return (z == center) and (y == center) and (x == center)

            # Trilinear updates with DC check
            for dz, dy, dx, dd in [
                (z0, y0, x0, dd000), (z0, y0, x1, dd001),
                (z0, y1, x0, dd010), (z0, y1, x1, dd011),
                (z1, y0, x0, dd100), (z1, y0, x1, dd101),
                (z1, y1, x0, dd110), (z1, y1, x1, dd111)
            ]:
                zi = dz + center
                yi = dy + center
                xi = dx + center

                V[zi, yi, xi] += dd * freq
                weight[zi, yi, xi] += dd * my_weight

    return V, weight
