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

import os
import torch
import time
import numpy as np

import starfile
import mrcfile
from tqdm import tqdm

import accelerate

import torch.nn.functional as F

from cryofm.core.fourier import fourier_to_primal_3d

from .backprojection import Backprojector
from .bp_utils import fourier_translation, create_so3


class BackprojectReconstruct(torch.nn.Module):
    def __init__(self, data_starfile, half_map_idx=None, batch_size=128, settings=None):
        super().__init__()
        # FFT(images) Consistent with notation in the paper, for now
        # self.X = f_images
        # mrc path = abs path + path in starfile
        self.mrc_abs_path = ""

        self.data_starfile = data_starfile
        self.half_map_idx = half_map_idx
        self.batch_size = batch_size

        self.radial_avg = False

        self.add_tau = False
        self.padding = None

        self.ctf_correction = True
        # If also generate fourier mask alongside
        self.fourier_mask = False
        # Default hyperparameters
        self.max_iter = 10

        self.default_optic_params = {
            'kV': 200.0,  # _rlnVoltage
            'Cs': 0.0,  # _rlnSphericalAberration
            'Bfac': 0.0,  # _rlnCtfBfactor
            'scale': 1.0,  # _rlnCtfScalefactor
            'Q0': 0.0,  # _rlnAmplitudeContrast
            'phase_shift': 0.0  # _rlnPhaseShift
        }

        self.default_particle_params = {
            'DeltafU': 0.0,  # _rlnDefocusU
            'DeltafV': None,  # _rlnDefocusV (fallback: use DeltafU)
            'azimuthal_angle': 0.0,  # _rlnDefocusAngle
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if settings is not None:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Unknown setting '{key}' ignored.")

        self.data_setup()
        print("Number of particles: ", self.N)

    def data_setup(self):
        df = starfile.read(self.data_starfile)
        self.df_optic = df['optics']
        self.df_particle = df['particles']
        if self.half_map_idx is not None:
            self.df_particle = self.df_particle[self.df_particle['rlnRandomSubset'] == self.half_map_idx]
        self.pixel_size = self.df_optic['rlnImagePixelSize'].values[0].item()
        self.voxel_size = self.pixel_size
        eulers = torch.tensor(self.df_particle[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].values,
                              dtype=torch.float32, device=self.device)
        self.eulers = torch.deg2rad(eulers)

        self.N = len(self.df_particle)
        self.L = int(self.df_optic['rlnImageSize'].values[0].item())

        image_name_split = self.df_particle["rlnImageName"].str.split("@", expand=True)
        image_idx = image_name_split[0]
        self.image_idx = image_idx.astype(int) - 1
        self.mrcs_path = self.mrc_abs_path + image_name_split[1]

    def get_particle_param(self, col, table, default):
        N = len(table)
        if col in table:
            return torch.tensor(table[col].values, dtype=torch.float32, device=self.device)
        else:
            if col == "rlnDefocusV":
                return default
            else:
                return torch.full((N,), default, dtype=torch.float32, device=self.device)

    def get_optic_param(self, col_name, table, default):
        # For now only support one optic group

        if col_name in table:
            param = table[col_name][0]
        else:
            param = default
        param = torch.tensor(param, dtype=torch.float32, device=self.device)
        return param

    def get_ctf_params(self, optic_table, particle_table):
        # Optic group parameters
        # TODO: Read optics by different groups
        N = len(particle_table)
        self.kV = self.get_optic_param('rlnVoltage', optic_table, self.default_optic_params['kV']).expand(N)
        self.Cs = self.get_optic_param('rlnSphericalAberration', optic_table, self.default_optic_params['Cs']).expand(N)
        self.Bfac = self.get_optic_param('rlnCtfBfactor', optic_table, self.default_optic_params['Bfac']).expand(N)
        self.scale = self.get_optic_param('rlnCtfScalefactor', optic_table, self.default_optic_params['scale']).expand(
            N)
        self.Q0 = self.get_optic_param('rlnAmplitudeContrast', optic_table, self.default_optic_params['Q0']).expand(N)
        self.phase_shift = self.get_optic_param('rlnPhaseShift', optic_table,
                                                self.default_optic_params['phase_shift']).expand(N)

        # Particle parameters
        self.DeltafU = self.get_particle_param('rlnDefocusU', particle_table, self.default_particle_params['DeltafU'])
        self.DeltafV = self.get_particle_param('rlnDefocusV', particle_table, self.DeltafU)
        self.azimuthal_angle = self.get_particle_param('rlnDefocusAngle', particle_table,
                                                       self.default_particle_params['azimuthal_angle'])

    def compute_ctf(self, optic_table, particle_table):
        '''
        Relion source: ctf.cpp/CTF::initialise, ctf.cpp/CTF::getFftwImage, ctf.h/getCTF

        Returns
        -------
        ctf : torch Tensor (N,L,L)

        '''
        self.get_ctf_params(optic_table, particle_table)

        local_Cs = self.Cs * 1e7
        local_kV = self.kV * 1e3
        rad_azimuth = torch.deg2rad(self.azimuthal_angle)

        # defocus_average = -(DeltafU + DeltafV) * 0.5
        # defocus_deviation = -(DeltafU - DeltafV) * 0.5

        # Relativistic electron wavelength (in Angstroms)
        lambda_A = 12.2643247 / torch.sqrt(local_kV * (1 + local_kV * 0.978466e-6))

        PI = torch.pi
        K1 = PI * lambda_A
        K2 = 0.5 * PI * local_Cs * lambda_A ** 3
        K3 = torch.atan(self.Q0 / torch.sqrt(1 - self.Q0 ** 2))
        K4 = -self.Bfac / 4.0
        K5 = torch.deg2rad(self.phase_shift)

        cos_az = torch.cos(rad_azimuth)
        sin_az = torch.sin(rad_azimuth)

        # Q: shape (N, 2, 2)
        Q = torch.stack([
            torch.stack([cos_az, sin_az], dim=-1),
            torch.stack([-sin_az, cos_az], dim=-1)
        ], dim=1).float().to(self.device)

        # D: shape (N, 2, 2)
        D = torch.zeros((self.DeltafU.shape[0], 2, 2), device=self.device)
        D[:, 0, 0] = -self.DeltafU
        D[:, 1, 1] = -self.DeltafV

        # A = Q^T @ D @ Q for each particle
        A = torch.bmm(Q.transpose(1, 2), torch.bmm(D, Q))  # shape: (N, 2, 2)

        Axx = A[:, 0, 0]
        Axy = A[:, 0, 1]
        Ayy = A[:, 1, 1]

        Axx = Axx[:, None, None]
        Axy = Axy[:, None, None]
        Ayy = Ayy[:, None, None]
        K1 = K1[:, None, None]
        K2 = K2[:, None, None]
        K3 = K3[:, None, None]
        K4 = K4[:, None, None]
        K5 = K5[:, None, None]
        scale = self.scale[:, None, None]

        freqs = torch.fft.fftshift(torch.fft.fftfreq(self.L, d=self.pixel_size)).to(self.device)
        Y, X = torch.meshgrid(freqs, freqs, indexing='ij')
        X = X[None, :, :]
        Y = Y[None, :, :]

        u2 = X ** 2 + Y ** 2

        gamma = (
                K1 * (Axx * X ** 2 + 2 * Axy * X * Y + Ayy * Y ** 2)
                + K2 * u2 ** 2
                - K5
                - K3
        )

        ctf = -torch.sin(gamma)
        ctf *= torch.exp(K4 * u2)  # B-factor decay
        ctf *= scale
        # Avoid exact zeros to improve numerical stability (GPU-safe)
        self.CTF = torch.where(torch.abs(ctf) < 1e-8, torch.sign(ctf) * 1e-8, ctf)
        return self.CTF

    def reconstruct(self):
        # TODO: Might prefer not updating self.L
        if self.padding is not None:
            self.pad = int(self.L * (self.padding - 1) // 2)
            self.L = self.L + 2 * self.pad

        self.V = torch.zeros((self.L, self.L, self.L), dtype=torch.complex64, device=self.device)
        self.W = torch.zeros((self.L, self.L, self.L), dtype=torch.float32, device=self.device)
        self.V_ctf = torch.zeros((self.L, self.L, self.L), device=self.device)
        if self.fourier_mask:
            self.V_mask = torch.zeros((self.L, self.L, self.L), device=self.device)
            fourier_mask = torch.ones((self.batch_size, self.L, self.L), device=self.device)

        # Nyquist
        self.max_r2 = (self.L // 2) ** 2
        for j in tqdm(range(0, self.N, self.batch_size)):
            # print(f'{j}/{self.N}')
            batch_indices = slice(j, min(j + self.batch_size, self.N))
            df_particle_batch = self.df_particle.iloc[batch_indices]
            eulers_batch = self.eulers[batch_indices]

            start_time = time.time()

            images = []
            for idx, path in zip(self.image_idx[batch_indices], self.mrcs_path[batch_indices]):
                with mrcfile.mmap(path, permissive=True) as mrcs:
                    if mrcs.data.ndim > 2:
                        images.append(np.array(mrcs.data[idx]))
                    else:
                        images.append(np.array(mrcs.data))
            print("Data time: ", time.time() - start_time)
            images = torch.from_numpy(np.stack(images)).float().to(device=self.device)
            N_b = images.shape[0]

            if self.ctf_correction:
                CTF = self.compute_ctf(self.df_optic, df_particle_batch)

            if self.padding is not None:
                images = F.pad(images, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

            images = torch.fft.ifftshift(images, dim=(-2, -1))
            X = torch.fft.fft2(images, dim=(-2, -1))
            X = torch.fft.fftshift(X, dim=(-2, -1))

            # from Angstrom to pixel unit
            if 'rlnOriginXAngst' in df_particle_batch:
                x_shifts = torch.tensor(df_particle_batch['rlnOriginXAngst'].values,
                                        device=self.device) / self.pixel_size
                y_shifts = torch.tensor(df_particle_batch['rlnOriginYAngst'].values,
                                        device=self.device) / self.pixel_size
            else:
                x_shifts = torch.tensor(df_particle_batch['rlnOriginX'].values, device=self.device)
                y_shifts = torch.tensor(df_particle_batch['rlnOriginY'].values, device=self.device)

            X = fourier_translation(X, x_shifts.float(), y_shifts.float())

            if self.ctf_correction:
                X *= CTF

            # Variance of noise
            sigma2 = torch.ones(N_b, self.L, self.L, device=self.device)
            # Variance of density
            tao2 = torch.ones(self.L, self.L, device=self.device)
            X /= sigma2

            pixel_weights = torch.ones(self.L, self.L, device=self.device)

            start_time = time.time()

            R = create_so3(-eulers_batch, device=self.device)

            backprojector = Backprojector(self.L, R, self.max_r2, pixel_weights=pixel_weights)
            V_b, W_b, V_b_idx, W_b_idx = backprojector.backproject2Dto3D(X)
            self.V[V_b_idx] += V_b[V_b_idx]
            self.W[W_b_idx] += W_b[W_b_idx]

            # for i in range(N_b):
            #     f_image = X[i]
            #     R = create_so3_single(-eulers_batch[i], device=self.device)
            #     V_i, W_i = backproject2Dto3D_single(f_image, R, max_r2, pixel_weights)
            #     self.V += V_i
            #     self.W += W_i

            if self.fourier_mask:
                fourier_mask = torch.ones((N_b, self.L, self.L), device=self.device)
                V_mask_b, _, V_mask_idx, _ = backprojector.backproject2Dto3D(fourier_mask)
                self.V_mask[V_mask_idx] += V_mask_b[V_mask_idx]

            if self.ctf_correction:
                CTF_term = CTF ** 2 / sigma2
                V_ctf_b, _, V_ctf_b_idx, _ = backprojector.backproject2Dto3D(CTF_term)
                self.V_ctf[V_ctf_b_idx] += V_ctf_b[V_ctf_b_idx]

                # for i in range(N_b):
                #     f_image = CTF_term[i]
                #     R = create_so3_single(-eulers_batch[i], device=self.device)
                #     V_i,_ = backproject2Dto3D_single(f_image, R, max_r2, pixel_weights)
                #     self.V_ctf += V_i

            print("Bp time: ", time.time() - start_time)

        # TODO: Might blow up GPU memory, covert to cpu for now
        self.V = self.V.cpu()
        self.W = self.W.cpu()
        self.V_ctf = self.V_ctf.cpu()

        mask = self.W > 1e-5
        self.V_ctf[mask] = self.V_ctf[mask] / self.W[mask]
        self.V_ctf[~mask] = 0.0

        if self.add_tau:
            self.V_ctf += 1 / tao2.cpu()

        self.V[mask] = self.V[mask] / self.W[mask]
        self.V[~mask] = 0.0

        # 3D Nyquist mask
        zs, ys, xs = torch.meshgrid(torch.arange(-self.L // 2, self.L // 2, device=self.V.device),
                                    torch.arange(-self.L // 2, self.L // 2, device=self.V.device),
                                    torch.arange(-self.L // 2, self.L // 2, device=self.V.device), indexing='ij')
        r2 = xs ** 2 + ys ** 2 + zs ** 2
        Nyq_mask3D = r2 <= self.max_r2
        self.V[~Nyq_mask3D] = 0

        if self.radial_avg:
            # To avoid dividing by 0, round V_ctf to its 1e-3 *radial average if it is too small
            V_ctf_radavg = self.radially_average(self.V_ctf)

            V_ctf_rounded = torch.where(
                self.V_ctf < 1e-3 * V_ctf_radavg,
                1e-3 * V_ctf_radavg,
                self.V_ctf
            )
            self.V = self.V / V_ctf_rounded

        else:
            mask = self.V_ctf > 1e-5
            self.V[mask] = self.V[mask] / self.V_ctf[mask].to(self.V.dtype)
            self.V[~mask] = 0.0

        # self.V = torch.where(V_ctf_rounded > 1e-8, self.V / V_ctf_rounded, torch.zeros_like(self.V))

        return

    def get_fourier_mask(self):
        self.V_mask = torch.zeros((self.L, self.L, self.L), device=self.device)

        # Nyquist
        self.max_r2 = (self.L // 2) ** 2
        for j in tqdm(range(0, self.N, self.batch_size)):
            # print(f'{j}/{self.N}')
            batch_indices = slice(j, min(j + self.batch_size, self.N))
            df_particle_batch = self.df_particle.iloc[batch_indices]
            eulers_batch = self.eulers[batch_indices]

            pixel_weights = torch.ones(self.L, self.L, device=self.device)

            R = create_so3(-eulers_batch, device=self.device)

            backprojector = Backprojector(self.L, R, self.max_r2, pixel_weights=pixel_weights)

            fourier_mask = torch.ones((eulers_batch.shape[0], self.L, self.L), device=self.device)
            V_mask_b, _, V_mask_idx, _ = backprojector.backproject2Dto3D(fourier_mask)
            self.V_mask[V_mask_idx] += V_mask_b[V_mask_idx]

        V_mask_np = self.V_mask.cpu().numpy().astype(np.float32)
        return V_mask_np

    def get_fourier_mask_dp(self, accelerator):
        self.V_mask = torch.zeros((self.L, self.L, self.L), device=self.device)
        # Nyquist
        self.max_r2 = (self.L // 2) ** 2
        batch_indices_list = []
        for j in range(0, self.N, self.batch_size):
            batch_indice = slice(j, min(j + self.batch_size, self.N))
            batch_indices_list.append(batch_indice)

        with accelerator.split_between_processes(batch_indices_list) as sub_batch_indices:
            for batch_indices in tqdm(sub_batch_indices):
                eulers_batch = self.eulers[batch_indices]
                pixel_weights = torch.ones(self.L, self.L, device=self.device)
                R = create_so3(-eulers_batch, device=self.device)
                backprojector = Backprojector(self.L, R, self.max_r2, pixel_weights=pixel_weights)
                fourier_mask = torch.ones((eulers_batch.shape[0], self.L, self.L), device=self.device)
                V_mask_b, _, V_mask_idx, _ = backprojector.backproject2Dto3D(fourier_mask)
                self.V_mask[V_mask_idx] += V_mask_b[V_mask_idx]

        accelerator.wait_for_everyone()
        self.V_mask = accelerate.utils.reduce(self.V_mask, "mean")
        V_mask_np = self.V_mask.cpu().numpy().astype(np.float32)
        return V_mask_np

    def radially_average(self, weight):
        '''
        Return the radial average of a 3D weight

        Relion source: backprojector::reconstruct, L1514

        Note:
            Could be redundant: The part out of Nyquist is rounded to the radial average on the Nyquist.

        '''
        zs, ys, xs = torch.meshgrid(torch.arange(-self.L // 2, self.L // 2, device=weight.device),
                                    torch.arange(-self.L // 2, self.L // 2, device=weight.device),
                                    torch.arange(-self.L // 2, self.L // 2, device=weight.device), indexing='ij')
        r2 = xs ** 2 + ys ** 2 + zs ** 2
        r2_floor = torch.floor(r2).to(torch.int64)
        r2_flat = r2_floor.view(-1)
        weight_flat = weight.view(-1)
        r2_clamped = torch.clamp(r2_flat, max=self.max_r2)
        num_bins = self.max_r2 + 1

        bin_sums = torch.zeros(num_bins, dtype=weight_flat.dtype, device=weight.device).scatter_add(0, r2_clamped,
                                                                                                    weight_flat)
        bin_counts = torch.zeros(num_bins, dtype=torch.int64, device=weight.device).scatter_add(0, r2_clamped,
                                                                                                torch.ones_like(
                                                                                                    weight_flat,
                                                                                                    dtype=torch.int64))
        radial_avg = torch.where(
            bin_counts > 0,
            bin_sums / bin_counts,
            torch.zeros_like(bin_sums)
        )

        avg_weight_flat = radial_avg[r2_clamped]
        avg_weight = avg_weight_flat.view(self.L, self.L, self.L)
        return avg_weight

    def get_result(self):

        V_spatial = fourier_to_primal_3d(self.V)

        V_real = V_spatial.real
        V_np = V_real.cpu().numpy().astype(np.float32)

        if self.padding is not None:
            lb = self.pad
            ub = -self.pad
            V_np = V_np[lb:ub, lb:ub, lb:ub]

        with mrcfile.new(self.save_path + '.mrc', overwrite=True) as mrc:
            mrc.set_data(V_np)
            mrc.voxel_size = self.voxel_size
            mrc.update_header_from_data()

        print(self.save_path + ' is saved')
        return
