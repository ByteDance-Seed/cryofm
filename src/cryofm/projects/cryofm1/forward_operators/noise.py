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

# here needs some refactor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch

from cryofm.core.fourier import primal_to_fourier_3d


def radial_average_3d_power_spectrum(volume):
    f_volume = primal_to_fourier_3d(volume.squeeze())
    D = f_volume.shape[-1]
    power_spectrum = torch.abs(f_volume) ** 2

    center = (D // 2, D // 2, D // 2)
    # Create a grid of indices
    z, y, x = np.indices(power_spectrum.shape)
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    distances = distances.astype(int)

    radial_avg = np.zeros(D)
    counts = np.zeros(D)

    for i in range(D):
        mask = (distances == i)
        radial_avg[i] = power_spectrum[mask].sum()
        counts[i] = mask.sum()

    # Avoid division by zero
    counts[counts == 0] = 1

    radial_avg /= counts
    radial_avg = torch.from_numpy(radial_avg)
    radial_avg = radial_avg[0: D // 2]

    return radial_avg


def radial_noise_from_snr(y, snr):
    """
    here y is the clean gt, not the observation. obs should be op(y).
    """
    # Calculate radially averaged power spectrum
    radial_signal = radial_average_3d_power_spectrum(y)
    radial_noise = radial_signal / snr
    return radial_noise


def noise_power_from_radial(D, radial_noise):
    center = (D // 2, D // 2, D // 2)
    # Create a grid of indices
    zz, yy, xx = np.indices((D, D, D))
    distances = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    distances = distances.astype(int)

    noise_power = np.zeros((D, D, D))
    for i in range(D // 2):
        noise_power[distances == i] = radial_noise[i]
    noise_power[distances >= D // 2] = radial_noise[-1]

    noise_power = torch.from_numpy(noise_power)
    return noise_power


def noise_power_from_fsc(y_obs, fsc):
    """
    here y_obs is op(y), fsc is the half maps fsc.
    when fsc < 0.05 (that snr < 0.05), high freq signal power estimation can be off by a lot.
    noise power is a bit off in low freq, but generally ok.
    """
    fsc = np.where(fsc <= 0, 1e-9, fsc)
    snr = fsc / (1 - fsc)
    radial_ps = radial_average_3d_power_spectrum(y_obs)  # radial ps should be |S^2 + N^2|
    noise_power = radial_ps / (1 + snr)
    signal_power = (radial_ps * snr) / (1 + snr)
    return noise_power, signal_power


def counts_of_radial_shell(D):
    center = (D // 2, D // 2, D // 2)
    # Create a grid of indices
    z, y, x = np.indices((D, D, D))
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    distances = distances.astype(int)
    num_voxel = np.zeros(D // 2)
    for i in range(D // 2):
        mask = (distances == i)
        num_voxel[i] = mask.sum()
    return num_voxel


def noise_power_from_amp_factor_matrix(y_obs, radial_noise_power, amp_factor_matrix):
    """
    here y_obs is op(y), radial_noise_power inferred from gsfsc,
    amp_factor_matrix from op.
    """
    D = amp_factor_matrix.shape[-1]
    counts_per_shell = counts_of_radial_shell(y_obs.shape[-1])
    counts_matrix = noise_power_from_radial(y_obs.shape[-1], counts_per_shell)
    noise_power_iso = noise_power_from_radial(y_obs.shape[-1], radial_noise_power)

    amp_factor_radial_sum = np.zeros(D // 2)
    center = (D // 2, D // 2, D // 2)
    z, y, x = np.indices(amp_factor_matrix.shape)
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    distances = distances.astype(int)
    for i in range(D // 2):
        mask = (distances == i)
        amp_factor_radial_sum[i] = amp_factor_matrix[mask].sum()

    denominator = noise_power_from_radial(y_obs.shape[-1], amp_factor_radial_sum)
    # denominator = amp_factor_matrix.sum()
    scale = 1.0
    noise_power = (scale * counts_matrix * noise_power_iso * amp_factor_matrix) / denominator
    return noise_power


def plot_radial_power(radial_powers, pix_freq, save_path=None, fontname=None):
    if fontname is not None:
        plt.rcParams["font.family"] = fontname  # "Times New Roman"

    max_freq = max(pix_freq)

    # pre-defined ticks
    xtick_locs = [i for i in (.01, .05, .0833, .125, .1667, .2, .25, .3333, .4, .5) if i <= max_freq]
    xtick_lbl = ["1/100", "1/20", "1/12", "1/8", "1/6", "1/5", "1/4", "1/3", "1/2.5", "1/2"][:len(xtick_locs)]

    for i, radial_power in enumerate(radial_powers):
        plt.plot(pix_freq, radial_power, linewidth=1, label=f'Power {i + 1}')
    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("Radially averaged power")

    plt.xticks(xtick_locs, xtick_lbl)
    plt.yscale('log')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.cla()
    plt.close()


def plot_slice_power(noise_powers, slice_axis=0, save_path=None, logscale=False, fontname=None):
    # Determine the slice index based on the chosen axis
    shape = noise_powers[0].shape
    center_idx = shape[slice_axis] // 2

    # Extract slices based on the slice axis
    center_slices = []
    for noise_power in noise_powers:
        try:
            noise_power = noise_power.detach().cpu().numpy()
        except AttributeError:
            pass
        if slice_axis == 0:  # Depth slice
            center_slice = noise_power[center_idx, :, :]
            if logscale:
                center_slices.append(np.log10(center_slice))
            else:
                center_slices.append(center_slice)
        elif slice_axis == 1:  # Height slice
            center_slice = noise_power[:, center_idx, :]
            if logscale:
                center_slices.append(np.log10(center_slice))
            else:
                center_slices.append(center_slice)
        elif slice_axis == 2:  # Width slice
            center_slice = noise_power[:, :, center_idx]
            if logscale:
                center_slices.append(np.log10(center_slice))
            else:
                center_slices.append(center_slice)
        else:
            raise ValueError("Invalid slice_axis. Must be 0, 1, or 2.")

    vmin = min([np.min(s) for s in center_slices])
    vmax = max([np.max(s) for s in center_slices])

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, len(noise_powers) + 1, width_ratios=[1]*len(noise_powers) + [0.05])  # Add space for the color bar
    axs = []
    ims = []

    for i in range(len(noise_powers)):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(center_slices[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Noise power {i}')
        axs.append(ax)
        ims.append(im)

    cbar_ax = fig.add_subplot(gs[-1])
    fig.colorbar(ims[0], cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.cla()
    plt.close()
