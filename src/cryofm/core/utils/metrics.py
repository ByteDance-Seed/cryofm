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

from typing import Union, List

import numpy as np
import numpy.linalg as LA
from scipy import ndimage
from scipy.spatial import cKDTree
from multiprocessing import cpu_count, Pool
import mrcfile

import matplotlib.pyplot as plt

from cryofm.core.fourier import np_real_to_rft, rfft3_freq, fft2_freq, fft3_freq, np_real_to_ft


def calc_frc(image1, image2, fourier_mask=None, num_shells=None):
    assert image1.shape == image2.shape, f"Shape mis-match {image1.shape} vs {image2.shape}"

    ny, nx = image1.shape

    if num_shells is None:
        num_shells = nx // 2 + 1

    image1 = np_real_to_ft(image1)
    image2 = np_real_to_ft(image2)

    ky, kx, radii = fft2_freq((ny, nx))
    # odd shape has max freq less than 0.5
    max_radii = min(np.abs(ky).max(), np.abs(kx).max())

    shell_radii = np.linspace(0, max_radii, num_shells, endpoint=True)

    labels = np.searchsorted(shell_radii, radii, side="left")
    # skip 0 freq
    index = np.arange(1, len(shell_radii))
    shell_count = len(index)

    numerator = _in_mask_ndi_sum(np.real(image1 * np.conj(image2)), mask=fourier_mask, labels=labels, index=index)
    denominator = np.sqrt(_in_mask_ndi_sum(np.abs(image1) ** 2, mask=fourier_mask, labels=labels, index=index) *
                          _in_mask_ndi_sum(np.abs(image2) ** 2, mask=fourier_mask, labels=labels, index=index))

    fsc_values = np.zeros(shell_count)
    for i in range(shell_count):
        if denominator[i] > 1e-5:
            fsc_values[i] = numerator[i] / denominator[i]

    fsc_freq = shell_radii[1:]

    return fsc_values, fsc_freq


def _in_mask_ndi_sum(data, mask=None, labels=None, index=None):
    if mask is None:
        data_masked = data
    else:
        data_masked = np.where(mask, data, 0)
    return ndimage.sum(data_masked, labels=labels, index=index)


# if fourier mask is not None, will calculate fsc in mask only
def calc_fsc(vol1, vol2, fourier_mask=None, num_shells=None):
    assert vol1.shape == vol2.shape, f"Shape mis-match {vol1.shape} vs {vol2.shape}"

    nz, ny, nx = vol1.shape

    if num_shells is None:
        num_shells = nx // 2 + 1

    vol1 = np_real_to_ft(vol1)
    vol2 = np_real_to_ft(vol2)

    kz, ky, kx, radii = fft3_freq((nz, ny, nx))
    # odd shape has max freq less than 0.5
    max_radii = min(np.abs(kz).max(), np.abs(ky).max(), np.abs(kx).max())

    shell_radii = np.linspace(0, max_radii, num_shells, endpoint=True)

    labels = np.searchsorted(shell_radii, radii, side="left")
    # skip 0 freq
    index = np.arange(1, len(shell_radii))
    shell_count = len(index)

    numerator = _in_mask_ndi_sum(np.real(vol1 * np.conj(vol2)), mask=fourier_mask, labels=labels, index=index)
    denominator = np.sqrt(_in_mask_ndi_sum(np.abs(vol1) ** 2, mask=fourier_mask, labels=labels, index=index) *
                          _in_mask_ndi_sum(np.abs(vol2) ** 2, mask=fourier_mask, labels=labels, index=index))

    fsc_values = np.zeros(shell_count)
    for i in range(shell_count):
        if denominator[i] > 1e-5:
            fsc_values[i] = numerator[i] / denominator[i]

    fsc_freq = shell_radii[1:]

    return fsc_values, fsc_freq


# if all fsc is small than threshold, this function will output the resolution defined by the minimum frequency
def fsc_to_resolution(fsc_values, shell_freq=None, threshold=0.143, voxel_size=1.0):
    assert len(fsc_values) == len(shell_freq)

    below_threshold = fsc_values <= threshold
    cross_index = np.where(below_threshold)[0][0] if np.any(below_threshold) else len(fsc_values) - 1

    if shell_freq is None:
        return cross_index

    # default freq for cross_index = 0 or = len(fsc_values) - 1, fsc_values[cross_index] = threshold
    freq = shell_freq[cross_index]

    # linear interp
    if cross_index > 0 and fsc_values[cross_index - 1] > threshold > fsc_values[cross_index]:
        frac = (threshold - fsc_values[cross_index]) / (fsc_values[cross_index - 1] - fsc_values[cross_index])
        freq = shell_freq[cross_index] * (1.0 - frac) + shell_freq[cross_index - 1] * frac

    return voxel_size / freq


def plot_fsc(fsc_values, pix_freq,
             threshold: Union[List, float] = 0.143, voxel_size=1.0, save_path=None, fontname=None):
    if fontname is not None:
        plt.rcParams["font.family"] = fontname  # "Times New Roman"

    max_freq = max(pix_freq)

    # pre-defined ticks
    xtick_locs = [i for i in (.01, .05, .0833, .125, .1667, .2, .25, .3333, .4, .5) if i <= max_freq]
    xtick_lbl = ["1/100", "1/20", "1/12", "1/8", "1/6", "1/5", "1/4", "1/3", "1/2.5", "1/2"][:len(xtick_locs)]
    ytick_locs = (0.0, .125, .143, .25, .375, .5, .625, .75, .875, 1.0)
    ytick_lbl = ("0", " ", "0.143", "0.25", " ", "0.5", " ", "0.75", " ", "1.0")

    if isinstance(threshold, float):
        threshold = [threshold, ]

    res_list = []
    for t in threshold:
        res = fsc_to_resolution(fsc_values, pix_freq, t, voxel_size)
        res_list.append(res)
        freq = voxel_size / res
        plt.annotate(r"{:1.2f} $\AA$".format(res), xy=(freq, t),
                     xytext=(freq, t + 0.05),
                     fontsize=12.,
                     # arrowprops={"width": 1, "shrink": .05}
                     )
        plt.axhline(t, color='k', linestyle=":")

    plt.plot(pix_freq, fsc_values, linewidth=1)
    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("FSC")

    plt.xticks(xtick_locs, xtick_lbl)
    plt.yticks(ytick_locs, ytick_lbl)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.cla()
    plt.close()
    return res_list


def plot_multiple_fsc(fsc_values_list: List, pix_freq,
             threshold: Union[List, float] = 0.143, labels=None, voxel_size=1.0, save_path=None, fontname=None):
    if fontname is not None:
        plt.rcParams["font.family"] = fontname  # "Times New Roman"

    max_freq = max(pix_freq)

    # pre-defined ticks
    xtick_locs = [i for i in (.01, .05, .0833, .125, .1667, .2, .25, .3333, .4, .5) if i <= max_freq]
    xtick_lbl = ["1/100", "1/20", "1/12", "1/8", "1/6", "1/5", "1/4", "1/3", "1/2.5", "1/2"][:len(xtick_locs)]
    ytick_locs = (0.0, .125, .143, .25, .375, .5, .625, .75, .875, 1.0)
    ytick_lbl = ("0", " ", "0.143", "0.25", " ", "0.5", " ", "0.75", " ", "1.0")

    if isinstance(threshold, float):
        threshold = [threshold, ]

    res_list = []
    for i, fsc_values in enumerate(fsc_values_list):
        tmp_res_list = []
        for t in threshold:
            res = fsc_to_resolution(fsc_values, pix_freq, t, voxel_size)
            res_list.append(res)
            freq = voxel_size / res
            plt.annotate(r"{:1.2f} $\AA$".format(res), xy=(freq, t),
                         xytext=(freq, t + 0.05),
                         fontsize=12.,
                         # arrowprops={"width": 1, "shrink": .05}
                         )
            plt.axhline(t, color='k', linestyle=":")
        res_list.append(tmp_res_list)

        label = None
        if labels is not None:
            label = labels[i]
        plt.plot(pix_freq, fsc_values, linewidth=1, label=label)

    plt.xlabel("Spatial Frequency (1/pixel)")
    plt.ylabel("FSC")

    if labels is not None:
        plt.legend()

    plt.xticks(xtick_locs, xtick_lbl)
    plt.yticks(ytick_locs, ytick_lbl)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.cla()
    plt.close()
    return res_list


def calc_cc(vol1, vol2, eps=1e-7):
    assert vol1.shape == vol2.shape, f"Shape mis-match {vol1.shape} vs {vol2.shape}"

    vol1 = vol1.ravel()
    vol2 = vol2.ravel()

    norm1 = LA.norm(vol1, 2)
    norm2 = LA.norm(vol2, 2)

    return np.sum(vol1 * vol2) / (norm1 * norm2 + eps)


def get_fsc_map(vol1, vol2, mask):
    """
    Adapted from spisonet.
    """
    h1 = vol1 * mask
    h2 = vol2 * mask
    f1 = np_real_to_ft(h1)
    f2 = np_real_to_ft(h2)
    ret = np.real(np.multiply(f1,np.conj(f2)))
    n1 = np.real(np.multiply(f1,np.conj(f1)))
    n2 = np.real(np.multiply(f2,np.conj(f2)))
    fsc_map = ret/np.sqrt(n1*n2)
    return fsc_map


def calculate_fsc(pixels_T, fsc_values, point_tree, r0):
    """
    Adapted from spisonet.
    """
    values = np.zeros(len(pixels_T[0]))
    for j, pixel in enumerate(zip(pixels_T[0], pixels_T[1], pixels_T[2])):
        filter_fsc_values = fsc_values[point_tree.query_ball_point(pixel, r0)]
        if len(filter_fsc_values) < 10:
            print(f"Num of filtered values: {len(filter_fsc_values)}")
        values[j] = filter_fsc_values.mean()
    return values


def ThreeD_FSC(fsc_map, angle=20, n_processes=16):
    """
    Adapted from spisonet.
    """
    from tqdm import tqdm

    nz, ny, nx = fsc_map.shape
    limit_r = nz//2
    
    threshold = 2 * np.sin(np.deg2rad(angle) / 2.0)

    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z ** 2 + Y ** 2 + X ** 2))

    fsc_map = fsc_map.astype(np.float32)
    out = np.zeros_like(fsc_map)
    out[nz // 2, nz // 2, nz // 2] = 1

    with Pool(processes=n_processes) as pool:
        results = []
        for i in tqdm(range(1, limit_r)):
            pixels_T = np.where(index == i)
            pixels = np.transpose(pixels_T)
            point_tree = cKDTree(pixels)
            r0 = i * threshold
            FSC_values = fsc_map[pixels_T]
            results.append(pool.apply_async(calculate_fsc, (pixels_T, FSC_values, point_tree, r0)))

        for i, result in enumerate(tqdm(results)):
            pixels_T = np.where(index == i + 1)
            out[pixels_T] = result.get()
    return out


def calc_3dfsc(vol1, vol2, ncpus=16, mask=None, save_path=None, cone_sampling_angle=10):
    """
    Adapted from spisonet.
    """
    cpu_system = cpu_count()
    if cpu_system < ncpus:
        ncpus = cpu_system

    if mask is None:
        mask = np.ones(vol1.shape, dtype=np.float32)

    fsc_map = get_fsc_map(vol1, vol2, mask=mask)
    fsc3d = ThreeD_FSC(fsc_map, angle=float(cone_sampling_angle), n_processes=ncpus)
    if save_path is not None:
        with mrcfile.new(save_path, overwrite=True) as mrc:
            mrc.set_data(fsc3d.astype(np.float32))
    return fsc3d
