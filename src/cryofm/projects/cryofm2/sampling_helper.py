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
import re
import time
import glob
import copy
import math
import random
import logging
import contextlib
import os.path as osp
from logging import getLogger
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Union, Optional, Dict, Any, Tuple, List

import mrcfile
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

import torch
import torch.nn.functional as F
import einops
import accelerate
from accelerate.utils import broadcast_object_list

from cryofm.core.utils.metrics import calc_fsc, plot_fsc
from cryofm.core.utils.mask import create_sphere_mask
from cryofm.core.utils.sampling_fm import sample_from_fm
from cryofm.core.datasets.transforms.patchify import GridPatches3D, GridAggregator, CenterCrop3D
from cryofm.core.utils.shape_utils import get_bbox, bbox_crop, ensure_min_bbox_size, get_pad_width
from cryofm.core.fourier import (fourier_to_primal_3d, primal_to_fourier_3d, np_real_to_rft, rfft3_freq_indices,
                                 fft3_freq_indices, shell_avg)

from cryofm.projects.cryofm2.utils.bp_reconstruct import BackprojectReconstruct
from cryofm.projects.cryofm2.utils.infer_relion_utils import (
    load_mrc, save_mrc, fft, ifft, normalize_voxel_size_fourier, resize_by_voxel_size, rescale_fourier,
    get_radial_mask, combine_map, create_radial_mask, res_from_fsc, get_crossover_grid, index_to_res, res_to_index)

logger = getLogger(__name__)

MODEL_VOXEL_SIZE = 1.5
BBOX_ENLARGE = 5
EPS_FSC = 1e-5
CRYOEM_DENSITY_MEAN = 0.04
CRYOEM_DENSITY_STD = 0.09


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} took {end - start} seconds")


@contextmanager
def change_dir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


class MainProcessFilter(logging.Filter):
    """Filter that only allows logs from the main process. Supports PartialState and Accelerator."""
    def __init__(self, state):
        super().__init__()
        self.state = state
    
    def filter(self, record):
        # Support PartialState and Accelerator, both have is_main_process attribute
        if hasattr(self.state, 'is_main_process'):
            return self.state.is_main_process
        # Backward compatibility: if no is_main_process, check process_index
        elif hasattr(self.state, 'process_index'):
            return self.state.process_index == 0
        return True


# powered by LLM
def auto_find_base_dir(star_file, relative_path):
    """
    Automatically find the base directory that can resolve a relative path.

    This function searches for the correct base directory by traversing upward
    from the star file's directory until it finds a directory where the relative
    path points to an existing file.

    Args:
        star_file (str): Absolute path to the .star file
        relative_path (str): Relative path read from the .star file (e.g., "./path/to/file.mrc")

    Returns:
        str: Absolute path to the base directory where the relative path is valid

    Raises:
        FileNotFoundError: If no valid base directory is found after traversing up to 10 levels

    Example:
        >>> star_file = "/path/to/project/subdir/file.star"
        >>> relative_path = "./data/image.mrc"
        >>> base_dir = auto_find_base_dir(star_file, relative_path)
        >>> # Returns "/path/to/project" if "/path/to/project/data/image.mrc" exists
    """
    # Remove leading "./" from relative path
    relative_path = relative_path.lstrip('./')

    # Start from the directory containing the star file
    current = os.path.dirname(os.path.abspath(star_file))

    # Traverse up to 10 parent directories
    for _ in range(10):
        # Check if the relative path resolves to an existing file in current directory
        if os.path.exists(os.path.join(current, relative_path)):
            return current

        parent = os.path.dirname(current)

        # Stop if we've reached the root directory
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(f"Unable to find base directory for relative path: {relative_path}")


# save
# assume output_path is always .mrc file, when use starfile input in RELION,
# the output_path will be ended with _external_reconstruct.mrc
def generate_debug_filename(output_path: str, suffix: str) -> str:
    directory = osp.dirname(output_path)
    filename = osp.basename(output_path)
    name, ext = osp.splitext(filename)

    if name.endswith("_external_reconstruct"):
        new_name = name[:-len("_external_reconstruct")]
    else:
        new_name = name

    # suffix contains file type
    new_filename = f"{new_name}_{suffix}"
    return osp.join(directory, new_filename)


patch_save_num = -1
map_save_num = -1


def next_numbered_path(path):
    dirpath, filename = osp.split(path)
    name, ext = osp.splitext(filename)

    pattern = re.compile(rf"^{re.escape(name)}_(\d{{3}}){re.escape(ext)}$")
    candidates = glob.glob(osp.join(dirpath, f"{name}_*{ext}"))

    max_n = 0
    for cand in candidates:
        m = pattern.match(osp.basename(cand))
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n

    next_n = max_n + 1
    return osp.join(dirpath, f"{name}_{next_n:03d}{ext}")


def save_patches_as_mrc_main(acl: accelerate.Accelerator, patches: torch.Tensor, base_path: str):
    """
    patches: Tensor[N, D, D, D]
    base_path: e.g. "./folder/file.mrc"  —— next_numbered_path will generate file_001.mrc, file_002.mrc, …
    """
    global patch_save_num

    for vol in tqdm(patches):
        if acl.is_main_process:
            patch_save_num = (patch_save_num + 1) % acl.num_processes

        acl.wait_for_everyone()
        patch_save_num = broadcast_object_list([patch_save_num], from_process=0)[0]

        if acl.process_index == patch_save_num:
            print(f"index:{acl.process_index} saving file")
            path = next_numbered_path(base_path)
            arr = vol.cpu().numpy().astype(np.float32)
            with mrcfile.new(path, overwrite=False) as mrc:
                mrc.set_data(arr)
            print(f"Saved patch to {path}")


def save_mrc_main(acl: accelerate.Accelerator, grid, filename, voxel_size=1., origin=0.):
    """
    patches: Tensor[N, D, D, D]
    base_path: e.g. "./folder/file.mrc"  —— next_numbered_path will generate file_001.mrc, file_002.mrc, …
    """
    global map_save_num
    if acl.is_main_process:
        map_save_num = (map_save_num + 1) % acl.num_processes
    acl.wait_for_everyone()
    map_save_num = broadcast_object_list([map_save_num], from_process=0)[0]

    if acl.process_index == map_save_num:
        print(f"{acl.process_index} saving file")
        save_mrc(grid, filename, voxel_size=voxel_size, origin=origin)


def center_crop(vol, patch_size):
    t = CenterCrop3D(patch_size)
    mask_result_curr = {"map_data": vol}
    return t.transform(mask_result_curr)["map_data"]


def bbox_crop_nd(data, bbox):
    """
    Generalized bbox crop: applies bbox_crop to the last three dims of `data`,
    preserving any leading dims (batch, channels, etc.).

    Args:
        data:  np.ndarray or torch.Tensor of shape [..., D, H, W]
        bbox:  array-like of shape [2, 3]

    Returns:
        cropped: same type as `data`, shape [..., d1-d0+1, h1-h0+1, w1-w0+1]
    """
    # Convert to numpy if torch
    is_tensor = isinstance(data, torch.Tensor)
    arr = data.detach().cpu().numpy() if is_tensor else np.asarray(data)

    # split off leading dims
    lead_shape = arr.shape[:-3]
    D, H, W = arr.shape[-3:]
    flat = arr.reshape(-1, D, H, W)  # shape [N, D, H, W]

    # crop each 3D volume
    cropped_list = [bbox_crop(flat[i], bbox) for i in range(flat.shape[0])]
    cropped = np.stack(cropped_list, axis=0)  # [N, d', h', w']

    # restore leading dims
    new_shape = (*lead_shape, *cropped.shape[1:])
    cropped = cropped.reshape(new_shape)

    # convert back to tensor if needed
    if is_tensor:
        cropped = torch.from_numpy(cropped).to(data.device).type(data.dtype)
    return cropped


def shannon_wavelet_transform_3d(x: torch.Tensor,
                                 n_bands: int = 8) -> torch.Tensor:
    """
    Perform Shannon wavelet transform in Fourier space with bandpass filters.

    Args:
        x: Tensor of shape [..., D, H, W], real-valued
        n_bands: number of radial frequency bands

    Returns:
        coeffs: Tensor of shape [..., n_bands, D, H, W], real-valued
    """
    *lead, D, H, W = x.shape
    # FFT (real to complex)
    f = torch.fft.fftn(x, dim=(-3, -2, -1))  # [..., D, H, W]

    # Compute normalized frequency grid
    fy = torch.fft.fftfreq(D, d=1.0).to(x.device)
    fx = torch.fft.fftfreq(H, d=1.0).to(x.device)
    fz = torch.fft.fftfreq(W, d=1.0).to(x.device)
    ky, kx, kz = torch.meshgrid(fy, fx, fz, indexing="ij")
    freq_radius = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)  # [D, H, W]
    freq_radius = freq_radius.to(x.device)

    # Max normalized freq = 0.5
    max_freq = 0.5
    band_edges = torch.linspace(0.0, max_freq, n_bands + 1, device=x.device)

    responses = []
    for i in range(n_bands):
        f_low = band_edges[i]
        f_high = band_edges[i + 1]
        mask = ((freq_radius >= f_low) & (freq_radius < f_high)).to(f.dtype)  # [D,H,W]
        f_band = f * mask  # [..., D, H, W]
        x_band = torch.fft.ifftn(f_band, dim=(-3, -2, -1)).real  # [..., D, H, W]
        responses.append(x_band)

    return torch.stack(responses, dim=-4)  # [..., n_bands, D, H, W]


def inverse_shannon_wavelet_transform_3d(coeffs: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct real-space volume from Shannon wavelet coefficients.

    Args:
        coeffs: Tensor of shape [..., n_bands, D, H, W]

    Returns:
        x_recon: Tensor of shape [..., D, H, W]
    """
    return coeffs.sum(dim=-4)  # Sum over bands


def maxvalue_norm_and_patchify(input_vol, patch_size, overlap, do_standardize=True):
    ori_data = copy.deepcopy(input_vol)
    if do_standardize:
        max_value = np.percentile(ori_data, 99.999)
        data = ori_data / max_value
        data = (data - CRYOEM_DENSITY_MEAN) / CRYOEM_DENSITY_STD
        results = {"map_data": data}
    else:
        max_value = None
        results = {"map_data": ori_data}
    grid_patcher = GridPatches3D(patch_size=patch_size, patch_overlap=overlap)
    results = grid_patcher.transform(results)
    results["ori_data"] = ori_data
    return max_value, results


def multi_channel_maxvalue_norm_and_patchify(input_vol, patch_size, overlap, do_standardize=True):
    """
    Returns:
      max_values: list of length C
      patched_results: dict with keys
        'patches' (np.ndarray of shape [C, n_patches, pD, pH, pW]),
        'locations', 'shape_before_patchify', 'ori_data'
    """
    arr = input_vol
    C = arr.shape[0]

    max_values = []
    patches_list = []
    locations = None
    shape0 = None

    for c in range(C):
        single = arr[c]
        max_v, results = maxvalue_norm_and_patchify(
            single,
            patch_size=patch_size,
            overlap=overlap,
            do_standardize=do_standardize
        )
        max_values.append(max_v)
        patches_list.append(results["patches"])
        if c == 0:
            locations = results["locations"]
            shape0 = results["shape_before_patchify"]
    patches_stack = np.stack(patches_list, axis=1)
    # print(f"DEBUG: {patches_stack.shape}")
    patched_results = {
        "patches": patches_stack,
        "locations": locations,
        "shape_before_patchify": shape0,
        "ori_data": arr,
    }
    return max_values, patched_results


def radial_shell_average(data: np.ndarray,
                         is_rft: bool = False,
                         already_freq: bool = False,
                         harmonic_mean: bool = False) -> np.ndarray:
    """
    Compute radial average amplitude of frequency shell layers.

    Args:
        data (np.ndarray): 3D real-space or frequency-domain data
        is_rft (bool): If True, data is in half-frequency RFT format; otherwise treated as full frequency domain
        already_freq (bool): If True, data is already in frequency domain; otherwise convert from real-space to half-frequency RFT first
        harmonic_mean (bool): If True, compute harmonic mean for each shell; otherwise compute arithmetic mean

    Returns:
        np.ndarray:
        - Half-frequency (RFT): shell average of length = n2
        - Full frequency domain: shell average of length = D//2+1
    """
    # 1. Get frequency domain data
    if already_freq:
        freq_data = data
    else:
        freq_data = np_real_to_rft(data)

    # 2. Select corresponding shell indices
    if is_rft:
        indices = rfft3_freq_indices(freq_data.shape)
    else:
        indices = fft3_freq_indices(freq_data.shape)

    # 3. Compute radial average amplitude
    if not harmonic_mean:
        # Arithmetic mean
        radial = shell_avg(np.abs(freq_data), indices)
    else:
        # Harmonic mean
        abs_vals = np.abs(freq_data).ravel()
        idx_flat = indices.ravel()

        # Compute reciprocal, avoid division by 0
        inv_vals = np.zeros_like(abs_vals, dtype=np.float64)
        nonzero = abs_vals > 0
        inv_vals[nonzero] = 1.0 / abs_vals[nonzero]

        # Count sum of reciprocals and sample count for each shell
        max_bin = idx_flat.max()
        sums_inv = np.bincount(idx_flat, weights=inv_vals, minlength=max_bin + 1).astype(np.float64)
        counts = np.bincount(idx_flat, minlength=max_bin + 1).astype(np.float64)

        # Harmonic mean H = counts / sums_inv
        # Avoid division by zero when sums_inv is 0
        radial = counts / np.clip(sums_inv, 1e-12, None)

    # 4. Crop full frequency domain input to Nyquist
    if already_freq and not is_rft:
        D = freq_data.shape[0]
        radial = radial[: (D // 2 + 1)]

    return radial


def volume_from_radial_shell_avg(radial_avg):
    """
    Reconstruct 3D concentric sphere volume from radial average values.

    Args:
        radial_avg (np.ndarray): Radial average array of shape=(K,)

    Returns:
        np.ndarray: Volume of shape=(D, D, D)
    """
    M = len(radial_avg)
    D = 2 * (M - 1)
    center = D // 2
    # Build coordinate grid
    coords = np.indices((D, D, D), dtype=np.float32)
    x = coords[0] - center
    y = coords[1] - center
    z = coords[2] - center
    # Compute normalized radius [0,1]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_norm = r / center
    # Map to radial_avg indices [0..K-1]
    idx = np.clip((r_norm * (M - 1)).astype(int), 0, M - 1)
    # Fill volume
    volume = radial_avg[idx]
    return volume


@dataclass
class InputData:
    file_type: str = None

    # basic information
    mode: str = None  # "refine", "refine_final", "classification"
    pixel_size: float = None
    original_size: int = None
    particle_diameter: float = None

    # output path
    output_path: str = None
    log_path: str = None

    # optional info
    this_half_index: Optional[int] = None
    fsc_spectra: Optional[np.ndarray] = None

    # here use the naming from Blush
    # we do posterior sampling on recons_unfil, recons is only used for post-processing mixing
    recons_unfil: Optional[np.ndarray] = None
    recons: Optional[np.ndarray] = None

    # half map A & B for calculating likelihood loss weights
    half_volume_a: Optional[np.ndarray] = None
    half_volume_b: Optional[np.ndarray] = None

    # for mixing etc.
    ref_volume: Optional[np.ndarray] = None


def compute_fsc_values(raw_half_a: np.ndarray, raw_half_b: np.ndarray,
                       fsc_spectra: np.ndarray, output_path: str,
                       pixel_size: float, debug: bool = False,
                       accelerator=None) -> np.ndarray:
    if raw_half_a is not None and raw_half_b is not None:
        fsc_vals, fsc_freqs = calc_fsc(raw_half_a, raw_half_b)
        if debug:
            save_path = generate_debug_filename(output_path, "fsc.png")
            if accelerator.is_main_process:
                plot_fsc(fsc_vals, fsc_freqs, voxel_size=pixel_size, save_path=save_path)
        fsc_vals = np.insert(fsc_vals, 0, (1 - 1e-9))
    else:
        if fsc_spectra is None:
            raise ValueError("fsc_spectra is None")
        fsc_vals = fsc_spectra
    fsc_vals = np.where(fsc_vals <= 0, 1e-9, fsc_vals)
    return fsc_vals


def compute_fourier_mask(data_starfile_path: str, half_map_idx: int, batch_size: int = 256):
    settings = {
        'ctf_correction': False,
    }
    model = BackprojectReconstruct(
        data_starfile=data_starfile_path,
        half_map_idx=half_map_idx,
        batch_size=batch_size,
        settings=settings
    )
    pose = getattr(model, "eulers", None)
    # this is an ad-hoc check in case there is no pose in the 1st iteration
    if pose is None or torch.all(pose == 0):
        logger.info(f"Pose is all zero, set mask to a sphere mask")
        # * 1000 to use mask threshold (usually 10)
        mask = create_sphere_mask(model.L, model.L, model.L, radius=model.L // 2).astype(np.float32) * 1000
    else:
        rel_path = model.mrcs_path.iloc[0]
        abs_path = auto_find_base_dir(data_starfile_path, rel_path)
        with change_dir(abs_path):
            mask = model.get_fourier_mask()
    return mask


# only use when 'inpaint' is in op arguments
def build_fourier_mask(starfile_path: str,
                       half_map_idx: int,
                       resized_volume: np.ndarray,
                       patch_size: int,
                       fmask_threshold: float,
                       output_path: str,
                       pixel_size: float,
                       debug: bool = False,
                       accelerator=None) -> tuple[np.ndarray, torch.Tensor]:
    raw_f_mask = compute_fourier_mask(starfile_path, half_map_idx)
    if debug:
        save_path = generate_debug_filename(output_path, "fmask.mrc")
        save_mrc_main(accelerator, raw_f_mask, save_path, pixel_size)
    # use crop & resize to get patch fourier mask
    resized_f_mask = center_crop(raw_f_mask, resized_volume.shape[-1])
    resized_f_mask = (resized_f_mask >= fmask_threshold).astype(bool)
    patch_f_mask = resize(resized_f_mask, (patch_size, ) * 3, order=0)
    nyquist_mask = create_sphere_mask(patch_size, patch_size, patch_size)
    patch_f_mask = patch_f_mask * nyquist_mask
    if debug:
        save_path = generate_debug_filename(output_path, "fmask_patch.mrc")
        save_mrc_main(accelerator, patch_f_mask, save_path, MODEL_VOXEL_SIZE)
    patch_f_mask = torch.from_numpy(patch_f_mask).float()
    return raw_f_mask, patch_f_mask


def compute_wavelet_weights(resized_half_volume_a: np.ndarray,
                            resized_half_volume_b: np.ndarray,
                            resized_input: np.ndarray,
                            nbands: int) -> np.ndarray:
    if resized_half_volume_a is not None and resized_half_volume_b is not None:
        resized_half_volume_a_wavelet = shannon_wavelet_transform_3d(
            torch.from_numpy(resized_half_volume_a).unsqueeze(0),
            n_bands=nbands)
        resized_half_volume_b_wavelet = shannon_wavelet_transform_3d(
            torch.from_numpy(resized_half_volume_b).unsqueeze(0),
            n_bands=nbands
        )
        wavelet_weights = 0.5 * (resized_half_volume_a_wavelet - resized_half_volume_b_wavelet) ** 2
        wavelet_weights = wavelet_weights.squeeze().cpu().numpy()
    else:
        logger.info("Doesn't have two half maps, set wavelet_weights to 1.")
        wavelet_weights = np.ones((nbands,) + resized_input.shape)
    return wavelet_weights


def check_and_pad_volumes_to_cubic(
        raw_input: Optional[np.ndarray],
        raw_recons: Optional[np.ndarray],
        raw_half_a: Optional[np.ndarray],
        raw_half_b: Optional[np.ndarray],
        raw_ref_volume: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[Tuple[int, int]]]]:
    # Check all volumes have the same shape
    volumes = [v for v in [raw_input, raw_recons, raw_half_a, raw_half_b, raw_ref_volume] if v is not None]
    pad_width = None
    if len(volumes) > 0:
        reference_shape = volumes[0].shape
        # Check shape consistency if there are multiple volumes
        if len(volumes) > 1:
            for vol in volumes[1:]:
                if vol.shape != reference_shape:
                    raise ValueError(f"All volumes must have the same shape. Got shapes: {[v.shape for v in volumes]}")
        
        # Check if it's a cube
        is_cube = len(set(reference_shape)) == 1
        
        if not is_cube:
            # Pad to cubic shape
            # Ensure target size is even to satisfy rfft requirements (normalize_voxel_size_fourier expects even sizes)
            max_dim = max(reference_shape)
            if max_dim % 2 != 0:
                max_dim += 1  # Round up to nearest even number
            target_shape = (max_dim, max_dim, max_dim)
            pad_width = get_pad_width(reference_shape, target_shape)
            
            logger.info(f"Padding volumes from {reference_shape} to {target_shape}")
            
            if raw_input is not None:
                raw_input = np.pad(raw_input, pad_width, mode='constant', constant_values=0)
            if raw_recons is not None:
                raw_recons = np.pad(raw_recons, pad_width, mode='constant', constant_values=0)
            if raw_half_a is not None:
                raw_half_a = np.pad(raw_half_a, pad_width, mode='constant', constant_values=0)
            if raw_half_b is not None:
                raw_half_b = np.pad(raw_half_b, pad_width, mode='constant', constant_values=0)
            if raw_ref_volume is not None:
                raw_ref_volume = np.pad(raw_ref_volume, pad_width, mode='constant', constant_values=0)
    
    return raw_input, raw_recons, raw_half_a, raw_half_b, raw_ref_volume, pad_width


def apply_bbox_crop(
        mask_path: str,
        do_bbox_crop: bool,
        resized_input: np.ndarray,
        resized_wavelet_weights: Optional[np.ndarray],
        patch_size: int,
        pixel_size: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    resized_mask = None
    if mask_path is not None:
        logger.info(f"Loading user provided mask from {mask_path}.")
        raw_mask, _, _ = load_mrc(mask_path)
        resized_mask = resize_by_voxel_size(raw_mask, pixel_size, MODEL_VOXEL_SIZE)
        resized_mask = resized_mask > 0.5  # binarize mask

    bbox = None
    bbox_input = resized_input
    bbox_wavelet_weights = resized_wavelet_weights
    if do_bbox_crop and resized_mask is not None:
        bbox = get_bbox(resized_mask > 0.5, BBOX_ENLARGE)
        bbox = ensure_min_bbox_size(bbox, resized_mask.shape, patch_size)
        bbox_input = bbox_crop(resized_input, bbox)
        if resized_wavelet_weights is not None:
            bbox_wavelet_weights = bbox_crop_nd(resized_wavelet_weights, bbox)
        logger.info(f"Cropping bbox from mask. New shape is {bbox_input.shape}")

    return resized_mask, bbox, bbox_input, bbox_wavelet_weights


def apply_normalize_and_patchify(
        do_maxval_norm: bool,
        bbox_resized_input: np.ndarray,
        bbox_wavelet_weights: Optional[np.ndarray],
        patch_size: int,
        patch_overlap: int,
        output_path: str,
        debug: bool,
        acl: accelerate.Accelerator,
        cryoem_density_std: float = CRYOEM_DENSITY_STD,
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    maxval_used_for_norm = 1.
    # TODO: here we should suppport general normalization, or just remove the do_maxval_norm flag
    if do_maxval_norm:
        maxval_used_for_norm, bbox_resized_prepared = maxvalue_norm_and_patchify(
            bbox_resized_input, patch_size=patch_size, overlap=patch_overlap
        )
        patched_input = torch.from_numpy(bbox_resized_prepared["patches"])
        patch_locations = torch.from_numpy(bbox_resized_prepared["locations"])
        if debug:
            # must reduce patch save num
            num_patches = patched_input.shape[0]
            step = max(num_patches // 10, 1)
            idx_to_save = list(range(0, num_patches, step))[:10]
            save_path = generate_debug_filename(output_path, "input_patch.mrc")
            save_patches_as_mrc_main(acl, patched_input[idx_to_save], save_path)
    else:
        raise NotImplementedError
    patched_wavelet_weights = None
    if bbox_wavelet_weights is not None:
        _, bbox_wavelet_weights_prepared = multi_channel_maxvalue_norm_and_patchify(
            bbox_wavelet_weights, patch_size=patch_size, overlap=patch_overlap, do_standardize=False)
        patched_wavelet_weights = torch.from_numpy(bbox_wavelet_weights_prepared["patches"])
        patched_wavelet_weights = patched_wavelet_weights / (maxval_used_for_norm ** 2 * cryoem_density_std ** 2)
    return maxval_used_for_norm, patched_input, patch_locations, patched_wavelet_weights


def build_target_energy_and_fsc_volume(
        raw_fourier_mask: np.ndarray,
        fsc_vals: np.ndarray,
        raw_input: np.ndarray,
        resized_input: np.ndarray,
        maxval_used_for_norm: float,
        op: List[str],
        patch_size: int,
        pixel_size: float,
        output_path: str,
        debug: bool,
        acl: accelerate.Accelerator,
        cryoem_density_std: float = CRYOEM_DENSITY_STD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_fourier_mask_ra = radial_shell_average(raw_fourier_mask, is_rft=False, already_freq=True, harmonic_mean=False)
    # here, if there is no inpaint in op, raw_fourier_mask and raw_fourier_mask_ra are all one.
    # therefore, numerator_curr is the same as fsc_values expanded to a volume.
    # the denominator_curr is just 1.
    # so that the raw_fsc_vol is just fsc_values expanded to a volume (there is no anisotropy).
    # if op contains inpaint, we will use another way to calculate the modulated target_energy_curr.
    # I keep this line of code only for consistency to the previous experiments.
    numerator = raw_fourier_mask * volume_from_radial_shell_avg(fsc_vals / raw_fourier_mask_ra)
    denominator = 1 - volume_from_radial_shell_avg(fsc_vals) + numerator
    raw_fsc_vol = numerator / (denominator + 1e-6)
    if debug:
        save_path = generate_debug_filename(output_path, "fscvol.mrc")
        save_mrc_main(acl, raw_fsc_vol, save_path, pixel_size)
    #
    raw_target_energy = (primal_to_fourier_3d(torch.from_numpy(raw_input.copy())).abs() ** 2).numpy()
    raw_target_energy_ra = radial_shell_average(raw_target_energy, is_rft=False, already_freq=True, harmonic_mean=False)
    if "inpaint" in op:
        # raw_target_energy = volume_from_radial_shell_avg(raw_target_energy_ra * fsc_vals) / (raw_fsc_vol + 1e-6)
        raw_target_energy = volume_from_radial_shell_avg(raw_target_energy_ra)
        # here, we calculate the target_energy_curr modulated by the factor (f_mask_ra / f_mask_ori).
        raw_target_energy = (volume_from_radial_shell_avg(raw_fourier_mask_ra) / (raw_fourier_mask + 1e-6)) * raw_target_energy
        raw_target_energy = raw_target_energy / (maxval_used_for_norm ** 2 * cryoem_density_std ** 2)
    else:
        raw_target_energy = volume_from_radial_shell_avg(raw_target_energy_ra)
        raw_target_energy = raw_target_energy / (maxval_used_for_norm ** 2 * cryoem_density_std ** 2)

    if debug:
        save_path = generate_debug_filename(output_path, "energy.mrc")
        save_mrc_main(acl, raw_target_energy, save_path, pixel_size)

    # resize
    resized_fsc_vol = center_crop(raw_fsc_vol, resized_input.shape[-1])
    patch_fsc_vol = resize(resized_fsc_vol, (patch_size, ) * 3, order=1)
    resized_target_energy = center_crop(raw_target_energy, resized_input.shape[-1])
    patch_target_energy = resize(resized_target_energy, (patch_size, ) * 3, order=1)

    if debug:
        save_path = generate_debug_filename(output_path, "fscvol_patch.mrc")
        save_mrc_main(acl, patch_fsc_vol, save_path, MODEL_VOXEL_SIZE)
        save_path = generate_debug_filename(output_path, "energy_patch.mrc")
        save_mrc_main(acl, patch_target_energy, save_path, MODEL_VOXEL_SIZE)

    #
    patch_fsc_vol = torch.from_numpy(patch_fsc_vol).float()
    patch_target_energy = torch.from_numpy(patch_target_energy).float()
    return patch_fsc_vol, patch_target_energy


def dps(cli_args, scheduler, unet, op, y, device, f_mask=None, ref_map=None, fsc_volume=None, vol_power=None,
        wt_sigma2=None):
    def _cond_v(x_t: torch.Tensor, t: torch.Tensor):
        with contextlib.ExitStack() as stack:
            if cli_args.bf16:
                stack.enter_context(torch.autocast(device.type, dtype=torch.bfloat16))

            _t = t.item()
            lamb_base = cli_args.lamb_base
            if op != "inpaint":
                x_t.requires_grad_(True)
                if x_t.grad is not None:
                    x_t.grad.zero_()
            # Current model can only handle channel=2, leave one value for future conversion to conditional model
            input_x = torch.concat([x_t, torch.zeros_like(x_t)], dim=1)

            # Where the current model should go
            v_t = unet(input_x, einops.repeat(t, "->b", b=len(x_t)).to(device)).sample
            # One step to restore to final state
            x_0_hat = x_t - v_t * t / scheduler.num_train_timesteps
            # y is my condition
            scaled_y = y
            loss = None

            f_x_0_hat = primal_to_fourier_3d(x_0_hat)
            f_y_scale = primal_to_fourier_3d(scaled_y).to(device)

            if "inpaint" in op and "denoise" not in op:
                # only inpaiting, loss is None. noise_power not needed.
                f_x_0_hat = f_y_scale * (f_mask) + f_x_0_hat * (1 - f_mask)
                x_0_hat = fourier_to_primal_3d(f_x_0_hat).real
                cond_v_t = (x_t - x_0_hat) / (_t / scheduler.num_train_timesteps)
            elif "denoise" in op and "inpaint" not in op:
                # only denoising, no inpainting. f_mask not needed.
                elem_err = F.mse_loss(
                    torch.view_as_real(f_x_0_hat),
                    torch.view_as_real(f_y_scale),
                    reduction='none'
                )
                sq_err = elem_err.mean(dim=-1)
                target_energy = vol_power
                fsc_safe = torch.clamp(fsc_volume, max=1.0 - EPS_FSC)
                weight = 1 / (1 - fsc_safe) / (target_energy + 1e-6)
                loss = (weight * sq_err).sum() / weight.sum()
            elif "inpaint" in op and "denoise" in op:
                # both inpaiting and denoise.
                f_x_0_hat = f_x_0_hat * f_mask
                elem_err = F.mse_loss(
                    torch.view_as_real(f_x_0_hat),
                    torch.view_as_real(f_y_scale),
                    reduction='none'
                )
                sq_err = elem_err.mean(dim=-1)
                target_energy = vol_power
                fsc_safe = torch.clamp(fsc_volume, max=1.0 - EPS_FSC)
                weight = 1 / (1 - fsc_safe) / (target_energy + 1e-6)
                loss = (weight * sq_err).sum() / weight.sum()
            elif "non-uniform" in op:
                # real space non-uniform variance operator.
                wt_x_0_hat = shannon_wavelet_transform_3d(x_0_hat, n_bands=cli_args.nbands)
                wt_y = shannon_wavelet_transform_3d(scaled_y.to(device), n_bands=cli_args.nbands)
                err2 = (wt_x_0_hat - wt_y).pow(2)  # [N,C,D,H,W]
                weight = 1 / (wt_sigma2 + 1e-6)
                # weight = wt_sigma2
                # weight = 1 / (1 - wt_sigma2 + 1e-6)
                loss = (weight * err2).sum() / weight.sum()
            if loss is not None:
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_t, allow_unused=True, materialize_grads=True)[0]
                if cli_args.use_lamb_w:
                    lamb_w = min(_t / max(.001, 1000 - _t), cli_args.lamb_w_max)
                else:
                    lamb_w = 1.0
                if cli_args.norm_grad:
                    cond_v_t = v_t + lamb_w * lamb_base * loss_grad / loss_grad.norm()
                else:
                    cond_v_t = v_t + lamb_w * lamb_base * loss_grad
            cond_v_t = cond_v_t.detach()
        return cond_v_t

    # Ablation: Number of time steps
    scheduler.set_timesteps(cli_args.num_timesteps, device=device)
    # sample
    # seed = cli_args.seed
    seed = cli_args.seed if cli_args.seed is not None else random.randrange(0, 2 ** 31)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x_t = torch.randn((y.shape[0], 1, cli_args.patch_size, cli_args.patch_size, cli_args.patch_size),
                      requires_grad=True,
                      device=device)

    # flow posterior sampling should be re-weighted by t/(1-t), i.e., inf -> 0
    assert cli_args.norm_grad is True, "Now we only support norm=True for flow"
    for t in scheduler.timesteps:
        _t = t.item()
        # scheduler.num_train_timesteps is immutable, set to 1000!!!
        h = scheduler.num_train_timesteps // cli_args.num_timesteps
        v_t = _cond_v(x_t, t)
        x_t = x_t - v_t * h / scheduler.num_train_timesteps
        # new_y = x_t.detach().cpu().numpy()
        new_y = x_t.detach()
    return new_y


def apply_posterior_sampling(
        args,
        unet,
        scheduler,
        patched_input: torch.Tensor,
        patch_locations: torch.Tensor,
        patched_wavelet_weights: Optional[torch.Tensor],
        patch_fsc_vol: Optional[torch.Tensor],
        patch_target_energy: Optional[torch.Tensor],
        patch_fourier_mask: Optional[torch.Tensor],
        bbox_input: np.ndarray,
        device: torch.device,
        output_path: str,
        debug: bool,
        acl: accelerate.Accelerator,
        cryoem_density_mean: float = CRYOEM_DENSITY_MEAN,
        cryoem_density_std: float = CRYOEM_DENSITY_STD,
):
    patch_overlap = args.patch_overlap
    batch_size = args.batch_size
    patch_fourier_mask = patch_fourier_mask.to(device)
    patch_fsc_vol = patch_fsc_vol.unsqueeze(1).to(device)
    patch_target_energy = patch_target_energy.unsqueeze(1).to(device)

    aggregator = GridAggregator(bbox_input.shape, patch_overlap)
    num_patches = patched_input.shape[0]
    pad = (acl.num_processes - (num_patches % acl.num_processes)) % acl.num_processes

    if pad > 0:
        patched_input = torch.cat([patched_input, torch.zeros([pad, *patched_input.shape[1:]])], dim=0)
        if patched_wavelet_weights is not None:
            patched_wavelet_weights = torch.cat(
                [patched_wavelet_weights, torch.zeros([pad, *patched_wavelet_weights.shape[1:]])], dim=0)

    payload: Dict[str, torch.Tensor] = {"x": patched_input}
    if patched_wavelet_weights is not None:
        payload["w"] = patched_wavelet_weights

    results = []
    with acl.split_between_processes(payload) as sub:
        sub_x = sub["x"]
        sub_w = sub.get("w")
        n_local = sub_x.shape[0]
        for i in tqdm(range(math.ceil(n_local / batch_size))):
            sl = slice(i * batch_size, min((i + 1) * batch_size, n_local))
            y_batch = sub_x[sl].unsqueeze(1)
            w_batch = sub_w[sl].to(device) if sub_w is not None else None

            out = dps(cli_args=args, scheduler=scheduler, unet=unet, op=args.op, y=y_batch, device=device,
                      f_mask=patch_fourier_mask, ref_map=None, fsc_volume=patch_fsc_vol, vol_power=patch_target_energy,
                      wt_sigma2=w_batch)
            results.append(out)

    outputs = torch.vstack(results)
    acl.wait_for_everyone()
    outputs = acl.gather(outputs)
    outputs = outputs[:num_patches]

    aggregator.add_batch(outputs, patch_locations)

    if debug:
        num_patches = outputs.shape[0]
        step = max(num_patches // 10, 1)
        idx_to_save = list(range(0, num_patches, step))[:10]
        save_path = generate_debug_filename(output_path, "output_patch.mrc")
        save_patches_as_mrc_main(acl, outputs[idx_to_save], save_path)

    bbox_output = aggregator.get_output_tensor().cpu().squeeze().numpy().astype(np.float32)
    bbox_output = (bbox_output * cryoem_density_std) + cryoem_density_mean
    return bbox_output


def cond_dps(cli_args, lit_model, op, y, device, f_mask=None, ref_map=None, fsc_volume=None, vol_power=None,
        wt_sigma2=None):
    def _cond_v(x_t: torch.Tensor, inputs: torch.Tensor, t: torch.Tensor):
        with contextlib.ExitStack() as stack:
            if cli_args.bf16:
                stack.enter_context(torch.autocast(device.type, dtype=torch.bfloat16))

            _t = t.item()
            lamb_base = cli_args.lamb_base
            if op != "inpaint":
                x_t.requires_grad_(True)
                if x_t.grad is not None:
                    x_t.grad.zero_()
            # Current model can only handle channel=2, leave one value for future conversion to conditional model
            # input_x = torch.concat([x_t, torch.zeros_like(x_t)], dim=1)

            bs = x_t.shape[0]
            inputs = lit_model.scale_from_data(lit_model.cfg, inputs)

            # here need a closure because sample_from_fm input arg v_xt_t should be a vector field that only
            # takes two arguments: xt and timestep.
            conditioned_generation_conds = {
                "input_cond": torch.tensor([cli_args.output_tag] * bs).to(device),
                "output_cond": torch.tensor([cli_args.output_tag] * bs).to(device),  # the tag of emready is 0
                "vol_cond": inputs.to(device),  # dimension should be [bs,h,n,w]
            }
            unconditioned_generation_conds = {
                "input_cond": None,
                "output_cond": torch.tensor([cli_args.output_tag] * bs).to(device),  # the tag of emready is 0
                "vol_cond": None,  # dimension should be [bs,h,n,w]
            }

            # Where the current model should go
            v_t = lit_model(x_t, _t, generation_conds=conditioned_generation_conds)

            # One step to restore to final state
            x_0_hat = x_t - v_t * t / lit_model.noise_scheduler.num_train_timesteps
            # y is my condition
            scaled_y = y
            loss = None

            f_x_0_hat = primal_to_fourier_3d(x_0_hat)
            f_y_scale = primal_to_fourier_3d(scaled_y).to(device)

            if "inpaint" in op and "denoise" not in op:
                # only inpaiting, loss is None. noise_power not needed.
                f_x_0_hat = f_y_scale * (f_mask) + f_x_0_hat * (1 - f_mask)
                x_0_hat = fourier_to_primal_3d(f_x_0_hat).real
                cond_v_t = (x_t - x_0_hat) / (_t / lit_model.noise_scheduler.num_train_timesteps)
            elif "denoise" in op and "inpaint" not in op:
                # only denoising, no inpainting. f_mask not needed.
                elem_err = F.mse_loss(
                    torch.view_as_real(f_x_0_hat),
                    torch.view_as_real(f_y_scale),
                    reduction='none'
                )
                sq_err = elem_err.mean(dim=-1)
                target_energy = vol_power
                fsc_safe = torch.clamp(fsc_volume, max=1.0 - EPS_FSC)
                weight = 1 / (1 - fsc_safe) / (target_energy + 1e-6)
                loss = (weight * sq_err).sum() / weight.sum()
            elif "inpaint" in op and "denoise" in op:
                # both inpaiting and denoise.
                f_x_0_hat = f_x_0_hat * f_mask
                elem_err = F.mse_loss(
                    torch.view_as_real(f_x_0_hat),
                    torch.view_as_real(f_y_scale),
                    reduction='none'
                )
                sq_err = elem_err.mean(dim=-1)
                target_energy = vol_power
                fsc_safe = torch.clamp(fsc_volume, max=1.0 - EPS_FSC)
                weight = 1 / (1 - fsc_safe) / (target_energy + 1e-6)
                loss = (weight * sq_err).sum() / weight.sum()
            elif "non-uniform" in op:
                # real space non-uniform variance operator.
                wt_x_0_hat = shannon_wavelet_transform_3d(x_0_hat, n_bands=cli_args.nbands)
                wt_y = shannon_wavelet_transform_3d(scaled_y.to(device), n_bands=cli_args.nbands)
                err2 = (wt_x_0_hat - wt_y).pow(2)  # [N,C,D,H,W]
                weight = 1 / (wt_sigma2 + 1e-6)
                # weight = wt_sigma2
                # weight = 1 / (1 - wt_sigma2 + 1e-6)
                loss = (weight * err2).sum() / weight.sum()
            if loss is not None:
                loss_grad = torch.autograd.grad(outputs=loss, inputs=x_t, allow_unused=True, materialize_grads=True)[0]
                if cli_args.use_lamb_w:
                    lamb_w = min(_t / max(.001, 1000 - _t), cli_args.lamb_w_max)
                else:
                    lamb_w = 1.0
                if cli_args.norm_grad:
                    cond_v_t = v_t + lamb_w * lamb_base * loss_grad / loss_grad.norm()
                else:
                    cond_v_t = v_t + lamb_w * lamb_base * loss_grad
            cond_v_t = cond_v_t.detach()

            # classifier-free
            if cli_args.cfg_weight != 0.:
                with torch.no_grad():
                    cond_v_t = ((1 + cli_args.cfg_weight) * cond_v_t -
                                cli_args.cfg_weight * lit_model(x_t, _t, generation_conds=unconditioned_generation_conds))
        return cond_v_t

    # Ablation: Number of time steps
    lit_model.noise_scheduler.set_timesteps(cli_args.num_timesteps, device=device)
    # sample
    x_t = torch.randn((y.shape[0], 1, cli_args.patch_size, cli_args.patch_size, cli_args.patch_size),
                      requires_grad=True,
                      device=device)

    # flow posterior sampling should be re-weighted by t/(1-t), i.e., inf -> 0
    assert cli_args.norm_grad is True, "Now we only support norm=True for flow"
    for t in lit_model.noise_scheduler.timesteps:
        _t = t.item()
        # scheduler.num_train_timesteps is immutable, set to 1000!!!
        h = lit_model.noise_scheduler.num_train_timesteps // cli_args.num_timesteps
        v_t = _cond_v(x_t, y, t)
        x_t = x_t - v_t * h / lit_model.noise_scheduler.num_train_timesteps
        # new_y = x_t.detach().cpu().numpy()
        new_y = x_t.detach()
    return new_y


def apply_cond_posterior_sampling(
        args,
        lit_model,
        patched_input: torch.Tensor,
        patch_locations: torch.Tensor,
        patched_wavelet_weights: Optional[torch.Tensor],
        patch_fsc_vol: Optional[torch.Tensor],
        patch_target_energy: Optional[torch.Tensor],
        patch_fourier_mask: Optional[torch.Tensor],
        bbox_input: np.ndarray,
        device: torch.device,
        output_path: str,
        debug: bool,
        acl: accelerate.Accelerator,
        cryoem_density_mean: float = CRYOEM_DENSITY_MEAN,
        cryoem_density_std: float = CRYOEM_DENSITY_STD,
):
    patch_overlap = args.patch_overlap
    batch_size = args.batch_size
    patch_fourier_mask = patch_fourier_mask.to(device)
    patch_fsc_vol = patch_fsc_vol.unsqueeze(1).to(device)
    patch_target_energy = patch_target_energy.unsqueeze(1).to(device)

    aggregator = GridAggregator(bbox_input.shape, patch_overlap)
    num_patches = patched_input.shape[0]
    pad = (acl.num_processes - (num_patches % acl.num_processes)) % acl.num_processes

    if pad > 0:
        patched_input = torch.cat([patched_input, torch.zeros([pad, *patched_input.shape[1:]])], dim=0)
        if patched_wavelet_weights is not None:
            patched_wavelet_weights = torch.cat(
                [patched_wavelet_weights, torch.zeros([pad, *patched_wavelet_weights.shape[1:]])], dim=0)

    payload: Dict[str, torch.Tensor] = {"x": patched_input}
    if patched_wavelet_weights is not None:
        payload["w"] = patched_wavelet_weights

    results = []
    with acl.split_between_processes(payload) as sub:
        sub_x = sub["x"]
        sub_w = sub.get("w")
        n_local = sub_x.shape[0]
        for i in tqdm(range(math.ceil(n_local / batch_size))):
            sl = slice(i * batch_size, min((i + 1) * batch_size, n_local))
            y_batch = sub_x[sl].unsqueeze(1)
            w_batch = sub_w[sl].to(device) if sub_w is not None else None

            out = cond_dps(cli_args=args, lit_model=lit_model, op=args.op, y=y_batch, device=device,
                      f_mask=patch_fourier_mask, ref_map=None, fsc_volume=patch_fsc_vol, vol_power=patch_target_energy,
                      wt_sigma2=w_batch)
            results.append(out)

    outputs = torch.vstack(results)
    acl.wait_for_everyone()
    outputs = acl.gather(outputs)
    outputs = outputs[:num_patches]

    aggregator.add_batch(outputs, patch_locations)

    if debug:
        num_patches = outputs.shape[0]
        step = max(num_patches // 10, 1)
        idx_to_save = list(range(0, num_patches, step))[:10]
        save_path = generate_debug_filename(output_path, "output_patch.mrc")
        save_patches_as_mrc_main(acl, outputs[idx_to_save], save_path)

    bbox_output = aggregator.get_output_tensor().cpu().squeeze().numpy().astype(np.float32)
    bbox_output = (bbox_output * cryoem_density_std) + cryoem_density_mean
    return bbox_output


@torch.no_grad()
def apply_cond_inference(
        args,
        lit_model,
        patched_input: torch.Tensor,
        patch_locations: torch.Tensor,
        bbox_input: np.ndarray,
        output_path: str,
        debug: bool,
        acl: accelerate.Accelerator,
        cryoem_density_mean: float = CRYOEM_DENSITY_MEAN,
        cryoem_density_std: float = CRYOEM_DENSITY_STD,
):
    def core_inference(inputs):
        device = lit_model.device
        bs = inputs.shape[0]
        with contextlib.ExitStack() as stack:
            stack.enter_context(lit_model.ema_scope(not args.no_ema))
            if args.bf16:
                stack.enter_context(torch.autocast(device.type, dtype=torch.bfloat16))

            inputs = lit_model.scale_from_data(lit_model.cfg, inputs)

            # here need a closure because sample_from_fm input arg v_xt_t should be a vector field that only
            # takes two arguments: xt and timestep.
            conditioned_generation_conds = {
                "input_cond": torch.tensor([args.output_tag] * bs).to(device),
                "output_cond": torch.tensor([args.output_tag] * bs).to(device),  # the tag of emready is 0
                "vol_cond": inputs.to(device),  # dimension should be [bs,h,n,w]
            }
            unconditioned_generation_conds = {
                "input_cond": None,
                "output_cond": torch.tensor([args.output_tag] * bs).to(device),  # the tag of emready is 0
                "vol_cond": None,  # dimension should be [bs,h,n,w]
            }

            def v_xt_t(_xt, _t):
                return (1 + args.cfg_weight) * lit_model(_xt, _t, generation_conds=conditioned_generation_conds) \
                    - args.cfg_weight * lit_model(_xt, _t, generation_conds=unconditioned_generation_conds)

            predicts = sample_from_fm(v_xt_t, lit_model.noise_scheduler, args.odeint, args.num_timesteps,
                                      len(inputs), device)

            # scale_to_data is not used now, just for compatibility
            predicts = lit_model.scale_to_data(lit_model.cfg, predicts)
            predicts = predicts.unsqueeze(1)
        return predicts

    patch_overlap = args.patch_overlap
    batch_size = args.batch_size

    aggregator = GridAggregator(bbox_input.shape, patch_overlap)
    num_patches = patched_input.shape[0]
    pad = (acl.num_processes - (num_patches % acl.num_processes)) % acl.num_processes

    if pad > 0:
        patched_input = torch.cat([patched_input, torch.zeros([pad, *patched_input.shape[1:]])], dim=0)

    payload: Dict[str, torch.Tensor] = {"x": patched_input}

    results = []
    with acl.split_between_processes(payload) as sub:
        sub_x = sub["x"]
        n_local = sub_x.shape[0]
        for i in tqdm(range(math.ceil(n_local / batch_size))):
            sl = slice(i * batch_size, min((i + 1) * batch_size, n_local))
            y_batch = sub_x[sl].unsqueeze(1)

            out = core_inference(y_batch)
            results.append(out)

    outputs = torch.vstack(results)
    acl.wait_for_everyone()
    outputs = acl.gather(outputs)
    outputs = outputs[:num_patches]

    aggregator.add_batch(outputs, patch_locations)

    if debug:
        num_patches = outputs.shape[0]
        step = max(num_patches // 10, 1)
        idx_to_save = list(range(0, num_patches, step))[:10]
        save_path = generate_debug_filename(output_path, "output_patch.mrc")
        save_patches_as_mrc_main(acl, outputs[idx_to_save], save_path)

    bbox_output = aggregator.get_output_tensor().cpu().squeeze().numpy().astype(np.float32)
    bbox_output = (bbox_output * cryoem_density_std) + cryoem_density_mean
    return bbox_output


# currently not very flexible, should be optimized
def apply_spisonet_mixing(
        raw_ref_volume: np.ndarray,
        resized_output: np.ndarray,
        resized_mask: np.ndarray,
        threshold_res: float,
        pixel_size: float,
        output_path: str = None,
        debug: bool = False,
):
    logger.info(f"Keeping the signal within {threshold_res} A from the referecence map.")
    # low_map = re.sub(r'run_it\d{3}', 'run_it000', input_starfile_path).replace("_external_reconstruct.star", ".mrc")
    # low_data, voxel_size, _ = load_mrc(low_map)
    resized_low_data = resize_by_voxel_size(raw_ref_volume, pixel_size, MODEL_VOXEL_SIZE)
    resized_output, voxel_size = combine_map(resized_low_data, resized_output, threshold_res, resized_mask, MODEL_VOXEL_SIZE)
    if debug:
        chimeric_mrc_name = output_path.replace("_external_reconstruct.mrc", "_output_chimeric.mrc")
        save_mrc(resized_output, chimeric_mrc_name, voxel_size)
    return resized_output


def apply_post_processing(
        file_type: str,
        raw_input: np.ndarray,
        raw_recons: np.ndarray,
        resized_input: np.ndarray,
        resized_output: np.ndarray,
        fsc_spectra: np.ndarray,
        pixel_size: float,
        particle_diameter: float,
        original_size: int,
        spectral_mixing: bool,
        skip_spectral_trailing: bool,
        output_path: str = None,
        debug: bool = False,
        acl: accelerate.Accelerator = None,
):
    if particle_diameter is not None:
        radial_mask = create_radial_mask(
            pixel_size=pixel_size, particle_diameter=particle_diameter, size=original_size).cpu().numpy()
        radial_mask_nv = create_radial_mask(
            pixel_size=MODEL_VOXEL_SIZE, particle_diameter=particle_diameter, size=resized_output.shape[0]).cpu().numpy()

        # Apply real space radial mask and resize to ori size
        resized_output = resized_output * radial_mask_nv  # apply real space radial mask (particle diameter)
    f_resized_output = fft(resized_output)
    f_final_output = rescale_fourier(f_resized_output, original_size)  # rescale back to original voxel size
    f_final_output *= (f_final_output.shape[0] / f_resized_output.shape[0]) ** 3  # preserve energy after resize

    if debug:
        save_path = generate_debug_filename(output_path, "debug_recons_unfil.mrc")
        save_mrc_main(acl, raw_input, save_path, pixel_size)
        save_path = generate_debug_filename(output_path, "debug_denoise_input.mrc")
        save_mrc_main(acl, resized_input, save_path, MODEL_VOXEL_SIZE)
        save_path = generate_debug_filename(output_path, "debug_denoise_output.mrc")
        save_mrc_main(acl, resized_output, save_path, MODEL_VOXEL_SIZE)

    # Apply mixing ---------------------------------
    if spectral_mixing:
        if file_type == "MRC":
            # Model output (MODEL_APIX, {MODEL_VOXEL_SIZE} voxel size) is used within the Nyquist frequency;
            # original signal is applied outside this range.
            model_nyquist_index = res_to_index(MODEL_VOXEL_SIZE * 2, pixel_size, original_size)
            if original_size // 2 > model_nyquist_index - 3:
                # TODO: this is a workaround to keep the same mixing with STAR file input
                radial_mask = create_radial_mask(
                    pixel_size=pixel_size, particle_diameter=original_size, size=original_size).cpu().numpy()
                logger.info(
                    f"File type {file_type} frequency selection: Use model output ({MODEL_VOXEL_SIZE} voxel size) "
                    f"within model Nyquist freq; use original signal beyond.")
                crossover_grid = get_crossover_grid(model_nyquist_index - 3, original_size, filter_edge_width=3)
                f_final_output = (f_final_output * crossover_grid + fft(raw_input * radial_mask) * (1 - crossover_grid))

        # determine resolution limit from iteration fsc, need fsc_spectra
        # the mixing logic is from https://github.com/3dem/relion-blush/blob/main/relion_blush/command_line.py#L120
        elif file_type == "STAR":
            if fsc_spectra is not None:
                max_denoised_index = res_from_fsc(fsc_spectra, res=None, threshold=1. / 7.) - 1
                if max_denoised_index > f_resized_output.shape[-1] - 3:
                    logger.info(f"Denoiser maximum resolution reached, mixing in data for higher frequencies.")
                    crossover_grid = get_crossover_grid(f_resized_output.shape[-1] - 3, original_size, filter_edge_width=3)
                    if raw_recons is not None:
                        f_final_output = (f_final_output * crossover_grid + fft(raw_recons * radial_mask) * (1 - crossover_grid))
                    # out_df_half2 = (f_denoised_output_ori_half2 * crossover_grid +
                    #           fft(recons_half2 * radial_mask) * (1 - crossover_grid))
                else:
                    if skip_spectral_trailing:
                        logger.info(f"Skipping spectral trailing")
                        # out_df_half2 = f_denoised_output_ori_half2
                    else:
                        logger.info(f"Applying spectral trailing")
                        crossover_grid = get_crossover_grid(max_denoised_index, original_size, filter_edge_width=3)
                        f_final_output = f_final_output * crossover_grid
                        # out_df_half2 = f_denoised_output_ori_half2 * crossover_grid

                        logger.info(f"Max denoised spectral index: {max_denoised_index}")
                        max_denoised_res = index_to_res(max_denoised_index, voxel_size=pixel_size, box_size=original_size)
                        logger.info(f"Max denoised resolution: {round(max_denoised_res, 2)}")
        else:
            raise Exception(f"Unsupported file type: {file_type}")

    final_output = ifft(f_final_output)

    return final_output


def apply_sqrt_fsc_weighting(density: np.ndarray, fsc: np.ndarray) -> np.ndarray:
    """
    Apply sqrt(FSC) frequency-domain weighting to real-space density and return weighted real-space density.

    Parameters
    ----------
    density : np.ndarray
        Real density of shape (D, D, D) (cube).
    fsc : np.ndarray
        Radial FSC of shape (D//2,), excluding DC term (i.e., from radius=1 to D//2).

    Returns
    -------
    np.ndarray
        Weighted real-space density of same shape (D, D, D).
    """
    if density.ndim != 3 or density.shape[0] != density.shape[1] or density.shape[1] != density.shape[2]:
        raise ValueError("density must be a cube (D, D, D)")
    D = density.shape[0]
    if D % 2 != 0:
        raise ValueError("D must be even to align with rFFT grid")
    # if fsc.shape != ((D // 2)):
    #     raise ValueError("fsc shape should be (D//2)")

    # 1) Real -> complex rFFT (keep only non-negative frequencies in last dimension)
    F = np.fft.rfftn(density, axes=(0, 1, 2))  # shape (D, D, D//2+1)

    # 2) Build integer frequency coordinates and compute radius shell indices (0..D//2)
    # np.fft.fftfreq(D) * D gives integer frequency indices (…,-2,-1,0,1,2,…; rfftfreq is 0..D//2)
    kx = (np.fft.fftfreq(D) * D).astype(np.float64)  # [-D/2..D/2)
    ky = (np.fft.fftfreq(D) * D).astype(np.float64)
    kz = (np.fft.rfftfreq(D) * D).astype(np.float64)  # [0..D/2]
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")  # shape (D, D, D//2+1)
    radii = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    # Map continuous radius to integer shell using floor; clip to [0, D//2]
    shell = np.floor(radii + 1e-12).astype(np.int64)
    shell = np.clip(shell, 0, D // 2)

    # 3) Assemble weights for each radius shell: DC(0) uses 1, others use sqrt(FSC)
    fsc_safe = np.sqrt(np.maximum(fsc, 0.0))  # length D//2
    shell_weights = np.empty(D // 2 + 1, dtype=F.real.dtype)
    shell_weights[0] = 1.0  # DC unchanged
    shell_weights[1:] = fsc_safe  # radius 1..D//2
    # shell_weights = fsc_safe

    # Map scalar weights to 3D frequency domain grid
    W = np.take(shell_weights, shell)  # shape (D, D, D//2+1)

    # 4) Weight and inverse transform back to real space
    Fw = F * W
    out = np.fft.irfftn(Fw, s=(D, D, D), axes=(0, 1, 2)).real
    return out


def apply_fsc_weighting(
        final_output1: np.ndarray,
        final_output2: np.ndarray,
        args,
        acl: accelerate.Accelerator = None,
):
    if acl.is_main_process:
        if "non-uniform" in args.op:
            before_weighting_wavelet1 = shannon_wavelet_transform_3d(
                torch.from_numpy(final_output1).unsqueeze(0), n_bands=args.nbands)
            before_weighting_wavelet2 = shannon_wavelet_transform_3d(
                torch.from_numpy(final_output2).unsqueeze(0), n_bands=args.nbands)
            n2 = 0.5 * (before_weighting_wavelet1 - before_weighting_wavelet2) ** 2
            p2 = 0.5 * (before_weighting_wavelet1 ** 2 + before_weighting_wavelet2 ** 2)
            s2 = torch.clamp((p2 - n2), min=0)
            filter = torch.sqrt(torch.clamp((s2 / (s2 + n2 + 1e-9)), max=1.0, min=0.0))
            final_output1 = inverse_shannon_wavelet_transform_3d(before_weighting_wavelet1 * filter).squeeze().numpy()
            final_output2 = inverse_shannon_wavelet_transform_3d(before_weighting_wavelet2 * filter).squeeze().numpy()
        else:
            denoised_fsc, _ = calc_fsc(final_output1, final_output2)
            final_output1 = apply_sqrt_fsc_weighting(final_output1, denoised_fsc)
            final_output2 = apply_sqrt_fsc_weighting(final_output2, denoised_fsc)

        acl.wait_for_everyone()
    return final_output1, final_output2
