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

"""
CryoEM Posterior Sampling
raw_ -> resized_ -> cropped_ -> patched_ -> cropped -> resized_ -> final_
"""

import os
# import re
# import glob
# import time
# import copy
import math
import argparse
import os.path as osp
from pathlib import Path
from logging import getLogger
# from dataclasses import dataclass
# from contextlib import contextmanager
from typing import Union, Optional, Dict, Any, Tuple, List, Callable

# import mrcfile
import numpy as np
# from torchvision.transforms.functional import resized_crop
# from tqdm import tqdm
# from skimage.transform import resize

import torch

from mmengine import Config

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list

# from cryostar.utils.fft_utils import fourier_to_primal_3d, primal_to_fourier_3d

from cryofm.core.training.accelerate_utils import setup_logging
from cryofm.projects.cryofm2.lit_modules import CryoFM2Cond
from cryofm.projects.cryofm2.utils.infer_relion_utils import (
    load_star, load_mrc, save_mrc, load_input_data, get_reconstructions, fft, ifft, normalize_voxel_size_fourier,
    resize_by_voxel_size, rescale_fourier, get_radial_mask)
from cryofm.projects.cryofm2.sampling_helper import (
    change_dir, auto_find_base_dir, InputData, timer, compute_fsc_values, build_fourier_mask, compute_wavelet_weights,
    apply_bbox_crop, apply_normalize_and_patchify, build_target_energy_and_fsc_volume, apply_cond_inference,
    apply_cond_posterior_sampling, apply_spisonet_mixing, apply_post_processing, apply_fsc_weighting, MODEL_VOXEL_SIZE,
    check_and_pad_volumes_to_cubic)


# hard-coded script name
def load_model(args, device):
    cfg = Config.fromfile(osp.join(args.model_dir, "config.yaml"))
    lit_model = CryoFM2Cond.load_from_safetensors(osp.join(args.model_dir, "model.safetensors"), cfg=cfg).to(device)
    lit_model.eval()
    return lit_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    data_group = parser.add_argument_group('data options')
    # MRC file or starfile as inputs
    # Given half maps, run posterior sampling on single input map (-i) or on two half maps (-i1, -i2), input file path
    # can be density map (.mrc, .map) or starfile (.starfile)
    data_group.add_argument("-i", "--input-path", type=str, default=None,
                            help="If this arg is not None, only do posterior sampling on this map.")
    data_group.add_argument("-i1", "--input-path1", type=str, default=None, help="Input half map 1 path")
    data_group.add_argument("-i2", "--input-path2", type=str, default=None, help="Input half map 2 path")
    data_group.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    # Other additional inputs
    # optional mask
    data_group.add_argument("--mask-path", type=str, default=None, help="Mask to control inference region")
    # optional particle set
    data_group.add_argument("--data-starfile-path", type=str, default=None,
                            help="Particle dataset starfile path, used for compute Fourier mask")

    model_group = parser.add_argument_group('model options')
    model_group.add_argument("--model-dir", action="store", type=str, required=True,
                             help="Model directory")
    # model_group.add_argument("--ckpt", action="store", type=str, default="last",
    #                          help="Checkpoint directory name")
    model_group.add_argument("--no-ema", action='store_true', default=False,
                             help="Disable EMA weights")
    model_group.add_argument("--bf16", action="store_true", default=False,
                             help="Use bf16 precision for fast inference")
    model_group.add_argument("--output-tag", type=int, default=1,
                             help="Conditional model output tag, to be removed")
    model_group.add_argument("--patch-size", action="store", type=int, default=64,
                             help="Model input patch size")
    model_group.add_argument("--batch-size", action="store", type=int, default=4,
                             help="Batch size")
    model_group.add_argument("--patch-overlap", action="store", type=int, default=32)
    # for sampling parameters
    model_group.add_argument("--odeint", action="store", type=str, default="euler")
    model_group.add_argument("--num-timesteps", action="store", type=int, default=200,
                             help="Number of sampling timesteps")
    model_group.add_argument("--cfg-weight", action="store", type=float, default=2.0,
                             help="Classifier-free-guidance weight")
    model_group.add_argument("--op", nargs="+", type=str, default=["denoise"],
                             help="Forward operator type, can be set as [denoise, denoise inpaint, non-uniform]")
    model_group.add_argument("--lamb-base", action='store', type=float, default=1000.0,
                             help="Likelihood gradient base step size")
    model_group.add_argument("--use-lamb-w", action='store_true', default=False,
                             help="Use decayed lamb scheduler")
    model_group.add_argument("--lamb-w-max", action='store', type=float, default=5.0,
                             help="Max weight for decayed lamb scheduler")
    model_group.add_argument("--norm-grad", action='store_true', default=False,
                             help="Use normalized gradient")
    model_group.add_argument("--fmask-threshold", action='store', type=float, default=1,
                             help="Threshold for inpaint fourier mask")
    model_group.add_argument("--threshold-res", action='store', type=float, default=10.0)
    model_group.add_argument("--nbands", action='store', type=int, default=64)
    model_group.add_argument("--bbox", action='store_true', default=False)
    extra_group = parser.add_argument_group('extra options')
    extra_group.add_argument("--fsc-weighting", action='store_true', default=False)
    extra_group.add_argument("--spectral-mixing", action='store_true', default=False, )
    extra_group.add_argument("--skip-spectral-trailing", action='store_true', default=False)
    # extra_group.add_argument("--verbose", action='store_true', default=False)
    extra_group.add_argument("--debug", action='store_true', default=False)
    extra_group.add_argument("--log-file-path", type=str, default=None)
    extra_group.add_argument("--seed", action="store", type=int, default=None)

    args = parser.parse_args()
    
    # Note: Set maxval_norm for code compatibility
    # This option previously supported multiple normalization methods, but now only maxval-norm is used.
    # We always set it to True and no longer expose it as a command-line argument to avoid confusion.
    args.maxval_norm = True
    
    return args


def detect_cryoem_file_type(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == '.star':
        return "STAR"
    elif ext in ['.mrc', '.map', ]:
        return "MRC"
    else:
        return "UNKNOWN"


def rln_ext_starfile_load_helper(starfile_path):
    d = load_star(starfile_path)
    rel_path = d['external_reconstruct_general']['rlnExtReconsDataReal']
    abs_path = auto_find_base_dir(starfile_path, rel_path)
    with change_dir(abs_path):
        data = load_input_data(starfile_path)
    return data


def prepare_input(args):
    # every input will go through a independent posterior sampling procedure
    # if args.input_path1 is None and args.input_path2 is None:
    #     raise RuntimeError("You must specify the two half map paths, either by mrc or by starfile.")
    input_data1 = InputData()
    input_data2 = InputData()
    if args.input_path1 is not None and args.input_path2 is not None:
        input_path1_file_type = detect_cryoem_file_type(args.input_path1)
        input_path2_file_type = detect_cryoem_file_type(args.input_path2)

        if input_path1_file_type != input_path2_file_type:
            raise ValueError(
                f"Half map file type must be the same, got {input_path1_file_type} and {input_path2_file_type}")

        if input_path1_file_type == "MRC":
            vol1, apix1, _ = load_mrc(args.input_path1)
            vol2, _, _ = load_mrc(args.input_path2)

            input_data1 = InputData(
                file_type="MRC",
                mode="refine",
                pixel_size=apix1,
                original_size=vol1.shape[0],
                output_path=osp.join(args.output_dir, Path(args.input_path1).stem + "_external_reconstruct.mrc"),
                recons_unfil=vol1,
                half_volume_a=vol1,
                half_volume_b=vol2, )
            input_data2 = InputData(
                file_type="MRC",
                mode="refine",
                pixel_size=apix1,
                original_size=vol2.shape[0],
                output_path=osp.join(args.output_dir, Path(args.input_path2).stem + "_external_reconstruct.mrc"),
                recons_unfil=vol2,
                half_volume_a=vol2,
                half_volume_b=vol1, )
        elif input_path2_file_type == "STAR":
            data1 = rln_ext_starfile_load_helper(args.input_path1)
            data2 = rln_ext_starfile_load_helper(args.input_path2)
            data1_recons_unfil, data1_recons = get_reconstructions(data1)
            data2_recons_unfil, data2_recons = get_reconstructions(data2)

            input_data1 = InputData(
                file_type="STAR",
                mode=data1["mode"],
                pixel_size=data1["pixel_size"],
                original_size=data1["original_size"],
                particle_diameter=data1["particle_diameter"],
                # output path
                output_path=osp.join(args.output_dir, Path(args.input_path1).stem + "_external_reconstruct.mrc"),
                this_half_index=data1["this_half_index"],
                fsc_spectra=data1["fsc_spectra"],
                recons_unfil=data1_recons_unfil,
                recons=data1_recons,
                half_volume_a=data1_recons_unfil,
                half_volume_b=data2_recons_unfil,
            )
            input_data2 = InputData(
                file_type="STAR",
                mode=data2["mode"],
                pixel_size=data2["pixel_size"],
                original_size=data2["original_size"],
                particle_diameter=data2["particle_diameter"],
                # output path
                output_path=osp.join(args.output_dir, Path(args.input_path2).stem + "_external_reconstruct.mrc"),
                this_half_index=data2["this_half_index"],
                fsc_spectra=data2["fsc_spectra"],
                recons_unfil=data2_recons_unfil,
                recons=data2_recons,
                half_volume_a=data2_recons_unfil,
                half_volume_b=data1_recons_unfil,
            )
        else:
            raise ValueError(f"Unknown input file type: {args.input_path1} or {args.input_path2}")

    # args.input_path is not None, only apply posterior
    if args.input_path is not None:
        input_path_file_type = detect_cryoem_file_type(args.input_path)
        if input_path_file_type == "MRC":
            vol, apix, _ = load_mrc(args.input_path)
            if input_data1.recons_unfil is not None:
                if input_data1.recons_unfil.shape != vol.shape:
                    raise ValueError(f"Input volumes input_path: {args.input_path} and "
                                     f"input_path1: {args.input_path1} shape mismatch")
                if not math.isclose(input_data1.pixel_size, apix, abs_tol=0.005):
                    raise ValueError(f"Input volumes input_path: {apix} and "
                                     f"input_path1: {input_data1.pixel_size} apix mismatch")

            input_data1.recons_unfil = vol
            input_data2 = None
            input_data1.file_type = "MRC"
            input_data1.mode = "refine"
            input_data1.pixel_size = apix
            input_data1.original_size = vol.shape[0]
            input_data1.output_path = osp.join(args.output_dir,
                                               Path(args.input_path).stem + "_external_reconstruct.mrc")
        elif input_path_file_type == "STAR":
            data = rln_ext_starfile_load_helper(args.input_path)
            data_recons_unfil, data_recons = get_reconstructions(data)
            if input_data1.recons_unfil is not None:
                if input_data1.recons_unfil.shape != data_recons_unfil.shape:
                    raise ValueError(f"Input volumes input_path: {args.input_path} and "
                                     f"input_path1: {args.input_path1} shape mismatch")
                if not math.isclose(input_data1.pixel_size, data['pixel_size'], abs_tol=0.005):
                    raise ValueError(f"Input volumes input_path: {data['pixel_size']} and "
                                     f"input_path1: {input_data1.pixel_size} apix mismatch")

            input_data1.recons_unfil = data_recons_unfil
            input_data2.recons = data_recons
            input_data2 = None
            input_data1.file_type = "STAR"
            input_data1.mode = data["mode"]
            input_data1.pixel_size = data["pixel_size"]
            input_data1.original_size = data["original_size"]
            input_data1.particle_diameter = data["particle_diameter"]
            input_data1.output_path = osp.join(args.output_dir,
                                               Path(args.input_path).stem + "_external_reconstruct.mrc")
            input_data1.fsc_spectra = data["fsc_spectra"]
        else:
            raise ValueError(f"Unknown input file type: {args.input_path}")
    return input_data1, input_data2


def _single_map_inference(
        args,
        input_data: InputData,
        lit_model,
        acl: accelerate.Accelerator, ):
    # device = acl.device

    # 1. prepare input data
    file_type = input_data.file_type
    mode = input_data.mode
    pixel_size = input_data.pixel_size
    output_path = input_data.output_path
    # Do map modification on this raw_volume
    raw_input = input_data.recons_unfil
    raw_recons = input_data.recons

    fsc_spectra = input_data.fsc_spectra
    # half_map_idx = input_data.this_half_index
    original_size = input_data.original_size
    raw_particle_diameter = input_data.particle_diameter

    # 1.5. Check and pad volumes to cubic shape if needed
    raw_input, raw_recons, _, _, _, pad_width = check_and_pad_volumes_to_cubic(
        raw_input, raw_recons, None, None, None
    )
    
    # Update original_size to match padded raw_input shape
    # This is needed because apply_post_processing uses original_size for rescaling
    # and mask creation, which should match the padded raw_input dimensions
    original_size = raw_input.shape[0]

    # 2. resize to model input Apix
    #    to resized_ volumes
    with timer("[Resize to model apix]"):
        resized_input = resize_by_voxel_size(raw_input, pixel_size, MODEL_VOXEL_SIZE)
        resized_input = broadcast_object_list([resized_input, ])[0]

    # 3. Do crop given real space mask for speedup
    #    controlled by args.bbox and args.mask_path
    resized_mask, bbox, bbox_input, _ = apply_bbox_crop(
        mask_path=args.mask_path, do_bbox_crop=args.bbox, resized_input=resized_input,
        resized_wavelet_weights=None, patch_size=args.patch_size, pixel_size=pixel_size, )

    # 5. Normalization and patchify
    maxval_used_for_norm, patched_input, patch_locations, patched_wavelet_weights = apply_normalize_and_patchify(
        do_maxval_norm=args.maxval_norm, bbox_resized_input=bbox_input, bbox_wavelet_weights=None,
        patch_size=args.patch_size, patch_overlap=args.patch_overlap, output_path=output_path, debug=args.debug,
        acl=acl, )

    # 4. Do inference on map
    bbox_output = apply_cond_inference(args, lit_model, patched_input=patched_input, patch_locations=patch_locations,
                                       bbox_input=bbox_input, output_path=output_path, debug=args.debug, acl=acl, )

    if args.maxval_norm:
        bbox_output = bbox_output * maxval_used_for_norm
    if bbox is not None:
        logger.info(f"Expanding output to original model Apix shape {resized_input.shape}")
        resized_output = resized_input.copy()
        resized_output[tuple(map(slice, bbox[0, :], bbox[1, :] + 1))] = bbox_output
    else:
        resized_output = bbox_output

    # 5. resize back
    final_output = apply_post_processing(
        file_type=file_type, raw_input=raw_input, raw_recons=raw_recons, resized_input=resized_input,
        resized_output=resized_output, fsc_spectra=fsc_spectra, pixel_size=pixel_size,
        particle_diameter=raw_particle_diameter, original_size=original_size, spectral_mixing=args.spectral_mixing,
        skip_spectral_trailing=args.skip_spectral_trailing, output_path=output_path, debug=args.debug, acl=acl, )
    
    # 6. Crop back to original shape if padding was applied
    if pad_width is not None:
        slice_fn = tuple([slice(pad_width[i][0], final_output.shape[i] - pad_width[i][1]) for i in range(3)])
        final_output = final_output[slice_fn]
    
    return final_output


def infer3d(
        args,
        input_data: InputData,
        lit_model,
        acl: accelerate.Accelerator,
):
    if input_data is None:
        raise ValueError(f"Input data is None")
    final_output = _single_map_inference(args, input_data, lit_model, acl)

    if acl.is_main_process:
        save_mrc(final_output, input_data.output_path, input_data.pixel_size, np.array([0, 0, 0]))
        logger.info(f'Output to file {input_data.output_path}')

    acl.wait_for_everyone()


def _single_map_process(
        args,
        input_data: InputData,
        lit_model,
        acl: accelerate.Accelerator, ):
    device = acl.device

    # 1. prepare input data
    file_type = input_data.file_type
    mode = input_data.mode
    pixel_size = input_data.pixel_size
    output_path = input_data.output_path
    # Do map modification on this raw_volume
    raw_input = input_data.recons_unfil
    raw_recons = input_data.recons
    # this two half maps is used for calculating operator
    raw_half_a = input_data.half_volume_a
    raw_half_b = input_data.half_volume_b

    fsc_spectra = input_data.fsc_spectra
    half_map_idx = input_data.this_half_index
    original_size = input_data.original_size
    raw_particle_diameter = input_data.particle_diameter
    data_starfile_path = args.data_starfile_path
    raw_ref_volume = input_data.ref_volume

    # 1.5. Check and pad volumes to cubic shape if needed
    raw_input, raw_recons, raw_half_a, raw_half_b, raw_ref_volume, pad_width = check_and_pad_volumes_to_cubic(
        raw_input, raw_recons, raw_half_a, raw_half_b, raw_ref_volume
    )
    
    # Update original_size to match padded raw_input shape
    # This is needed because apply_post_processing uses original_size for rescaling
    # and mask creation, which should match the padded raw_input dimensions
    original_size = raw_input.shape[0]

    # 2. resize to model input Apix
    #    to resized_ volumes
    with timer("[Resize to model apix]"):
        resized_input = resize_by_voxel_size(raw_input, pixel_size, MODEL_VOXEL_SIZE)
        resized_input = broadcast_object_list([resized_input, ])[0]

        resized_half_a, resized_half_b = None, None
        if raw_half_a is not None:
            resized_half_a = resize_by_voxel_size(raw_half_a, pixel_size, MODEL_VOXEL_SIZE)
            resized_half_a = broadcast_object_list([resized_half_a, ])[0]
        if raw_half_b is not None:
            resized_half_b = resize_by_voxel_size(raw_half_b, pixel_size, MODEL_VOXEL_SIZE)
            resized_half_b = broadcast_object_list([resized_half_b, ])[0]

    # 3. prepare forward operator
    # 3.1 FSC
    fsc_vals = compute_fsc_values(raw_half_a=raw_half_a, raw_half_b=raw_half_b, fsc_spectra=fsc_spectra,
                                  output_path=output_path, pixel_size=pixel_size, debug=args.debug, accelerator=acl)
    fsc_vals = broadcast_object_list([fsc_vals, ])[0]

    # 3.2 Fourier mask
    if "inpaint" in args.op:
        with timer("[Create inpaint fourier mask]"):
            raw_fourier_mask, patch_fourier_mask = build_fourier_mask(
                starfile_path=data_starfile_path, half_map_idx=half_map_idx, resized_volume=resized_input,
                patch_size=args.patch_size, fmask_threshold=args.fmask_threshold, output_path=output_path,
                pixel_size=pixel_size, debug=args.debug, accelerator=acl)
    else:
        raw_fourier_mask = np.ones_like(raw_input)
        patch_fourier_mask = torch.ones(args.patch_size, args.patch_size, args.patch_size)
    raw_fourier_mask, patch_fourier_mask = broadcast_object_list([raw_fourier_mask, patch_fourier_mask, ])

    # 3.3 non-uniform wavelet weights
    if "non-uniform" in args.op:
        resized_wavelet_weights = compute_wavelet_weights(
            resized_half_volume_a=resized_half_a, resized_half_volume_b=resized_half_b,
            resized_input=resized_input, nbands=args.nbands, )
        resized_wavelet_weights = broadcast_object_list([resized_wavelet_weights, ])[0]
    else:
        resized_wavelet_weights = None

    # 4. Do crop given real space mask for speedup
    #    controlled by args.bbox and args.mask_path
    resized_mask, bbox, bbox_input, bbox_wavelet_weights = apply_bbox_crop(
        mask_path=args.mask_path, do_bbox_crop=args.bbox, resized_input=resized_input,
        resized_wavelet_weights=resized_wavelet_weights, patch_size=args.patch_size, pixel_size=pixel_size, )

    # 5. Normalization and patchify
    maxval_used_for_norm, patched_input, patch_locations, patched_wavelet_weights = apply_normalize_and_patchify(
        do_maxval_norm=args.maxval_norm, bbox_resized_input=bbox_input, bbox_wavelet_weights=bbox_wavelet_weights,
        patch_size=args.patch_size, patch_overlap=args.patch_overlap, output_path=output_path, debug=args.debug,
        acl=acl, )

    # 6. FSC volume and target energy, relative to noise power
    # TODO: this step now needs maxval_used_for_norm, should be changed
    patch_fsc_vol, patch_target_energy = build_target_energy_and_fsc_volume(
        raw_fourier_mask=raw_fourier_mask, fsc_vals=fsc_vals, raw_input=raw_input, resized_input=resized_input,
        maxval_used_for_norm=maxval_used_for_norm, op=args.op, patch_size=args.patch_size, pixel_size=pixel_size,
        output_path=output_path, debug=args.debug, acl=acl, )
    patch_fsc_vol, patch_target_energy = broadcast_object_list([patch_fsc_vol, patch_target_energy, ])

    # 7. posterior sampling
    bbox_output = apply_cond_posterior_sampling(
        args=args, lit_model=lit_model,
        patched_input=patched_input, patch_locations=patch_locations, patched_wavelet_weights=patched_wavelet_weights,
        patch_fsc_vol=patch_fsc_vol, patch_target_energy=patch_target_energy, patch_fourier_mask=patch_fourier_mask,
        bbox_input=bbox_input, device=device, output_path=output_path, debug=args.debug, acl=acl)

    if args.maxval_norm:
        bbox_output = bbox_output * maxval_used_for_norm
    if bbox is not None:
        logger.info(f"Expanding output to original model Apix shape {resized_input.shape}")
        resized_output = resized_input.copy()
        resized_output[tuple(map(slice, bbox[0, :], bbox[1, :] + 1))] = bbox_output
    else:
        resized_output = bbox_output

    # 8. post-process
    # 8.1 same as spIsoNet mixing
    if "keep_lowres" in args.op and not mode == "refine_final":
        resized_output = apply_spisonet_mixing(
            raw_ref_volume=raw_ref_volume, resized_output=resized_output, resized_mask=resized_mask,
            threshold_res=args.threshold_res, pixel_size=pixel_size, output_path=output_path, debug=args.debug, )

    final_output = apply_post_processing(
        file_type=file_type, raw_input=raw_input, raw_recons=raw_recons, resized_input=resized_input,
        resized_output=resized_output, fsc_spectra=fsc_spectra, pixel_size=pixel_size,
        particle_diameter=raw_particle_diameter, original_size=original_size, spectral_mixing=args.spectral_mixing,
        skip_spectral_trailing=args.skip_spectral_trailing, output_path=output_path, debug=args.debug, acl=acl, )
    
    # 9. Crop back to original shape if padding was applied
    if pad_width is not None:
        slice_fn = tuple([slice(pad_width[i][0], final_output.shape[i] - pad_width[i][1]) for i in range(3)])
        final_output = final_output[slice_fn]
    
    return final_output


def refine3d(
        args,
        input_data1: Optional[InputData],
        input_data2: Optional[InputData],
        lit_model,
        acl: accelerate.Accelerator,
):
    def _save_helper(cur_input, cur_output):
        if cur_input.mode == "refine":
            save_mrc(cur_output, cur_input.output_path, cur_input.pixel_size, np.array([0, 0, 0]))
            logger.info(f'Output to file {cur_input.output_path}')
        elif cur_input.mode == "refine_final":
            save_mrc(cur_input.recons, cur_input.output_path, cur_input.pixel_size, np.array([0, 0, 0]))
            logger.info(f'Final reconstruction output to file {cur_input.output_path}')

            save_path = cur_input.output_path[:-4] + "_denoised.mrc"
            save_mrc(cur_output, save_path, cur_input.pixel_size, np.array([0, 0, 0]))
            logger.info(f'Final reconstruction denoised output to file {save_path}')

    final_output1 = None
    final_output2 = None
    if input_data1 is not None:
        if input_data1.mode == "refine_final":
            final_output1 = input_data1.recons_unfil
        else:
            final_output1 = _single_map_process(args, input_data1, lit_model, acl)
    if input_data2 is not None:
        if input_data2.mode == "refine_final":
            final_output2 = input_data2.recons_unfil
        else:
            final_output2 = _single_map_process(args, input_data2, lit_model, acl)

    # fsc weighting
    if args.fsc_weighting and (final_output1 is not None and final_output2 is not None):
        logger.info(f"Apply FSC weighting.")
        final_output1, final_output2 = apply_fsc_weighting(final_output1, final_output2, args, acl=acl)

    if acl.is_main_process:
        if final_output1 is not None:
            _save_helper(input_data1, final_output1)
        if final_output2 is not None:
            _save_helper(input_data2, final_output2)
        # save avg
        if final_output1 is not None and final_output2 is not None:
            save_mrc(np.asarray((final_output1 + final_output2) / 2, dtype=np.float32),
                     osp.join(args.output_dir, "avg_external_reconstruct.mrc"), input_data1.pixel_size,
                     np.array([0, 0, 0]))

    acl.wait_for_everyone()


if __name__ == "__main__":
    cli_args = parse_args()
    setup_logging(cli_args.log_file_path)
    logger = get_logger(__name__)

    accelerator = accelerate.Accelerator()

    lit_model_ = load_model(cli_args, device=accelerator.device)

    input_data1_, input_data2_ = prepare_input(cli_args)

    if not osp.exists(cli_args.output_dir):
        os.makedirs(cli_args.output_dir, exist_ok=True)

    # do pure inference
    if cli_args.input_path is not None and cli_args.input_path1 is None:
        logger.info(f"Apply pure conditional inference on density map {cli_args.input_path}.")
        infer3d(cli_args, input_data=input_data1_, lit_model=lit_model_, acl=accelerator)
    # do posterior sampling
    else:
        if input_data1_.mode == "classification":
            raise NotImplementedError(f"Mode not supported: {input_data1_.mode}")
        elif input_data1_.mode == "refine" or input_data1_.mode == "refine_final":
            refine3d(
                cli_args,
                input_data1=input_data1_,
                input_data2=input_data2_,
                lit_model=lit_model_,
                acl=accelerator
            )
        else:
            raise NotImplementedError(f"Mode not supported: {input_data1_.mode}")

    accelerator.wait_for_everyone()
