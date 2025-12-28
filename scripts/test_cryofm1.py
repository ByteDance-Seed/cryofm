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

import time
import os.path as osp
from pathlib import Path
from typing import List, Callable
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import einops
from diffusers import DDPMScheduler
from mmengine.registry import DATASETS
from mmengine import Config, ConfigDict, mkdir_or_exist
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from cryofm.core.demo_data import load_fsc_curves
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.utils.misc import pretty_dict
from cryofm.core.utils.metrics import calc_fsc, plot_fsc
from cryofm.core.utils.setup_env import register_custom_modules
from cryofm.core.training.lightning_utils import autoload_model
from cryofm.core.training.accelerate_utils import setup_logging, _get_next_version
from cryofm.core.fourier import primal_to_fourier_3d, hartley_to_primal_3d, primal_to_hartley_3d

from cryofm.projects.cryofm1.lit_modules import CryoFM1
from cryofm.projects.cryofm1.forward_operators import (BaseOperator, NoiseOperator, GaussianNoiseOperator,
                                                       SpectralNoiseOperator, AnisotropicSpectralNoiseOperator,
                                                       MissingWedgeOperator, LowpassOperator, CompositeOperator)
from cryofm.projects.cryofm1.forward_operators.noise import (radial_average_3d_power_spectrum, radial_noise_from_snr,
                                                             noise_power_from_radial, noise_power_from_fsc,
                                                             noise_power_from_amp_factor_matrix, plot_radial_power,
                                                             plot_slice_power)

LINE = '-' * 18
BASE_CONFIG = Config(
    dict(
        work_dir="work_dirs",
        data_root=None,
        exp_name=None,
        seed=1,
        log_level="DEBUG",

        model_dir=None,
        ckpt=None,

        prior=dict(
            prior_type="fm",  # "fm" or "ddpm"
            apix=1.5,
            side_shape=64,
            num_timesteps=1000,
        ),

        dps=dict(
            lamb_base=1.0,
            lamb_w_max=5.0,
            norm_grad=True,
            normalization=None,  # normalization before DPS. "clip-norm", "linear", or "clip-linear"
            loss_fn="f-mse",  # r-l2, r-l1, f-mse, f-msf; only support f-mse for now
            postprocess_value=None
            # rescale DPS modified volume value range. "clip01", "clip01-rescale", or "clip_norm-rescale"
        ),

        task=dict(
            task_names=["lowpass_filter"],
            # gaussian_noise, spectral_noise, anisotropic_spectral_noise, missing_wedge, lowpass_filter
            snr_idx=1,  # 1 to 5
            gaussian_noise_power=0.05,  # 0.02, 0.05, 0.1
            spectral_noise_snr=None,
            amp_factor=50.0,
            tilt_angle=60,
            lowpass_diameter=36,
            eval_n_samples=None,  # only eval n samples, set to None to eval all samples
        ),
    )
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--exp-name", type=str)
    group = parser.add_argument_group('Model', 'Model configuration')
    group.add_argument("--model-dir", action="store", type=str, required=True)
    group.add_argument('--log-level', default='INFO',
                       help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    group = parser.add_argument_group('DPS', 'DPS hyperparameters')
    group.add_argument("--num-timesteps", action="store", type=int, default=200)
    group.add_argument("--lamb-base", action="store", type=float, default=1.)
    group.add_argument("--lamb-w-max", action="store", type=float, default=5.)

    group = parser.add_argument_group('Task', 'Task configuration')
    group.add_argument("--task-names", nargs='+', action="store", type=str, required=True, default="lowpass_filter")
    group.add_argument("--snr-idx", action="store", type=int, default=1)  # 1-5
    group.add_argument("--tilt-angle", action="store", type=int, default=60)
    group.add_argument("--lowpass-diameter", action="store", type=int, default=None)
    group.add_argument("--eval-n-samples", action="store", type=int, default=None)
    # group.add_argument("--perturb-ratio", action="store", type=float, default=None)

    args = parser.parse_args()
    return args


def setup_config(base_config, cli_args):
    cfg = base_config.copy()

    def _merge(cfg, args):
        for k in cfg.keys():
            if isinstance(cfg[k], ConfigDict):
                cfg[k] = _merge(cfg[k], args)
            else:
                if hasattr(cli_args, k):
                    cfg[k] = getattr(cli_args, k)
        return cfg

    cfg = _merge(cfg, cli_args)
    return cfg


def get_test_dataset(data_root):
    if data_root is None:
        raise ValueError("data_root is not set")

    test_dataset = dict(
        ann_file='split/test.csv',
        data_root=data_root,
        pipeline=[
            dict(type='LoadMapFromFile'),
            dict(patch_size=64, type='CenterCrop3D'),
            dict(max_percentile=99.999, min_value=0, type='ClipNorm3D'),
            dict(keys=[
                'map_data',
            ], size=64, type='Pad3D'),
            # dict(num_patches=2, patch_size=64, type='RandomPatches3D'),
            dict(keys=[
                'map_id',
                'src_shape',
                'map_data',
            ], type='PackInputs'),
        ],
        type='EMDBMapDataset')
    test_dataset = DATASETS.build(test_dataset)
    return test_dataset


def create_forward_operator(cfg, device, shape, clean_signal=None):
    """Create degradation operator based on configuration."""
    operators = []

    for task in cfg.task.task_names:
        if task == "gaussian_noise":
            op = GaussianNoiseOperator(device=device, shape=shape, noise_power=cfg.task.gaussian_noise_power)
        elif task == "spectral_noise":
            op = SpectralNoiseOperator(device=device, shape=shape)
            op.setup_from_snr(signal=clean_signal, snr=cfg.task.spectral_noise_snr)
        elif task == "anisotropic_spectral_noise":
            op = AnisotropicSpectralNoiseOperator(device=device, shape=shape)
            op.setup_from_snr(signal=clean_signal, snr=cfg.task.spectral_noise_snr)
            op.setup_cryoet_anisotropic(tilt_angle=cfg.task.tilt_angle, amp_factor=cfg.task.amp_factor)
        elif task == "missing_wedge":
            op = MissingWedgeOperator(device=device, shape=shape, tilt_angle=cfg.task.tilt_angle)
        elif task == "lowpass_filter":
            op = LowpassOperator(device=device, shape=shape, cutoff_diameter=cfg.task.lowpass_diameter)
        else:
            raise NotImplementedError(f"Operator {task} not implemented")

        operators.append(op)

    if len(operators) == 1:
        return operators[0]

    return CompositeOperator(operators)


def estimate_operator_from_fsc(device, shape, half_map1, half_maps_fsc):
    noise_radial_power, signal_radial_power = noise_power_from_fsc(half_map1, half_maps_fsc)
    logger.debug(f"Estimated noise radial power: {noise_radial_power}.")
    D = half_map1.shape[-1]
    op = NoiseOperator(device=device, shape=shape,
                       noise_power=noise_power_from_radial(D, noise_radial_power).to(device))
    return op, signal_radial_power, noise_radial_power


def estimate_operator_from_fsc_and_amp_matrix(device, shape, half_map1, half_maps_fsc, amp_matrix):
    noise_radial_power, signal_radial_power = noise_power_from_fsc(half_map1, half_maps_fsc)
    logger.debug(f"Estimated noise radial power: {noise_radial_power}.\n"
                 f"min: {torch.min(noise_radial_power):.4e}, max: {torch.max(noise_radial_power):.4e}, mean: {torch.mean(noise_radial_power):.4e}.")
    noise_power = noise_power_from_amp_factor_matrix(half_map1, noise_radial_power, amp_matrix).to(device)
    op = NoiseOperator(device=device, shape=shape, noise_power=noise_power)
    return op, signal_radial_power, noise_radial_power


def scale_from_data(x, cfg):
    if hasattr(cfg, "z_scale"):
        x = (x - cfg.z_scale.mean) / cfg.z_scale.std
    return x


def scale_to_data(x, cfg):
    if hasattr(cfg, "z_scale"):
        x = x * cfg.z_scale.std + cfg.z_scale.mean
    return x


def save_volumes(volumes, filenames, apix: float, save_dir: str):
    if not osp.exists(save_dir):
        mkdir_or_exist(save_dir)
    for volume, filename in zip(volumes, filenames):
        save_mrc(volume, osp.join(save_dir, filename), apix)


def calculate_and_plot_fsc(vol1, vol2, thresholds, voxel_size, save_path, description="", logger_handler=None):
    fsc_values, fsc_freq = calc_fsc(vol1, vol2)
    mean_fsc = np.mean(fsc_values)
    logger_handler.info(f"Mean FSC {description} {mean_fsc:.4f}")

    res_list = plot_fsc(fsc_values, fsc_freq, thresholds, voxel_size=voxel_size,
                        save_path=save_path)
    threshold_str = ", ".join(map(str, thresholds))
    logger_handler.info(f"FSC res {description} at [{threshold_str}]: {', '.join(map(lambda x: f'{x:.4f}', res_list))}")
    return fsc_values, fsc_freq, mean_fsc


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def observation_half_map_stats(y, y1, y2, cfg, save_dir: str, logger_handler):
    logger_handler.info(f"{LINE} Observation stats:")
    apix = cfg.prior.apix
    # save input volumes
    save_volumes([y.cpu().numpy(), y1.cpu().numpy(), y2.cpu().numpy()],
                 ["y.mrc", "src_y1.mrc", "src_y2.mrc"],
                 apix, save_dir)

    y = to_numpy(y)
    y1 = to_numpy(y1)
    y2 = to_numpy(y2)
    y12 = (y1 + y2) / 2

    # Calculate FSC between half maps y1 y2
    y1_y2_fsc, fsc_freq, _ = calculate_and_plot_fsc(y1, y2, [0.5, 0.143], apix, osp.join(save_dir, "half-maps-fsc.png"),
                                                    "between half maps", logger_handler)

    # Calculate FSC between averaged half maps and ground truth
    _, _, gt_y12_mean_fsc = calculate_and_plot_fsc(y, y12, [0.5, 0.143], apix, osp.join(save_dir, "cref-fsc.png"),
                                                   "\033[4m\033[1mbetween original and averaged noised volumes\033[0m",
                                                   logger_handler)

    # Calculate FSC between individual half maps and ground truth
    calculate_and_plot_fsc(y, y1, [0.5, 0.378, 0.143], apix, osp.join(save_dir, "half-map1-gt-fsc.png"),
                           "between original and noised volume 1", logger_handler)

    calculate_and_plot_fsc(y, y2, [0.5, 0.378, 0.143], apix, osp.join(save_dir, "half-map2-gt-fsc.png"),
                           "between original and noised volume 2", logger_handler)
    return y1_y2_fsc, fsc_freq, gt_y12_mean_fsc


def prediction_half_map_stats(y, y1, y2, cfg, save_dir: str, logger_handler):
    logger_handler.info(f"{LINE} Prediction stats:")
    apix = cfg.prior.apix

    y = to_numpy(y)
    y1 = to_numpy(y1)
    y2 = to_numpy(y2)
    y12 = (y1 + y2) / 2

    save_volumes([y1, y2, y12],
                 ["new_y1.mrc", "new_y2.mrc", "new_y_avg.mrc"],
                 apix, save_dir)

    # new_y1 vs gt fsc
    _, _, y1_y_mean_fsc = calculate_and_plot_fsc(y1, y, [0.5, 0.378, 0.143], apix,
                                                 osp.join(save_dir, "half-map1-dps-gt-fsc.png"),
                                                 "between original and output volume 1", logger_handler)

    # new_y2 vs gt fsc
    _, _, y2_y_mean_fsc = calculate_and_plot_fsc(y2, y, [0.5, 0.378, 0.143], apix,
                                                 osp.join(save_dir, "half-map2-dps-gt-fsc.png"),
                                                 "between original and output volume 2", logger_handler)

    # half maps fsc after dps
    _, _, _ = calculate_and_plot_fsc(y1, y2, [0.5, 0.143], apix,
                                     osp.join(save_dir, "half-maps-dps-fsc.png.png"),
                                     "between half maps after dps", logger_handler)

    # avg dps half maps vs gt fsc (dps cref)
    _, _, y12_y_mean_fsc = calculate_and_plot_fsc(y12, y, [0.5, 0.143], apix,
                                                  osp.join(save_dir, "cref-dps-fsc.png"),
                                                  "\033[4m\033[1mbetween original and output avg half map (cref dps)\033[0m",
                                                  logger_handler)
    return y1_y_mean_fsc, y2_y_mean_fsc, y12_y_mean_fsc


def estimate_half_map_operators(y, y1, y2, cfg, y1_y2_fsc, fsc_freq, gt_op, save_dir: str):
    device = y.device
    shape = y.shape[-3:]

    gt_signal_radial_power = radial_average_3d_power_spectrum(y)
    gt_radial_noise = radial_noise_from_snr(y, cfg.task.spectral_noise_snr)

    if set(cfg.task.task_names).issubset({'gaussian_noise', 'spectral_noise'}):
        logger.info(f"Noise is isotropic and no fourier mask. Estimating noise power from the half map FSC curve.")
        # the returned power are all radial power
        op1, signal_power1, noise_power1 = estimate_operator_from_fsc(device, shape, y1, y1_y2_fsc)
        op2, signal_power2, noise_power2 = estimate_operator_from_fsc(device, shape, y2, y1_y2_fsc)
        plot_radial_power([gt_radial_noise, noise_power1, noise_power2], fsc_freq,
                          save_path=osp.join(save_dir, "noise_powers.png"))
        plot_radial_power([gt_signal_radial_power, signal_power1, signal_power2], fsc_freq,
                          save_path=osp.join(save_dir, "signal_powers.png"))
    elif "anisotropic_spectral_noise" in cfg.task.task_names:
        logger.info(f"Noise is anisotropic and no fourier mask. Estimating noise power for the entire volume.")
        logger.info(
            f"GT noise power is min: {torch.min(gt_op.noise_power):.4e}, max: {torch.max(gt_op.noise_power):.4e}, mean: {torch.mean(gt_op.noise_power):.4e}.")
        op1, _, _ = estimate_operator_from_fsc_and_amp_matrix(device, shape, y1, y1_y2_fsc,
                                                              gt_op.amp_matrix.cpu().numpy())

        plot_slice_power(
            [gt_op.noise_power.cpu(), op1.noise_power.cpu(), (gt_op.noise_power.cpu() - op1.noise_power.cpu())],
            slice_axis=2, logscale=True,
            save_path=osp.join(save_dir, "noise_powers_1.png"))

        op2, _, _ = estimate_operator_from_fsc_and_amp_matrix(device, shape, y2, y1_y2_fsc,
                                                              gt_op.amp_matrix.cpu().numpy())

        plot_slice_power(
            [gt_op.noise_power.cpu(), op2.noise_power.cpu(), (gt_op.noise_power.cpu() - op2.noise_power.cpu())],
            slice_axis=2, logscale=True,
            save_path=osp.join(save_dir, "noise_powers_2.png"))
    else:
        logger.info(f"Use GT Operator as estimated operator")
        op1 = gt_op
        op2 = gt_op
    return op1, op2


def determine_dsp_task_type(task_names: List[str]) -> str:
    """Determine DPS task type (denoising/inpainting/inpainting_and_denoising).
    """
    has_mask = "lowpass_filter" in task_names or "missing_wedge" in task_names
    has_noise = any(
        noise_type in task_names for noise_type in ["spectral_noise", "gaussian_noise", "anisotropic_spectral_noise"])

    if has_mask:
        return "inpainting_and_denoising" if has_noise else "inpainting"
    elif has_noise:
        return "denoising"
    else:
        raise ValueError(f"Unknown task type: {task_names}")


# In CryoFM1 we use modify xt by loss grad
def run_dps(exp_cfg: Config, model_cfg: Config, scheduler, op: BaseOperator,
            unet: nn.Module, y: torch.Tensor, device: torch.device,
            map_id: str, idx: int) -> np.ndarray:
    """Run Diffusion(Flow) posterior sampling.

    Args:
        exp_cfg: Configuration object
        model_cfg: Model configuration
        scheduler: Noise scheduler
        op: Degradation operator
        unet: UNet model
        y: Input tensor
        device: PyTorch device
        map_id: Map identifier
        idx: Sample index

    Returns:
        np.ndarray: DPS output volume
    """
    task_names = exp_cfg.task.task_names
    task_type = determine_dsp_task_type(task_names)
    logger.info(f"Forward operators are: {task_names}")
    logger.info(f"DPS task type: {task_type}")
    side_shape = exp_cfg.prior.side_shape

    logger.info(f"Num time steps {exp_cfg.prior.num_timesteps}")
    scheduler.set_timesteps(exp_cfg.prior.num_timesteps, device=device)
    # fork a custom scheduler, only used for DDPM,
    if exp_cfg.prior.prior_type.startswith("ddpm"):
        custom_ddpm_scheduler = DDPMScheduler.from_config(scheduler.config, prediction_type="sample")
        custom_ddpm_scheduler.set_timesteps(exp_cfg.prior.num_timesteps, device=device)
    else:
        custom_ddpm_scheduler = None

    # customize t
    # ts = scheduler.timesteps
    # ts = ts[ts >= 50]
    # ts2 = np.arange(0, 50)[::-1].copy().tolist()
    # ts = ts.tolist() + ts2
    # scheduler.set_timesteps(timesteps=ts, device=device)
    # if custom_ddpm_scheduler is not None:
    #     custom_ddpm_scheduler.set_timesteps(timesteps=ts, device=device)

    # initial sample
    x_t = torch.randn((1, 1, side_shape, side_shape, side_shape),
                      requires_grad=True, device=device)

    logger.debug(f"Given y " + pretty_dict({
        "min": y.min(), "max": y.max(), "mean": y.mean(), "var": y.var()
    }, sci_mode=True))

    # some heuristic settings
    if exp_cfg.prior.prior_type.startswith("fm"):
        # flow posterior sampling should be re-weighted by t/(1-t), i.e., inf -> 0
        assert exp_cfg.dps.norm_grad is True, "Now we only support norm=True for flow"
        lamb_base = exp_cfg.dps.lamb_base
    else:
        if exp_cfg.dps.norm_grad:
            lamb_base = exp_cfg.dps.lamb_base
            lamb_decay = 0
        else:
            lamb_base = exp_cfg.dps.lamb_base
            lamb_decay = 100

    for t in scheduler.timesteps:
        _t = t.item()

        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
        op.zero_grad()

        model_output = unet(x_t, einops.repeat(t, "->b", b=len(x_t)).to(device))
        step_output = scheduler.step(model_output, t, x_t)
        x_t_prev = step_output.prev_sample
        logger.debug(f"Time step {t} xt_prev " + pretty_dict({
            "min": x_t_prev.min(), "max": x_t_prev.max(), "mean": x_t_prev.mean(), "var": x_t_prev.var()
        }, sci_mode=True))
        x_0_hat = step_output.pred_original_sample
        logger.debug(f"Time step {t} x0_hat " + pretty_dict({
            "min": x_0_hat.min(), "max": x_0_hat.max(), "mean": x_0_hat.mean(), "var": x_0_hat.var()
        }, sci_mode=True))
        scaled_x_0_hat = scale_to_data(x_0_hat, model_cfg)
        logger.debug(f"Time step {t} scaled x0^hat " + pretty_dict({
            "min": scaled_x_0_hat.min(), "max": scaled_x_0_hat.max(), "mean": scaled_x_0_hat.mean(),
            "var": scaled_x_0_hat.var()
        }, sci_mode=True))
        scaled_y = scale_from_data(y, model_cfg)

        if task_type == "inpainting_and_denoising":
            """
            in-painting + denoising
            don't have to maintain the integrity of the unmasked area
            only calculate the loss in the unmasked area (masked area snr=0)
            """
            noise_power = op.noise_power
            f_mask = op.f_mask
            op_x_0_hat = op(scaled_x_0_hat)
        elif task_type == "inpainting":
            """
            in-painting only
            mix x_0_hat with gt y to maintain the interity of the unmasked area
            """
            noise_power = None
            f_mask = op.f_mask
            h_x_0_hat = primal_to_hartley_3d(x_0_hat)
            scaled_h_y = primal_to_hartley_3d(scaled_y)
            h_x_0_hat = scaled_h_y * f_mask + h_x_0_hat * f_mask.logical_not()
            x_0_hat = hartley_to_primal_3d(h_x_0_hat)
            # x_0_hat = op(x_0_hat)
            if exp_cfg.prior.prior_type.startswith("fm"):
                a = (_t - 1000 // exp_cfg.prior.num_timesteps) / _t
                b = 1000 // exp_cfg.prior.num_timesteps / _t
                x_t_prev = a * x_t + b * x_0_hat
            else:
                # prediction_type = "sample", given x_0_hat
                step_output = custom_ddpm_scheduler.step(x_0_hat, t, x_t)
                x_t_prev = step_output.prev_sample
        elif task_type == "denoising":
            noise_power = op.noise_power
            f_mask = None
            op_x_0_hat = op(scaled_x_0_hat)

        if exp_cfg.dps.loss_fn == "r-l2":
            raise NotImplemented
            # loss = F.mse_loss(x_0_hat, y)
        elif exp_cfg.dps.loss_fn == "r-l1":
            raise NotImplemented
            # loss = F.l1_loss(x_0_hat, y)
        elif exp_cfg.dps.loss_fn == "f-mse":
            if task_type == "denoising":
                # reweight loss with noise power
                f_op_x_0_hat = torch.view_as_real(primal_to_fourier_3d(op_x_0_hat) / torch.sqrt(noise_power))
                f_y = torch.view_as_real(primal_to_fourier_3d(y) / torch.sqrt(noise_power))
                loss = F.mse_loss(f_op_x_0_hat, f_y)
            elif task_type == "inpainting":
                # unmasked area is the same as y
                loss = None
            elif task_type == "inpainting_and_denoising":
                # reweight loss with noise power AND only calculate the unmasked area
                f_op_x_0_hat = torch.view_as_real(primal_to_fourier_3d(op_x_0_hat) / torch.sqrt(noise_power) * f_mask)
                f_y = torch.view_as_real(primal_to_fourier_3d(y) / torch.sqrt(noise_power) * f_mask)
                loss = F.mse_loss(f_op_x_0_hat, f_y)
            else:
                raise NotImplementedError
        elif exp_cfg.dps.loss_fn == "f-msf":
            # f_x_0_hat = torch.view_as_real(primal_to_fourier_3d(x_0_hat))
            # f_y = torch.view_as_real(primal_to_fourier_3d(y))
            # loss = - torch.sum(f_x_0_hat * f_y) + torch.sum(f_x_0_hat * f_x_0_hat) / 2
            raise NotImplementedError
        else:
            raise NotImplementedError

        if loss is not None:
            loss_grad = torch.autograd.grad(outputs=loss, inputs=x_t, allow_unused=True, materialize_grads=True)[0]
            logger.debug(f"Time step {_t} " + pretty_dict({
                "loss": loss, "grad-mean": loss_grad.mean(), "grad-var": loss_grad.var()
            }, sci_mode=True))

            if exp_cfg.prior.prior_type.startswith("fm"):
                lamb_w = min(_t / max(.001, 1000 - _t), exp_cfg.dps.lamb_w_max)
            else:
                if _t < lamb_decay:
                    lamb_w = _t / lamb_decay
                else:
                    lamb_w = 1

            if exp_cfg.dps.norm_grad:
                x_t_prev = x_t_prev - lamb_w * lamb_base * loss_grad / loss_grad.norm()
                logger.debug(f"Total lamb is {(lamb_w * lamb_base):.4e}")
            else:
                x_t_prev = x_t_prev - lamb_w * lamb_base * loss_grad
        x_t = x_t_prev.detach()

        if _t % 100 == 0:
            with torch.no_grad():
                new_y = scale_to_data(x_t[0, 0].detach().cpu().numpy(), model_cfg)
            save_mrc(new_y, osp.join(exp_cfg.work_dir, exp_cfg.exp_name, map_id, f"{idx}-{_t}.mrc"), exp_cfg.prior.apix)

    with torch.no_grad():
        new_y = scale_to_data(x_t[0, 0], model_cfg)

    logger.debug(f"Output volume value min {new_y.min():.4e}, max {new_y.max():.4e},"
                 f" p99.999 {torch.quantile(new_y, 0.99999):.4e}")

    new_y = new_y.detach().cpu().numpy()
    return new_y


def main(acl: Accelerator, cfg: Config):
    """Main function for map modification.

    Args:
        acl: Accelerator object
        cfg: Configuration object
    """
    device = acl.device

    lit_model = load_model_from_oss(cfg, device)

    model = lit_model.model.to(device)
    scheduler = lit_model.noise_scheduler
    test_dataset = get_test_dataset(cfg.data_root)

    results = []
    n_samples = cfg.task.eval_n_samples or len(test_dataset)

    time_elapsed = []

    for i in range(n_samples):
        logger.info(f"{LINE} Processing sample {i}/{n_samples} {LINE}")

        map_id = str(test_dataset[i]["map_id"][0].item())
        save_dir = osp.join(cfg.work_dir, cfg.exp_name, map_id)
        mkdir_or_exist(save_dir)

        y = test_dataset[i]["map_data"][None, None].to(device)

        op = create_forward_operator(cfg, device, y.shape[-3:], clean_signal=y)

        with torch.no_grad():
            y1 = op(y)
            y2 = op(y)

        # Observation stats
        y1_y2_fsc, fsc_freq, gt_y12_mean_fsc = observation_half_map_stats(y.squeeze(), y1.squeeze(), y2.squeeze(), cfg,
                                                                          save_dir, logger)

        # Estimate degradation operators
        tic = time.time()
        op1, op2 = estimate_half_map_operators(y, y1, y2, cfg, y1_y2_fsc, fsc_freq, op, save_dir)

        # Run DPS
        new_y1 = run_dps(cfg, lit_model.cfg, scheduler, op1, model, y1, device, map_id, 1)
        new_y2 = run_dps(cfg, lit_model.cfg, scheduler, op2, model, y2, device, map_id, 2)
        toc = time.time()
        time_elapsed.append(toc - tic)

        y1_y_mean_fsc, y2_y_mean_fsc, y12_y_mean_fsc = prediction_half_map_stats(
            y.squeeze(), new_y1.squeeze(), new_y2.squeeze(), cfg, save_dir, logger)

        results.append([gt_y12_mean_fsc, y1_y_mean_fsc, y2_y_mean_fsc, y12_y_mean_fsc])

    time_elapsed = np.mean(time_elapsed)
    logger.info("Average running time per sample: {:.4f}".format(time_elapsed))

    results = np.array(results).T
    results = np.hstack([np.mean(results, axis=1, keepdims=True), np.std(results, axis=1, keepdims=True), results])
    results = np.around(results, decimals=4)
    df = pd.DataFrame(results)
    df.columns = ["mean", "std"] + ["id " + str(i) for i in range(n_samples)]
    df.insert(0, "type", pd.Index(["observation", "pred1", "pred2", "pred half avg"]))

    # if wandb is not None:
    #     wandb.summary["running_time"] = time_elapsed
    #     wandb.log({"summary": wandb.Table(dataframe=df)})

    logger.info("\n" + df.to_string(float_format=lambda x: "{:.2f}".format(x)))


# load model from trained checkpoint (for development/testing)
def load_model_from_checkpoint(cfg: Config):
    lit_model = autoload_model(
        model_dir=cfg.model_dir,
        ckpt_dir=cfg.ckpt,
        model_file_name="model_uncond.py",
        model_name="DDPM3D",
        config_file_name="config.py",
        ckpt_file_name="ckpt.pt"
    )
    lit_model.eval()
    return lit_model


# load model from open-sourced weights
def load_model_from_oss(cfg: Config, device: torch.device):
    model_cfg = Config.fromfile(osp.join(cfg.model_dir, "config.yaml"))
    lit_model = CryoFM1.load_from_safetensors(
        osp.join(cfg.model_dir, "model.safetensors"),
        cfg=model_cfg
    ).to(device)
    lit_model.eval()
    return lit_model


if __name__ == "__main__":
    register_custom_modules()

    accelerator = Accelerator()
    logger = get_logger(__name__)

    args_from_cli = parse_args()
    config = setup_config(BASE_CONFIG, args_from_cli)

    logger.setLevel(config.log_level)

    set_seed(config.seed)
    if config.exp_name is None:
        config.exp_name = (f"dps-{Path(config.model_dir).name}-{'_'.join(config.degrade_type)}"
                           f"-ts{config.prior.num_timesteps}-lamb{config.lamb_w_max}"
                           f"-snr{config.task.snr_idx}-angle{config.degrade.top_angle_degree}")
    if not osp.exists(config.work_dir):
        mkdir_or_exist(config.work_dir)
    version = _get_next_version(config.work_dir, config.exp_name)
    config.exp_name += f"_{version:02d}"

    # if wandb is not None:
    #     wandb.init(project="cryoem_dps_mapmod", name=getattr(config, "exp_name", None))
    #     wandb.config.update(vars(args_from_cli))

    fsc_curves = load_fsc_curves()

    # set SNR here
    logger.info(f"Use FSC SNR at {fsc_curves.columns[config.task.snr_idx]}")
    fsc = torch.from_numpy(fsc_curves.iloc[:, config.task.snr_idx].values).float()
    fsc = torch.where(fsc <= 0, 1e-9, fsc)
    snr = fsc / (1 - fsc)

    config.task.spectral_noise_snr = snr

    mkdir_or_exist(osp.join(config.work_dir, config.exp_name))
    setup_logging(osp.join(config.work_dir, config.exp_name, "run.log"))

    logger.info(config.pretty_text)

    main(accelerator, config)
