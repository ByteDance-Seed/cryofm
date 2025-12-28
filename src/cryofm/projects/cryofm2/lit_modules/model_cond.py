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
import math
import warnings
import subprocess
import os.path as osp
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import einops

from diffusers import DDPMScheduler
from mmengine import mkdir_or_exist
from safetensors.torch import load_file as load_safetensors

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from cryofm.core.optimizers import FairseqAdam
from cryofm.core.models.ema import LitEma
from cryofm.core.utils.scheduling_fm import FMScheduler
from cryofm.core.utils.sampling_fm import sample_from_fm
from cryofm.core.datasets.cryofm import CryoFMMapDataset, CryoFMMapCropDataset
from cryofm.core.datasets.transforms.patchify import GridPatches3D, GridAggregator
from cryofm.core.models.unet3d import UNet3DModel
from cryofm.core.training.lightning_utils import get_1st_unique_indices, log_to_current, init_pl_w_cfg
from cryofm.core.utils.misc import pretty_dict
from cryofm.core.training.misc import count_params
from cryofm.core.utils.setup_env import register_custom_modules
from cryofm.core.utils.shape_utils import get_pad_width
from cryofm.core.utils.metrics import calc_fsc
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.training.timestep_sampling import sample_timesteps

log_to_current = rank_zero_only(log_to_current)


class CryoFM2Cond(pl.LightningModule):
    """LightningModule for CryoFM2 conditional 3D diffusion and flow matching.

    This class implements a PyTorch LightningModule for conditional generation
    in the CryoFM2 project, supporting both DDPM (Denoising Diffusion Probabilistic Model)
    and flow matching processes on 3D cryo-EM data using UNet3DModel architecture.
    
    The model supports conditional generation with volume conditions (vol_cond) and
    output tags (output_cond), enabling tasks such as volume-to-volume translation
    and conditional refinement. It includes EMA (Exponential Moving Average) support
    for stable training and evaluation, and supports patch-based inference for
    handling arbitrary-sized input volumes.
    """

    def __init__(
            self,
            cfg,
            ignore_keys=None,
            input_key="patches", ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []

        self.cfg = cfg
        self.input_key = input_key
        # check some info here
        log_to_current(f"Use z_scale mean {self.cfg.z_scale.mean} std {self.cfg.z_scale.std}")
        self.use_apix_cond = False
        self.use_vol_cond = False

        log_to_current(f"~~~~~~~~ Use model type UNet ~~~~~~~~")
        self.model = UNet3DModel(**cfg.model)
        self.use_vol_cond = True

        if cfg.process == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                clip_sample=False,
                prediction_type=cfg.ddpm.prediction_type
            )
        elif cfg.process == "fm":
            self.noise_scheduler = FMScheduler(1000)
        else:
            raise NotImplementedError

        self.history_saved_dirs = []
        self.within_ema = False
        self.model_ema = LitEma(self.model, decay=0.99)
        log_to_current(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.ori_val_loss = defaultdict(lambda: [])
        self.ema_val_loss = defaultdict(lambda: [])

    @contextmanager
    def ema_scope(self, use_ema: bool, context=None):
        if use_ema:
            self.within_ema = True
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                log_to_current(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if use_ema:
                self.model_ema.restore(self.model.parameters())
                self.within_ema = False
                if context is not None:
                    log_to_current(f"{context}: Restored training weights")

    def on_fit_start(self):
        self.cfg.work_dir = self.trainer.strategy.broadcast(self.cfg.work_dir)

    # resample on the every epoch beginning
    def on_train_epoch_start(self):
        if isinstance(self.trainer.train_dataloader.dataset, ConcatDataset):
            for dataset in self.trainer.train_dataloader.dataset.datasets:
                dataset.set_epoch(self.trainer.current_epoch)
        else:
            self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)

    # con_flag differential condition, task flag differential task.
    def forward(self, noisy_latents, timesteps, generation_conds=None):
        output_cond = generation_conds["output_cond"]  # one dimension
        vol_cond = generation_conds["vol_cond"]
        if vol_cond is None:
            vol_cond = torch.zeros_like(noisy_latents)  # vol cond is 0
            con_flag = torch.zeros_like(noisy_latents)  # con flag is 0
            inputs = torch.concat([noisy_latents, vol_cond, con_flag], dim=1)
        else:
            con_flag = torch.ones_like(noisy_latents)  # con flag is 1
            inputs = torch.concat([noisy_latents, vol_cond, con_flag], dim=1)
        return self.model(inputs, timestep=timesteps, class_labels=output_cond).sample

    @staticmethod
    def scale_from_data(cfg, x):
        if hasattr(cfg, "z_scale") and cfg.z_scale.mean is not None and cfg.z_scale.std is not None:
            x = (x - cfg.z_scale.mean) / cfg.z_scale.std
        return x

    @staticmethod
    def scale_to_data(cfg, x):
        if hasattr(cfg, "z_scale") and cfg.z_scale.mean is not None and cfg.z_scale.std is not None:
            x = x * cfg.z_scale.std + cfg.z_scale.mean
        return x

    def training_step(self, batch, batch_idx):
        # [Training implementation omitted for open-source release]
        # The full training step includes:
        # - Data preprocessing and normalization
        # - Conditional information processing (input/output tags, volume conditions)
        # - Conditional dropout for robustness
        # - Timestep sampling and noise addition
        # - Model forward pass with conditional inputs
        # - Loss computation
        # - Logging and EMA updates
        
        raise NotImplementedError("Training implementation not included in open-source release")

    def _get_save_dir(self):
        save_dir = osp.join(self.cfg.work_dir, f"{self.current_epoch:05d}_{self.global_step:07d}")
        mkdir_or_exist(save_dir)
        return save_dir

    def maybe_ema_validation_step(self, batch, batch_idx):
        val_loss = self.ema_val_loss if self.within_ema else self.ori_val_loss
        t_to_eval = [50, 100, 200, 400, 800]
        t = torch.tensor(t_to_eval, device=self.device, dtype=torch.int32)

        patch1 = batch["map_path1"].unsqueeze(1)
        patch2 = batch["map_path2"].unsqueeze(1)
        input_cond = batch["input_tag"]
        output_cond = batch["output_tag"]

        z_target = patch2
        z_target = self.scale_from_data(self.cfg, z_target)
        z_noise = torch.randn(z_target.shape, device=self.device)
        zt = self.noise_scheduler.add_noise(z_target, z_noise, t)
        vol_cond = patch1
        if vol_cond is not None:
            vol_cond = einops.repeat(vol_cond, 'b c d h w -> (b n) c d h w', n=len(t))
            input_cond = einops.repeat(input_cond, 'b c -> (b n c)', n=len(t))
            output_cond = einops.repeat(output_cond, 'b c -> (b n c)', n=len(t))

        generation_conds = {
            "input_cond": input_cond,
            "output_cond": output_cond,
            "vol_cond": vol_cond
        }
        noise_pred = self(zt, t, generation_conds=generation_conds)

        if self.cfg.process == "fm":
            targets = z_noise - z_target
        elif self.cfg.process == "ddpm":
            if self.cfg.ddpm.prediction_type == 'epsilon':
                targets = z_noise
            elif self.cfg.ddpm.prediction_type == 'sample':
                targets = z_target
            elif self.cfg.ddpm.prediction_type == 'v_prediction':
                targets = self.noise_scheduler.get_velocity(z_target, z_noise, t)
        if targets.size(0) != noise_pred.size(0):
            targets = einops.repeat(targets, "b c d h w -> (b n) c d h w", n=len(noise_pred))
        loss = F.mse_loss(noise_pred, targets, reduction="none").mean(dim=(1, 2, 3, 4))
        for i, ele in enumerate(t_to_eval):
            val_loss[ele].append(loss[i])

    def validation_step(self, batch, batch_idx):
        pass

    # assume the batch size equal 1
    def predict_step(self, batch, batch_idx):
        patch_size = self.cfg.patch_size
        batch_size = self.cfg.inference.batch_size
        patch_overlap = self.cfg.inference.patch_overlap
        input_map = batch["map_path1"][0].cpu().numpy()
        gt_map = batch["map_path2"][0].cpu().numpy()
        input_tag = batch["input_tag"]
        output_rag = batch["output_tag"]

        def core_inference(inputs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inputs = self.scale_from_data(self.cfg, inputs)
                input_tag_copy = einops.repeat(input_tag, "b d -> (b n d)", n=inputs.shape[0])
                output_tag_copy = einops.repeat(output_rag, "b d -> (b n d)", n=inputs.shape[0])
                conditioned_generation_conds = {
                    "input_cond": input_tag_copy.cuda(),
                    "output_cond": output_tag_copy.cuda(),
                    "vol_cond": inputs.cuda()
                }
                un_conditioned_generation_conds = {
                    "input_cond": None,
                    "output_cond": output_tag_copy.cuda(),
                    "vol_cond": None
                }

                def v_xt_t(_xt, _t):
                    cfg_weight = 2.0
                    return (1 + cfg_weight) * self(_xt, _t, generation_conds=conditioned_generation_conds) \
                        - cfg_weight * self(_xt, _t, generation_conds=un_conditioned_generation_conds)

                predicts = sample_from_fm(v_xt_t, self.noise_scheduler, method="midpoint", num_steps=50,
                                          num_samples=len(inputs), device=self.device)

                predicts = self.scale_to_data(self.cfg, predicts)
                predicts = predicts.unsqueeze(1)
            return predicts

        # input vol may be smaller than (64, 64, 64)
        pad_width = None
        if np.any(np.asarray(input_map.shape) < patch_size):
            target_shape = np.maximum(patch_size, input_map.shape)
            pad_width = get_pad_width(input_map.shape, target_shape)
            input_map = np.pad(input_map, pad_width, mode='constant', constant_values=0)

        # crop & splice patches utilities
        results = {"map_data": input_map}
        grid_patcher = GridPatches3D(patch_size=patch_size, patch_overlap=patch_overlap)
        results = grid_patcher.transform(results)
        patch_agg = GridAggregator(input_map.shape, patch_overlap)

        patches, locations = torch.from_numpy(results["patches"]), torch.from_numpy(results['locations'])

        num_patches = patches.shape[0]
        for i in range(math.ceil(num_patches / batch_size)):
            slice_fn = slice(i * batch_size, min((i + 1) * batch_size, num_patches))
            tmp_patches = patches[slice_fn].unsqueeze(1).to(self.device)

            predictions = core_inference(tmp_patches)

            patch_agg.add_batch(predictions, locations[slice_fn])

        pred_map = patch_agg.get_output_tensor().cpu().numpy()[0].astype(np.float32)

        if pad_width is not None:
            slice_fn = tuple([slice(pad_width[i][0], pred_map.shape[i] - pad_width[i][1]) for i in range(3)])
            pred_map = pred_map[slice_fn]

        fsc_vals, _ = calc_fsc(pred_map, gt_map)
        return pred_map, np.mean(fsc_vals)

    def maybe_ema_gather_validation_loss(self):
        val_loss = self.ema_val_loss if self.within_ema else self.ori_val_loss
        signature = "ema" if self.within_ema else "ori"

        log_dict = {}
        t_to_eval = [50, 100, 200, 400, 800]
        for i, t in enumerate(t_to_eval):
            t_loss = torch.tensor(val_loss[t])
            t_loss = self.all_gather(t_loss).flatten()
            idx_to_select = get_1st_unique_indices(t_loss)
            t_loss = t_loss[idx_to_select]
            log_dict[f"{signature}-t={t}"] = t_loss.mean().item()

        if self.global_rank == 0:
            log_to_current(f"Epoch {self.current_epoch} Validation " + pretty_dict(log_dict))

        val_loss.clear()

    # Run a test at the end of each validation epoch
    def on_validation_end(self):
        # Use test samples
        tmp_loader = self.trainer.val_dataloaders or self.trainer.test_dataloaders
        # Collect tag list
        tag_list = []
        for i, batch in enumerate(tmp_loader):
            if i >= 1:
                break
            tag_list.append(batch["output_tag"][0][0].cpu().numpy().item())

        # Create folder
        tmp_folder = Path("tmp") / Path(self._get_save_dir()).name
        if self.trainer.is_global_zero:
            if tmp_folder.exists():
                os.system(f"rm {tmp_folder}/*")
            else:
                tmp_folder.mkdir(exist_ok=True, parents=True)

        self.trainer.strategy.barrier()

        tag_list = torch.tensor(tag_list).to(self.device)
        # one device generate one uncondition sample！
        for use_ema in [True, False]:
            with self.ema_scope(use_ema=use_ema):
                signature = "ema" if use_ema else "ori"
                # tag_list = tag_list.clone().detach()
                uncondiction_generation_conds = {
                    "input_cond": None,
                    "output_cond": tag_list,
                    "vol_cond": None
                }

                # need to merge cryo_em1
                def _uncond_v(_zt, _t):
                    return self(_zt, _t, generation_conds=uncondiction_generation_conds)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vol_outs = sample_from_fm(
                        v_xt_t=_uncond_v, scheduler=self.noise_scheduler,
                        method="midpoint", num_steps=50, num_samples=1,
                        device=self.device
                    )
                vol_outs = self.scale_to_data(self.cfg, vol_outs)

                save_mrc(vol_outs[0].cpu().numpy().astype(np.float32),
                         f"{tmp_folder}/{signature}-sampled_{i}_uncond_tag_{tag_list[0]}_device_{self.device}.mrc")

        self.trainer.strategy.barrier()
        # save ckpt
        self.trainer.save_checkpoint(f"{tmp_folder}/ckpt.pt")
        if self.trainer.strategy.is_global_zero:
            os.system(f"mv {tmp_folder}/* {self._get_save_dir()}/")
            save_dir = self._get_save_dir()
            self.history_saved_dirs.append(save_dir)
            if hasattr(self.cfg, "keep_last_k"):
                keep_last_k = self.cfg.keep_last_k
            else:
                keep_last_k = 3
            if keep_last_k is not None and len(self.history_saved_dirs) >= keep_last_k:
                for to_remove in self.history_saved_dirs[:-keep_last_k]:
                    for p in [Path(to_remove) / "ckpt.pt", Path(to_remove) / "lora.pt"]:
                        if p.exists():
                            p.unlink()
                            log_to_current(f"delete {p} to keep last {keep_last_k} ckpts")
        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        lr = self.cfg.optimizer["lr"]
        opt = FairseqAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

        # Define a multi-stage learning rate
        def lr_lambda(step):
            warmup_steps = self.cfg.optimizer.warmup
            if step < warmup_steps:
                # Warmup phase: linear increase from 0.01 * lr to 1.0 * lr
                return 0.01 + (1.0 - 0.01) * step / warmup_steps
            else:
                # Cosine decay
                n_min = 1e-6
                T_max = self.trainer.max_steps
                # Learning rate at current step for cosine schedule
                cur_cos_lr = n_min + (self.cfg.optimizer.lr - n_min) * (
                            1 + math.cos(math.pi * (step - warmup_steps) / T_max)) / 2
                # Compute scaling factor
                scaling = cur_cos_lr / self.cfg.optimizer.lr
                return scaling

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def sample_from_fm_with_vol(
            self,
            scheduler: FMScheduler,
            num_steps: int,
            num_samples: int,
            tmp_loader: DataLoader,
            device: torch.device,
    ):
        """
            v_xt_t should be a vector field that only takes two arguments: xt and t,
            where t is in the range of (0, scheduler.num_train_timesteps).
            Not sure whether we should implement this in `scheduling_fm`.
            But it could be a place to showcase the customization of more complex solvers.
        """

        result_vout = []
        result_vinput = []

        # Default to Euler method, assume batch_size=1, then i represents the number of samples
        for i, batch in enumerate(tmp_loader):
            if i >= num_samples:
                break
            noise = torch.normal(0.0, 1.0, size=(1, 1, 64, 64, 64), device=device, dtype=torch.float32)
            scheduler = FMScheduler(scheduler.num_train_timesteps)
            T = scheduler.num_train_timesteps
            scheduler.set_timesteps(num_steps)
            zt = noise

            # Process data
            batch = self.trainer.strategy.batch_to_device(batch)
            patch1 = batch["map_path1"].unsqueeze(1)  # Add a channel dimension
            vol_cond = patch1
            input_cond = batch["input_tag"].squeeze(1)
            output_cond = batch["output_tag"].squeeze(1)
            generation_conds = {
                "input_cond": input_cond,
                "output_cond": output_cond,
                "vol_cond": vol_cond
            }

            # Closure function for generation
            def _cfg_v(_zt, _t):
                cfg_weight = 2.0
                return (1 + cfg_weight) * self(_zt, _t, generation_conds=generation_conds) \
                    - cfg_weight * self(_zt, _t, generation_conds=None)

            # Generate samples
            for t in tqdm(scheduler.timesteps):
                step_idx = scheduler._steps.index(t)
                step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]
                vt = _cfg_v(zt, einops.repeat(t - step_size, "->b", b=len(zt)).to(device))
                zt = zt - vt * step_size / T

            # Store output samples and input samples
            result_vout.append(zt)
            result_vinput.append(vol_cond.squeeze(1))

        # Post-process generated samples
        out = torch.vstack(result_vout)
        result_vinput = torch.vstack(result_vinput)
        out = out.float()
        out = einops.rearrange(out, "b 1 d h w -> b d h w")
        return out, result_vinput

    @classmethod
    def load_from_safetensors(cls, ckpt_path, cfg, map_location="cpu", strict=True):
        lit_model = cls(cfg=cfg)
        state_dict = load_safetensors(str(ckpt_path))

        # If your original state_dict contains a prefix (e.g., "model."), you can handle it here:
        # state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        lit_model.model.load_state_dict(state_dict, strict=strict)
        lit_model.model_ema = LitEma(lit_model.model, decay=0.99)

        lit_model.to(map_location)
        return lit_model

    @classmethod
    def load_from_any_checkpoint(cls, ckpt_path, cfg, map_location="cpu", strict=True):
        ckpt_path = str(ckpt_path)
        if ckpt_path.endswith(".safetensors"):
            return cls.load_from_safetensors(ckpt_path, cfg, map_location, strict)
        elif ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pt"):
            return cls.load_from_checkpoint(
                ckpt_path,
                map_location=map_location,
                cfg=cfg,
                strict=strict,
            )
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
