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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import einops
from diffusers import DDPMScheduler
from mmengine import mkdir_or_exist

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from safetensors.torch import load_file as load_safetensors

from cryofm.core.models.ema import LitEma
from cryofm.core.optimizers import FairseqAdam
from cryofm.core.models.unet3d import UNet3DModel
from cryofm.core.training.misc import count_params
from cryofm.core.training.lightning_utils import get_1st_unique_indices, log_to_current
from cryofm.core.utils.misc import pretty_dict
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.utils.scheduling_fm import FMScheduler
from cryofm.core.utils.sampling_fm import sample_from_fm
from cryofm.core.training.timestep_sampling import sample_timesteps

log_to_current = rank_zero_only(log_to_current)


class CryoFM2Uncond(pl.LightningModule):
    """LightningModule for CryoFM2 unconditional 3D diffusion and flow matching.

    This class implements a PyTorch LightningModule for unconditional generation
    in the CryoFM2 project, supporting both DDPM (Denoising Diffusion Probabilistic Model)
    and flow matching processes on 3D cryo-EM data using UNet3DModel architecture.
    
    The model includes EMA (Exponential Moving Average) support for stable training
    and evaluation.
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

    # resample on the every epoch begining
    def on_train_epoch_start(self):
        if isinstance(self.trainer.train_dataloader.dataset, ConcatDataset):
            for dataset in self.trainer.train_dataloader.dataset.datasets:
                dataset.set_epoch(self.trainer.current_epoch)
        else:
            self.trainer.train_dataloader.dataset.set_epoch(self.trainer.current_epoch)

    # con_flag differential condition, task flag differential task.
    def forward(self, noisy_latents, timesteps):
        # Add an extra channel for future conversion to conditional model
        noisy_latents = torch.concat([noisy_latents, torch.zeros_like(noisy_latents)], dim=1)
        return self.model(noisy_latents, timestep=timesteps).sample

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
        # - Timestep sampling and noise addition
        # - Model forward pass
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

        # add a channel dim
        patch2 = batch["map_path2"].unsqueeze(1)

        z_target = patch2
        z_target = self.scale_from_data(self.cfg, z_target)
        z_noise = torch.randn(z_target.shape, device=self.device)
        zt = self.noise_scheduler.add_noise(z_target, z_noise, t)
        noise_pred = self(zt, t)

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

        val_loss.clear()

    def on_validation_end(self):
        # tmp_loader = self.trainer.val_dataloaders or self.trainer.test_dataloaders
        if self.cfg.num_val_samples != 0:
            if self.trainer.is_global_zero:
                for use_ema in [True, False]:
                    with self.ema_scope(use_ema=use_ema):
                        signature = "ema" if use_ema else "ori"
                        tmp_folder = Path("tmp") / Path(self._get_save_dir()).name

                        if tmp_folder.exists():
                            # if the current dir is empty, will send warning.
                            os.system(f"rm {tmp_folder}/*")
                        else:
                            tmp_folder.mkdir(exist_ok=True, parents=True)

                        # need to merge cryo_em
                        def _uncond_v(_zt, _t):
                            return self(_zt, _t)

                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            vol_outs = sample_from_fm(
                                v_xt_t=_uncond_v, scheduler=self.noise_scheduler,
                                method="midpoint", num_steps=50, num_samples=self.cfg.num_val_samples,
                                device=self.device
                            )
                        vol_outs = self.scale_to_data(self.cfg, vol_outs)
                        for i in range(self.cfg.num_val_samples):
                            save_mrc(vol_outs[i].cpu().numpy().astype(np.float32),
                                     f"{tmp_folder}/{signature}-sampled_{i}_uncond.mrc")
                        os.system(f"mv {tmp_folder}/* {self._get_save_dir()}/")

        # saving checkpoint
        self.trainer.save_checkpoint(f"{self._get_save_dir()}/ckpt.pt")

        if self.trainer.is_global_zero:
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

    def configure_optimizers(self):
        lr = self.cfg.optimizer["lr"]
        opt = FairseqAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0,
                                                      total_iters=self.cfg.optimizer.warmup)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

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
