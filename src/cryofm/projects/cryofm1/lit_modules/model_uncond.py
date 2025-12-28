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

import os.path as osp
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import einops

from diffusers import DDPMScheduler

from mmengine import mkdir_or_exist, Config

from safetensors.torch import load_file as load_safetensors

import lightning.pytorch as pl

from cryofm.core.models.ema import LitEma
from cryofm.core.models.hdit import make_model
from cryofm.core.optimizers import FairseqAdam
from cryofm.core.utils.misc import pretty_dict
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.training.misc import count_params
from cryofm.core.utils.scheduling_fm import FMScheduler
from cryofm.core.training.lightning_utils import get_1st_unique_indices, log_to_current


class CryoFM1(pl.LightningModule):
    """LightningModule for CryoFM 3D diffusion and flow matching.

    This class implements a PyTorch LightningModule for the CryoFM project,
    supporting both traditional DDPM (Denoising Diffusion Probabilistic Model)
    and flow matching processes on 3D data.
    """

    def __init__(
        self,
        cfg,
        ignore_keys=None,
        input_key="patches",
    ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []

        self.cfg = cfg
        self.input_key = input_key
        # check some info here
        log_to_current(f"Use z_scale mean {self.cfg.z_scale.mean} std {self.cfg.z_scale.std}")

        if self.cfg.model_type == "hdit":
            assert self.cfg.hdit_model.type == "image_transformer_v2", f"Now only supports image_transformer_v2"
            log_to_current(f"\n~~~~~~~~\nUse model type HDiT\n~~~~~~~~")
            self.model = make_model(Config(cfg.hdit_model))
        else:
            raise NotImplementedError(f"Only supports model type 'hdit', got {self.cfg.model_type}")

        if cfg.process == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                clip_sample=False,
                prediction_type=cfg.ddpm.prediction_type
            )
        elif cfg.process == "fm":
            self.noise_scheduler = FMScheduler(1000)

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

    def forward(self, noisy_latents, timesteps):
        return self.model(noisy_latents, timesteps)

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
        x = batch["map_data"][None, ...]
        z0 = x
        z0 = (z0 - self.cfg.z_scale.mean) / self.cfg.z_scale.std
        z1 = torch.randn(z0.shape, device=self.device)
        zt = self.noise_scheduler.add_noise(z0, z1, t)
        noise_pred = self(zt, t)

        if self.cfg.process == "fm":
            targets = z1 - z0
        elif self.cfg.process == "ddpm":
            if self.cfg.ddpm.prediction_type == 'epsilon':
                targets = z1
            elif self.cfg.ddpm.prediction_type == 'sample':
                targets = z0
            elif self.cfg.ddpm.prediction_type == 'v_prediction':
                targets = self.noise_scheduler.get_velocity(z0, z1, t)
        
        loss = F.mse_loss(noise_pred, targets, reduction="none").mean(dim=(1, 2, 3, 4))
        for i, ele in enumerate(t_to_eval):
            val_loss[ele].append(loss[i])

    def validation_step(self, batch, batch_idx):
        for use_ema in [True, False]:
            with self.ema_scope(use_ema):
                self.maybe_ema_validation_step(batch, batch_idx)

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
            wandb.log(log_dict, step=self.global_step)

        val_loss.clear()

    def maybe_ema_sample(self):
        z_size = self.cfg.patch_size
        with torch.no_grad():
            zt = torch.randn((1, 1, z_size, z_size, z_size), device=self.device)
            self.noise_scheduler.set_timesteps(1000)
            for t in tqdm(self.noise_scheduler.timesteps):
                model_output = self(zt, einops.repeat(t, "->b", b=len(zt)).to(self.device))
                zt = self.noise_scheduler.step(model_output, t, zt).prev_sample
            vol = zt * self.cfg.z_scale.std + self.cfg.z_scale.mean
        return vol

    def maybe_ema_sample_fast(self, num_samples):
        scheduler = FMScheduler(1000)
        zt = torch.normal(0.0, 1.0, size=(num_samples, 1, 64, 64, 64), device=self.device, dtype=torch.float32)
        T = scheduler.num_train_timesteps
        scheduler.set_timesteps(50)
        for t in tqdm(scheduler.timesteps):
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]
            tt = einops.repeat(t, "->b", b=len(zt)).to(self.device)
            k1 = self(zt, tt)
            k2 = self(zt - k1 * step_size // 2 / T, tt - step_size // 2)
            zt = zt - k2 * step_size / T
        vol = zt * self.cfg.z_scale.std + self.cfg.z_scale.mean
        return vol
            
    def on_validation_end(self):
        # validation
        for use_ema in [True, False]:
            with self.ema_scope(use_ema=use_ema):
                self.maybe_ema_gather_validation_loss()

        # unconditional generation
        if self.cfg.num_val_samples != 0:
            if self.trainer.is_global_zero:
                for use_ema in [True, False]:
                    with self.ema_scope(use_ema=use_ema):
                        signature = "ema" if use_ema else "ori"
                        # for i in range(self.cfg.num_val_samples):
                        #     vol = self.maybe_ema_sample()
                        #     save_mrc(vol[0, 0].cpu().numpy().astype(np.float32), f"{self._get_save_dir()}/{signature}-sampled_{i}.mrc")
                        vols = self.maybe_ema_sample_fast(self.cfg.num_val_samples)
                        for i in range(self.cfg.num_val_samples):
                            save_mrc(vols[i, 0].cpu().numpy().astype(np.float32), f"{self._get_save_dir()}/{signature}-sampled_{i}.mrc")

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
                    p = Path(to_remove) / "ckpt.pt"
                    if p.exists():
                        p.unlink()
                        log_to_current(f"delete {p} to keep last {keep_last_k} ckpts")

    def configure_optimizers(self):
        lr = self.cfg.optimizer["lr"]
        # opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.98))
        # return opt
        opt = FairseqAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=self.cfg.optimizer.warmup)
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
