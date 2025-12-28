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

from typing import List, Optional

import numpy as np

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


class FMScheduler:
    """
        Designed following the API of diffusers's DDPMScheduler.

        **An important ramark**:

        When training flow matching models in `train_ddpm3d.py`, we are using timesteps 
        of (1, 2, ..., 1000) instead of (0, 1, ..., 999).


        In DDPMScheduler, 
            scheduler.timesteps is (0, 1, 2, ..., 999),
            scheduler.step(t) will get x{t} given x{t+1}
            scheduler.add_noise(t) will produce x{t+1} for DDPM to predict x{t}

        check it by:
            ```
                scheduler = DDPMScheduler()
                sample = torch.rand(1, 1)
                noise_0 = scheduler.add_noise(sample, torch.rand(1, 1), torch.tensor(0, dtype=torch.long))
                assert sample.item() != noise_0.item()
                # below code throws exception
                # noise_1000 = scheduler.add_noise(sample, torch.rand(1, 1), torch.tensor(1000, dtype=torch.long)) 
            ```

        This can be a bug (mathematically) in diffusers, checkout:
        - https://github.com/huggingface/diffusers/issues/2585
        - Common Diffusion Noise Schedules and Sample Steps are Flawed

        check it by:
            ```
                scheduler = DDIMScheduler()
                scheduler.set_timesteps(10)
                scheduler.timesteps

                # The result is tensor([900, 800, 700, 600, 500, 400, 300, 200, 100,   0])
                # I think we expect it to start with 1000 and end with 100?
            ```
    """

    def __init__(self, num_train_timesteps=1000, prediction_type='v_prediction'):
        assert prediction_type in ["v_prediction", "x0", ]
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

    def set_timesteps(self, num_inference_steps: Optional[int] = None, device=None, timesteps: Optional[List[int]] = None,):
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or custom `timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("custom `timesteps` must be in descending order.")

            if timesteps[0] > self.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.num_train_timesteps}."
                )

            _steps = torch.from_numpy(np.array(timesteps, dtype=np.int64))
        elif num_inference_steps is not None:
            if num_inference_steps > self.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.num_train_timesteps} timesteps."
                )

            _steps = torch.from_numpy(np.linspace(self.num_train_timesteps, 0, num_inference_steps + 1, dtype=np.int64))
        else:
            raise ValueError("`num_inference_steps` or `timesteps` must be provided.")

        if device is not None:
            _steps = _steps.to(device)
        self.timesteps = _steps[:-1]
        self._steps = _steps.tolist()

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
    ) -> DDPMSchedulerOutput:
        assert timestep > 0
        step_idx = self._steps.index(timestep)
        step_size = self._steps[step_idx] - self._steps[step_idx + 1]

        if self.prediction_type == 'v_prediction':
            prev_sample = sample - step_size / self.num_train_timesteps * model_output
            pred_original_sample = sample - timestep / self.num_train_timesteps * model_output
            ret = DDPMSchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample
            )
        elif self.prediction_type == 'x0':
            pred_original_sample = model_output
            prev_sample = (pred_original_sample * (step_size / self.num_train_timesteps) +
                           sample * ((timestep - step_size) / self.num_train_timesteps))
            return DDPMSchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample
            )
        else:
            raise ValueError("prediction_type must be 'v_prediction' or 'x0'")
        return ret

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        t_ = timesteps[:, None, None, None, None]
        noisy_samples = noise * t_ / self.num_train_timesteps + original_samples * (1 - t_ / self.num_train_timesteps)
        return noisy_samples
