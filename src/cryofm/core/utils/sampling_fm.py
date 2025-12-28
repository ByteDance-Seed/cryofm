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

import einops
import numpy as np
import torch

from torchdiffeq import odeint
from tqdm import tqdm

from .scheduling_fm import FMScheduler


def wrap_v_for_tde_ode_solver(v_xt_t: callable, num_train_timesteps: int):
    """
        The function wraps the model for the ODE solver.

        - The ODE solver requires a function with signature func(t, y) while 
            our model may take (y, t) as the input.
        - The velocity v=dx/dt=(x1-x0)/1, where the total time is 1.
            Therefore, the timestep (between 0 and 1000) of the model should be scaled 
            to (0, 1) for the ODE solver. (Otherwise, the step size dt will be 1, instead
            of 1/1000, in case we take 1000 steps)
    """
    def func(t, y):
        t = (num_train_timesteps * t).long().reshape(-1)
        v = v_xt_t(y, t)
        return v
    return func


# @torch.no_grad()
def sample_from_fm(
        # model: torch.nn.Module, 
        v_xt_t: callable,
        scheduler: FMScheduler, 
        method: str, 
        num_steps: int, 
        num_samples: int, 
        device: torch.device,
        side_shape: int = 64
    ):
    """
        v_xt_t should be a vector field that only takes two arguments: xt and t,
        where t is in the range of (0, scheduler.num_train_timesteps).

        Not sure whether we should implement this in `scheduling_fm`.
        But it could be a place to showcase the customization of more complex solvers.
    """
    assert method in ("rk4", "euler", "midpoint", "heun", "ralston", "midpoint_no_bar")
    noise = torch.normal(0.0, 1.0, size=(num_samples, 1, side_shape, side_shape, side_shape), device=device, dtype=torch.float32)
    scheduler = FMScheduler(scheduler.num_train_timesteps)
    T = scheduler.num_train_timesteps

    scheduler.set_timesteps(num_steps)
    zt = noise
    if method == "rk4":
        timesteps_01 = scheduler.timesteps.float().reshape(-1).to(device) / T
        for i in tqdm(range(len(timesteps_01) - 1)):
            small_ts = timesteps_01[i: i+2]
            zt = odeint(
                wrap_v_for_tde_ode_solver(v_xt_t, T), 
                zt, 
                small_ts, 
                method="rk4"
            )[-1]
    elif method == "euler":
        for t in tqdm(scheduler.timesteps):
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]
            vt = v_xt_t(zt, einops.repeat(t - step_size, "->b", b=len(zt)).to(device))
            zt = zt - vt * step_size / T
    # when muti_gpu inference, the log is heavy, so we use tqdm.disable() to disable it
    elif method == "euler_no_bar":
        for t in scheduler.timesteps:
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]
            vt = v_xt_t(zt, einops.repeat(t - step_size, "->b", b=len(zt)).to(device))
            zt = zt - vt * step_size / T
    elif method in ("midpoint", "heun", "ralston"):
        # These methods belong to rk2 methods, and the notations alpha/b1/b2/k1/k2 follows the wikipedia.
        # See: the "Second-order methods with two stages" part in the
        #   Runge-Kutta's wikipedia, https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        for t in tqdm(scheduler.timesteps):
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]

            alpha = {"midpoint": 1/2, "heun": 1, "ralston": 2/3}[method]
            b1, b2 = 1 - 1 / (2 * alpha), 1 / (2 * alpha)

            k1 = v_xt_t(zt, einops.repeat(t, "->b", b=len(zt)).to(device))
            k1_step_size = torch.tensor(int(step_size * alpha), device=device)
            zt_k1 = zt - k1 * k1_step_size / T
            k2 = v_xt_t(zt_k1, einops.repeat(t - k1_step_size, "->b", b=len(zt)).to(device))
            zt = zt - (b1 * k1 + b2 * k2) * step_size / T
    elif method in ("midpoint_no_bar", "heun_no_bar", "ralston_no_bar"):
        # These methods belong to rk2 methods, and the notations alpha/b1/b2/k1/k2 follows the wikipedia.
        # See: the "Second-order methods with two stages" part in the
        #   Runge-Kutta's wikipedia, https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        for t in scheduler.timesteps:
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]

            alpha = {"midpoint_no_bar": 1/2, "heun": 1, "ralston": 2/3}[method]
            b1, b2 = 1 - 1 / (2 * alpha), 1 / (2 * alpha)

            k1 = v_xt_t(zt, einops.repeat(t, "->b", b=len(zt)).to(device))
            k1_step_size = torch.tensor(int(step_size * alpha), device=device)
            zt_k1 = zt - k1 * k1_step_size / T
            k2 = v_xt_t(zt_k1, einops.repeat(t - k1_step_size, "->b", b=len(zt)).to(device))
            zt = zt - (b1 * k1 + b2 * k2) * step_size / T
    else:
        raise NotImplementedError
    out = zt

    out = out.float()
    out = einops.rearrange(out, "b 1 d h w -> b d h w")
    return out

@torch.no_grad()
def sample_from_fm_with(
        # model: torch.nn.Module, 
        v_xt_t: callable,
        scheduler: FMScheduler, 
        method: str, 
        num_steps: int, 
        num_samples: int, 
        device: torch.device,
    ):
    """
        v_xt_t should be a vector field that only takes two arguments: xt and t,
        where t is in the range of (0, scheduler.num_train_timesteps).

        Not sure whether we should implement this in `scheduling_fm`.
        But it could be a place to showcase the customization of more complex solvers.
    """
    assert method in ("rk4", "euler", "midpoint", "heun", "ralston")
    noise = torch.normal(0.0, 1.0, size=(num_samples, 1, 64, 64, 64), device=device, dtype=torch.float32)
    scheduler = FMScheduler(scheduler.num_train_timesteps)
    T = scheduler.num_train_timesteps

    scheduler.set_timesteps(num_steps)
    zt = noise
    if method == "rk4":
        timesteps_01 = scheduler.timesteps.float().reshape(-1).to(device) / T
        for i in tqdm(range(len(timesteps_01) - 1)):
            small_ts = timesteps_01[i: i+2]
            zt = odeint(
                wrap_v_for_tde_ode_solver(v_xt_t, T), 
                zt, 
                small_ts, 
                method="rk4"
            )[-1]
    elif method == "euler":
        for t in tqdm(scheduler.timesteps):
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]
            shape = einops.repeat(t - step_size, "->b", b=len(zt)).shape
            vt = v_xt_t(zt, einops.repeat(t - step_size, "->b", b=len(zt)).to(device))
            zt = zt - vt * step_size / T
    elif method in ("midpoint", "heun", "ralston"):
        # These methods belong to rk2 methods, and the notations alpha/b1/b2/k1/k2 follows the wikipedia.
        # See: the "Second-order methods with two stages" part in the
        #   Runge-Kutta's wikipedia, https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        for t in tqdm(scheduler.timesteps):
            step_idx = scheduler._steps.index(t)
            step_size = scheduler._steps[step_idx] - scheduler._steps[step_idx + 1]

            alpha = {"midpoint": 1/2, "heun": 1, "ralston": 2/3}[method]
            b1, b2 = 1 - 1 / (2 * alpha), 1 / (2 * alpha)

            k1 = v_xt_t(zt, einops.repeat(t, "->b", b=len(zt)).to(device))
            k1_step_size = torch.tensor(int(step_size * alpha), device=device)
            zt_k1 = zt - k1 * k1_step_size / T
            k2 = v_xt_t(zt_k1, einops.repeat(t - k1_step_size, "->b", b=len(zt)).to(device))
            zt = zt - (b1 * k1 + b2 * k2) * step_size / T
    else:
        raise NotImplementedError
    out = zt

    out = out.float()
    out = einops.rearrange(out, "b 1 d h w -> b d h w")
    return out


