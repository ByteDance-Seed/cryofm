# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
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
#
# This file contains code modified from k-diffusion (https://github.com/crowsonkb/k-diffusion/),
# which is licensed under the MIT License.

import math

import numpy as np

import torch
import torch._dynamo
from torch import nn

from . import flags

if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(d_pos, h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(d_pos, h_pos, w_pos, indexing='ij'), dim=-1)
    d, h, w, c3 = grid.shape
    return grid.view(d * h * w, c3)


def bounding_box(d, h, w):
    # Choose the largest side, it will be -1, 1 min max
    if len({d, h, w}) == 1:
        z_min, z_max, y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
    else:
        max_val = np.max([d, h, w])
        z_min, z_max = - d / max_val, d / max_val
        y_min, y_max = - h / max_val, h / max_val
        x_min, x_max = - w / max_val, w / max_val
    return z_min, z_max, y_min, y_max, x_min, x_max


def make_axial_pos(d, h, w, align_corners=False, dtype=None, device=None):
    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box(d, h, w)
    if align_corners:
        d_pos = torch.linspace(z_min, z_max, h, dtype=dtype, device=device)
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        d_pos = centers(z_min, z_max, d, dtype=dtype, device=device)
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(d_pos, h_pos, w_pos)
