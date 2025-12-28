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

# This file is adapted from the CryoSTAR project:
# https://github.com/bytedance/cryostar
# Original code is licensed under the Apache License, Version 2.0.
# Copyright (c) 2023 ByteDance.
import torch

from cryofm.core.fourier import primal_to_fourier_3d, fourier_to_primal_3d, primal_to_fourier_2d, fourier_to_primal_2d
from cryofm.core.utils.mask import create_circular_mask, create_sphere_mask


def downsample_3d(r: torch.Tensor, down_side: int) -> torch.Tensor:
    f_density = primal_to_fourier_3d(r)
    D = f_density.shape[0]

    # even: D = 8, DC_loc = 4
    # - - - - 0 + + +
    # odd: D = 7, DC_loc = 3
    # - - - 0 + + +
    DC_loc = D // 2

    # even: L = 8, [DC-4, DC+4)
    # - - - - 0 + + +
    # odd:  L = 7, [DC-3, DC+4)
    # - - - 0 + + +
    DC_left = down_side // 2
    DC_right = down_side - DC_left

    s = DC_loc - DC_left
    e = DC_loc + DC_right
    f_density_down = f_density[s:e, s:e, s:e]
    mask = torch.tensor(create_sphere_mask(down_side, down_side, down_side, radius=down_side // 2))
    f_density_down[~mask] = 0
    density_down = fourier_to_primal_3d(f_density_down).real
    return density_down


def downsample_2d(r: torch.Tensor, down_side: int) -> torch.Tensor:
    f_image = primal_to_fourier_2d(r)
    D = f_image.shape[0]
    DC_loc = D // 2
    DC_left = down_side // 2
    DC_right = down_side - DC_left

    s = DC_loc - DC_left
    e = DC_loc + DC_right
    f_image_down = f_image[s:e, s:e]
    mask = torch.tensor(create_circular_mask(down_side, down_side, radius=down_side // 2))
    f_image_down[~mask] = 0
    image_down = fourier_to_primal_2d(f_image_down).real
    return image_down
