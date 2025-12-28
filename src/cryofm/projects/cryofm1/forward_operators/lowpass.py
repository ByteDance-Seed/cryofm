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

import torch

from cryofm.core.utils.mask import create_sphere_mask

from .base import FilterOperator


class LowpassOperator(FilterOperator):
    """Low pass filter operator."""

    def __init__(self, device: torch.device, shape: tuple, cutoff_diameter: int ):
        f_mask = create_sphere_mask(shape[0], shape[1], shape[2], radius=cutoff_diameter // 2)

        super().__init__(device, shape, torch.from_numpy(f_mask))

        self.cutoff_diameter = cutoff_diameter
