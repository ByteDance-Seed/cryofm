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

from typing import Dict, Any, Union, Optional

import torch
import torch.nn as nn

from cryofm.core.fourier import hartley_to_primal_3d, primal_to_hartley_3d


class BaseOperator(nn.Module):
    """Base forward operator"""

    def __init__(self, device: torch.device, shape: tuple):
        super().__init__()
        self.device = device
        self.shape = shape


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NoiseOperator(BaseOperator):
    def __init__(self,
                 device: torch.device,
                 shape: tuple,
                 noise_power: Optional[Union[float, torch.Tensor]] = None,
                 seed: Optional[int] = None):
        super().__init__(device, shape)
        self.noise_power = torch.zeros(shape).to(device) + 1e-9
        if noise_power is not None:
            if isinstance(noise_power, float):
                self.noise_power += noise_power
            else:
                self.noise_power += noise_power.to(device)

        self.seed = seed
        self.g = torch.Generator(device)
        if seed is not None:
            self.g.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_x = primal_to_hartley_3d(x)
        noise = torch.normal(0, torch.sqrt(self.noise_power), generator=self.g)
        h_x_noisy = noise + h_x
        return hartley_to_primal_3d(h_x_noisy)
        
    @property
    def noise_properties(self) -> Dict[str, torch.Tensor]:
        return {
            "noise_power": self.noise_power
        }


class FilterOperator(BaseOperator):
    def __init__(self, device: torch.device, shape: tuple, f_mask: torch.Tensor):
        super().__init__(device, shape)
        self.f_mask = f_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_x = primal_to_hartley_3d(x)
        h_x_filtered = h_x * self.f_mask
        return hartley_to_primal_3d(h_x_filtered)
    
    @property
    def mask_properties(self) -> Dict[str, torch.Tensor]:
        return {
            "f_mask": self.f_mask
        }


class CompositeOperator(BaseOperator):
    """Composite operator combining multiple degradation operators"""

    def __init__(self, operators: list):
        super().__init__(operators[0].device, operators[0].shape)
        self.operators = nn.ModuleList(operators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.operators:
            x = op(x)
        return x

    @property
    def noise_properties(self) -> Union[Dict[str, torch.Tensor], None]:
        """Combine noise properties from all noise operators"""
        noise_ops = [op for op in self.operators if isinstance(op, NoiseOperator)]
        if not noise_ops:
            return None

        combined_noise_power = noise_ops[0].noise_power.clone()
        for op in noise_ops[1:]:
            combined_noise_power += op.noise_power

        return {"noise_power": combined_noise_power}

    @property
    def mask_properties(self) -> Union[Dict[str, torch.Tensor], None]:
        """Combine mask properties from all filter operators"""
        filter_ops = [op for op in self.operators if isinstance(op, FilterOperator)]
        if not filter_ops:
            return None

        combined_f_mask = filter_ops[0].f_mask.clone()
        for op in filter_ops[1:]:
            combined_f_mask *= op.f_mask

        return {"f_mask": combined_f_mask}
