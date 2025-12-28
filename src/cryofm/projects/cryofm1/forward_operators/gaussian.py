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

from typing import Optional, Union

import torch

from .base import NoiseOperator


class GaussianNoiseOperator(NoiseOperator):
    """Gaussian noise forward operator"""

    def __init__(
            self,
            device: torch.device,
            shape: tuple,
            noise_power: Union[float, torch.Tensor],
            seed: Optional[int] = None
    ):
        super().__init__(device, shape, noise_power=noise_power, seed=seed)
