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

import random
from typing import Union, Tuple, List, Dict

import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS


@TRANSFORMS.register_module()
class RandomNoise3D(BaseTransform):

    def __init__(self, keys: List[str], p: float = 0.5, mean: float = 0.0, std: Tuple[float, float] = (0, 0.1)):
        if isinstance(keys, str):
            keys = [keys]

        self.keys = keys
        self.p = p
        self.mean = mean
        self.std = std

    def transform(self, results: Dict) -> Dict:
        if random.random() >= self.p:
            return results

        for k in self.keys:
            std = random.uniform(self.std[0], self.std[1])
            results[k] = results[k] + std * np.random.randn(*results[k].shape).astype(np.float32) + self.mean
        return results
