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
from typing import Union, Tuple, List, Dict, Optional

import numpy as np
import scipy.ndimage as ndi

from mmcv.transforms import BaseTransform, TRANSFORMS


def sample_uniform_sextet(params):
    results = []
    for a, b in zip(params[::2], params[1::2]):
        results.append(np.random.uniform(a, b))
    return results


@TRANSFORMS.register_module()
class RandomBlur3D(BaseTransform):

    def __init__(self, keys: List[str], p: float = 1.0, std: Tuple[float] = (0, 2)):
        if isinstance(keys, str):
            keys = [keys]

        if len(std) == 2:
            std = std * 3
        elif len(std) == 6:
            pass
        else:
            raise ValueError('std must be either a tuple of length 2 or 6')

        self.keys = keys
        self.p = p
        self.std = std

    def transform(self, results: Dict) -> Dict:
        if random.random() >= self.p:
            return results

        for k in self.keys:
            std = sample_uniform_sextet(self.std)
            results[k] = ndi.gaussian_filter(results[k], std)
        return results
