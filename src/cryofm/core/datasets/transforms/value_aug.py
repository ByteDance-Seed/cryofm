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
from typing import Union, List, Dict

import numpy as np

from mmcv.transforms import BaseTransform, TRANSFORMS


def directional_scale_arbitrary(data, direction, scale_min=0.5, scale_max=1.5, method="linear"):
    assert method in ["linear", "sin", "cos"]
    direction = np.array(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)

    D, H, W = data.shape

    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack((zz, yy, xx), axis=-1)  # (D, H, W, 3)

    projection = np.tensordot(coords, direction, axes=([-1], [0]))  # (D, H, W)

    p_min, p_max = projection.min(), projection.max()
    if method == "linear":
        scale_map = scale_min + (projection - p_min) / (p_max - p_min) * (scale_max - scale_min)
    elif method == "sin":
        normalized_p = (projection - p_min) / (p_max - p_min)  # in [0, 1]
        scale_map = scale_min + np.sin(normalized_p * np.pi) * (scale_max - scale_min)
    elif method == "cos":
        normalized_p = (projection - p_min) / (p_max - p_min)  # in [0, 1]
        scale_map = scale_min + np.cos(normalized_p * np.pi) * (scale_max - scale_min)
    else:
        raise NotImplementedError

    scaled_data = data * scale_map
    return scaled_data


def random_unit_vector():
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)
    return vec


@TRANSFORMS.register_module()
class RandomValueScale(BaseTransform):

    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 scale_min: float = 0.5,
                 scale_max: float = 1.5,
                 scale_method_list: List[str] = ("linear", "sin", "cos"),
                 ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_method_list = scale_method_list

    def transform(self, results: Dict) -> Dict:
        vec = random_unit_vector()
        method = random.choice(self.scale_method_list)

        for k in self.keys:
            results[k] = directional_scale_arbitrary(results[k],
                                                     vec,
                                                     scale_min=self.scale_min,
                                                     scale_max=self.scale_max,
                                                     method=method).astype(np.float32)
        return results
