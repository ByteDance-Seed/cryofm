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

from typing import Dict, List, Tuple, Union
from copy import deepcopy

import mrcfile
import numpy as np

from mmcv.transforms import BaseTransform, TRANSFORMS


def mrc_get_data(file_path, to_float32=False):
    with mrcfile.open(file_path) as m:
        data = m.data.copy()
    if to_float32:
        data = data.astype(np.float32)
    return data


# load mrc & normalize
# path from path_keys, load data and save to map_keys
@TRANSFORMS.register_module()
class LoadMapFromFile(BaseTransform):
    def __init__(self, path_keys: Union[str, List[str]] = "map_path", map_keys: Union[str, List[str]] = "map_data"):
        if isinstance(path_keys, str):
            path_keys = [path_keys]
        if isinstance(map_keys, str):
            map_keys = [map_keys]

        self.path_keys = path_keys
        self.map_keys = map_keys

    def transform(self, results: dict) -> dict:
        for p_key, m_key in zip(self.path_keys, self.map_keys):
            map_path = results[p_key]
            with mrcfile.open(map_path) as m:
                map_data = m.data.copy()

            results[m_key] = map_data.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class CopyData(BaseTransform):

    def __init__(self, src_key, dst_key):
        self.src_key = src_key
        self.dst_key = dst_key

    def transform(self, results: dict) -> dict:
        if isinstance([self.dst_key], np.ndarray):
            results[self.dst_key] = results[self.src_key].copy()
        elif isinstance([self.dst_key], (list, tuple, dict)):
            results[self.dst_key] = deepcopy(results[self.src_key])
        else:
            results[self.dst_key] = results[self.src_key]
        return results


@TRANSFORMS.register_module()
class LoadExtraMapFromFile(BaseTransform):

    def __init__(self, path_exchange: List, extra_map_key="cond_map"):
        self.path_exchange = path_exchange
        self.extra_map_key = extra_map_key

    def transform(self, results: dict) -> dict:
        map_path = results["map_path"]
        extra_map_path = map_path.replace(self.path_exchange[0], self.path_exchange[1])
        with mrcfile.open(extra_map_path) as m:
            extra_map_data = m.data.copy()

        results[self.extra_map_key] = extra_map_data
        return results
