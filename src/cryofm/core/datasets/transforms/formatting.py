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

from typing import List, Tuple

from mmcv.transforms import to_tensor
from mmcv.transforms import BaseTransform, TRANSFORMS


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):

    def __init__(self, keys: Tuple[List[str], str] = None, ) -> None:

        assert keys is not None, 'keys in PackInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]

        # print(f"{self.keys}")

    def transform(self, results: dict) -> dict:
        out = {}
        for k in self.keys:
            value = results.get(k, None)
            if value is not None:
                out[k] = to_tensor(value)
        return out

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str
