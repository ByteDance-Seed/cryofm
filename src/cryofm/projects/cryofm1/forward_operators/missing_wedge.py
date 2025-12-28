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

from cryofm.core.utils.mask import create_cryoet_wedge_mask

from .base import FilterOperator


class MissingWedgeOperator(FilterOperator):
    """Missing wedge forward operator.
    """
    def __init__(self, device, shape, tilt_angle, plane_axis_ids=(0, 1), rot_axis_id=0):
        f_mask = create_cryoet_wedge_mask(
            shape[-1],
            tilt_angle,
            plane_axis_ids=plane_axis_ids,
            rot_axis_id=rot_axis_id
        )

        super().__init__(device, shape, torch.from_numpy(f_mask).to(device))

        self.tilt_angle = tilt_angle
