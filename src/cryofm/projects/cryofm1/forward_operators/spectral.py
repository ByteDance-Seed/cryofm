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

from typing import Tuple, Optional, Union

import torch

from cryofm.core.utils.mask import create_cryoet_wedge_mask

from .base import NoiseOperator
from .noise import radial_average_3d_power_spectrum, noise_power_from_radial


class SpectralNoiseOperator(NoiseOperator):
    """Spectral noise degradation operator"""

    def __init__(
            self,
            device: torch.device,
            shape: tuple,
            noise_power: Optional[Union[float, torch.Tensor]] = None,
            seed: Optional[int] = None
    ):
        super().__init__(device, shape, noise_power=noise_power, seed=seed)

    # calc spectral noise power
    def setup_from_snr(self, signal: torch.Tensor, snr: torch.Tensor):
        radial_signal = radial_average_3d_power_spectrum(signal)
        radial_noise = radial_signal / snr
        noise_power = noise_power_from_radial(signal.shape[-1], radial_noise)
        self.noise_power += noise_power.to(self.device)


class AnisotropicSpectralNoiseOperator(SpectralNoiseOperator):
    """Anisotropic spectral noise degradation operator"""

    def __init__(
            self,
            device: torch.device,
            shape: tuple,
            noise_power: Optional[float] = None,
            amp_matrix: torch.Tensor = None,
            seed: Optional[int] = None
    ):
        super().__init__(device, shape, noise_power=noise_power, seed=seed)

        if amp_matrix is None:
            self.amp_matrix = torch.ones(shape).to(device)
        else:
            self.amp_matrix = amp_matrix

    def setup_cryoet_anisotropic(self, tilt_angle: int, amp_factor: float):
        f_mask = create_cryoet_wedge_mask(self.shape[-1], tilt_angle, plane_axis_ids=[0, 1], rot_axis_id=0)
        f_mask = torch.from_numpy(f_mask).to(self.device)
        self.amp_matrix = torch.where(f_mask, self.amp_matrix, self.amp_matrix * amp_factor)
        self.noise_power = torch.where(f_mask, self.noise_power, self.noise_power * amp_factor)
