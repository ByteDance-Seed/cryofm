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

import numpy as np

from cryofm.core.fourier import np_real_to_rft, np_rft_to_real, rfft3_freq_indices, shell_avg


def spectral_shell_whitening(data, epsilon=1e-10):
    rft = np_real_to_rft(data)

    n0, n1, n2 = rft.shape

    spectral_index = rfft3_freq_indices(rft.shape)

    spectral_amplitude = shell_avg(np.abs(rft), spectral_index)[:n2]
    spectral_size = len(spectral_amplitude)
    spectral_index[spectral_index > spectral_size - 1] = spectral_size - 1

    power_grid = np.take((1 / (spectral_amplitude + epsilon)), spectral_index)
    # deal with DC
    dc_pos = (n0 // 2, n1 // 2, 0)
    power_grid[dc_pos] = 1

    norm_rft = power_grid * rft
    real = np_rft_to_real(norm_rft).real
    return real
