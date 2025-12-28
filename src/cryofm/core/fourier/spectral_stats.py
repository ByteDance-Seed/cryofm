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
from scipy import ndimage

from cryofm.core.fourier import fft3_freq_indices, rfft3_freq_indices


def shell_avg(arr: np.ndarray, indices: np.ndarray = None):
    if indices is None:
        indices = fft3_freq_indices(arr.shape)

    max_radii = np.amin(arr.shape)
    shell_radii = np.arange(0, max_radii)
    labels = np.searchsorted(shell_radii, indices, side="left")
    return ndimage.mean(arr, labels=labels, index=shell_radii)


def rft_shell_avg(arr: np.ndarray, indices: np.ndarray = None):
    if indices is None:
        indices = rfft3_freq_indices(arr.shape)

    return shell_avg(arr, indices)
