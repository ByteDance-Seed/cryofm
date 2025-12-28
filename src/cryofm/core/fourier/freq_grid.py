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

from typing import Union
from collections.abc import Iterable

import numpy as np


def line_freq(size: int):
    return np.fft.fftshift(np.fft.fftfreq(size))


def rfft2_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    k0 = line_freq(n0)
    k1 = np.fft.rfftfreq(n1)

    radii = np.sqrt(k0.reshape((-1, 1, 1)) ** 2 +
                    k1.reshape((1, -1, 1)) ** 2)
    return k0, k1, radii


def rfft3_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size

    k0 = line_freq(n0)
    k1 = line_freq(n1)
    k2 = np.fft.rfftfreq(n2)

    radii = np.sqrt(k0.reshape((-1, 1, 1)) ** 2 +
                    k1.reshape((1, -1, 1)) ** 2 +
                    k2.reshape((1, 1, -1)) ** 2)
    return k0, k1, k2, radii


def fft2_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    k0 = line_freq(n0)
    k1 = line_freq(n1)

    radii = np.sqrt(k0.reshape((-1, 1)) ** 2 +
                    k1.reshape((1, -1)) ** 2)
    return k0, k1, radii


def fft3_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size

    k0 = line_freq(n0)
    k1 = line_freq(n1)
    k2 = line_freq(n2)

    radii = np.sqrt(k0.reshape((-1, 1, 1)) ** 2 +
                    k1.reshape((1, -1, 1)) ** 2 +
                    k2.reshape((1, 1, -1)) ** 2)
    return k0, k1, k2, radii


def fft1_freq_indices(size: int):
    return np.arange(- (size // 2), (size - 1) // 2 + 1)


def fft2_freq_indices(size: Union[int, Iterable[int]]) -> np.ndarray:
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    indices = np.round(np.sqrt(fft1_freq_indices(n0).reshape((-1, 1)) ** 2 +
                               fft1_freq_indices(n1).reshape((1, -1)) ** 2))
    return indices.astype(int)


def fft3_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size
    indices = np.round(np.sqrt(fft1_freq_indices(n0).reshape((-1, 1, 1)) ** 2 +
                               fft1_freq_indices(n1).reshape((1, -1, 1)) ** 2 +
                               fft1_freq_indices(n2).reshape((1, 1, -1)) ** 2))
    return indices.astype(int)


def rfft2_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    indices = np.round(np.sqrt(fft1_freq_indices(n0).reshape((-1, 1)) ** 2 +
                               np.arange(0, n1).reshape((1, -1)) ** 2))
    return indices.astype(int)


def rfft3_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size
    indices = np.round(np.sqrt(fft1_freq_indices(n0).reshape((-1, 1, 1)) ** 2 +
                               fft1_freq_indices(n1).reshape((1, -1, 1)) ** 2 +
                               np.arange(0, n2).reshape((1, 1, -1)) ** 2))
    return indices.astype(int)
