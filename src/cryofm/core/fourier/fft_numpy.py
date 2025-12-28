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

"""
Convert data between real, Fourier (ft), Hartley (ht) space.
Note:
    - We use the default normalization mode "backward" used by numpy and PyTorch, i.e. normalization only
    happens in backward direction, e.g. ifft.
    - For ht <-> ft, no normalization happens.
    - If we convert data between ft & ht, we must make sure that DC term is at center.
    - All transformation functions within this module (real <-> ft <-> ht) are internally self-consistent w.r.t. fftshift/ifftshift.
    - When using the output from these functions in other modules/codes, you must carefully consider whether the DC (zero-frequency) 
      component is centered or not (i.e., if further fftshift or ifftshift is needed).
"""
import inspect
from typing import Callable, Tuple

import numpy as np


def np_real_to_ft(real: np.ndarray) -> np.ndarray:
    """
    Convert real space data to complex Fourier space using numpy FFT.

    Parameters
    ----------
    real : np.ndarray
        Real-space input array.

    Returns
    -------
    np.ndarray
        Complex array in Fourier space (same shape as input, DC at the center).
    """
    real = np.fft.ifftshift(real)
    ft = np.fft.fftshift(np.fft.fftn(real))
    return ft


def np_ft_to_real(ft: np.ndarray) -> np.ndarray:
    """Convert Fourier space to real space."""
    ft = np.fft.ifftshift(ft)
    real = np.fft.fftshift(np.fft.ifftn(ft))
    return real


def np_real_to_rft(real: np.ndarray) -> np.ndarray:
    real = np.fft.ifftshift(real)
    rft = np.fft.fftshift(np.fft.rfftn(real), axes=list(range(real.ndim))[:-1])
    return rft


def np_rft_to_real(rft: np.ndarray, s: Tuple[int] = None) -> np.ndarray:
    rft = np.fft.ifftshift(rft, axes=list(range(rft.ndim))[:-1])
    real = np.fft.fftshift(np.fft.irfftn(rft, s=s))
    return real


def np_real_to_ht(real: np.ndarray) -> np.ndarray:
    ht = np_real_to_ft(real)
    return np.real(ht) - np.imag(ht)


def np_ht_to_real(ht: np.ndarray) -> np.ndarray:
    r = np_real_to_ft(ht)
    r = r / r.size
    return np.real(r) - np.imag(r)


def np_ft_to_ht(ft: np.ndarray) -> np.ndarray:
    return np.real(ft) - np.imag(ft)


def np_ht_to_ft(ht: np.ndarray) -> np.ndarray:
    # H(-w) = flip(H(w)) only if DC term is shifted to center
    pad_width = tuple((0, 0) if s % 2 else (0, 1) for s in ht.shape)
    ht_aug = np.pad(ht, pad_width=pad_width, mode='wrap')
    ht_aug_neg = np.flip(ht_aug)
    ft = (ht_aug + ht_aug_neg) * 0.5 - (ht_aug - ht_aug_neg) * 0.5 * 1j
    return ft[tuple(slice(0, s) for s in ht.shape)]


# tests
def _cycle_test(function: Callable, data_function: Callable = np.random.rand):
    def _item(*args):
        nda = data_function(*args)
        sig = inspect.signature(function)
        if 's' in sig.parameters:
            assert np.allclose(nda, function(nda, s=args)), (f"{function.__name__} does not pass cycle test "
                                                             f"for array shape {args}.")
        else:
            assert np.allclose(nda, function(nda)), (f"{function.__name__} does not pass cycle test "
                                                     f"for array shape {args}.")
    # 2d
    _item(6, 6)
    _item(7, 7)
    _item(7, 8)
    # 3d
    _item(6, 6, 6)
    _item(7, 7, 7)
    _item(6, 7, 8)
    _item(7, 8, 9)


def _rand_ft_nda(*args):
    real = np.fft.ifftshift(np.random.rand(*args))
    return np.fft.fftshift(np.fft.fftn(real))


def _rand_rft_nda(*args):
    real = np.fft.ifftshift(np.random.rand(*args))
    return np.fft.fftshift(np.fft.rfftn(real), axes=list(range(real.ndim))[:-1])


def test_np_ft_to_real():
    _cycle_test(lambda x: np.real_if_close(np_ft_to_real(np_real_to_ft(x))))


def test_np_real_to_ft():
    _cycle_test(lambda x: np_real_to_ft(np.real_if_close(np_ft_to_real(x))),
                data_function=_rand_ft_nda)


# def test_np_real_to_rft():
#     _cycle_test(lambda x, s: np_real_to_rft(np.real_if_close(np_rft_to_real(x, s=s))),
#                 data_function=_rand_rft_nda)


def test_np_rft_to_real():
    _cycle_test(lambda x, s: np.real_if_close(np_rft_to_real(np_real_to_rft(x), s=s)))


def test_np_real_to_ht():
    # ht -> real
    _cycle_test(lambda x: np_real_to_ht(np_ht_to_real(x)))
    # ht -> ft -> real
    _cycle_test(lambda x: np_real_to_ht(np.real_if_close(np_ft_to_real(np_ht_to_ft(x)))))


def test_np_ht_to_real():
    # real -> ht
    _cycle_test(lambda x: np_ht_to_real(np_real_to_ht(x)))
    # real -> ft -> ht
    _cycle_test(lambda x: np.real_if_close(np_ht_to_real(np_ft_to_ht(np_real_to_ft(x)))))


def test_np_ft_to_ht():
    _cycle_test(lambda x: np_ft_to_ht(np_ht_to_ft(x)))


def test_np_ht_to_ft():
    _cycle_test(lambda x: np_ht_to_ft(np_ft_to_ht(x)),
                data_function=_rand_ft_nda)


def test_np_real_to_rft():
    def _test_with_shape(shape):
        tmp = np.random.rand(*shape)
        ft = np_real_to_ft(tmp)
        rft = np_real_to_rft(tmp)

        last_dim = tmp.shape[-1]
        if last_dim % 2 == 0:
            assert np.allclose(ft[..., -last_dim // 2:], rft[..., :-1])
            assert np.allclose(ft[..., 0], rft[..., -1])
        else:
            assert np.allclose(ft[..., -last_dim // 2:], rft)

    _test_with_shape((10, 12, 10))
    _test_with_shape((10, 12, 11))
