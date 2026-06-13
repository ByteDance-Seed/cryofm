"""
NumPy-based implementations for conversions between real, Fourier (ft), and Hartley (ht) space.
Note:
    - We use the default normalization mode "backward" used by numpy, i.e., normalization only
      happens in the backward direction, e.g., ifft.
    - For ht <-> ft, no normalization happens.
    - If we convert data between ft & ht, we must make sure that the DC term is at the center.
"""
import inspect
from typing import Callable, Tuple, Union
from collections.abc import Iterable
from scipy import ndimage
import numpy as np


def np_real_to_rft(real: np.ndarray):
    real = np.fft.ifftshift(real)
    ft = np.fft.fftshift(np.fft.rfftn(real), axes=list(range(real.ndim))[:-1])
    return ft


def np_real_to_ft(real: np.ndarray) -> np.ndarray:
    real = np.fft.ifftshift(real)
    ft = np.fft.fftshift(np.fft.fftn(real))
    return ft


def np_ft_to_real(ft: np.ndarray) -> np.ndarray:
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
    ht_aug = np.pad(ht, pad_width=pad_width, mode="wrap")
    ht_aug_neg = np.flip(ht_aug)
    ft = (ht_aug + ht_aug_neg) * 0.5 - (ht_aug - ht_aug_neg) * 0.5 * 1j
    return ft[tuple(slice(0, s) for s in ht.shape)]


def line_freq(size: int):
    return np.fft.fftshift(np.fft.fftfreq(size))


def rfft2_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    k0 = line_freq(n0)
    k1 = np.fft.rfftfreq(n1)

    radii = np.sqrt(k0.reshape((-1, 1, 1)) ** 2 + k1.reshape((1, -1, 1)) ** 2)
    return k0, k1, radii


def rfft3_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size

    k0 = line_freq(n0)
    k1 = line_freq(n1)
    k2 = np.fft.rfftfreq(n2)

    radii = np.sqrt(
        k0.reshape((-1, 1, 1)) ** 2
        + k1.reshape((1, -1, 1)) ** 2
        + k2.reshape((1, 1, -1)) ** 2
    )
    return k0, k1, k2, radii


def fft3_freq(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size

    # 各个方向上的频率大小
    k0 = line_freq(n0)
    k1 = line_freq(n1)
    k2 = line_freq(n2)

    # 进行广播加权，每一个数据代表的是这个点离中心的频率大小
    radii = np.sqrt(
        k0.reshape((-1, 1, 1)) ** 2
        + k1.reshape((1, -1, 1)) ** 2
        + k2.reshape((1, 1, -1)) ** 2
    )
    return k0, k1, k2, radii


def fft1_freq_indices(size: int):
    return np.arange(-(size // 2), (size - 1) // 2 + 1)


def fft2_freq_indices(size: Union[int, Iterable[int]]) -> np.ndarray:
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    indices = np.round(
        np.sqrt(
            fft1_freq_indices(n0).reshape((-1, 1)) ** 2
            + fft1_freq_indices(n1).reshape((1, -1)) ** 2
        )
    )
    return indices.astype(int)


def fft3_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size
    indices = np.round(
        np.sqrt(
            fft1_freq_indices(n0).reshape((-1, 1, 1)) ** 2
            + fft1_freq_indices(n1).reshape((1, -1, 1)) ** 2
            + fft1_freq_indices(n2).reshape((1, 1, -1)) ** 2
        )
    )
    return indices.astype(int)


def rfft2_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 2
    assert len(size) == 2, "Input arg size should be int or tuple with length 2."

    n0, n1 = size

    indices = np.round(
        np.sqrt(
            fft1_freq_indices(n0).reshape((-1, 1)) ** 2
            + np.arange(0, n1).reshape((1, -1)) ** 2
        )
    )
    return indices.astype(int)


def rfft3_freq_indices(size: Union[int, Iterable[int]]):
    if isinstance(size, int):
        size = (size,) * 3
    assert len(size) == 3, "Input arg size should be int or tuple with length 3."

    n0, n1, n2 = size
    indices = np.round(
        np.sqrt(
            fft1_freq_indices(n0).reshape((-1, 1, 1)) ** 2
            + fft1_freq_indices(n1).reshape((1, -1, 1)) ** 2
            + np.arange(0, n2).reshape((1, 1, -1)) ** 2
        )
    )
    return indices.astype(int)


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


# tests
def _cycle_test(function: Callable, data_function: Callable = np.random.rand):
    def _item(*args):
        nda = data_function(*args)
        sig = inspect.signature(function)
        if "s" in sig.parameters:
            assert np.allclose(nda, function(nda, s=args)), (
                f"{function.__name__} does not pass cycle test "
                f"for array shape {args}."
            )
        else:
            assert np.allclose(nda, function(nda)), (
                f"{function.__name__} does not pass cycle test "
                f"for array shape {args}."
            )

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
    _cycle_test(
        lambda x: np_real_to_ft(np.real_if_close(np_ft_to_real(x))),
        data_function=_rand_ft_nda,
    )


def test_np_real_to_rft():
    _cycle_test(
        lambda x, s: np_real_to_rft(np.real_if_close(np_rft_to_real(x, s=s))),
        data_function=_rand_rft_nda,
    )


def test_np_rft_to_real():
    _cycle_test(lambda x, s: np.real_if_close(np_rft_to_real(np_real_to_rft(x), s=s)))


def test_np_real_to_ht():
    # ht -> real
    _cycle_test(lambda x: np_real_to_ht(np_ht_to_real(x)))
    # ht -> ft -> real
    _cycle_test(
        lambda x: np_real_to_ht(np.real_if_close(np_ft_to_real(np_ht_to_ft(x))))
    )


def test_np_ht_to_real():
    # real -> ht
    _cycle_test(lambda x: np_ht_to_real(np_real_to_ht(x)))
    # real -> ft -> ht
    _cycle_test(
        lambda x: np.real_if_close(np_ht_to_real(np_ft_to_ht(np_real_to_ft(x))))
    )


def test_np_ft_to_ht():
    _cycle_test(lambda x: np_ft_to_ht(np_ht_to_ft(x)))


def test_np_ht_to_ft():
    _cycle_test(lambda x: np_ht_to_ft(np_ft_to_ht(x)), data_function=_rand_ft_nda)


def test_np_real_to_rft():
    def _test_with_shape(shape):
        tmp = np.random.rand(*shape)
        ft = np_real_to_ft(tmp)
        rft = np_real_to_rft(tmp)

        last_dim = tmp.shape[-1]
        if last_dim % 2 == 0:
            assert np.allclose(ft[..., -last_dim // 2 :], rft[..., :-1])
            assert np.allclose(ft[..., 0], rft[..., -1])
        else:
            assert np.allclose(ft[..., -last_dim // 2 :], rft)

    _test_with_shape((10, 12, 10))
    _test_with_shape((10, 12, 11))


if __name__ == "__main__":
    test_np_ft_to_real()
    test_np_real_to_ft()
    test_np_rft_to_real()
    test_np_real_to_rft()
    test_np_ht_to_real()
    test_np_real_to_ht()
    test_np_ht_to_ft()
    test_np_ft_to_ht()