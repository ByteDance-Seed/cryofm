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

from typing import Tuple, List

import numpy as np


def get_pad_width(src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    Calculate the pad width along each axis to reshape the source shape to the destination shape.

    For each axis, computes the amount of padding to add before and after
    to reach the destination shape from the source shape. If the destination
    size is smaller than or equal to the source, zero pad is used.

    Parameters
    ----------
    src_shape : sequence of int
        The original shape of the array.
    dst_shape : sequence of int
        The desired shape after padding.

    Returns
    -------
    pad_width : list of tuple of int
        Each tuple specifies (pad_before, pad_after) for the corresponding axis.

    Examples
    --------
    >>> get_pad_width((28, 28), (32, 32))
    [(2, 2), (2, 2)]
    >>> get_pad_width((32, 32, 32), (40, 40, 40))
    [(4, 4), (4, 4), (4, 4)]
    """
    return [((dst - src) // 2, (dst - src) - (dst - src) // 2) if dst > src else (0, 0)
            for src, dst in zip(src_shape, dst_shape)]


def get_crop_slice_fn(src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...]) -> List[slice]:
    """
    Get slicing indices to crop a centered region of shape `dst_shape` from an array of shape `src_shape`.

    For each axis, returns a slice object that crops the array from the center so that the result has size
    `dst_shape` along that axis. If `dst_shape` is greater than or equal to `src_shape` along any axis,
    returns slice(None) for that axis (i.e., no cropping).

    Parameters
    ----------
    src_shape : sequence of int
        The shape of the source array.
    dst_shape : sequence of int
        The desired shape after cropping.

    Returns
    -------
    slices : list of slice
        List of slice objects for each axis that extracts a centered crop of size `dst_shape` from `src_shape`.

    Examples
    --------
    >>> get_crop_slice_fn((32, 32), (28, 28))
    [slice(2, 30, None), slice(2, 30, None)]
    >>> get_crop_slice_fn((32, 32, 32), (32, 32, 32))
    [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    """
    return [slice((src - dst) // 2, (src - dst) // 2 + dst) if dst < src else slice(None)
            for src, dst in zip(src_shape, dst_shape)]


def get_extra_index(shape: Tuple[int, int, int], existing_index: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Compute extra indices for 3D patches that do not overlap with the existing ones.

    Given the shape of a 3D array, an array of existing patch start indices (typically N x 3),
    and the patch size, this function returns the start and end indices for all extra patches
    of the given size that do not overlap with any of those in `existing_index` and that fit 
    inside the array bounds.

    Parameters
    ----------
    shape : tuple of int
        The shape of the 3D array (depth, height, width).
    existing_index : numpy.ndarray
        Array of start indices of existing patches. Should have shape (N, 3).
    patch_size : int
        The edge length of the cubic patch (applied to all axes).

    Returns
    -------
    extra_index : numpy.ndarray
        Array of shape (M, 6). Each row gives [z_start, y_start, x_start, z_end, y_end, x_end] 
        for one extra patch (start inclusive, end exclusive).

    Examples
    --------
    >>> shape = (8, 8, 8)
    >>> existing_index = np.array([[0, 0, 0], [2, 2, 2]])
    >>> patch_size = 3
    >>> extra_index = get_extra_index(shape, existing_index, patch_size)
    >>> extra_index.shape  # (62, 6)
    """
    mask = np.ones(shape, dtype=bool)
    mask[:, :, -(patch_size - 1):] = False
    mask[:, -(patch_size - 1):, :] = False
    mask[-(patch_size - 1):, :, :] = False

    mask[existing_index[:, 0], existing_index[:, 1], existing_index[:, 2]] = False
    start_ini = np.argwhere(mask)
    end_ini = start_ini + patch_size
    extra_index = np.hstack((start_ini, end_ini))
    return extra_index


def get_bbox(mask: np.ndarray, enlarge: int = 0) -> np.ndarray:
    """
    Compute the bounding box of nonzero (foreground) regions in a mask of arbitrary dimension, optionally enlarging it.

    Parameters
    ----------
    mask : numpy.ndarray
        The input mask array of any shape (2D, 3D, ...), with dtype bool or float.
        Values > 0.5 are considered foreground.
    enlarge : int, optional
        The number of elements to enlarge each bound of the bounding box (default is 0).
        Enlarge will not extend beyond the valid array shape.

    Returns
    -------
    bbox : numpy.ndarray
        An array of shape (2, N) where N is the number of dimensions of `mask`.
        The first row gives the minimal indices of the bounding box,
        and the second row gives the maximal indices (inclusive) for each axis.

    Examples
    --------
    >>> mask = np.zeros((10, 10, 10))  # 3D example
    >>> mask[2:5, 3:6, 4:7] = 1
    >>> bbox = get_bbox(mask, enlarge=1)
    >>> bbox  # doctest: +SKIP
    array([[1, 2, 3],
           [5, 6, 7]])
    """
    assert enlarge >= 0, "enlarge must be >= 0"
    indices = np.argwhere(mask > 0.5)
    bbox = np.vstack([np.amin(indices, axis=0), np.amax(indices, axis=0)])
    # ensure not out of index
    bbox[0, :] = np.maximum(bbox[0, :] - enlarge, 0)
    bbox[1, :] = np.minimum(bbox[1, :] + enlarge, np.asarray(mask.shape) - 1)
    return bbox


# bbox should be [[start indices], [end indices]]
def bbox_crop(data: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Crop an array to a bounding box.

    Parameters
    ----------
    data : numpy.ndarray
        The input array to crop, of arbitrary shape.
    bbox : numpy.ndarray
        An array of shape (2, N) where N is the number of dimensions of `data`.
        bbox[0, :] contains the minimum indices (inclusive) and
        bbox[1, :] contains the maximum indices (inclusive) for each axis.

    Returns
    -------
    cropped : numpy.ndarray
        The cropped sub-array of `data` defined by the bounding box.

    Examples
    --------
    >>> arr = np.arange(1000).reshape(10, 10, 10)
    >>> bbox = np.array([[2, 3, 4], [5, 6, 7]])
    >>> cropped = bbox_crop(arr, bbox)
    >>> cropped.shape
    (4, 4, 4)
    """
    return data[tuple(map(slice, bbox[0, :], bbox[1, :] + 1))]


def ensure_min_bbox_size(bbox, max_shape, min_size=64):
    """
    Adjusts a bounding box to ensure it meets a minimum size along each axis.

    Parameters
    ----------
    bbox : np.ndarray
        An array of shape (2, N), where N is the number of dimensions. Contains
        the start (bbox[0]) and end (bbox[1]) coordinates for each axis of a bounding box.
    max_shape : tuple or list
        The maximum allowed shape for each axis. Typically the shape of the target array.
    min_size : int, optional
        The minimum desired size for each axis of the bounding box. Default is 64.

    Returns
    -------
    bbox : np.ndarray
        A new bounding box array of the same shape as input, with at least `min_size` along each axis,
        and clamped within the [0, max_shape-1] range.

    Examples
    --------
    >>> bbox = np.array([[10, 15, 8], [12, 18, 10]])
    >>> max_shape = (32, 32, 32)
    >>> ensure_min_bbox_size(bbox, max_shape, min_size=8)
    array([[ 8, 12,  4],
           [15, 19, 11]])
    """
    def adjust_axis(bbox_axis, limit):
        size = bbox_axis[1] - bbox_axis[0] + 1
        if size >= min_size:
            return bbox_axis

        center = sum(bbox_axis) / 2
        half_size = min_size / 2
        new_min = max(0, min(center - half_size, limit - min_size))
        return int(new_min), int(new_min + min_size - 1)

    bbox = bbox.copy()
    for axis, limit in enumerate(max_shape):
        bbox[:, axis] = adjust_axis(bbox[:, axis], limit)

    return bbox
