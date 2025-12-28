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

from typing import Union, List

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as R
from skimage import measure
from skimage.morphology import binary_dilation, disk


def _set_center_value(arr, val):
    arr[tuple(np.asarray(arr.shape) // 2)] = val
    return arr


def _centered_ogrid(shape):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_x, center_y, center_z = shape[0] // 2, shape[1] // 2, shape[2] // 2
    z = z - center_z
    y = y - center_y
    x = x - center_x
    return z, y, x


def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_sphere_mask(d, h, w, center=None, radius=None) -> np.ndarray:
    if center is None:
        center = (int(d / 2), int(h / 2), int(w / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    z, y, x = np.ogrid[:d, :h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    mask = dist_from_center <= radius
    return mask


def create_cryoet_wedge_mask(
        shape: Union[int, List[int]],
        tilt_angle: int = 60,
        plane_axis_ids: List[int] = (1, 2),
        rot_axis_id: int = 1,
        return_sphere_mask: bool = False
    ):
    """
    Create Cryo-ET style missing wedge mask.
    In the literature on cryo-electron tomography (cryo-ET), the "missing wedge" refers to the signal loss attributed
    to the range of sample tilting angles [-tilt, +tilt] (preserved signal). This function is defined by plane_axis_ids, which specifies
    the two axes comprising the sample plane, and by rot_axis_id, which determines the axis around which the sample
    rotates within the plane. For 3D data, the conventional order of the axes is z, y, x, with the default assumption
    being that the sample rotates around the y-axis on the yx plane.

    @param shape: The lengths of the sides of a three-dimensional matrix.
    @param tilt_angle: Maximum tilt angle in degrees (preserved signal).
    @param plane_axis_ids: sample plane axis ids.
    @param rot_axis_id: sample rotate alone which axis.
    @return: np.ndarray
    """
    if isinstance(shape, int):
        shape = (shape, ) * 3

    assert len(shape) == 3, f"Shape must be 3D, got {len(shape)}D."
    assert 0 < tilt_angle < 90, "Tilt_angle must be in (0, 90)"
    assert rot_axis_id in plane_axis_ids, f"rot_axis_id must be in plane_axis_ids"

    indices = _centered_ogrid(shape)

    # wedge plane
    v_axis_id = ({0, 1, 2} - set(plane_axis_ids)).pop()
    h_axis_id = (set(plane_axis_ids) - {rot_axis_id}).pop()

    theta = np.arctan2(np.abs(indices[v_axis_id]), np.abs(indices[h_axis_id]))
    # make sure center is 0
    theta = _set_center_value(theta, 0)

    theta = np.degrees(theta)
    mask = theta <= tilt_angle
    mask = np.repeat(mask, repeats=shape[rot_axis_id], axis=rot_axis_id)

    if return_sphere_mask:
        mask &= create_sphere_mask(*shape)
    return mask


def create_axial_cone_mask(shape: Union[int, List[int]], angle: int = 60, axis_id: int = 0):
    """
    Creating a conical mask with its axis of symmetry along the standard coordinate axis.

    @param shape: The lengths of the sides of a three-dimensional matrix.
    @param angle: angle in degrees.
    @param axis_id: symmetry axis id.
    @return: np.ndarray
    """
    if isinstance(shape, int):
        shape = (shape, ) * 3

    assert len(shape) == 3, f"Shape must be 3D, got {len(shape)}D."
    assert 0 < angle <= 180, "Tilt_angle must be in (0, 180]"
    assert axis_id in (0, 1, 2), "axis_id must be in (0, 1, 2)"
    half_angle = angle / 2
    tan_half_angle = np.tan(np.radians(half_angle))

    indices = _centered_ogrid(shape)

    extra_axis_ids = [0, 1, 2]
    extra_axis_ids.remove(axis_id)

    r = np.sqrt(indices[extra_axis_ids[0]] ** 2 + indices[extra_axis_ids[1]] ** 2)
    mask = np.asarray(r <= (np.abs(indices[axis_id]) * tan_half_angle))
    mask = _set_center_value(mask, True)
    return mask
