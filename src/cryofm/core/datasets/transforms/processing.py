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

from typing import Dict, Optional, Union, Tuple, List

import numpy as np
from mmcv.transforms import BaseTransform, TRANSFORMS
import random
import torch
import torch.nn.functional as F

from cryofm.core.utils.mask import create_circular_mask, create_sphere_mask


# normalize
class _ClipNorm(BaseTransform):
    """Clip and Normalize data to [0, 1]"""

    def __init__(self, key, min_value=None, max_value=None, min_percentile=None, max_percentile=None):
        self.key = key
        self.min_value = min_value
        self.max_value = max_value
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def transform(self, results: Dict) -> Dict:
        data = results[self.key]

        min_value = data.min()
        max_value = data.max()

        if self.min_value is not None:
            min_value = self.min_value
        if self.min_percentile is not None:
            min_value = np.percentile(data, self.min_percentile)
        if self.max_value is not None:
            max_value = self.max_value
        if self.max_percentile is not None:
            max_value = np.percentile(data, self.max_percentile)

        np.clip(data, a_min=min_value, a_max=max_value, out=data)

        value_range = (max_value - min_value)

        data -= min_value

        if "contour_lvl" in results:
            results["contour_lvl"] = results["contour_lvl"] - min_value

        if value_range > 1e-8:
            data /= value_range
            if "contour_lvl" in results:
                results["contour_lvl"] = results["contour_lvl"] / value_range

        results[self.key] = data

        return results


@TRANSFORMS.register_module()
class NoiseRemove(BaseTransform):
    # Page 5 of CryoBench: Datasets and Benchmarks for the Heterogeneous Reconstruction Tasks in CryoEM
    # Just an implementation but seems not very useful?

    def __init__(self, percentile=0.9999, ratio=0.5):
        self.percentile = percentile
        self.ratio = ratio

    def transform(self, results: Dict) -> Dict:
        data = results["map_data"]

        a_min = np.percentile(data, self.percentile) * self.ratio
        data[data < a_min] = 0.0
        results["map_data"] = data
        return results


@TRANSFORMS.register_module()
class ClipNorm2D(_ClipNorm):
    def __init__(self, min_value=None, max_value=None, min_percentile=None, max_percentile=None):
        super().__init__(key="img", min_value=min_value, max_value=max_value, min_percentile=min_percentile,
                         max_percentile=max_percentile)
        self.min_value = min_value
        self.max_value = max_value
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile


@TRANSFORMS.register_module()
class Scale2D(BaseTransform):
    def __init__(self, scale=1.0):
        self.scale = scale

    def transform(self, results: Dict) -> Dict:
        results["img"] = results["img"] * self.scale
        return results


@TRANSFORMS.register_module()
class ClipNorm3D(_ClipNorm):
    """Clip and Normalize map_data to [0, 1]"""

    def __init__(self, key="map_data", min_value=None, max_value=None, min_percentile=None, max_percentile=None):
        super().__init__(key=key, min_value=min_value, max_value=max_value, min_percentile=min_percentile,
                         max_percentile=max_percentile)
    
@TRANSFORMS.register_module()
class RandomGlobalScale(BaseTransform):
    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 scale_min: float = 0.5,
                 scale_max: float = 1.5
                 ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
    def transform(self, results: Dict) -> Dict:
        if random.random() >= self.p:
            return results
        s = random.uniform(self.scale_min, self.scale_max)
        for k in self.keys:
            results[k] = (results[k] * s).astype(np.float32)
        return results
    

@TRANSFORMS.register_module()
class MaxValueStandardize(BaseTransform):
    def __init__(self, 
                 keys: Union[str, List[str]],
                 max_value_key: str):
    
        self.keys = keys
        self.max_value_key = max_value_key

    def transform(self, results: Dict) -> Dict:
        for key in self.keys:
            map_data = results[key]
            max_value = results[self.max_value_key]
            map_data = map_data / max_value
            results[key] = map_data.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class ClipMaxValueStandardize(BaseTransform):
    def __init__(self, 
                 keys: Union[str, List[str]],
                 max_value_key: str):
    
        self.keys = keys
        self.max_value_key = max_value_key

    def transform(self, results: Dict) -> Dict:
        for key in self.keys:
            map_data = results[key]
            max_value = results[self.max_value_key]
            map_data = np.clip(map_data, a_min=None, a_max=max_value)
            map_data = map_data / max_value
            results[key] = map_data.astype(np.float32)
        return results
    


@TRANSFORMS.register_module()
class Standardize3D(BaseTransform):

    def __init__(self, map_key="map_data"):
        self.map_key = map_key

    def transform(self, results: Dict) -> Dict:
        map_data = results[self.map_key]
        mean, std = np.mean(map_data), np.std(map_data)
        # do inplace calc
        map_data -= mean
        if "contour_lvl" in results:
            results["contour_lvl"] = results["contour_lvl"] - mean

        if std > 1e-8:
            map_data /= std
            if "contour_lvl" in results:
                results["contour_lvl"] = results["contour_lvl"] / std
        results[self.map_key] = map_data
        return results


@TRANSFORMS.register_module()
class AnchorKeyStandardize3D(BaseTransform):

    def __init__(self, anchor_key="map_data", map_keys="map_data"):
        self.anchor_key = anchor_key
        if isinstance(map_keys, str):
            map_keys = list(map_keys)
        self.map_keys = map_keys

    def transform(self, results: Dict) -> Dict:
        anchor_data = results[self.anchor_key]
        mean, std = np.mean(anchor_data), np.std(anchor_data)

        for k in self.map_keys:
            tmp = results[k]
            tmp -= mean
            if std > 1e-8:
                tmp /= std
            results[k] = tmp

        if "contour_lvl" in results:
            results["contour_lvl"] = results["contour_lvl"] - mean

        if std > 1e-8:
            if "contour_lvl" in results:
                results["contour_lvl"] = results["contour_lvl"] / std
        return results


@TRANSFORMS.register_module()
class PrecomputedStandardize3D(BaseTransform):

    def __init__(self, map_key, mean_key=None, std_key=None, mean=None, std=None):
        assert (mean_key is not None and std_key is not None) or (
                mean is not None and std is not None)
        self.map_key = map_key
        self.mean_key = mean_key
        self.std_key = std_key
        self.mean = mean
        self.std = std

    def transform(self, results: Dict) -> Dict:
        map_data = results[self.map_key]
        if self.mean_key is not None and self.std_key is not None:
            mean, std = results[self.mean_key], results[self.std_key]
        else:
            mean, std = self.mean, self.std
        # do inplace calc
        map_data -= mean

        if std > 1e-8:
            map_data /= std
        results[self.map_key] = map_data
        return results


@TRANSFORMS.register_module()
class SliceVolumeWithPose(BaseTransform):

    def __init__(self, del_map_data: bool = True):
        self.del_map_data = del_map_data

    def transform(self, results: Dict) -> Dict:
        map_data = results["map_data"]
        rot_mat = results["rot_mat"]

        side_shape = results["src_shape"]

        x = torch.linspace(-1, 1, side_shape)
        y = torch.linspace(-1, 1, side_shape)
        xx, yy = torch.meshgrid([x, y], indexing="ij")
        plane_coords = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)

        rot_plane_coords = torch.einsum("ijkn, nm -> ijkm", plane_coords.unsqueeze(2), rot_mat)

        img = F.grid_sample(torch.from_numpy(map_data[None, None, ...].copy()),
                            rot_plane_coords[None, ...],
                            align_corners=True)
        # 1, 1, x, y, 1 -> x, y -> y, x
        img = img.squeeze().permute(1, 0).contiguous()
        results["img"] = img.numpy()

        if self.del_map_data:
            del results["map_data"]
        return results


@TRANSFORMS.register_module()
class SamplePointsFromVolume(BaseTransform):

    def __init__(self, num_points: int, order: str = "zyx", del_map_data: bool = True):
        assert order in ["zyx", "xyz"]
        self.num_points = num_points
        self.order = order
        self.del_map_data = del_map_data

    def transform(self, results: Dict) -> Dict:
        # map is z, y, x array
        map_data = results["map_data"]
        side_shape = results["src_shape"]

        if "sphere_mask" in results:
            mask = results["sphere_mask"]
            valid_indices = np.transpose(np.nonzero(mask))
        else:
            grid = np.indices((side_shape,) * 3)
            valid_indices = np.transpose(grid, (1, 2, 3, 0)).reshape(-1, 3)

        num_valid_points = len(valid_indices)

        if num_valid_points > self.num_points:
            select = np.random.choice(num_valid_points, self.num_points, replace=False)
        else:
            select = np.random.permutation(num_valid_points)
            if self.num_points - num_valid_points > 0:
                extra_select = np.random.choice(num_valid_points, self.num_points - num_valid_points, replace=True)
                select = np.append(select, extra_select)

        indices = valid_indices[select]

        values = map_data[tuple(np.hsplit(indices, 3))]

        if self.order == "xyz":
            indices = np.flip(indices, axis=1)

        results["points"] = np.hstack((indices.astype(np.float32), values))

        if self.del_map_data:
            del results["map_data"]
        return results


@TRANSFORMS.register_module()
class SamplePointsFromVolumeWithHCP(BaseTransform):

    def __init__(self, num_points: int, max_pad_size: int, sphere_radius: int = 5, order: str = "zyx",
                 del_map_data: bool = True):
        assert order in ["zyx", "xyz"]

        self.num_points = num_points
        self.box_size = max_pad_size * 2
        self.sphere_radius = sphere_radius
        self.order = order
        self.del_map_data = del_map_data

        self.hcp_mask = create_hcp_mask(self.box_size, radius=sphere_radius)

    def transform(self, results: Dict) -> Dict:
        # map is z, y, x array
        map_data = results["map_data"]
        side_shape = results["src_shape"]

        rand_start_z = np.random.randint(0, self.box_size + 1 - side_shape)
        rand_start_y = np.random.randint(0, self.box_size + 1 - side_shape)
        rand_start_x = np.random.randint(0, self.box_size + 1 - side_shape)

        tmp_hcp_mask = self.hcp_mask[rand_start_z:rand_start_z + side_shape,
                                     rand_start_y:rand_start_y + side_shape,
                                     rand_start_x:rand_start_x + side_shape].copy()

        if "sphere_mask" in results:
            mask = results["sphere_mask"]
            valid_indices = np.transpose(np.nonzero(mask))
            tmp_hcp_mask = tmp_hcp_mask * mask
        else:
            grid = np.indices((side_shape,) * 3)
            valid_indices = np.transpose(grid, (1, 2, 3, 0)).reshape(-1, 3)

        # unique is slow, need optimize
        u, counts = np.unique(tmp_hcp_mask, return_counts=True)
        unzero_mask = u != 0
        u = u[unzero_mask]
        counts = counts[unzero_mask]
        num_hcp_valid_points = np.sum(counts)

        if num_hcp_valid_points > self.num_points:
            perm = np.random.permutation(len(u))
            perm_u = u[perm]
            perm_counts = counts[perm]
            cumsum_counts = np.cumsum(perm_counts)
            num_balls = np.searchsorted(cumsum_counts, self.num_points, side="left") + 1
            tmp_mask = np.isin(tmp_hcp_mask, perm_u[:num_balls])
            valid_indices = np.transpose(np.nonzero(tmp_mask))

        num_valid_points = len(valid_indices)

        if num_valid_points > self.num_points:
            select = np.random.choice(num_valid_points, self.num_points, replace=False)
        else:
            select = np.random.permutation(num_valid_points)
            if self.num_points - num_valid_points > 0:
                extra_select = np.random.choice(num_valid_points, self.num_points - num_valid_points, replace=True)
                select = np.append(select, extra_select)

        indices = valid_indices[select]

        values = map_data[tuple(np.hsplit(indices, 3))]

        if self.order == "xyz":
            indices = np.flip(indices, axis=1)

        results["points"] = np.hstack((indices.astype(np.float32), values))

        if self.del_map_data:
            del results["map_data"]
        return results


@TRANSFORMS.register_module()
class CreateCircularMask(BaseTransform):
    def __init__(self, radius_ratio: float = 1.0):
        self.radius_ratio = radius_ratio
        self.cache = {}

    def transform(self, results: Dict) -> Dict:
        src_shape = results["src_shape"]
        if src_shape in self.cache:
            mask = self.cache[src_shape]
        else:
            mask = create_circular_mask(src_shape, src_shape, radius=self.radius_ratio * src_shape / 2)
            mask = mask.astype(bool)
        results["circular_mask"] = mask
        return results


@TRANSFORMS.register_module()
class CreateSphereMask(BaseTransform):
    def __init__(self, radius_ratio: float = 1.0):
        self.radius_ratio = radius_ratio
        self.cache = {}

    def transform(self, results: Dict) -> Dict:
        src_shape = results["src_shape"]
        if src_shape in self.cache:
            mask = self.cache[src_shape]
        else:
            mask = create_sphere_mask(src_shape, src_shape, src_shape, radius=self.radius_ratio * src_shape / 2)
            mask = mask.astype(bool)
        results["sphere_mask"] = mask
        return results


def _set_array_values_by_mask(array, val_array, mask):
    array[mask] = val_array[mask].copy()


# Hexagonal closed-packing
def _hcp_layer_mask(plane_size: Union[int, List[int]], radius: int, layer_type: str = 'A'):
    if isinstance(plane_size, int):
        plane_size = (plane_size,) * 2

    diameter = radius * 2
    assert np.all(np.asarray(plane_size) >= diameter), f"Box size {plane_size} must >= 2 * radius {radius}"

    sqrt_three = np.sqrt(3)

    # small ball mask for changing values quickly
    small_ball_select_mask = create_sphere_mask(diameter, diameter, diameter, radius=radius).astype(bool)
    line_mask = np.zeros((max(plane_size), diameter, diameter), dtype=int)
    for i, s in enumerate(range(0, max(plane_size) + 1 - diameter, diameter)):
        line_mask[s:s + diameter, ...][small_ball_select_mask] = i + 1
    line_select_mask = line_mask.astype(bool)

    result_mask = np.zeros((*plane_size, diameter), dtype=int)

    def _create_line_mask(start):
        tmp_line_mask = np.zeros((plane_size[0], diameter, diameter), dtype=int)
        num_balls = int((plane_size[0] - start) / diameter)
        length = num_balls * diameter

        _set_array_values_by_mask(tmp_line_mask[start:start + length], line_mask[:length], line_select_mask[:length])

        return tmp_line_mask, tmp_line_mask.max()

    if layer_type == 'A':
        # centers:
        # 1: (2r, r), (4r, r), ...
        # 2: (r, r + sqrt(3) * r, (4r, r + sqrt(3) * r), ...
        # 3: (2r, r + 2 * sqrt(3) * r), (4r, r + 2 * sqrt(3) * r), ...
        first_line, first_line_num_balls = _create_line_mask(0)
        second_line, second_line_num_balls = _create_line_mask(radius - 1)
        y_origin = 0

    elif layer_type == 'B':
        # centers:
        # 1: (r, r + sqrt(3)/3 * r), (3r, r + sqrt(3)/3 * r), ...
        # 2: (2r, r + 4 * sqrt(3)/3 * r, (4r, r + 4 * sqrt(3)/3 * r), ...
        # 3: (r, r + 7 * sqrt(3)/3 * r), (3r, 7 * sqrt(3)/3 * r), ...
        first_line, first_line_num_balls = _create_line_mask(radius)
        second_line, second_line_num_balls = _create_line_mask(0)
        y_origin = sqrt_three / 3 * radius

    else:
        raise NotImplementedError

    ny = int((plane_size[1] - diameter - y_origin) / (sqrt_three * radius)) + 1
    total_num_balls = 0

    for y_id in range(ny):
        y_s = int(y_origin + (sqrt_three * radius) * y_id)
        if y_s + radius >= plane_size[1]:
            break

        if y_id % 2 == 0:
            _set_array_values_by_mask(result_mask[:, y_s:y_s + diameter, :],
                                      first_line + total_num_balls,
                                      first_line.astype(bool))
            total_num_balls += first_line_num_balls
        else:
            _set_array_values_by_mask(result_mask[:, y_s:y_s + diameter, :],
                                      second_line + total_num_balls,
                                      second_line.astype(bool))
            total_num_balls += second_line_num_balls

    return result_mask


# https://www.wikiwand.com/en/Close-packing_of_equal_spheres
def create_hcp_mask(box_size: Union[int, List[int]], radius: int):
    if isinstance(box_size, int):
        box_size = (box_size,) * 3

    diameter = radius * 2
    assert np.all(np.asarray(box_size) >= diameter), f"Box size {box_size} must >= 2 * radius {radius}"

    sqrt_six = np.sqrt(6)

    result_mask = np.zeros(box_size, dtype=int)

    # loop with A-B-A-B pattern
    plane_a = _hcp_layer_mask(box_size[:2], radius, 'A')
    plane_a_balls = plane_a.max()
    plane_b = _hcp_layer_mask(box_size[:2], radius, 'B')
    plane_b_balls = plane_b.max()

    # centers in z-axis: r, r + 2 * sqrt(6) / 3 * r, r + 2 * 2 * sqrt(6) / 3 * r, ...
    nz = int((box_size[2] - diameter) / sqrt_six * 2 * radius / 3) + 1
    total_num_balls = 0

    for z_id in range(nz):
        z_s = int(z_id * 2 * sqrt_six * radius / 3)
        if z_s + diameter >= box_size[2]:
            break

        if z_id % 2 == 0:
            _set_array_values_by_mask(result_mask[:, :, z_s:z_s + diameter],
                                      plane_a + total_num_balls,
                                      plane_a.astype(bool))
            total_num_balls += plane_a_balls
        else:
            _set_array_values_by_mask(result_mask[:, :, z_s:z_s + diameter],
                                      plane_b + total_num_balls,
                                      plane_b.astype(bool))
            total_num_balls += plane_b_balls

    return result_mask


@TRANSFORMS.register_module()
class PadAll(BaseTransform):

    def __init__(self,
                 keys: Union[str, List[str]] = "img",
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_square: bool = False,
                 pad_val: Union[int, float, dict] = None,
                 padding_mode: str = 'constant') -> None:

        if isinstance(keys, str):
            keys = list(keys)
        self.keys = keys
        self.size = size
        self.size_divisor = size_divisor

        if pad_val is None:
            pad_val = {k: 0 for k in keys}
        elif isinstance(pad_val, (int, float)):
            pad_val = {k: pad_val for k in keys}
        assert isinstance(pad_val, dict), 'pad_val '

        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def transform(self, results: dict) -> dict:
        # use the first key to determine output size
        src_shape = results[self.keys[0]].shape

        size = None
        if self.pad_to_square:
            max_size = max(src_shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (src_shape[0], src_shape.shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]

        results[f'{self.keys[0]}_shape'] = size

        for k in self.keys:
            pad_val = self.pad_val.get(k, 0)

            if isinstance(pad_val, int) and results[k].ndim == 3:
                pad_val = tuple(pad_val for _ in range(results[k].shape[2]))

            padded = center_pad(results[k], shape=size, mode=self.padding_mode, pad_val=pad_val)
            results[k] = padded
        return results


@TRANSFORMS.register_module()
class Pad3D(BaseTransform):

    def __init__(self,
                 keys: Union[str, List[str]] = "img",
                 size: Union[int, Tuple[int, int, int]] = None,
                 pad_val: Union[int, float, dict] = None,
                 padding_mode: str = 'constant') -> None:

        if isinstance(keys, str):
            keys = list(keys)
        self.keys = keys
        if isinstance(size, int):
            size = (size, ) * 3
        self.size = size

        if pad_val is None:
            pad_val = {k: 0 for k in keys}
        elif isinstance(pad_val, (int, float)):
            pad_val = {k: pad_val for k in keys}
        assert isinstance(pad_val, dict), 'pad_val '

        self.pad_val = pad_val

        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def transform(self, results: dict) -> dict:
        for k in self.keys:
            pad_val = self.pad_val.get(k, 0)

            padded = center_pad(results[k], shape=self.size, mode=self.padding_mode, pad_val=pad_val)
            results[k] = padded
        return results


def center_pad(arr, shape, mode='constant', pad_val=0):
    src_shape = arr.shape
    dst_shape = np.maximum(src_shape, shape)
    pad_width = np.asarray(
        [((d - s) // 2, (d - s) - (d - s) // 2) if d > s else (0, 0) for s, d in zip(src_shape, dst_shape)])

    return np.pad(arr, pad_width, mode=mode, constant_values=pad_val)
