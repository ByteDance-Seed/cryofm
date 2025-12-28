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

# follow `torchio` design, need further optimization
import logging
from typing import Dict, Union, List, Tuple, Optional

import torch

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from mmcv.transforms import BaseTransform, TRANSFORMS

logger = logging.getLogger(__name__)


@TRANSFORMS.register_module()
class CenterCrop3D(BaseTransform):

    def __init__(self, patch_size: Union[int, Tuple[int, int, int]], keys="map_data"):
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if isinstance(keys, str):
            keys = [keys]

        self.patch_size = np.asarray(patch_size)
        self.keys = keys

    def transform(self, results: Dict):
        map_data = results[self.keys[0]]

        shape = np.asarray(map_data.shape)

        start_i, start_j, start_k = [(s - p) // 2 if s - p > 0 else 0 for p, s in zip(self.patch_size, shape)]

        for k in self.keys:
            results[k] = results[k][start_i:start_i + self.patch_size[0],
                         start_j:start_j + self.patch_size[1],
                         start_k:start_k + self.patch_size[2]]

        return results


@TRANSFORMS.register_module()
class RandomPatches3D(BaseTransform):

    def __init__(self, patch_size: Union[int, Tuple[int, int, int]], num_patches: int = 1, keys="map_data"):
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if isinstance(keys, str):
            keys = [keys]

        self.patch_size = np.asarray(patch_size)
        self.num_patches = num_patches
        self.keys = keys

    def transform(self, results: Dict):
        dtype = results[self.keys[0]].dtype
        shape = np.asarray(results[self.keys[0]].shape)
        valid_range = shape - self.patch_size

        rand_i, rand_j, rand_k = [np.random.randint(0, x + 1, size=self.num_patches) for x in valid_range]

        # logger.debug(f"{self.__class__.__name__} rand_i: {rand_i}, rand_j: {rand_j}, rand_k: {rand_k}")
        out = {}
        for k in self.keys:
            if k == "map_data":
                out_key = "patches"
            else:
                out_key = k + "_patches"
            out[out_key] = np.empty([self.num_patches, ] + self.patch_size.tolist(), dtype=dtype)

        for idx, index_ini in enumerate(zip(rand_i, rand_j, rand_k)):
            index_fin = np.asarray(index_ini) + self.patch_size

            for k in self.keys:
                if k == "map_data":
                    out["patches"][idx] = results["map_data"][tuple(map(slice, index_ini, index_fin))]
                else:
                    out[k + "_patches"][idx] = results[k][tuple(map(slice, index_ini, index_fin))]

        for k in out.keys():
            results[k] = out[k]
        results["shape_before_patchify"] = shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_size={self.patch_size}, num_patches={self.num_patches})'
        return repr_str


@TRANSFORMS.register_module()
class CreateProbabilityMap3D(BaseTransform):

    def __init__(self, threshold: float, dilation_iter: int = None):
        self.threshold = threshold
        self.dilation_iter = dilation_iter

    def transform(self, results: Dict):
        map_data = results['map_data']
        d, h, w = map_data.shape

        prob_map = map_data > self.threshold
        # make sure it has a center patch, in case mask is all zero
        if prob_map.sum() == 0:
            prob_map[d // 2, h // 2, w // 2] = 1

        s = generate_binary_structure(3, 2)

        if self.dilation_iter is not None:
            prob_map = binary_dilation(prob_map, structure=s, iterations=self.dilation_iter)

        results["prob_map"] = prob_map
        return results


@TRANSFORMS.register_module()
class WeightedPatches3D(BaseTransform):

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int, int]],
            probability_map_name: Optional[str],
            num_patches: int = 1,
            keys="map_data",
            margin=None,
            stride=None,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if isinstance(keys, str):
            keys = [keys]

        self.patch_size = np.asarray(patch_size)
        self.probability_map_name = probability_map_name
        self.num_patches = num_patches
        self.keys = keys
        self.margin = margin
        self.stride = stride
        self.cdf = None

    def process_probability_map(
            self,
            probability_map: np.ndarray,
    ) -> np.ndarray:
        if isinstance(probability_map, torch.Tensor):
            probability_map = probability_map.detach().cpu().numpy()
        # Using float32 can create cdf with maximum very far from 1, e.g. 0.92!
        data = probability_map.astype(np.float64)
        assert data.ndim == 3
        self.clear_probability_borders(data, self.patch_size)
        total = data.sum()
        if total == 0:
            half_patch_size = tuple(n // 2 for n in self.patch_size)
            message = (
                'Empty probability map found:'
                '\nVoxels with positive probability might be near the image'
                ' border.\nIf you suspect that this is the case, try adding a'
                ' padding transform\nwith half the patch size:'
                f' torchio.Pad({half_patch_size})'
            )
            raise RuntimeError(message)
        data /= total  # normalize probabilities
        return data

    @staticmethod
    def apply_margin_stride(probability_map: np.ndarray, margin: int = None, stride: int = None):
        if stride is None:
            stride = 1
        if margin is None:
            margin = 0
        assert stride >= 1 and margin >= 0, "stride must larger than or equal to 1, margin must be positive."
        indices = np.argwhere(probability_map > 0)
        bounding_box = np.vstack((np.min(indices, axis=0), np.max(indices, axis=0)))
        for i in range(3):
            if bounding_box[1, i] - bounding_box[0, i] > 2 * margin:
                bounding_box[:, i] += np.asarray([margin, -margin])
            else:
                bounding_box[:, i] = np.mean(bounding_box[:, i]).astype(int)

        x, y, z = np.mgrid[bounding_box[0, 0]:bounding_box[1, 0] + 1:stride,
                  bounding_box[0, 1]:bounding_box[1, 1] + 1:stride,
                  bounding_box[0, 2]:bounding_box[1, 2] + 1:stride]

        new_mask = np.zeros_like(probability_map).astype(bool)
        new_mask[x, y, z] = 1
        probability_map[~new_mask] = 0


    def transform(self, results: Dict):
        map_data = results['map_data']
        if isinstance(map_data, torch.Tensor):
            map_data = map_data.detach().cpu().numpy()
        prob_map = results[self.probability_map_name]

        #
        if self.margin is not None and self.stride is not None:
            self.apply_margin_stride(prob_map, self.margin, self.stride)

        probability_map_array = self.process_probability_map(prob_map)

        cdf = self.get_cumulative_distribution_function(probability_map_array)

        shape = np.asarray(map_data.shape)
        # valid_range = shape - self.patch_size

        out = {}
        for k in self.keys:
            if k == "map_data":
                out_key = "patches"
            else:
                out_key = k + "_patches"
            out[out_key] = np.empty([self.num_patches, ] + self.patch_size.tolist(), dtype=map_data.dtype)

        for idx in range(self.num_patches):
            index_ini = self.get_random_index_ini(probability_map_array, cdf)
            index_fin = np.asarray(index_ini) + self.patch_size
            for k in self.keys:
                if k == "map_data":
                    out["patches"][idx] = map_data[tuple(map(slice, index_ini, index_fin))]
                else:
                    out[k + "_patches"][idx] = results[k][tuple(map(slice, index_ini, index_fin))]

        for k in out.keys():
            results[k] = out[k]
        results["shape_before_patchify"] = shape
        return results

    @staticmethod
    def clear_probability_borders(
            probability_map: np.ndarray,
            patch_size: np.ndarray,
    ) -> None:
        # Set probability to 0 on voxels that wouldn't possibly be sampled
        # given the current patch size
        # We will arbitrarily define the center of an array with even length
        # using the // Python operator
        # For example, the center of an array (3, 4) will be on (1, 2)
        #
        #   Patch         center
        #  . . . .        . . . .
        #  . . . .   ->   . . x .
        #  . . . .        . . . .
        #
        #
        #    Prob. map      After preprocessing
        #
        #  x x x x x x x       . . . . . . .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x       . . . . . . .
        #
        # The dots represent removed probabilities, x mark possible locations
        crop_ini = patch_size // 2
        crop_fin = (patch_size - 1) // 2
        crop_i, crop_j, crop_k = crop_ini
        probability_map[:crop_i, :, :] = 0
        probability_map[:, :crop_j, :] = 0
        probability_map[:, :, :crop_k] = 0

        # The call tolist() is very important. Using np.uint16 as negative
        # index will not work because e.g. -np.uint16(2) == 65534
        crop_i, crop_j, crop_k = crop_fin.tolist()
        if crop_i:
            probability_map[-crop_i:, :, :] = 0
        if crop_j:
            probability_map[:, -crop_j:, :] = 0
        if crop_k:
            probability_map[:, :, -crop_k:] = 0

    @staticmethod
    def get_cumulative_distribution_function(
            probability_map: np.ndarray,
    ) -> np.ndarray:
        """Return the cumulative distribution function of a probability map."""
        flat_map = probability_map.flatten()
        flat_map_normalized = flat_map / flat_map.sum()
        cdf = np.cumsum(flat_map_normalized)
        return cdf

    def get_random_index_ini(
            self,
            probability_map: np.ndarray,
            cdf: np.ndarray,
    ) -> np.ndarray:
        center = self.sample_probability_map(probability_map, cdf)
        assert np.all(center >= 0)
        # See self.clear_probability_borders
        index_ini = center - self.patch_size // 2
        assert np.all(index_ini >= 0)
        return index_ini

    @classmethod
    def sample_probability_map(
            cls,
            probability_map: np.ndarray,
            cdf: np.ndarray,
    ) -> np.ndarray:
        """Inverse transform sampling.

        Example:
            >>> probability_map = np.array(
            ...    ((0,0,1,1,5,2,1,1,0),
            ...     (2,2,2,2,2,2,2,2,2)))
            >>> probability_map
            array([[0, 0, 1, 1, 5, 2, 1, 1, 0],
                   [2, 2, 2, 2, 2, 2, 2, 2, 2]])
            >>> histogram = np.zeros_like(probability_map)
            >>> for _ in range(100000):
            ...     histogram[WeightedSampler.sample_probability_map(probability_map, cdf)] += 1  # doctest:+SKIP
            ...
            >>> histogram  # doctest:+SKIP
            array([[    0,     0,  3479,  3478, 17121,  7023,  3355,  3378,     0],
                   [ 6808,  6804,  6942,  6809,  6946,  6988,  7002,  6826,  7041]])
        """  # noqa: B950
        # Get first value larger than random number ensuring the random number
        # is not exactly 0 (see https://github.com/fepegar/torchio/issues/510)
        random_number = max(torch.finfo(torch.float32).eps, torch.rand(1).item()) * cdf[-1]

        random_location_index = np.searchsorted(cdf, random_number)

        center = np.unravel_index(
            random_location_index,
            probability_map.shape,
        )

        probability = probability_map[center]
        if probability <= 0:
            message = (
                'Error retrieving probability in weighted sampler.'
                ' Please report this issue at'
                ' https://github.com/fepegar/torchio/issues/new?labels=bug&template=bug_report.md'  # noqa: B950
            )
            raise RuntimeError(message)

        return np.array(center)


@TRANSFORMS.register_module()
class GridPatches3D(BaseTransform):

    def __init__(self,
                 patch_size: Union[int, Tuple[int, int, int]],
                 patch_overlap: Union[int, Tuple[int, int, int]] = 0,
                 keys="map_data", ):
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if isinstance(patch_overlap, int):
            patch_overlap = (patch_overlap,) * 3
        if isinstance(keys, str):
            keys = [keys]

        self.patch_size = np.asarray(patch_size)
        self.patch_overlap = patch_overlap
        self.keys = keys

    # copied from torchio.data.sampler.grid, useful code!
    @staticmethod
    def _get_patches_locations(image_size, patch_size, patch_overlap, ) -> np.ndarray:
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # Locations:
        # [[0, 5],
        #  [3, 8],
        #  [5, 10]]
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if indices_dim[-1] != im_size_dim - patch_size_dim:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))

    def transform(self, results: Dict):
        map_data = results['map_data']

        locations = self._get_patches_locations(map_data.shape,
                                                patch_size=self.patch_size, patch_overlap=self.patch_overlap)

        out = {}
        for k in self.keys:
            if k == "map_data":
                out_key = "patches"
            else:
                out_key = k + "_patches"
            out[out_key] = np.empty([len(locations), ] + self.patch_size.tolist(), dtype=map_data.dtype)

        for idx, crop_idx in enumerate(locations):

            for k in self.keys:
                if k == "map_data":
                    out["patches"][idx] = map_data[tuple(map(slice, crop_idx[:3], crop_idx[3:]))]
                else:
                    out[k + "_patches"][idx] = results[k][tuple(map(slice, crop_idx[:3], crop_idx[3:]))]

        for k in out.keys():
            results[k] = out[k]

        results['locations'] = locations
        # results["patches"] = patches
        results["shape_before_patchify"] = map_data.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_size={self.patch_size}, patch_overlap={self.patch_overlap})'
        return repr_str


# TODO: support crop mode, etc
class GridAggregator:

    def __init__(self, shape, patch_overlap, overlap_mode: str = 'average'):
        self.shape = shape
        self.patch_overlap = patch_overlap
        self.overlap_mode = overlap_mode
        self._output_tensor = None
        self._avgmask_tensor = None

    def _initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.size(1)
        self._output_tensor = torch.zeros(
            num_channels,
            *self.shape,
            dtype=batch.dtype,
        )

    def _initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.size(1)
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.shape,
            dtype=batch.dtype,
        )

    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
    ) -> None:
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        patch_sizes = locations[:, 3:] - locations[:, :3]
        # There should be only one patch size
        assert len(np.unique(patch_sizes, axis=0)) == 1
        input_spatial_shape = tuple(batch.shape[-3:])
        target_spatial_shape = tuple(patch_sizes[0])
        if input_spatial_shape != target_spatial_shape:
            message = (
                f'The shape of the input batch, {input_spatial_shape},'
                ' does not match the shape of the target location,'
                f' which is {target_spatial_shape}'
            )
            raise RuntimeError(message)
        self._initialize_output_tensor(batch)
        assert isinstance(self._output_tensor, torch.Tensor)
        if self.overlap_mode == 'average':
            self._initialize_avgmask_tensor(batch)
            assert isinstance(self._avgmask_tensor, torch.Tensor)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                :,
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,
                ] += 1
        else:
            raise NotImplementedError

    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        assert isinstance(self._output_tensor, torch.Tensor)
        if self.overlap_mode in ['average', 'hann']:
            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/fepegar/torchio/issues/526
            output = torch.true_divide(
                self._output_tensor,
                self._avgmask_tensor,
            )
        else:
            output = self._output_tensor
        return output
