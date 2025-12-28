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

import random
import logging
from typing import Dict, Union, List, Tuple

import numpy as np

from scipy.ndimage import affine_transform, zoom
from mmcv.transforms import BaseTransform, TRANSFORMS
from monai.transforms import RandAffined, RandZoomd

import torch

from cryofm.core.utils.rotation_conversion import random_rotation
from cryofm.core.fourier import downsample_3d

logger = logging.getLogger(__name__)


def nda_int_center(arr: np.ndarray) -> List[int]:
    shape = arr.shape
    return [s // 2 for s in shape]


def nda_float_center(arr: np.ndarray) -> List[float]:
    shape = arr.shape
    return [s / 2 for s in shape]


def rotation_to_affine_matrix(rot_mat, center):
    trans_to_origin = np.eye(4)
    trans_to_origin[:3, 3] = -np.array(center)

    trans_back = np.eye(4)
    trans_back[:3, 3] = np.array(center)

    affine_mat = trans_back @ np.block([[rot_mat, np.zeros((3, 1))],
                                        [np.zeros((1, 3)), np.ones((1, 1))]]) @ trans_to_origin
    return affine_mat


def random_rotation_as_affine_matrix(center):
    rot_mat = random_rotation().numpy()

    return rotation_to_affine_matrix(rot_mat, center)


@TRANSFORMS.register_module()
class RandomRotation3D(BaseTransform):

    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        # probability doing this transformation
        self.p = p
        # args for affine transform:
        self.order = order
        self.mode = mode
        self.cval = cval

    def transform(self, results: Dict):
        if random.random() >= self.p:
            return results

        rot_mat = random_rotation().numpy()
        center = nda_float_center(results[self.keys[0]])
        affine_mat = rotation_to_affine_matrix(rot_mat, center)

        for k in self.keys:
            # actually this will do a inverse rot_mat transformation
            results[k] = affine_transform(results[k], affine_mat, order=self.order, mode=self.mode, cval=self.cval)

        # logger.debug(f"{self.__class__.__name__} use rotation matrix:\n {np.around(rot_mat, decimals=3)}")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, p={self.p}, order={self.order}, mode={self.mode}, cval={self.cval})'
        return repr_str


# cw rot
def rot90(vol, axis_idx=0):
    assert vol.ndim == 3
    assert axis_idx in (0, 1, 2)

    axes = [0, 1, 2]
    axes.remove(axis_idx)
    return np.flip(np.swapaxes(vol, *axes), axes[1])


def rot180(vol, axis_idx=0):
    assert vol.ndim == 3
    assert axis_idx in (0, 1, 2)

    axes = [0, 1, 2]
    axes.remove(axis_idx)
    return np.flip(vol, axis=axes)


def rot270(vol, axis_idx=0):
    assert vol.ndim == 3
    assert axis_idx in (0, 1, 2)

    axes = [0, 1, 2]
    axes.remove(axis_idx)
    return np.flip(np.swapaxes(vol, *axes), axes[0])


def rot_aug(vol, face_id, camera_id):
    assert face_id in (0, 1, 2, 3, 4, 5)
    assert camera_id in (0, 1, 2, 3)

    if face_id == 0:
        vol = vol
    elif face_id == 1:
        vol = rot90(vol, axis_idx=1)
    elif face_id == 2:
        vol = rot180(vol, axis_idx=1)
    elif face_id == 3:
        vol = rot270(vol, axis_idx=1)
    elif face_id == 4:
        vol = rot90(vol, axis_idx=0)
    elif face_id == 5:
        vol = rot270(vol, axis_idx=0)
    else:
        raise NotImplementedError

    if camera_id == 0:
        vol = vol
    elif camera_id == 1:
        vol = rot90(vol, axis_idx=2)
    elif camera_id == 2:
        vol = rot180(vol, axis_idx=2)
    elif camera_id == 3:
        vol = rot270(vol, axis_idx=2)
    else:
        raise NotImplementedError
    return vol


@TRANSFORMS.register_module()
class RotCube24(BaseTransform):
    def __init__(self, keys: Union[str, List[str]], p: float = 1.0,):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.p = p

    def transform(self, results: Dict):
        if random.random() >= self.p:
            return results

        rand_face_id = random.randint(0, 5)
        rand_camera_id = random.randint(0, 3)

        for k in self.keys:
            # .copy is important to avoid stride error
            results[k] = rot_aug(results[k], rand_face_id, rand_camera_id).copy()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, p={self.p})'
        return repr_str


class _ResizeApix(BaseTransform):

    def __init__(self,
                 keys: Union[str, List[str]],
                 resize_method: str = 'fft',
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.resize_method = resize_method
        assert self.resize_method in ("fft", "interp")
        # args for interpolation
        self.order = order
        self.mode = mode
        self.cval = cval

    def resize(self, results: Dict, tgt_apix: float):
        src_apix = results['apix']
        results['apix'] = tgt_apix

        zoom_factor = src_apix / tgt_apix

        for k in self.keys:
            if self.resize_method == "fft":
                src_side = results[k].shape[0]
                tgt_side = round(src_side * zoom_factor)
                results[k] = downsample_3d(torch.from_numpy(results[k]), tgt_side).numpy()
            else:
                results[k] = zoom(results[k], zoom=zoom_factor, order=self.order, mode=self.mode, cval=self.cval)
        return results


@TRANSFORMS.register_module()
class ResizeApix3D(_ResizeApix):

    def __init__(self,
                 keys: Union[str, List[str]],
                 tgt_apix: float = 2.,
                 resize_method: str = 'fft',
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0
                 ):
        self.tgt_apix = tgt_apix

        super().__init__(keys=keys, resize_method=resize_method, order=order, mode=mode, cval=cval)

    def transform(self, results: Dict):
        results = self.resize(results, self.tgt_apix)
        return results


@TRANSFORMS.register_module()
class RandomScaleApixInList3D(_ResizeApix):

    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 target_apix: Union[float, List[float]] = None,
                 resize_method: str = 'fft',
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0
                 ):
        assert target_apix is not None, 'Target apix must be specified.'

        self.p = p

        if isinstance(target_apix, float):
            target_apix = [target_apix]
        self.target_apix = target_apix

        super().__init__(keys=keys, resize_method=resize_method, order=order, mode=mode, cval=cval)

    def transform(self, results: Dict):
        if random.random() >= self.p:
            return results

        tgt_apix = np.random.choice(self.target_apix, size=1, replace=False)[0]
        return self.resize(results, tgt_apix)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, p={self.p}, target_apix={self.target_apix}, resize_method={self.resize_method}, '
            f'order={self.order}, mode={self.mode}, cval={self.cval})')
        return repr_str


@TRANSFORMS.register_module()
class RandomUpScaleApixInList3D(RandomScaleApixInList3D):

    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 target_apix: Union[float, List[float]] = None,
                 resize_method: str = 'fft',
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0
                 ):
        super().__init__(keys=keys, p=p, target_apix=target_apix, resize_method=resize_method, order=order, mode=mode,
                         cval=cval)

    def transform(self, results: Dict):
        if random.random() >= self.p:
            return results

        src_apix = results['apix']
        target_apix_list = np.asarray(self.target_apix)
        target_apix_list = target_apix_list[target_apix_list > src_apix]
        if len(target_apix_list) > 0:
            tgt_apix = np.random.choice(target_apix_list, size=1, replace=False)[0]
            if tgt_apix > src_apix:
                results = self.resize(results, tgt_apix)
        return results


@TRANSFORMS.register_module()
class RandomUpScaleApix3D(_ResizeApix):

    def __init__(self,
                 keys: Union[str, List[str]],
                 p: float = 0.5,
                 max_scale: float = 2.,
                 resize_method: str = 'fft',
                 order: int = 3,
                 mode: str = 'constant',
                 cval: float = 0.0
                 ):
        assert max_scale > 1., "Max scale must be larger than 1."
        self.p = p
        self.max_scale = max_scale

        super().__init__(keys=keys, resize_method=resize_method, order=order, mode=mode, cval=cval)

    def transform(self, results: Dict):
        if random.random() >= self.p:
            return results

        src_apix = results['apix']
        scale_factor = np.around(np.random.rand() * (self.max_scale - 1), decimals=2) + 1
        scale_factor = max(min(scale_factor, self.max_scale), 1.)
        tgt_apix = src_apix * scale_factor
        results = self.resize(results, tgt_apix)
        return results


@TRANSFORMS.register_module()
class RandomShift(BaseTransform):

    def __init__(self, keys: Dict, mode: List, prob: float, translate_range: Union[List, Tuple], padding_mode: str = "zeros"):
        self.keys = keys
        self.rand_affine = RandAffined(
            keys=keys,
            mode=mode,
            prob=prob,
            spatial_size=None,
            translate_range=translate_range,
            rotate_range=None,
            scale_range=None,
            padding_mode=padding_mode,
        )

    def transform(self, results: Dict):
        tmp_results = {k: results[k] for k in self.keys}
        tmp_results = self.rand_affine(tmp_results)
        results.update(tmp_results)
        return results


@TRANSFORMS.register_module()
class RandomZoom(BaseTransform):

    def __init__(self, keys: Dict, prob: float = 1.0, min_zoom: float = 0.97, max_zoom: float = 1.03,
                 mode: str = "trilinear", padding_mode: str = "constant"):
        self.keys = keys
        self.rand_zoom = RandZoomd(
            keys=keys,
            prob=prob,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            mode=mode,
            padding_mode=padding_mode,
        )

    def transform(self, results: Dict):
        tmp_results = {k: results[k] for k in self.keys}
        tmp_results = self.rand_zoom(tmp_results)
        results.update(tmp_results)
        return results
