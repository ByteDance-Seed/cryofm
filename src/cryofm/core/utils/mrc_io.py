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

import shutil
import logging
import warnings
from typing import Tuple, Union

import mrcfile
import numpy as np

logger = logging.getLogger(__name__)


def save_mrc(vol,
             path,
             voxel_size: Union[int, float, Tuple, np.recarray] = None,
             origin: Union[int, float, Tuple, np.recarray] = None):
    """
    Save volumetric data to an MRC file, and set voxel size and origin if specified.

    Parameters
    ----------
    vol : array-like
        The volumetric data to save (typically a 3D numpy array).
    path : str
        The filepath where the MRC file will be saved.
    voxel_size : int, float, tuple, or np.recarray, optional
        Voxel size to set for the file header. Can be a single number or a 3-tuple (x, y, z),
        or a compatible numpy record. If not provided, defaults to 1.
    origin : int, float, tuple, or np.recarray, optional
        Origin to set for the file header. Can be a single number or a 3-tuple (x, y, z),
        or a compatible numpy record. If not provided, defaults to 0.

    See Also
    --------
    https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.mrcobject.MrcObject.voxel_size
    """
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)

        if voxel_size is not None:
            m.voxel_size = voxel_size

        if origin is not None:
            m.header.origin = origin


def ensure_zyx_vol(src_path, dst_path):
    """
    Ensure a volumetric MRC file uses ZYX (section, row, column) axis order and save to a new file.

    Converts an MRC file with arbitrary axis ordering (according to header's mapc, mapr, maps)
    into the standard ZYX axis order (MAPC=3, MAPR=2, MAPS=1). If the input file is already in 
    this order, optionally copies to a new location, otherwise reorders axes and adjusts 
    metadata for output.

    Parameters
    ----------
    src_path : str
        Path to the input MRC file.
    dst_path : str
        Path for the output MRC file. Can be same as `src_path` to overwrite, or different to copy/convert.

    Notes
    -----
    Standard ZYX order means that:
        - array axis 0 = sections (z), axis 1 = rows (y), axis 2 = columns (x)
        - m.header.mapc == 3, m.header.mapr == 2, m.header.maps == 1

    The voxel size, origin, and nstart metadata are preserved and/or correctly reordered
    when axes are modified.
    """
    with mrcfile.open(src_path) as m:
        # these entry values are numpy.ndarray, array index is [section(s), row(r), column(c)]
        crs_order = [int(m.header[k]) for k in ("mapc", "mapr", "maps")]
        src_order = crs_order[::-1]

        # z, y, x
        if src_order == [3, 2, 1]:
            if src_path != dst_path:
                shutil.copyfile(src_path, dst_path)
            return

        # data array dimension s (arr axis 0), r (arr axis 1), c (arr axis 2) should be src 3,2,1
        src_to_321 = [src_order.index(i + 1) for i in reversed(range(3))]
        new_data = m.data.copy()
        new_data = np.transpose(new_data, axes=src_to_321)

        voxel_size = m.voxel_size
        origin = m.header.origin
        nstart = m.nstart
        crs_to_123 = [crs_order.index(i + 1) for i in range(3)]
        new_nstart = [nstart.tolist()[axis] for axis in crs_to_123]
        new_nstart = np.rec.array(tuple(new_nstart), dtype=nstart.dtype)

    with mrcfile.new(dst_path, overwrite=True) as m:
        m.set_data(new_data)
        m.voxel_size = voxel_size
        m.header.origin = origin
        m.nstart = new_nstart
    return


def load_vol(file_path):
    with mrcfile.open(file_path) as m:
        vol = m.data.copy()
        apix = m.voxel_size.x
        if m.voxel_size.y != apix or m.voxel_size.z != apix:
            warnings.warn(f"Volume has different Apix along XYZ: {m.voxel_size}")
    logger.info(f"Volume shape {vol.shape}")
    logger.info(f"Volume apix {apix}")
    return vol, round(apix.item(), 4)


def load_images(file_path):
    with mrcfile.open(file_path) as m:
        images = m.data.copy()
        apix = m.voxel_size.x
        if m.voxel_size.y != apix or m.voxel_size.z != apix:
            warnings.warn(f"Image has different Apix along YX: {m.voxel_size}")
    logger.info(f"Total {len(images) if images.ndim == 3 else 1} images")
    logger.info(f"Image shape is {images.shape[-2:]}")
    logger.info(f"Image apix {apix}")
    return images, round(apix.item(), 4)
