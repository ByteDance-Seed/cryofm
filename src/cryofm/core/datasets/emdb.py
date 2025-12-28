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

import os
import os.path as osp
from typing import Optional, List, Union, Callable, Mapping, Sequence, Any

import pandas as pd

# mmengine
from mmengine import Config
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.registry import DATASETS

from cryofm.core.utils.rotation_conversion import random_rotations

MAP_EXTENSIONS = (".mrc", ".map")

META_KEEP_INFO = {
    "nz": ["src_shape", ],
    "apix": ["src_apix", "apix"],
    "apix_class": ["apix_class", ],
    "resolution": ["resolution"],
    "contour_lvl": ["contour_lvl", ],
    "input_tag": ["input_tag", ],
    "output_tag": ["output_tag", ],
}


# data_root + ann_file: train.txt location, contain map relative path
# data_root + data_prefix['map']: map location prefix
@DATASETS.register_module()
class EMDBMapDataset(BaseDataset):
    METAINFO = dict(dataset_type='emdb_map_dataset', task_name='generating')

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        # to be compatible with base class
        if data_prefix is None:
            data_prefix = dict(map='')

        if pipeline is None:
            pipeline = list()

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def __len__(self):
        return self.num_maps()

    @force_full_init
    def num_maps(self):
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data_info["map_id"] = idx
        return self.pipeline(data_info)

    def _process_data_path(self, path, data_prefix_key):
        if path.startswith(os.sep):
            # Avoid absolute-path-like annotations
            path = path[1:]

        file_name = osp.splitext(osp.basename(path))[0]
        path = osp.join(self.data_prefix[data_prefix_key], path)
        return path, file_name

    def load_data_list(self) -> List[dict]:
        df = pd.read_csv(self.ann_file, dtype={'rel_path': str})

        data_list = []

        for index, row in df.iterrows():
            map_path, file_name  = self._process_data_path(row["rel_path"], data_prefix_key="map")

            meta = {
                "file_name": file_name,
                "map_path": map_path,
            }
            for k in META_KEEP_INFO:
                if k in row.keys():
                    for new_k in META_KEEP_INFO[k]:
                        meta[new_k] = row[k]

            for k in ("mean", "std", ):
                if k in row.keys():
                    meta[k] = float(row[k])

            data_list.append(meta)
        return data_list


# this is used for create 2D projections
@DATASETS.register_module()
class EMDBMapPoseDataset(EMDBMapDataset):
    METAINFO = dict(dataset_type='emdb_pose_dataset', task_name='generating')

    def __init__(self,
                 num_poses: int = 5000,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        # custom arguments
        self.num_poses = num_poses

        # init poses,
        self.poses = None

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def __len__(self):
        return super().__len__() * self.num_poses

    def set_rand_poses(self):
        # may need sync
        self.poses = random_rotations(self.num_poses)

    def full_init(self):
        self.set_rand_poses()
        super().full_init()

    def prepare_data(self, idx) -> Any:
        map_id = idx // self.num_poses
        pose_id = idx % self.num_poses
        data_info = self.get_data_info(map_id)
        data_info["map_id"] = map_id
        data_info['pose_id'] = pose_id
        data_info["rot_mat"] = self.poses[pose_id]
        return self.pipeline(data_info)


@DATASETS.register_module()
class EMDBPairDataset(EMDBMapDataset):
    METAINFO = dict(dataset_type='emdb_pair_dataset', task_name='generating')

    def load_data_list(self) -> List[dict]:
        df = pd.read_csv(self.ann_file, dtype={'rel_path1': str, 'rel_path2': str})

        data_list = []

        for index, row in df.iterrows():
            rel_path1, file_name1 = self._process_data_path(row["rel_path1"], data_prefix_key="map")
            rel_path2, file_name2 = self._process_data_path(row["rel_path2"], data_prefix_key="map")

            meta = {
                "file_name1": file_name1,
                "rel_path1": rel_path1,
                "file_name2": file_name2,
                "rel_path2": rel_path2,
            }
            for k in META_KEEP_INFO:
                if k in row.keys():
                    for new_k in META_KEEP_INFO[k]:
                        meta[new_k] = row[k]

            for k in ("mean1", "std1", "mean2", "std2"):
                if k in row.keys():
                    meta[k] = float(row[k])

            data_list.append(meta)
        return data_list
