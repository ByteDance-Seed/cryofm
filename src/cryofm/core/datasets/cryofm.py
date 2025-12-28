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
import re
import os.path as osp
from itertools import product
from typing import List, Optional, Callable, Dict, Union

import mrcfile
import numpy as np
import pandas as pd
from tqdm import trange

import torch
from torch.utils.data import Dataset

from mmengine.dataset import Compose


class CryoFMMapDataset(Dataset):
    _map_key_prefix = "map_path"

    def __init__(
            self,
            data_info_path: str,
            data_dir: str,
            input_tag: Optional[int] = None,
            output_tag: Optional[int] = None,
            transform: List[Union[dict, Callable]] = None,
            seed: int = 42,
            data_exclude_path: str = None
    ):
        # the two CSVs contain data with matching row indices:
        self.data_info_df = pd.read_csv(data_info_path)
        self.map_keys = [k for k in self.data_info_df.columns.tolist() if k.startswith(self._map_key_prefix)]
        ori_len = len(self.data_info_df)
        if data_exclude_path is not None:
            specfic_key = self.map_keys[0]
            data_exclude_df = pd.read_csv(data_exclude_path)
            exclude_values = data_exclude_df.iloc[:, 0].tolist()
            pattern = "|".join(map(re.escape, exclude_values))
            mask = self.data_info_df[specfic_key].str.contains(pattern, na=False)
            self.data_info_df = self.data_info_df[~mask]
            self.data_info_df = self.data_info_df.reset_index(drop=True)
            # print(f"filter over {ori_len}------>{len(self.data_info_df)}")

        self.num_items = len(self.data_info_df)
        self.data_dir = data_dir
        self.input_tag = input_tag
        self.output_tag = output_tag
        self.transform = Compose(transform)
        self.seed = seed

        self.mrc_handlers: Dict[str, np.memmap] = {}

        # initialize data and pre-sample indices for first epoch
        self._initialize_data()

    def _initialize_data(self):
        """Initialize memory-mapped handlers and load patch indices"""
        for idx in range(self.num_items):
            # map
            for k in self.map_keys:
                rel_path = self.data_info_df.loc[idx, k]
                abs_path = osp.join(self.data_dir, rel_path)
                self.mrc_handlers[rel_path] = mrcfile.mmap(abs_path, mode='r').data

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        results = {}
        data_info_row = self.data_info_df.iloc[idx]

        for k in self.map_keys:
            map_rel_path = data_info_row[k]
            results[k] = self.mrc_handlers[map_rel_path].copy().astype(np.float32)

        results.update({index: value for index, value in data_info_row.items() if index not in self.map_keys})

        results["map_id"] = idx

        if self.input_tag is not None:
            results["input_tag"] = self.input_tag
        if self.output_tag is not None:
            results["output_tag"] = self.output_tag

        return self.transform(results)

    def __del__(self):
        """Clean up memory-mapped files"""
        for handler in self.mrc_handlers.values():
            if hasattr(handler, 'flush'):
                handler.flush()


class CryoFMMapCropDataset(Dataset):
    """Dataset for CryoFM map training that requires:

    - A data info CSV file specifying relative paths to density maps
    - A crop info CSV file specifying relative paths to numpy arrays containing valid crop indices
    - Data directory containing density map files
    - Crop directory containing numpy arrays with valid crop indices
    """
    _map_key_prefix = "map_path"
    _crop_key_prefix = "crop_path"

    def __init__(
            self,
            data_info_path: str,
            crop_info_path: str,
            data_dir: str,
            crop_dir: str,
            blank_crop_dir: str,
            patches_per_map: int,
            input_tag: Optional[int] = None,
            output_tag: Optional[int] = None,
            transform: List[Union[dict, Callable]] = None,
            seed: int = 42,
            position_index_scale: int = 50,
            crop_size: int = 64,
            split_ratio: float = 0.7,
            data_exclude_path: str = None
    ):
        # the two CSVs contain data with matching row indices:
        self.data_info_df = pd.read_csv(data_info_path)
        ori_len = len(self.data_info_df)
        self.crop_info_df = pd.read_csv(crop_info_path)    

        self.map_keys = [k for k in self.data_info_df.columns.tolist() if k.startswith(self._map_key_prefix)]
        # if you need to exclude some entries, use an extra csv to list them
        if data_exclude_path is not None:
            specfic_key = self.map_keys[0]
            data_exclude_df = pd.read_csv(data_exclude_path)
            exclude_values = data_exclude_df.iloc[:, 0].tolist()
            pattern = "|".join(map(re.escape, exclude_values))
            mask = self.data_info_df[specfic_key].str.contains(pattern, na=False)
            self.data_info_df = self.data_info_df[~mask]
            self.data_info_df = self.data_info_df.reset_index(drop=True)
            self.crop_info_df = self.crop_info_df[~mask]
            self.crop_info_df = self.crop_info_df.reset_index(drop=True)
            # print(f"filter over {ori_len}------>{len(self.data_info_df)}")

        self.num_items = len(self.data_info_df)

        self.crop_key = self._crop_key_prefix
        self.data_dir = data_dir
        self.crop_dir = crop_dir
        self.blank_crop_dir = blank_crop_dir
        self.patches_per_map = patches_per_map
        self.input_tag = input_tag
        self.output_tag = output_tag
        self.transform = Compose(transform)
        self.seed = seed
        self.position_index_scale = position_index_scale
        self.crop_size = crop_size
        self.split_ratio = split_ratio

        # store mmap handlers and patch indices for each MRC
        self.mrc_handlers: Dict[str, np.memmap] = {}
        self.crop_handlers: Dict[str, np.memmap] = {}
        self.blank_crop_handlers: Dict[str, np.memmap] = {}

        # add new attributes for pre-sampled indices
        self.epoch = 0
        self.pre_sampled_indices: Dict[int, np.ndarray] = {}
        self.blank_pre_sampled_indices: Dict[int, np.ndarray] = {}

        # add new attribute to store available patches count
        self.available_patches: Dict[int, int] = {}
        self.blank_available_patches: Dict[int, int] = {}

        # initialize data and pre-sample indices for first epoch
        self._initialize_data()
        self._sample_epoch_indices()

    def _initialize_data(self):
        """Initialize memory-mapped handlers and load patch indices"""

        root_dir = self.blank_crop_dir

        for idx in trange(self.num_items):
            # map
            for k in self.map_keys:
                map_rel_path = self.data_info_df.loc[idx, k]
                abs_path = osp.join(self.data_dir, map_rel_path)
                self.mrc_handlers[map_rel_path] = mrcfile.mmap(abs_path, mode='r').data

            # crop index
            rel_path = self.crop_info_df.loc[idx, self.crop_key]
            abs_path = osp.join(self.crop_dir, rel_path)
            index_data = np.load(abs_path, mmap_mode='r')
            blank_index_data = np.load(os.path.join(root_dir, rel_path), mmap_mode='r')

            self.blank_crop_handlers[rel_path] = blank_index_data
            self.crop_handlers[rel_path] = index_data

            self.available_patches[idx] = len(self.crop_handlers[rel_path])
            self.blank_available_patches[idx] = len(self.blank_crop_handlers[rel_path])

    def _sample_epoch_indices(self):
        """
        Pre-sample patch indices for each MRC file at the start of epoch
        Handles cases where available patches < patches_per_mrc by repeating indices

        """
        rng = np.random.RandomState(self.epoch + self.seed)

        # non-blank patch
        num_patch_non_blank = int(self.patches_per_map * self.split_ratio)
        for item_idx, num_valid_patches in self.available_patches.items():
            if num_valid_patches >= num_patch_non_blank:
                sampled_indices = rng.randint(0, num_valid_patches, size=num_patch_non_blank)
            else:
                repeats = num_patch_non_blank // num_valid_patches + 1
                all_indices = np.tile(np.arange(num_valid_patches), repeats)
                rng.shuffle(all_indices)
                sampled_indices = all_indices[:num_patch_non_blank]

            self.pre_sampled_indices[item_idx] = sampled_indices

        # blank patch
        num_patch_blank = self.patches_per_map - num_patch_non_blank
        for item_idx, num_valid_patches in self.blank_available_patches.items():
            if num_valid_patches >= num_patch_blank:
                blank_sampled_indices = rng.randint(0, num_valid_patches, size=num_patch_blank)
            else:
                repeats = num_patch_blank // num_valid_patches + 1
                all_indices = np.tile(np.arange(num_valid_patches), repeats)
                rng.shuffle(all_indices)
                blank_sampled_indices = all_indices[:num_patch_blank]

            self.blank_pre_sampled_indices[item_idx] = blank_sampled_indices

    def set_epoch(self, epoch: int):
        """
        Set epoch number and resample indices
        Should be called at the start of each epoch
        """
        if self.epoch != epoch:
            self.epoch = epoch
            self._sample_epoch_indices()

    def get_permutation(self, axis_domains):
        shape0 = np.arange(axis_domains[0])
        shape1 = np.arange(axis_domains[1])
        shape2 = np.arange(axis_domains[2])
        combinations = np.array(list(product(shape0, shape1, shape2)))
        return combinations

    # def rows_diff(self, a, b):
    #     a_view = a.view(dtype=[('', a.dtype)] * a.shape[1]).reshape(-1)
    #     b_view = b.view(dtype=[('', b.dtype)] * b.shape[1]).reshape(-1)
    #     diff = np.setdiff1d(a_view, b_view)
    #
    #     return diff.view(a.dtype).reshape(-1, a.shape[1])

    def __len__(self) -> int:
        """Return total number of samples (patches_per_mrc * number of MRCs)"""
        return self.patches_per_map * self.num_items

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a patch or patch pair using pre-sampled indices
        """
        # calculate which MRC and which patch to use
        item_idx = idx // self.patches_per_map
        random_patch_idx = idx % self.patches_per_map
        start_point = np.array([0., 0., 0.])
        crop_rel_path = self.crop_info_df.loc[item_idx, self.crop_key]
        threshold = int(self.patches_per_map * self.split_ratio)
        if random_patch_idx < threshold:
            patch_idx = self.pre_sampled_indices[item_idx][random_patch_idx]
            index_ini, index_fin = np.split(self.crop_handlers[crop_rel_path][patch_idx], 2)
        else:
            patch_idx = self.blank_pre_sampled_indices[item_idx][random_patch_idx - threshold]
            index_ini, index_fin = np.split(self.blank_crop_handlers[crop_rel_path][patch_idx], 2)

        results = {}
        data_info_row = self.data_info_df.iloc[item_idx]

        for k in self.map_keys:
            map_rel_path = data_info_row[k]
            results[k] = self.mrc_handlers[map_rel_path][tuple(map(slice, index_ini, index_fin))].copy().astype(
                np.float32)


        results.update({index: value for index, value in data_info_row.items() if index not in self.map_keys})

        results["map_id"] = item_idx
        if self.input_tag is not None:
            results["input_tag"] = self.input_tag
        if self.output_tag is not None:
            results["output_tag"] = self.output_tag

        return self.transform(results)

    def __del__(self):
        """Clean up memory-mapped files"""
        for handlers in (self.mrc_handlers, self.crop_handlers):
            for handler in handlers.values():
                if hasattr(handler, 'flush'):
                    handler.flush()
