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

import re
import os
import time
import argparse
import logging
import os.path as osp
from pathlib import Path
from shutil import copyfile
from typing import Union

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import wait_for_everyone, broadcast
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.logging.logger import MMFormatter


def _get_next_version(root_dir, dir_name_prefix):
    dir_list = [d for d in os.listdir(root_dir) if d.startswith(dir_name_prefix)]

    new_id = 0

    if len(dir_list) > 0:
        exist_ids = [int(i.split('_')[-1]) for i in dir_list if len(i.split('_')) > 0 and i.split('_')[-1].isdigit()]
        if len(exist_ids) > 0:
            new_id = max(exist_ids) + 1
    return new_id


def mmengine_parser():
    parser = argparse.ArgumentParser(description='Init with mmengine style configuration.')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    return args, cfg


def flatten_nested_dict(nested: Union[dict, Config]) -> dict:
    """
        Inputs:
            {"a": {"b": 1}, "c": 2}
        Returns:
            {"a.b": 1, "c": 2}
    """
    stack = list(nested.items())
    ans = {}
    while stack:
        key, val = stack.pop()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                stack.append((f"{key}.{sub_key}", sub_val))
        else:
            ans[key] = val
    return ans


def init_mmengine_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        flatten_cfg = flatten_nested_dict(cfg)
        for ele in args.cfg_options.keys():
            if ele not in flatten_cfg:
                raise Exception(f"{ele} not in config!")
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def kv_parser():
    """
    Use case:
        args, unknown = parser.parse_known_args(["config", "--data", "1", "--flag", "false"])
        ...
        -> Namespace(cfg_options={'data': 1, 'flag': False}, config='config')
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', help='config file path')
    args, unknown = parser.parse_known_args()
    assert len(unknown) % 2 == 0, "extra arguments not paired"
    cfg_options = {}
    while len(unknown) != 0:
        k = unknown.pop(0)
        v = unknown.pop(0)
        assert k.startswith("--") and not v.startswith("--")
        cfg_options[k[2:]] = DictAction._parse_iterable(v)
    args.cfg_options = cfg_options

    cfg = init_mmengine_config(args)
    return args, cfg


def setup_logging(filename=None, log_level="INFO"):
    if isinstance(log_level, str):
        log_level = [log_level, log_level]
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(MMFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
    console_handler.setLevel(log_level[0])
    handlers = [console_handler, ]

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(MMFormatter(color=False, datefmt='%m/%d %H:%M:%S'))
        file_handler.setLevel(log_level[1])
        handlers.append(file_handler)

    logging.basicConfig(
        level="INFO",
        handlers=handlers,
    )


def timestamp(fmt='%Y%m%d_%H%M%S'):
    return time.strftime(fmt, time.localtime())


def count_params(model, verbose=False, print_fn=print):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print_fn(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params
