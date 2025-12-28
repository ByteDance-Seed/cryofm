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

import time
import argparse
import logging
import os.path as osp
from shutil import copyfile

import torch

from mmengine.logging.logger import MMFormatter
from mmengine import Config, DictAction, mkdir_or_exist
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import wait_for_everyone, broadcast

from .misc import mmengine_parser, kv_parser, _get_next_version, setup_logging, timestamp


def str2ord(s):
    return [ord(c) for c in s]


def ord2str(ascii_list):
    return ''.join([chr(int(i)) for i in ascii_list])


def broadcast_str(accelerator, string, from_process=0):
    device = accelerator.device
    length = len(string)
    # sync str length
    sync_length = broadcast(torch.tensor(length, device=device), from_process=from_process)

    tmp = str2ord(string)
    if sync_length > length:
        tmp.extend([0, ] * (sync_length - length))
    tmp = tmp[:sync_length]

    tmp = broadcast(torch.tensor(tmp, device=device), from_process=from_process)
    return ord2str(tmp.cpu().numpy().tolist())


# use hard-coded keys: exp_name, acl_config
def init_accelerator_w_cfg(
        override_mode="default",
        exp_prefix='',
        backup_list=None,
        work_dir_name="work_dirs",
        inplace=False):
    if override_mode == "default":
        args, cfg = mmengine_parser()
    elif override_mode == "key-value":
        args, cfg = kv_parser()
    else:
        raise NotImplementedError(f"{override_mode} mode not implemented.")

    if len(exp_prefix) and not exp_prefix.endswith('_'):
        exp_prefix += '_'

    if not hasattr(cfg, "exp_name") or cfg.exp_name == "" or cfg.exp_name is None:
        exp_name = exp_prefix + osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = cfg.exp_name

    mkdir_or_exist(work_dir_name)
    cfg.work_dir = osp.join(work_dir_name, exp_name)

    # init accelerator
    if hasattr(cfg, "acl_config"):
        accelerator = Accelerator(**cfg.acl_config)
    else:
        accelerator = Accelerator()

    if accelerator.is_main_process:
        if not inplace:
            version = _get_next_version(work_dir_name, exp_name)
            cfg.work_dir += f"_{version:02d}"

        # create work_dir
        mkdir_or_exist(osp.abspath(cfg.work_dir))
        # backup scripts
        if backup_list is not None:
            for p in backup_list:
                copyfile(p, osp.join(cfg.work_dir, osp.basename(p)))

        # dump cfg
        cfg.dump(osp.join(cfg.work_dir, "config.py"))

        # setup logging
        setup_logging(osp.join(cfg.work_dir, timestamp() + ".log"))

    wait_for_everyone()
    cfg.work_dir = broadcast_str(accelerator, cfg.work_dir)

    logger = get_logger(__name__)
    logger.info(f"working directory set to {cfg.work_dir}")
    logger.info("Config:\n" + cfg.pretty_text)
    return accelerator, cfg
