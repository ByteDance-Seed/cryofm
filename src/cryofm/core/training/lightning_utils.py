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

import functools
import os.path as osp
from pathlib import Path
from shutil import copyfile

import torch

from lightning.fabric.utilities import rank_zero_only
from mmengine import mkdir_or_exist, MMLogger, Config, print_log

from cryofm.core.utils.ckpt_loading import get_last_ckpt
from cryofm.core.utils.dynamic_import import import_module_from_path

from .misc import _get_next_version, mmengine_parser, kv_parser, timestamp

log_to_current = rank_zero_only(functools.partial(print_log, logger="current"))


def merge_step_outputs(outputs):
    ks = outputs[0].keys()
    res = {}
    for k in ks:
        res[k] = torch.concat([out[k] for out in outputs], dim=0)
    return res


def squeeze_dict_outputs_1st_dim(outputs):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k].flatten(start_dim=0, end_dim=1)
    return res


def filter_outputs_by_indices(outputs, indices):
    res = {}
    for k in outputs.keys():
        res[k] = outputs[k][indices]
    return res


def get_1st_unique_indices(t):
    _, idx, counts = torch.unique(t, dim=None, sorted=True, return_inverse=True, return_counts=True)
    # ind_sorted: the index corresponding to same unique value will be grouped by these indices
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((cum_sum.new_tensor([0, ]), cum_sum[:-1]))
    first_idx = ind_sorted[cum_sum]
    return first_idx


#
def init_pl_w_cfg(override_mode="default",
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
    rank = getattr(rank_zero_only, "rank", None)

    if rank is None:
        raise RuntimeError("There must be some error with lightning.")
    if rank == 0:
        if not inplace:
            version = _get_next_version(work_dir_name, exp_name)
            cfg.work_dir += f"_{version:02d}"

        mkdir_or_exist(osp.abspath(cfg.work_dir))
        # backup scripts
        if backup_list is not None:
            for p in backup_list:
                copyfile(p, osp.join(cfg.work_dir, osp.basename(p)))

        # dump cfg
        cfg.dump(osp.join(cfg.work_dir, "config.py"))

        # log
        log_file = osp.join(cfg.work_dir, f'{timestamp()}.log')
        logger = MMLogger.get_instance(name="MMLogger", logger_name="cryostar", log_file=log_file)
        logger.info("Config:\n" + cfg.pretty_text)
    else:
        _ = MMLogger.get_instance(name="MMLogger", logger_name="cryostar_fork")
    return cfg


def autoload_model(model_dir, ckpt_dir, model_file_name, model_name, config_file_name="config.py", ckpt_file_name="ckpt.pt"):
    p = Path(model_dir)
    assert p.exists()

    model_path = p / model_file_name
    cfg_path = p / config_file_name

    if ckpt_dir != "last":
        assert (p / ckpt_dir).exists()
        ckpt_path = p / ckpt_dir / ckpt_file_name
    else:
        ckpt_path = p / get_last_ckpt(p) / ckpt_file_name
    print(f"load checkpoint from {ckpt_path}")
    assert ckpt_path.exists()

    print(f"load model spec from {model_path}")
    print(f"load config from {cfg_path}")

    module = import_module_from_path(osp.abspath(model_path))
    model = getattr(module, model_name)
    model_instance = model.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        cfg=Config.fromfile(cfg_path)
    )

    return model_instance
