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
from pathlib import Path
from typing import Union


def get_last_ckpt(path):
    return get_ckpt_by_step(path, "last")


def get_ckpt_by_step(path: str, step: Union[int, str] = "last") -> str:
    """
    Get the checkpoint folder name by step number.

    Parameters
    ----------
    path : str
        Path to the root directory containing checkpoint folders.
    step : int or str, optional
        Step number of the checkpoint to retrieve, or "last" for the most recent one. Default is "last".

    Returns
    -------
    str
        Checkpoint folder name, e.g., '03333_0090000'.

    Raises
    ------
    AssertionError
        If step is not an integer or "last".
    Exception
        If the requested step is not found.

    """
    assert isinstance(step, int) or step == "last", "please pass the step number only!"

    p = Path(path)
    assert p.exists()

    steps, folders = [], []
    for ele in p.glob("*"):
        if ele.is_dir() and re.match(r"\d+\_\d+", str(ele.name)):
            steps.append(int(ele.name.split("_")[1]))
            folders.append(ele.name)

    steps, folders = zip(*sorted(zip(steps, folders), key=lambda x: x[0]))

    ckpt_path = None
    if step == "last":
        ckpt_path = folders[-1]
    else:
        for s, f in zip(steps, folders):
            if s == step:
                ckpt_path = f
                break
    if ckpt_path is None:
        raise Exception(f"step {step} is not found, available steps {steps}")

    return ckpt_path
