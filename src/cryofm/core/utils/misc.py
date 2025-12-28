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

import numpy as np

import torch


def pretty_dict(x, precision=3, sci_mode=False):
    """
    Format a dictionary of various types into a pretty single-line string.

    Parameters
    ----------
    x : dict
        The input dictionary whose values can be of types such as `torch.Tensor`, `np.ndarray`, `float`, etc.
    precision : int, optional
        The number of decimal places for floating-point formatting (default is 3).
    sci_mode : bool, optional
        Whether to use scientific notation for floating-point numbers (default is False).

    Returns
    -------
    str
        A formatted string representing dictionary items in the format: `k1: v1 | k2: v2 | ...`.

    Examples
    --------
    >>> pretty_dict({
    ...     "k1": 1,
    ...     "k3": torch.tensor(3.1415926535)
    ... }, precision=3)
    'k1: 1 | k3: 3.141'
    """
    ret = None
    if isinstance(x, dict):
        ret = []
        for k, v in x.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                v = v.item()
            else:
                v = v
            if isinstance(v, float):
                # ret.append("{}: {:.3f}".format(k, v))
                if sci_mode:
                    float_formatter = "{}: {:.<PCS>e}".replace("<PCS>", str(precision))
                else:
                    float_formatter = "{}: {:.<PCS>f}".replace("<PCS>", str(precision))
                ret.append(float_formatter.format(k, v))
            else:
                ret.append("{}: {}".format(k, v))
        ret = " | ".join(ret)
    return ret
