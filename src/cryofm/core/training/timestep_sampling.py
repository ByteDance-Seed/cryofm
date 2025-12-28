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

import torch


def sample_timesteps(n, process: str, strategy: str):
    assert process in ("fm", "ddpm")
    assert strategy in ("logit-normal", "uniform", "increase", "decrease")

    timesteps = None
    if process == "ddpm":
        if strategy == "logit-normal":
            timesteps = torch.clamp((torch.sigmoid(torch.randn(n)) * 1040 - 20).long(), 0, 999)
        elif strategy == "uniform":
            timesteps = torch.randint(0, 1000, (n,), dtype=torch.int32)
        elif strategy == "increase":
            weights = torch.linspace(1, 1000, steps=1000)
            prob = weights / weights.sum()
            timesteps = torch.multinomial(prob, n, replacement=True)
        elif strategy == "decrease":
            weights = torch.linspace(1000, 1, steps=1000)
            prob = weights / weights.sum()
            timesteps = torch.multinomial(prob, n, replacement=True)

    elif process == "fm":
        if strategy == "logit-normal":
            timesteps = torch.clamp((torch.sigmoid(torch.randn(n)) * 1040 - 20).long(), 0, 1000)
        elif strategy == "uniform":
            timesteps = torch.randint(0, 1001, (n,), dtype=torch.int32)
        elif strategy == "increase":
            weights = torch.linspace(1, 1000, steps=1001)
            prob = weights / weights.sum()
            timesteps = torch.multinomial(prob, n, replacement=True)
        elif strategy == "decrease":
            weights = torch.linspace(1000, 1, steps=1001)
            prob = weights / weights.sum()
            timesteps = torch.multinomial(prob, n, replacement=True)
    return timesteps
