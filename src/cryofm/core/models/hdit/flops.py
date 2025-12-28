# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
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
#
# This file contains code modified from k-diffusion (https://github.com/crowsonkb/k-diffusion/),
# which is licensed under the MIT License.

from contextlib import contextmanager
import math
import threading


state = threading.local()
state.flop_counter = None


@contextmanager
def flop_counter(enable=True):
    try:
        old_flop_counter = state.flop_counter
        state.flop_counter = FlopCounter() if enable else None
        yield state.flop_counter
    finally:
        state.flop_counter = old_flop_counter


class FlopCounter:
    def __init__(self):
        self.ops = []

    def op(self, operator, *args, **kwargs):
        self.ops.append((operator, args, kwargs))

    @property
    def flops(self):
        flops = 0
        for operator, args, kwargs in self.ops:
            flops += operator(*args, **kwargs)
        return flops


def op(operator, *args, **kwargs):
    if getattr(state, "flop_counter", None):
        state.flop_counter.op(operator, *args, **kwargs)


def op_linear(x, weight):
    return math.prod(x) * weight[0]


def op_attention(q, k, v):
    *b, s_q, d_q = q
    *b, s_k, d_k = k
    *b, s_v, d_v = v
    return math.prod(b) * s_q * s_k * (d_q + d_v)


def op_natten(q, k, v, kernel_size):
    *q_rest, d_q = q
    *_, d_v = v
    return math.prod(q_rest) * (d_q + d_v) * kernel_size**2
