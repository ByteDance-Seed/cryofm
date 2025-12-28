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

import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from . import flops
from . import flags
from .utils import apply_wd, zero_init, tag_module


@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


class Linear(nn.Linear):
    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, None, :] + 1, self.eps)


# Mapping network
class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting
class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2, 2)):
        super().__init__()
        self.d = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = apply_wd(Linear(in_features * self.d * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        # Merge two blocks, dimension becomes 8 times the original
        x = rearrange(x, "... (d nd) (h nh) (w nw) e -> ... d h w (nd nh nw e)", nd=self.d, nh=self.h, nw=self.w)
        # Then perform a projection
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2, 2)):
        super().__init__()
        self.d = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = apply_wd(Linear(in_features, out_features * self.d * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... d h w (nd nh nw e) -> ... (d nd) (h nh) (w nw) e", nd=self.d, nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2, 2)):
        super().__init__()
        self.d = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = apply_wd(Linear(in_features, out_features * self.d * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... d h w (nd nh nw e) -> ... (d nd) (h nh) (w nw) e", nd=self.d, nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# Embeddings and NTK Fourier transform features
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1., use_fp32=False):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)
        self.use_fp32 = use_fp32

    def forward(self, inputs):
        if self.use_fp32:
            # bsz in 1 * in out -> bsz in out
            f = (2 * math.pi * inputs.float()[:, :, None] * self.weight.T).sum(1)
        else:
            f = 2 * math.pi * inputs @ self.weight.T
        ret = torch.cat([f.cos(), f.sin()], dim=-1)
        return ret

