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
# specifically from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
# k-diffusion is licensed under the MIT License.

from dataclasses import dataclass
from functools import reduce
import math
from typing import Union

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from mmengine import Config, ConfigDict

from . import layers
from . import flags
from . import flops
from .axial_rope import make_axial_pos
from .utils import (apply_wd, zero_init, filter_params, tag_module, checkpoint, downscale_pos,
                    scale_for_cosine_sim, scale_for_cosine_sim_qkv)

try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Rotary position embeddings
# Here maybe an issue with the implementation of apply_rotary_emb in k-diffusion, the trick that can be derived
# from the formula requires reversing x, but for now, the original implementation is maintained.
@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    # d here contains partial (x, y, z)
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d: d * 2], x[..., d * 2:]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d: d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE3D(nn.Module):
    def __init__(self, dim, n_heads, frac=4):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        # The "4" here represents that each axial component accounts for 1/4 of the head dimension,
        # and the 3D version will maintain this design for the time being. If you modify this part,
        # you will also need to change the calculation method in apply_rotary_emb.
        freqs = torch.linspace(log_min, log_max, n_heads * (dim // frac) + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // frac, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_d = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_h = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 2:3] * self.freqs.to(pos.dtype)
        return torch.cat((theta_d, theta_h, theta_w), dim=-1)


# Transformer layers
def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock3D(nn.Module):
    def __init__(self, d_model, d_head, cond_features=None, dropout=0.0, use_fp32_rope=False):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        if cond_features is None:
            self.norm = layers.RMSNorm(d_model)
        else:
            self.norm = layers.AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(layers.Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        # Divide embedding dim to 3 axis separately
        self.pos_emb = AxialRoPE3D(d_head // 3, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(layers.Linear(d_model, d_model, bias=False)))
        self.use_fp32_rope = use_fp32_rope

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond=None):
        skip = x
        if cond is None:
            x = self.norm(x)
        else:
            x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        if not self.use_fp32_rope:
            pos = rearrange(pos, "... d h w e -> ... (d h w) e").to(qkv.dtype)
        else:
            pos = rearrange(pos, "... d h w e -> ... (d h w) e")
        theta = self.pos_emb(pos)
        # print(self.use_fp32_rope, pos.dtype, theta.dtype)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n d h w (t nh e) -> n (d h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (d h w) nh e -> n d h w (nh e)", d=skip.shape[-4], h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n d h w (t nh e) -> t n nh (d h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (d h w) e -> n d h w (nh e)", d=skip.shape[-4], h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class CrossAttentionBlock3D(nn.Module):
    def __init__(self, d_model, d_head, cond_features=None, cross_attention_dim=768, dropout=0.0, use_fp32_rope=False):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        if cond_features is None:
            self.norm = layers.RMSNorm(d_model)
        else:
            self.norm = layers.AdaRMSNorm(d_model, cond_features)
        self.q_proj = apply_wd(layers.Linear(d_model, d_model, bias=False))
        self.kv_proj = apply_wd(layers.Linear(cross_attention_dim, d_model * 2, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        # Divide embedding dim to 3 axis separately
        self.pos_emb = AxialRoPE3D(d_head // 3, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(layers.Linear(d_model, d_model, bias=False)))
        self.use_fp32_rope = use_fp32_rope

    def forward(self, x, pos, cond=None, encoder_hidden_states=None):
        skip = x
        if cond is None:
            x = self.norm(x)
        else:
            x = self.norm(x, cond)
        q = self.q_proj(x)
        kv = self.kv_proj(encoder_hidden_states)
        qkv = torch.concat((q, kv), dim=-1)
        if not self.use_fp32_rope:
            pos = rearrange(pos, "... d h w e -> ... (d h w) e").to(qkv.dtype)
        else:
            pos = rearrange(pos, "... d h w e -> ... (d h w) e")
        theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n d h w (t nh e) -> n (d h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (d h w) nh e -> n d h w (nh e)", d=skip.shape[-4], h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n d h w (t nh e) -> t n nh (d h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (d h w) e -> n d h w (nh e)", d=skip.shape[-4], h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock3D(nn.Module):
    def __init__(self, d_model, d_head, cond_features=None, kernel_size=7, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        if cond_features is None:
            self.norm = layers.RMSNorm(d_model)
        else:
            self.norm = layers.AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(layers.Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        # Divide embedding dim to 3 axis separately
        self.pos_emb = AxialRoPE3D(d_head // 3, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(layers.Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond=None):
        skip = x
        if cond is None:
            x = self.norm(x)
        else:
            x = self.norm(x, cond)
            
        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, "n d h w (t nh e) -> t n d h w nh e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            theta = self.pos_emb(pos)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na3d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n d h w nh e -> n d h w (nh e)")
        else:
            q, k, v = rearrange(qkv, "n d h w (t nh e) -> t n nh d h w e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None, None], 1e-6)
            theta = self.pos_emb(pos).movedim(-2, -5)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na3d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1).to(v.dtype)
            x = natten.functional.na3d_av(a, v, self.kernel_size, 1)
            x = rearrange(x, "n nh d h w e -> n d h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features=None, dropout=0.0):
        super().__init__()
        if cond_features is None:
            self.norm = layers.RMSNorm(d_model)
        else:
            self.norm = layers.AdaRMSNorm(d_model, cond_features)
        self.up_proj = apply_wd(layers.LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(layers.Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        if cond is None:
            x = self.norm(x)
        else:
            x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features=None, cross_attention_dim=None, dropout=0.0, use_fp32_rope=False):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.self_attn = SelfAttentionBlock3D(d_model, d_head, cond_features=cond_features, dropout=dropout, use_fp32_rope=use_fp32_rope)
        if cross_attention_dim is None:
            self.cross_attn = None
        else:
            self.cross_attn = CrossAttentionBlock3D(d_model, d_head, cond_features=cond_features,
                                                    cross_attention_dim=cross_attention_dim, dropout=dropout, use_fp32_rope=use_fp32_rope)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond=None, encoder_hidden_states=None):
        x = checkpoint(self.self_attn, x, pos, cond)
        if self.cross_attention_dim is not None and encoder_hidden_states is not None:
            x = checkpoint(self.cross_attn, x, pos, cond, encoder_hidden_states)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features=None, kernel_size=7, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock3D(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond=None):
        x = checkpoint(self.self_attn, x, pos, cond)
        x = checkpoint(self.ff, x, cond)
        return x


class NoAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, cond_features=None, dropout=0.0):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)

    def forward(self, x, pos, cond=None):
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


# Configuration
@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


class ImageTransformerDenoiserModelV2(nn.Module):
    def __init__(self, levels, mapping, in_channels, out_channels, patch_size, num_classes=0, mapping_cond_dim=0, use_fp32_time_emb=False, use_fp32_rope=False):
        super().__init__()
        self.num_classes = num_classes

        self.patch_in = layers.TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width, use_fp32=use_fp32_time_emb)
        self.time_in_proj = layers.Linear(mapping.width, mapping.width, bias=False)
        # self.aug_emb = layers.FourierFeatures(9, mapping.width, use_fp32=use_fp32_time_emb)
        # self.aug_in_proj = layers.Linear(mapping.width, mapping.width, bias=False)
        # self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        # self.mapping_cond_in_proj = layers.Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        # self.mapping = tag_module(layers.MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head,
                                                                 cond_features=mapping.width, dropout=spec.dropout,
                                                                 use_fp32_rope=use_fp32_rope)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head,
                                                                       cond_features=mapping.width,
                                                                       kernel_size=spec.self_attn.kernel_size,
                                                                       dropout=spec.dropout)
            # elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
            #     layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head,
            #                                                             mapping.width, spec.self_attn.window_size,
            #                                                             i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff,
                                                                      cond_features=mapping.width,
                                                                      dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([layers.TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([layers.TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = layers.RMSNorm(levels[0].width)
        self.patch_out = layers.TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, timesteps=None):
        # Patching
        x = x.movedim(1, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-4], x.shape[-3], x.shape[-2], device=x.device).view(
            x.shape[-4], x.shape[-3], x.shape[-2], 3)

        # Mapping network
        # if class_cond is None and self.class_emb is not None:
        #     raise ValueError("class_cond must be specified if num_classes > 0")
        # if mapping_cond is None and self.mapping_cond_in_proj is not None:
        #     raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        time_emb = self.time_in_proj(self.time_emb(timesteps[..., None]))
        cond = time_emb

        # c_noise = torch.log(sigma) / 4
        # time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        # aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        # aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        # class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        # mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        # cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # Un-patching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, 1)

        return x


def build_model_config(config: Union[Config, ConfigDict, dict]):
    defaults = {
        'mapping_width': 256,
        'mapping_depth': 2,
        'mapping_d_ff': None,
        'mapping_cond_dim': 0,
        'mapping_dropout_rate': 0.,
        'd_ffs': None,
        'self_attns': None,
        'dropout_rate': None,
        'augment_wrapper': False,
        'skip_stages': 0,
        'has_variance': False,
    }
    if isinstance(config, Config):
        config.merge_from_dict(defaults)
    elif isinstance(config, ConfigDict):
        config.merge(defaults)
    else:
        config.update(defaults)

    if "use_fp32_time_emb" not in config:
        config["use_fp32_time_emb"] = False
    if "use_fp32_rope" not in config:
        config["use_fp32_rope"] = False

    if not config['mapping_d_ff']:
        config['mapping_d_ff'] = config['mapping_width'] * 3
    if not config['d_ffs']:
        d_ffs = []
        for width in config['widths']:
            d_ffs.append(width * 3)
        config['d_ffs'] = d_ffs
    if not config['self_attns']:
        self_attns = []
        default_neighborhood = {"type": "neighborhood", "d_head": 64, "kernel_size": 7}
        default_global = {"type": "global", "d_head": 64}
        for i in range(len(config['widths'])):
            self_attns.append(default_neighborhood if i < len(config['widths']) - 1 else default_global)
        config['self_attns'] = self_attns
    if config['dropout_rate'] is None:
        config['dropout_rate'] = [0.0] * len(config['widths'])
    elif isinstance(config['dropout_rate'], float):
        config['dropout_rate'] = [config['dropout_rate']] * len(config['widths'])

    assert len(config['widths']) == len(config['depths'])
    assert len(config['widths']) == len(config['d_ffs'])
    assert len(config['widths']) == len(config['self_attns'])
    assert len(config['widths']) == len(config['dropout_rate'])
    levels = []
    for depth, width, d_ff, self_attn, dropout in zip(config['depths'], config['widths'], config['d_ffs'],
                                                      config['self_attns'], config['dropout_rate']):
        if self_attn['type'] == 'global':
            self_attn = GlobalAttentionSpec(self_attn.get('d_head', 64))
        elif self_attn['type'] == 'neighborhood':
            self_attn = NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        # elif self_attn['type'] == 'shifted-window':
        #     self_attn = ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
        elif self_attn['type'] == 'none':
            self_attn = NoAttentionSpec()
        else:
            raise ValueError(f'unsupported self attention type {self_attn["type"]}')
        levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
    mapping = MappingSpec(config['mapping_depth'], config['mapping_width'], config['mapping_d_ff'],
                          config['mapping_dropout_rate'])
    return levels, mapping, config


def make_model(config: Union[Config, ConfigDict, dict]):
    levels, mapping, config = build_model_config(config)

    assert config["type"] == "image_transformer_v2", f"This function now only supports image_transformer_v2"

    model = ImageTransformerDenoiserModelV2(
        levels=levels,
        mapping=mapping,
        in_channels=config['input_channels'],
        out_channels=config['input_channels'],
        patch_size=config['patch_size'],
        num_classes=0,  # dummy arg
        mapping_cond_dim=config['mapping_cond_dim'],
        use_fp32_time_emb=config['use_fp32_time_emb'],
        # use_fp32_rope=config['use_fp32_rope']
    )
    return model


# if __name__ == '__main__':
#     # Use case
#     test_config = Config({
#         "type": "image_transformer_v2",
#         "input_channels": 1,
#         "input_size": [64, 64],
#         "patch_size": [4, 4, 4],
#         "depths": [2, 2, 4],
#         "widths": [128, 256, 512],
#         "self_attns": [
#             {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
#             {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
#             {"type": "global", "d_head": 64}
#         ]
#     })
#
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
#     test_model = make_model(test_config).to(device)
#     test_input = torch.randn(1, 1, 64, 64, 64, device=device)
#     test_output = test_model(test_input, torch.ones(1, device=device))
#     print(test_output.shape)
