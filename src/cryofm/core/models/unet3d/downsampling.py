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
# This file contains code modified from Hugging Face Diffusers
# (https://github.com/huggingface/diffusers), which is licensed under the Apache License 2.0.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.normalization import RMSNorm
from diffusers.utils.deprecation_utils import deprecate


class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 3D layer.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


def create_3d_kernel_einsum(kernel_1d):
    kernel_1d = kernel_1d.squeeze()
    return torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)


def upfirdn3d_native(
        tensor: torch.Tensor,
        kernel: torch.Tensor,
        up: int = 1,
        down: int = 1,
        pad: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    up_x = up_y = up_z = up
    down_x = down_y = down_z = down
    pad_x0 = pad_y0 = pad_z0 = pad[0]
    pad_x1 = pad_y1 = pad_z1 = pad[1]

    _, channel, in_d, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_d, in_h, in_w, 1)

    _, in_d, in_h, in_w, minor = tensor.shape
    kernel_d, kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_d, 1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_z - 1])
    out = out.view(-1, in_d * up_z, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out,
        [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)]
    )
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
          :,
          max(-pad_z0, 0) : out.shape[1] - max(-pad_z1, 0),
          max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
          :,
          ]

    out = out.permute(0, 4, 1, 2, 3)
    out = out.reshape \
        ([-1, 1, in_d * up_z + pad_z0 + pad_z1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1, 2]).view(1, 1, kernel_d, kernel_h, kernel_w)
    out = F.conv3d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        )
    out = out.permute(0, 2, 3, 4, 1)
    out = out[:, ::down_z, ::down_y, ::down_x, :]

    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d) // down_z + 1
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_d, out_h, out_w)


class FirDownsample3D(nn.Module):
    """A 3D FIR downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(
            self,
            channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            use_conv: bool = False,
            fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv3d_0 = nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_3d(
            self,
            hidden_states: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
            kernel: Optional[torch.Tensor] = None,
            factor: int = 2,
            gain: float = 1,
    ) -> torch.Tensor:
        """Fused `Conv3d()` followed by `downsample_3d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
            weight (`torch.Tensor`, *optional*):
                Weight tensor of the shape `[filterD, filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`torch.Tensor`, *optional*):
                FIR filter of the shape `[firD, firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, D // factor, H // factor, W // factor]` or `[N, D // factor, H // factor, W // factor, C]`,
                and same datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = create_3d_kernel_einsum(kernel)
        kernel /= torch.sum(kernel)

        kernel = kernel * gain

        if self.use_conv:
            _, _, convD, convH, convW = weight.shape
            pad_value = (kernel.shape[0] - factor) + (convW - 1)
            stride_value = [factor, factor, factor]
            upfirdn_input = upfirdn3d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = F.conv3d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn3d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            downsample_input = self._downsample_3d(hidden_states, weight=self.Conv3d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv3d_0.bias.reshape(1, -1, 1, 1, 1)
        else:
            hidden_states = self._downsample_3d(hidden_states, kernel=self.fir_kernel, factor=2)

        return hidden_states


# downsample/upsample layer used in k-upscaler, might be able to use FirDownsample3D/DirUpsample3D instead
class KDownsample3D(nn.Module):
    r"""A 3D K-downsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = kernel_1d.shape[1] // 2 - 1

        kernel_3d = create_3d_kernel_einsum(kernel_1d)
        self.register_buffer("kernel", kernel_3d, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.pad(inputs, (self.pad,) * 6, self.pad_mode)
        weight = inputs.new_zeros(
            [
                inputs.shape[1],
                inputs.shape[1],
                self.kernel.shape[0],
                self.kernel.shape[1],
                self.kernel.shape[2],
            ]
        )
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        kernel = self.kernel.to(weight)[None, :].expand(inputs.shape[1], -1, -1, -1)
        weight[indices, indices] = kernel
        return F.conv3d(inputs, weight, stride=2)


def downsample_3d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""Downsample3D a batch of 3D images with the given filter.
    Accepts a batch of 3D images of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`torch.Tensor`)
            Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firD, firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude.

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, D // factor, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = create_3d_kernel_einsum(kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn3d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output
