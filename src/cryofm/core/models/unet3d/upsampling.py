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

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.normalization import RMSNorm
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.import_utils import is_torch_version

from .downsampling import create_3d_kernel_einsum


class Upsample3D(nn.Module):
    """A 3D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 3D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose3d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv3d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest3d_out_frame' op does not support bfloat16 until PyTorch 2.1
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            scale_factor = (
                2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-3:])])
            )
            if hidden_states.numel() * scale_factor > pow(2, 31):
                hidden_states = hidden_states.contiguous()

            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv3d_0(hidden_states)

        return hidden_states


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

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)])
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
        :,
        max(-pad_z0, 0) : out.shape[1] - max(-pad_z1, 0),
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 4, 1, 2, 3)
    out = out.reshape([-1, 1, in_d * up_z + pad_z0 + pad_z1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
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


class FirUpsample3D(nn.Module):
    """A 3D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`, optional):
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
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_3d(
        self,
        hidden_states: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        kernel: Optional[torch.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> torch.Tensor:
        """Fused `upsample_3d()` followed by `Conv3d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of the shape `[N, C, D, H, W]` or `[N, C, D, H, W`.
            weight (`torch.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`torch.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to nearest-neighbor upsampling.
            factor (`int`, *optional*): Integer upsampling factor (default: 2).
            gain (`float`, *optional*): Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, D * factor, H * factor, W * factor]` or `[N, D * factor, H * factor, W * factor, C]`, and same
                datatype as `hidden_states`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = create_3d_kernel_einsum(kernel)
        kernel /= torch.sum(kernel)

        kernel = kernel * (gain * (factor**3))

        if self.use_conv:
            convD = weight.shape[2]
            convH = weight.shape[3]
            convW = weight.shape[4]
            inC = weight.shape[1]

            pad_value = (kernel.shape[0] - factor) - (convW - 1)

            stride = (factor, factor, factor)
            # Determine data dimensions.
            output_shape = (
                (hidden_states.shape[2] - 1) * factor + convD,
                (hidden_states.shape[3] - 1) * factor + convH,
                (hidden_states.shape[4] - 1) * factor + convW,
            )
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convD,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convH,
                output_shape[2] - (hidden_states.shape[4] - 1) * stride[2] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0 and output_padding[2] >= 0
            num_groups = hidden_states.shape[1] // inC

            # Transpose weights.
            weight = torch.reshape(weight, (num_groups, -1, inC, convD, convH, convW))
            weight = torch.flip(weight, dims=[3, 4, 5]).permute(0, 2, 1, 3, 4, 5)
            weight = torch.reshape(weight, (num_groups * inC, -1, convD, convH, convW))

            inverse_conv = F.conv_transpose3d(
                hidden_states,
                weight,
                stride=stride,
                output_padding=output_padding,
                padding=0,
            )

            output = upfirdn3d_native(
                inverse_conv,
                torch.tensor(kernel, device=inverse_conv.device),
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn3d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            height = self._upsample_3d(hidden_states, self.Conv3d_0.weight, kernel=self.fir_kernel)
            height = height + self.Conv3d_0.bias.reshape(1, -1, 1, 1, 1)
        else:
            height = self._upsample_3d(hidden_states, kernel=self.fir_kernel, factor=2)

        return height


class KUpsample3D(nn.Module):
    r"""A 3D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1

        kernel_3d = create_3d_kernel_einsum(kernel_1d)
        self.register_buffer("kernel", kernel_3d, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.pad(inputs, ((self.pad + 1) // 2,) * 6, self.pad_mode)
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
        return F.conv_transpose3d(inputs, weight, stride=2, padding=self.pad * 2 + 1)


def upsample_3d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""Upsample3D a batch of 3D images with the given filter.
    Accepts a batch of 3D images of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`torch.Tensor`):
            Input tensor of the shape `[N, C, D, H, W]` or `[N, D, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firD, firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, D * factor, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = create_3d_kernel_einsum(kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * (factor**3))
    pad_value = kernel.shape[0] - factor
    output = upfirdn3d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output
