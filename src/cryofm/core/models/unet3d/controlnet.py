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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.utils import BaseOutput, logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

from .unet import UNet3DModel
from .up_blocks import get_up_block
from .down_blocks import get_down_block
from .mid_blocks import get_mid_block, UNetMidBlock3D

logger = logging.get_logger(__name__)


@dataclass
class ControlNet3DOutput(BaseOutput):
    """
    The output of [`ControlNet3DModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, depth // resolution, height // resolution, width // resolution)`.
        mid_block_res_sample (`torch.Tensor`):
            The activation of the middle block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, depth // lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNet3DConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 "latent images" for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNet3DModel(ModelMixin, ConfigMixin):
    """
    A 3D ControlNet model.

    Args:
        sample_size (`Union[int, Tuple[int, int]]`, *optional*, defaults to `None`):
            The size of the input image.
        in_channels (`int`, defaults to 3):
            The number of channels in the input sample.
        out_channels (`int`, defaults to 3):
            The number of channels in the output sample.
        center_input_sample (`bool`, defaults to `False`):
            Whether to center the input sample.
        time_embedding_type (`str`, defaults to `"positional"`):
            The type of time embedding to use.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            The dimension of the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        down_block_types (`tuple[str]`, defaults to `("DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D")`):
            The tuple of downsample blocks to use.
        up_block_types (`tuple[str]`, defaults to `("AttnUpBlock3D", "AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`tuple[int]`, defaults to `(224, 448, 672, 896)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        downsample_type (`str`, defaults to `"conv"`):
            The type of downsampling to use.
        upsample_type (`str`, defaults to `"conv"`):
            The type of upsampling to use.
        dropout (`float`, defaults to 0.0):
            The dropout rate to use.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to 8):
            The dimension of the attention heads.
        norm_num_groups (`int`, defaults to 32):
            The number of groups to use for the normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the attention normalization.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks.
        add_attention (`bool`, defaults to `True`):
            Whether to add attention to the mid block.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`.
        num_train_timesteps (`int`, *optional*, defaults to `None`):
            The number of training timesteps.
        conditioning_channels (`int`, defaults to 3):
            The number of channels in the conditional input.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D"),
        up_block_types: Tuple[str, ...] = ("AttnUpBlock3D", "AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        conditioning_channels: int = 1,
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNet3DConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            conditioning_channels=conditioning_channels,
            block_out_channels=conditioning_embedding_out_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv3d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

    @classmethod
    def from_unet(
        cls,
        unet: UNet3DModel,
        # controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 1,
    ):
        r"""
        Instantiate a [`ControlNet3DModel`] from [`UNet3DModel`].

        Parameters:
            unet (`UNet3DModel`):
                The UNet model weights to copy to the [`ControlNet3DModel`]. All configuration options are also copied
                where applicable.
        """
        controlnet = cls(
            sample_size=unet.config.sample_size,
            in_channels=unet.config.in_channels,
            out_channels=unet.config.out_channels,
            center_input_sample=unet.config.center_input_sample,
            time_embedding_type=unet.config.time_embedding_type,
            time_embedding_dim=unet.config.time_embedding_dim,
            freq_shift=unet.config.freq_shift,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            down_block_types=unet.config.down_block_types,
            up_block_types=unet.config.up_block_types,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            downsample_padding=unet.config.downsample_padding,
            downsample_type=unet.config.downsample_type,
            upsample_type=unet.config.upsample_type,
            dropout=unet.config.dropout,
            act_fn=unet.config.act_fn,
            attention_head_dim=unet.config.attention_head_dim,
            norm_num_groups=unet.config.norm_num_groups,
            attn_norm_num_groups=unet.config.attn_norm_num_groups,
            norm_eps=unet.config.norm_eps,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            add_attention=unet.config.add_attention,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            num_train_timesteps=unet.config.num_train_timesteps,
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ControlNet3DOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        The [`ControlNet3DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            controlnet_cond (`torch.Tensor`, *optional*, defaults to `None`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
                If None, the model will run in unconditional mode.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnets.controlnet.ControlNet3DOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.controlnets.controlnet.ControlNet3DOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnets.controlnet.ControlNet3DOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        sample = self.conv_in(sample)

        if controlnet_cond is not None:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
            sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNet3DOutput(down_block_res_samples=down_block_res_samples,
                                  mid_block_res_sample=mid_block_res_sample)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# if __name__ == "__main__":
#     bs = 2
#     in_ch = 1
#     patch_size = 64
#     a = torch.randn(bs, in_ch, patch_size, patch_size, patch_size)
#     b = torch.randn(bs, in_ch, patch_size, patch_size, patch_size)
#     t = torch.zeros(bs, dtype=torch.long)
#
#     unet = UNet3DModel(
#         sample_size=patch_size,
#         in_channels=in_ch,
#         out_channels=1,
#         # center_input_sample=False,
#         time_embedding_type="positional",
#         time_embedding_dim=None,
#         freq_shift=0,
#         flip_sin_to_cos=True,
#         down_block_types=("DownBlock3D", "DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D"),
#         up_block_types=("AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D", "UpBlock3D"),
#         block_out_channels=(32, 64, 128, 256),
#         layers_per_block=2,
#         mid_block_scale_factor=1,
#         downsample_padding=1,
#         downsample_type="conv",
#         upsample_type="conv",
#         dropout=0.0,
#         act_fn="silu",
#         attention_head_dim=8,
#         norm_num_groups=16,
#         attn_norm_num_groups=None,
#         norm_eps=1e-5,
#         resnet_time_scale_shift="default",
#         # add_attention=True,
#         class_embed_type=None,
#         num_class_embeds=None,
#     )
#
#     controlnet = ControlNet3DModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 16, 32),)
#
#     print(f'Total UNet parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f} M')
#     print(f'Total ControlNet parameters: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.2f} M')
#
#     down_block_res_samples, mid_block_res_sample = controlnet(a, t, controlnet_cond=b, return_dict=False,)
#
#     model_output = unet(
#         a,
#         t,
#         down_block_additional_residuals=down_block_res_samples,
#         mid_block_additional_residual=mid_block_res_sample,
#         return_dict=False,
#     )[0]
#
#     print(model_output.shape)
