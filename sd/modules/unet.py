from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sd.modules.util import conv_nd, normalization, timestep_embedding
from sd.modules.attention import SpatialTransformer
from sd.util import zero_module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.

    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    """

    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4,2,1],
        dropout=0,
        channel_mult=[1,2,4,4],
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=-1,
        transformer_depth=1,
        context_dim=768,
        use_xformers=False,
        use_linear_in_transformer=False
        ):
        super().__init__()

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dims = dims
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.use_xformers = use_xformers
        self.use_linear_in_transformer = use_linear_in_transformer

        self.build_time_embed(model_channels)
        ch, ds = self.build_input_blocks(in_channels, model_channels)
        self.build_middle_block(ch)
        ch = self.build_output_blocks(ch, model_channels, ds)
        self.build_out(ch, out_channels)

    def build_time_embed(self, model_channels):
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

    def build_input_resblock(self, level, i, ch_in, ch_out):
        return ResBlock(ch_in, self.time_embed_dim, self.dropout, out_channels=ch_out, dims=self.dims, use_checkpoint=self.use_checkpoint)

    def build_input_transformer(self, level, i, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels
                    
        return SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint, use_xformers=self.use_xformers, use_linear=self.use_linear_in_transformer)

    def build_input_downsample(self, level, ch):
        return Downsample(ch, dims=self.dims, out_channels=ch)

    def build_input_blocks(self, in_channels, model_channels):
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(self.dims, in_channels, model_channels, 3, padding=1))])
        
        self.input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for i in range(self.num_res_blocks[level]):
                layers = [self.build_input_resblock(level, i, ch, mult * model_channels)]
                ch = mult * model_channels
                if ds in self.attention_resolutions:
                    layers.append(self.build_input_transformer(level, i, ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_block_chans.append(ch)
            
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(self.build_input_downsample(level, ch)))
                self.input_block_chans.append(ch)
                ds *= 2
        return ch, ds

    def build_middle_resblock(self, i, ch):
        return ResBlock(ch, self.time_embed_dim, self.dropout, dims=self.dims, use_checkpoint=self.use_checkpoint)

    def build_middle_transformer(self, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels
            
        return SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint, use_xformers=self.use_xformers, use_linear=self.use_linear_in_transformer)

    def build_middle_block(self, ch):
        self.middle_block = TimestepEmbedSequential(
            self.build_middle_resblock(0, ch),
            self.build_middle_transformer(ch),
            self.build_middle_resblock(1, ch)
        )

    def build_output_resblock(self, level, i, ch_in, ch_out):
        return ResBlock(ch_in, self.time_embed_dim, self.dropout, out_channels=ch_out, dims=self.dims, use_checkpoint=self.use_checkpoint)

    def build_output_transformer(self, level, i, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels

        return SpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint, use_xformers=self.use_xformers, use_linear=self.use_linear_in_transformer)

    def build_output_upsample(self, level, ch):
        return Upsample(ch, dims=self.dims)

    def build_output_blocks(self, ch, model_channels, ds):
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = self.input_block_chans.pop()
                layers = [self.build_output_resblock(level, i, ch + ich, model_channels * mult)]
                ch = model_channels * mult
                
                if ds in self.attention_resolutions:
                    layers.append(self.build_output_transformer(level, i, ch))
                    
                if level and i == self.num_res_blocks[level]:
                    layers.append(self.build_output_upsample(level, ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        return ch
        
    def build_out(self, ch_in, ch_out):
        self.out = nn.Sequential(
            normalization(ch_in),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, ch_in, ch_out, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h)

class UNetModelV2(UNetModel):
    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4,2,1],
        dropout=0,
        channel_mult=[1,2,4,4],
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        transformer_depth=1,
        context_dim=1024,
        use_xformers=False,
        use_linear_in_transformer=True
        ):
        super().__init__(in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout,
                         channel_mult, dims, use_checkpoint, use_fp16, num_heads, num_head_channels, transformer_depth,
                         context_dim, use_xformers, use_linear_in_transformer)

