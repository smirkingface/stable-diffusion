import torch
import torch.nn as nn
from einops import rearrange

from sd.modules.util import conv_nd, normalization
from sd.modules.attention import SpatialTransformer
from sd.util import zero_module
from sd.modules.unet import ResBlock, UNetModel, Upsample, Downsample


class ExtendedResBlock(ResBlock):
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
        kernel_size=3,
        skip_connection_conv=False
    ):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=kernel_size//2),
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
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=kernel_size//2)),
        )

        if self.out_channels == channels and not skip_connection_conv:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=kernel_size//2)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


class ExtendedSpatialTransformer(SpatialTransformer):
    def __init__(self, *args, use_spatial_encoding=False, image_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.proj_in.in_channels
        self.use_spatial_encoding = use_spatial_encoding
        if use_spatial_encoding:
            self.pos_embedding = nn.Parameter(torch.zeros((1,inner_dim) + tuple(image_shape)))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.use_spatial_encoding:
            x += self.pos_embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

        
class ExtendedUNetModel(UNetModel):
    def __init__(self, kernel_size=3, skip_connection_conv=False, use_spatial_encoding=False, **kwargs):
        self.kernel_size = kernel_size
        self.skip_connection_conv = skip_connection_conv
        self.use_spatial_encoding = use_spatial_encoding
        super().__init__(**kwargs)
        
    def build_input_resblock(self, level, i, ch_in, ch_out):
        return ExtendedResBlock(ch_in, self.time_embed_dim, self.dropout, out_channels=ch_out, dims=self.dims, use_checkpoint=self.use_checkpoint,
                                kernel_size=self.kernel_size, skip_connection_conv=self.skip_connection_conv)

    def build_input_transformer(self, level, i, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels
                    
        return ExtendedSpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint,
                                          use_spatial_encoding=self.use_spatial_encoding, image_shape=[int(64/(2**level)),int(64/(2**level))])
    
    def build_middle_resblock(self, i, ch):
        return ExtendedResBlock(ch, self.time_embed_dim, self.dropout, dims=self.dims, use_checkpoint=self.use_checkpoint,
                                kernel_size=self.kernel_size, skip_connection_conv=self.skip_connection_conv, use_conv=self.skip_connection_conv)

    def build_middle_transformer(self, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels
            
        return ExtendedSpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint,
                                          use_spatial_encoding=self.use_spatial_encoding, image_shape=[int(64/(2**(len(self.channel_mult)-1))),int(64/(2**(len(self.channel_mult)-1)))])
   
    def build_output_resblock(self, level, i, ch_in, ch_out):
        return ExtendedResBlock(ch_in, self.time_embed_dim, self.dropout, out_channels=ch_out, dims=self.dims, use_checkpoint=self.use_checkpoint,
                                kernel_size=self.kernel_size, skip_connection_conv=self.skip_connection_conv)

    def build_output_transformer(self, level, i, ch):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels

        return ExtendedSpatialTransformer(ch, num_heads, dim_head, depth=self.transformer_depth, context_dim=self.context_dim, use_checkpoint=self.use_checkpoint,
                                          use_spatial_encoding=self.use_spatial_encoding, image_shape=[int(64/(2**level)),int(64/(2**level))])
