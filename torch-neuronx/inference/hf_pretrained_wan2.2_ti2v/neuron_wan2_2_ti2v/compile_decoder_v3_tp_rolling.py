"""
Wan2.2 VAE Decoder Compilation - TP + Rolling Cache.

Combines tensor parallelism (TP) with rolling feat_cache for high-resolution
video generation (720P+) on Trainium2.

TP shards channels across ranks to reduce per-rank instruction count below
the 5M limit. Rolling cache passes feat_cache as explicit I/O for temporal
coherence (no flickering).

Architecture (Wan2.2-TI2V-5B VAE decoder):
  conv_in:    48 -> 1024        (ColumnParallel)
  mid_block:  1024 -> 1024      (ColumnRow, 2 resnets + attention)
  up_block_0: 1024 -> 1024      (ColumnRow, 3 resnets + upsample3d)
  up_block_1: 1024 -> 1024      (ColumnRow, 3 resnets + upsample3d)
  up_block_2: 1024 -> 512       (ColumnRow, 3 resnets + upsample2d)
  up_block_3: 512 -> 256        (ColumnRow, 3 resnets)
  norm_out:   256               (ShardedRMSNorm)
  conv_out:   256 -> 12         (RowParallel, all-reduce)

All internal activations stay sharded. Only conv_out gathers via all-reduce.
Cache tensors have sharded channel dimensions (except conv_in cache = 48 ch).
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from functools import reduce
import operator

from neuronx_distributed import ModelBuilder, NxDParallelState
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from safetensors.torch import save_file

try:
    from neuron_commons import attention_wrapper
except ModuleNotFoundError:
    from neuron_wan2_2_ti2v.neuron_commons import attention_wrapper
torch.nn.functional.scaled_dot_product_attention = attention_wrapper

CACHE_T = 2


# ============================================================================
# Parallel Conv3d layers
# ============================================================================

class ColumnParallelConv3d(nn.Module):
    """full input -> sharded output (no gather)."""
    def __init__(self, in_channels, out_channels_full, kernel_size, stride=1, padding=0,
                 bias=True, tp_degree=1, causal_padding=None):
        super().__init__()
        self.tp_degree = tp_degree
        self.out_channels_full = out_channels_full
        self.sharded_out = out_channels_full // tp_degree
        self._causal_padding = causal_padding  # 6-tuple for F.pad

        self.conv = nn.Conv3d(in_channels, self.sharded_out,
                              kernel_size=kernel_size, stride=stride,
                              padding=(0, 0, 0), bias=bias)

    def forward(self, x, cache_x=None):
        padding = list(self._causal_padding) if self._causal_padding else [0]*6
        if cache_x is not None and padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class RowParallelConv3d(nn.Module):
    """sharded input -> full output (all-reduce)."""
    def __init__(self, in_channels_full, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, tp_degree=1, causal_padding=None):
        super().__init__()
        self.tp_degree = tp_degree
        self.sharded_in = in_channels_full // tp_degree
        self._causal_padding = causal_padding

        self.conv = nn.Conv3d(self.sharded_in, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=(0, 0, 0), bias=bias)

    def forward(self, x, cache_x=None):
        padding = list(self._causal_padding) if self._causal_padding else [0]*6
        if cache_x is not None and padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        output = self.conv(x)
        return reduce_from_tensor_model_parallel_region(output)


class ColumnRowParallelConv3d(nn.Module):
    """sharded input -> sharded output (no communication)."""
    def __init__(self, in_channels_full, out_channels_full, kernel_size, stride=1, padding=0,
                 bias=True, tp_degree=1, causal_padding=None):
        super().__init__()
        self.tp_degree = tp_degree
        self.sharded_in = in_channels_full // tp_degree
        self.sharded_out = out_channels_full // tp_degree
        self._causal_padding = causal_padding

        self.conv = nn.Conv3d(self.sharded_in, self.sharded_out,
                              kernel_size=kernel_size, stride=stride,
                              padding=(0, 0, 0), bias=bias)

    def forward(self, x, cache_x=None):
        padding = list(self._causal_padding) if self._causal_padding else [0]*6
        if cache_x is not None and padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class ColumnRowParallelConv2d(nn.Module):
    """sharded input -> sharded output for 2D conv (no communication)."""
    def __init__(self, in_channels_full, out_channels_full, kernel_size, stride=1,
                 padding=0, bias=True, tp_degree=1):
        super().__init__()
        self.tp_degree = tp_degree
        self.sharded_in = in_channels_full // tp_degree
        self.sharded_out = out_channels_full // tp_degree
        self.conv = nn.Conv2d(self.sharded_in, self.sharded_out,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Sharded Norm / Attention / Upsample
# ============================================================================

def _get_causal_padding(original_conv):
    """Extract the 6-tuple causal padding from a WanCausalConv3d."""
    return tuple(original_conv._padding)


class ShardedWanRMSNorm(nn.Module):
    """RMS Norm with all-reduce for sharded channels."""
    def __init__(self, dim_full, tp_degree, channel_first=True, images=False):
        super().__init__()
        self.tp_degree = tp_degree
        self.channel_first = channel_first
        self.scale = dim_full ** 0.5
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        sharded_dim = dim_full // tp_degree
        shape = (sharded_dim, *broadcastable_dims) if channel_first else (sharded_dim,)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        local_sq_sum = (x ** 2).sum(dim=dim, keepdim=True)
        global_sq_sum = reduce_from_tensor_model_parallel_region(local_sq_sum)
        global_norm = torch.sqrt(global_sq_sum + 1e-8)
        x_normalized = x / global_norm
        return x_normalized * self.scale * self.gamma + self.bias


class ShardedWanAttentionBlock(nn.Module):
    """Attention block with sharded channels. Heads split across TP ranks."""
    def __init__(self, dim_full, tp_degree, num_heads=8):
        super().__init__()
        self.tp_degree = tp_degree
        sharded_dim = dim_full // tp_degree
        self.heads = num_heads // tp_degree

        self.norm = ShardedWanRMSNorm(dim_full, tp_degree, channel_first=True, images=True)
        self.to_qkv = nn.Conv2d(sharded_dim, sharded_dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(sharded_dim, sharded_dim, kernel_size=1, bias=True)

    def forward(self, x):
        batch, channels, frames, height, width = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

        x_normed = self.norm(x_2d)
        qkv = self.to_qkv(x_normed)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = q.shape[1] // self.heads
        q = q.reshape(batch * frames, self.heads, head_dim, -1).permute(0, 1, 3, 2)
        k = k.reshape(batch * frames, self.heads, head_dim, -1).permute(0, 1, 3, 2)
        v = v.reshape(batch * frames, self.heads, head_dim, -1).permute(0, 1, 3, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(batch * frames, -1, height, width)

        out = self.proj(attn_out)
        out = out.reshape(batch, frames, -1, height, width).permute(0, 2, 1, 3, 4)
        return x + out


class ShardedWanResidualBlock(nn.Module):
    """Residual block with sharded convolutions and feat_cache support."""
    def __init__(self, in_dim_full, out_dim_full, tp_degree, in_causal_pad, out_causal_pad,
                 shortcut_causal_pad=None):
        super().__init__()
        self.tp_degree = tp_degree
        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout(0.0)

        self.norm1 = ShardedWanRMSNorm(in_dim_full, tp_degree, images=False)
        self.conv1 = ColumnRowParallelConv3d(
            in_dim_full, out_dim_full, kernel_size=3, bias=True,
            tp_degree=tp_degree, causal_padding=in_causal_pad)
        self.norm2 = ShardedWanRMSNorm(out_dim_full, tp_degree, images=False)
        self.conv2 = ColumnRowParallelConv3d(
            out_dim_full, out_dim_full, kernel_size=3, bias=True,
            tp_degree=tp_degree, causal_padding=out_causal_pad)

        if in_dim_full != out_dim_full:
            self.conv_shortcut = ColumnRowParallelConv3d(
                in_dim_full, out_dim_full, kernel_size=1, bias=False,
                tp_degree=tp_degree, causal_padding=shortcut_causal_pad)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x, feat_cache, feat_idx):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv2(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        return x + h


class ShardedWanUpsample(nn.Module):
    """Upsample that preserves dtype (matches WanUpsample behavior)."""
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x.float(), scale_factor=self.scale_factor, mode=self.mode).type_as(x)


class ShardedWanResample(nn.Module):
    """Sharded upsampler with feat_cache support."""
    def __init__(self, dim_full, mode, tp_degree, upsample_out_dim_full=None):
        super().__init__()
        self.tp_degree = tp_degree
        self.mode = mode
        if upsample_out_dim_full is None:
            upsample_out_dim_full = dim_full // 2

        if mode in ("upsample2d", "upsample3d"):
            self.spatial_upsample = ShardedWanUpsample(scale_factor=(2.0, 2.0))
            self.spatial_conv = ColumnRowParallelConv2d(
                dim_full, upsample_out_dim_full, kernel_size=3, padding=1,
                bias=True, tp_degree=tp_degree)

        if mode == "upsample3d":
            # time_conv: dim -> dim*2 (both sharded)
            causal_pad = (0, 0, 0, 0, 2, 0)  # WanCausalConv3d with kernel=(3,1,1), padding=(1,0,0)
            self.time_conv = ColumnRowParallelConv3d(
                dim_full, dim_full * 2, kernel_size=(3, 1, 1), bias=True,
                tp_degree=tp_degree, causal_padding=causal_pad)
        else:
            self.time_conv = None

    def forward(self, x, feat_cache, feat_idx):
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d" and self.time_conv is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.time_conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            # Reshape: (b, 2*c_sharded, t, h, w) -> (b, c_sharded, t*2, h, w)
            c_sharded = c
            x = x.reshape(b, 2, c_sharded, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
            x = x.reshape(b, c_sharded, t * 2, h, w)

        # Spatial upsample (2D per frame)
        t = x.shape[2]
        c = x.shape[1]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.spatial_upsample(x)
        x = self.spatial_conv(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x


class ShardedDupUp3D(nn.Module):
    """Sharded DupUp3D (parameter-free, works with sharded channels)."""
    def __init__(self, in_channels_full, out_channels_full, factor_t, factor_s, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.out_channels_sharded = out_channels_full // tp_degree
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s

        in_sharded = in_channels_full // tp_degree
        self.repeats = out_channels_full * self.factor // in_channels_full

    def forward(self, x):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0), self.out_channels_sharded,
            self.factor_t, self.factor_s, self.factor_s,
            x.size(2), x.size(3), x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0), self.out_channels_sharded,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        return x


# ============================================================================
# Sharded Decoder Blocks
# ============================================================================

# Standard causal padding for kernel=3,padding=1: (1,1,1,1,2,0)
CAUSAL_PAD_3 = (1, 1, 1, 1, 2, 0)
# For kernel=1: no padding
CAUSAL_PAD_1 = (0, 0, 0, 0, 0, 0)


class ShardedWanMidBlock(nn.Module):
    def __init__(self, dim_full, tp_degree):
        super().__init__()
        self.resnets = nn.ModuleList([
            ShardedWanResidualBlock(dim_full, dim_full, tp_degree,
                                   CAUSAL_PAD_3, CAUSAL_PAD_3),
            ShardedWanResidualBlock(dim_full, dim_full, tp_degree,
                                   CAUSAL_PAD_3, CAUSAL_PAD_3),
        ])
        self.attentions = nn.ModuleList([
            ShardedWanAttentionBlock(dim_full, tp_degree),
        ])

    def forward(self, x, feat_cache, feat_idx):
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class ShardedWanResidualUpBlock(nn.Module):
    def __init__(self, in_dim_full, out_dim_full, tp_degree,
                 upsample_mode=None, temporal_upsample=False):
        super().__init__()
        self.tp_degree = tp_degree

        shortcut_pad = CAUSAL_PAD_1 if in_dim_full != out_dim_full else None
        # 3 resnets (num_res_blocks + 1 = 3)
        self.resnets = nn.ModuleList()
        current_dim = in_dim_full
        for _ in range(3):
            self.resnets.append(
                ShardedWanResidualBlock(
                    current_dim, out_dim_full, tp_degree,
                    CAUSAL_PAD_3, CAUSAL_PAD_3,
                    shortcut_causal_pad=shortcut_pad if current_dim != out_dim_full else None)
            )
            current_dim = out_dim_full

        if upsample_mode is not None:
            self.upsampler = ShardedWanResample(
                out_dim_full, upsample_mode, tp_degree,
                upsample_out_dim_full=out_dim_full)
        else:
            self.upsampler = None

        # avg_shortcut (DupUp3D) for blocks with upsampling
        if upsample_mode is not None:
            factor_t = 2 if temporal_upsample else 1
            self.avg_shortcut = ShardedDupUp3D(
                in_dim_full, out_dim_full, factor_t=factor_t, factor_s=2,
                tp_degree=tp_degree)
        else:
            self.avg_shortcut = None

    def forward(self, x, feat_cache, feat_idx):
        x_copy = x.clone()

        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy)

        return x


class ShardedWanDecoder(nn.Module):
    """Full TP-sharded Wan decoder with rolling feat_cache."""
    def __init__(self, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree

        # conv_in: full 48 -> sharded 1024
        self.conv_in = ColumnParallelConv3d(
            48, 1024, kernel_size=3, bias=True,
            tp_degree=tp_degree, causal_padding=CAUSAL_PAD_3)

        self.nonlinearity = nn.SiLU()

        # mid_block: sharded 1024
        self.mid_block = ShardedWanMidBlock(1024, tp_degree)

        # up_blocks
        # up_block_0: 1024->1024, upsample3d (temporal=True)
        # up_block_1: 1024->1024, upsample3d (temporal=True)
        # up_block_2: 1024->512,  upsample2d (temporal=False)
        # up_block_3: 512->256,   no upsample
        self.up_blocks = nn.ModuleList([
            ShardedWanResidualUpBlock(1024, 1024, tp_degree,
                                     upsample_mode="upsample3d", temporal_upsample=True),
            ShardedWanResidualUpBlock(1024, 1024, tp_degree,
                                     upsample_mode="upsample3d", temporal_upsample=True),
            ShardedWanResidualUpBlock(1024, 512, tp_degree,
                                     upsample_mode="upsample2d", temporal_upsample=False),
            ShardedWanResidualUpBlock(512, 256, tp_degree,
                                     upsample_mode=None),
        ])

        # norm_out + conv_out: sharded 256 -> full 12
        self.norm_out = ShardedWanRMSNorm(256, tp_degree, images=False)
        self.conv_out = RowParallelConv3d(
            256, 12, kernel_size=3, bias=True,
            tp_degree=tp_degree, causal_padding=CAUSAL_PAD_3)

    def forward(self, x, feat_cache, feat_idx):
        # conv_in with cache (input is full 48 channels)
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        # mid_block
        x = self.mid_block(x, feat_cache, feat_idx)

        # up_blocks
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        # output
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        return x


# ============================================================================
# Rolling Cache Wrapper (35 inputs, 35 outputs)
# ============================================================================

NUM_FEAT_CACHE = 34

class DecoderWrapperTPRolling(nn.Module):
    """Wrapper: (x, c0..c33) -> (output, c0..c33)."""
    def __init__(self, tp_degree):
        super().__init__()
        self.decoder = ShardedWanDecoder(tp_degree)

    def forward(self, x,
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
                c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                c30, c31, c32, c33):
        feat_cache = [
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
            c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
            c30, c31, c32, c33,
        ]
        feat_idx = [0]
        output = self.decoder(x, feat_cache, feat_idx)
        return tuple([output] + feat_cache)


# ============================================================================
# Cache shapes (sharded for TP)
# ============================================================================

def get_feat_cache_shapes_tp(batch_size, latent_height, latent_width, tp_degree,
                              dtype=torch.bfloat16):
    """
    Return 34 feat_cache tensor shapes with sharded channels for TP.

    Cache[0] (conv_in input) has full 48 channels.
    All other caches have channels divided by tp_degree.
    """
    lh, lw = latent_height, latent_width
    tp = tp_degree

    return [
        # [0] conv_in input: full channels (not sharded)
        (batch_size, 48, 2, lh, lw),
        # [1-4] mid_block: 2 resnets x 2 convs = 4
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        # [5-11] up_block_0: 3 resnets x 2 convs + 1 upsampler = 7
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),
        (batch_size, 1024 // tp, 2, lh, lw),          # upsampler time_conv
        # [12-18] up_block_1: 3 resnets x 2 convs + 1 upsampler = 7
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),
        (batch_size, 1024 // tp, 2, lh*2, lw*2),      # upsampler time_conv
        # [19-24] up_block_2: 3 resnets x 2 convs = 6 (upsample2d, no time_conv)
        (batch_size, 1024 // tp, 2, lh*4, lw*4),      # resnet0 conv1 (in=1024)
        (batch_size, 512 // tp, 2, lh*4, lw*4),       # resnet0 conv2 (out=512)
        (batch_size, 512 // tp, 2, lh*4, lw*4),
        (batch_size, 512 // tp, 2, lh*4, lw*4),
        (batch_size, 512 // tp, 2, lh*4, lw*4),
        (batch_size, 512 // tp, 2, lh*4, lw*4),
        # [25-30] up_block_3: 3 resnets x 2 convs = 6 (no upsampler)
        (batch_size, 512 // tp, 2, lh*8, lw*8),       # resnet0 conv1 (in=512)
        (batch_size, 256 // tp, 2, lh*8, lw*8),       # resnet0 conv2 (out=256)
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        # [31] conv_out input: sharded 256
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        # [32-33] placeholders
        (batch_size, 256 // tp, 2, lh*8, lw*8),
        (batch_size, 12, 2, lh*8, lw*8),              # placeholder (full channels)
    ]


# ============================================================================
# Weight sharding
# ============================================================================

def _shard(tensor, dim, tp_degree, rank):
    """Slice tensor along dim for given rank."""
    size = tensor.shape[dim] // tp_degree
    start = size * rank
    return tensor.narrow(dim, start, size).clone()


def create_sharded_checkpoint(original_decoder, wrapper, tp_degree, rank, dtype=torch.bfloat16):
    """
    Create a checkpoint for a specific TP rank by sharding the original VAE decoder weights.

    Maps sharded model parameter names -> sliced original weights.
    """
    orig_sd = original_decoder.state_dict()
    sharded_sd = {}

    for name, param in wrapper.state_dict().items():
        sharded_sd[name] = torch.zeros_like(param, dtype=dtype)

    # Helper: shard conv3d weight/bias
    def shard_conv(prefix_sharded, prefix_orig, shard_type):
        """shard_type: 'column' (out), 'row' (in), 'column_row' (both), 'none'"""
        w = orig_sd[f"{prefix_orig}.weight"].to(dtype)
        key_w = f"{prefix_sharded}.conv.weight"
        if shard_type == "column":
            sharded_sd[key_w] = _shard(w, 0, tp_degree, rank)
        elif shard_type == "row":
            sharded_sd[key_w] = _shard(w, 1, tp_degree, rank)
        elif shard_type == "column_row":
            sharded_sd[key_w] = _shard(_shard(w, 0, tp_degree, rank), 1, tp_degree, rank)
        else:
            sharded_sd[key_w] = w

        bias_key_orig = f"{prefix_orig}.bias"
        bias_key_sharded = f"{prefix_sharded}.conv.bias"
        if bias_key_orig in orig_sd and bias_key_sharded in sharded_sd:
            b = orig_sd[bias_key_orig].to(dtype)
            if shard_type in ("column", "column_row"):
                sharded_sd[bias_key_sharded] = _shard(b, 0, tp_degree, rank)
            elif shard_type == "row":
                sharded_sd[bias_key_sharded] = b / tp_degree
            else:
                sharded_sd[bias_key_sharded] = b

    def shard_conv2d(prefix_sharded, prefix_orig, shard_type):
        """For 2D convolutions in the upsampler."""
        w = orig_sd[f"{prefix_orig}.weight"].to(dtype)
        key_w = f"{prefix_sharded}.conv.weight"
        if shard_type == "column_row":
            sharded_sd[key_w] = _shard(_shard(w, 0, tp_degree, rank), 1, tp_degree, rank)
        else:
            sharded_sd[key_w] = w

        bias_key_orig = f"{prefix_orig}.bias"
        bias_key_sharded = f"{prefix_sharded}.conv.bias"
        if bias_key_orig in orig_sd and bias_key_sharded in sharded_sd:
            b = orig_sd[bias_key_orig].to(dtype)
            if shard_type == "column_row":
                sharded_sd[bias_key_sharded] = _shard(b, 0, tp_degree, rank)
            else:
                sharded_sd[bias_key_sharded] = b

    def shard_norm(prefix_sharded, prefix_orig):
        gamma = orig_sd[f"{prefix_orig}.gamma"].to(dtype)
        sharded_sd[f"{prefix_sharded}.gamma"] = _shard(gamma, 0, tp_degree, rank)

    def shard_attn(prefix_sharded, prefix_orig):
        """Shard attention qkv and proj Conv2d."""
        for sub in ("to_qkv", "proj"):
            w = orig_sd[f"{prefix_orig}.{sub}.weight"].to(dtype)
            w = _shard(_shard(w, 0, tp_degree, rank), 1, tp_degree, rank)
            sharded_sd[f"{prefix_sharded}.{sub}.weight"] = w
            b = orig_sd[f"{prefix_orig}.{sub}.bias"].to(dtype)
            sharded_sd[f"{prefix_sharded}.{sub}.bias"] = _shard(b, 0, tp_degree, rank)
        shard_norm(f"{prefix_sharded}.norm", f"{prefix_orig}.norm")

    def shard_resblock(prefix_sharded, prefix_orig, in_dim, out_dim):
        """Shard a WanResidualBlock."""
        shard_norm(f"{prefix_sharded}.norm1", f"{prefix_orig}.norm1")
        shard_conv(f"{prefix_sharded}.conv1", f"{prefix_orig}.conv1", "column_row")
        shard_norm(f"{prefix_sharded}.norm2", f"{prefix_orig}.norm2")
        shard_conv(f"{prefix_sharded}.conv2", f"{prefix_orig}.conv2", "column_row")
        if in_dim != out_dim:
            shard_conv(f"{prefix_sharded}.conv_shortcut", f"{prefix_orig}.conv_shortcut", "column_row")

    # --- conv_in: ColumnParallel ---
    shard_conv("decoder.conv_in", "conv_in", "column")

    # --- mid_block ---
    for i in range(2):
        shard_resblock(f"decoder.mid_block.resnets.{i}", f"mid_block.resnets.{i}", 1024, 1024)
    shard_attn("decoder.mid_block.attentions.0", "mid_block.attentions.0")

    # --- up_blocks ---
    block_configs = [
        (1024, 1024, True),   # up_block_0: has upsampler (upsample3d)
        (1024, 1024, True),   # up_block_1: has upsampler (upsample3d)
        (1024, 512, True),    # up_block_2: has upsampler (upsample2d)
        (512, 256, False),    # up_block_3: no upsampler
    ]

    for bi, (in_dim, out_dim, has_upsample) in enumerate(block_configs):
        sp = f"decoder.up_blocks.{bi}"
        op = f"up_blocks.{bi}"

        current_dim = in_dim
        for ri in range(3):
            shard_resblock(f"{sp}.resnets.{ri}", f"{op}.resnets.{ri}", current_dim, out_dim)
            current_dim = out_dim

        if has_upsample:
            # upsampler resample Conv2d
            shard_conv2d(f"{sp}.upsampler.spatial_conv", f"{op}.upsampler.resample.1", "column_row")
            # upsampler time_conv (upsample3d only)
            time_conv_key = f"{op}.upsampler.time_conv.weight"
            if time_conv_key in orig_sd:
                shard_conv(f"{sp}.upsampler.time_conv", f"{op}.upsampler.time_conv", "column_row")

    # --- norm_out ---
    shard_norm("decoder.norm_out", "norm_out")

    # --- conv_out: RowParallel ---
    shard_conv("decoder.conv_out", "conv_out", "row")

    return sharded_sd


# ============================================================================
# Main compile function
# ============================================================================

def save_model_config(output_path, config):
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_tp_rolling(args):
    latent_height = args.height // 16
    latent_width = args.width // 16
    tp_degree = args.tp_degree
    world_size = args.world_size
    batch_size = 1
    decoder_frames = args.decoder_frames
    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 VAE Decoder TP + Rolling Cache Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"Decoder frames: {decoder_frames}")
    print(f"TP degree: {tp_degree}")
    print(f"World size: {world_size}")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # Get sharded cache shapes
        feat_cache_shapes = get_feat_cache_shapes_tp(
            batch_size, latent_height, latent_width, tp_degree, dtype)

        total_cache_bytes = sum(reduce(operator.mul, s) * 2 for s in feat_cache_shapes)
        print(f"\nCache: {len(feat_cache_shapes)} tensors, {total_cache_bytes/1024/1024:.0f} MB per direction")
        for i, s in enumerate(feat_cache_shapes):
            size_mb = reduce(operator.mul, s) * 2 / 1024 / 1024
            print(f"  [{i:2d}] {s}  ({size_mb:.1f} MB)")

        # Create wrapper
        print("\nCreating sharded decoder...")
        wrapper = DecoderWrapperTPRolling(tp_degree)
        wrapper = wrapper.to(dtype).eval()

        # Build trace kwargs
        decoder_input = torch.rand(
            (batch_size, 48, decoder_frames, latent_height, latent_width), dtype=dtype)
        trace_kwargs = {"x": decoder_input}
        for i, shape in enumerate(feat_cache_shapes):
            trace_kwargs[f"c{i}"] = torch.zeros(shape, dtype=dtype)

        print(f"  Input x: {decoder_input.shape}")
        print(f"  Cache I/O: ~{total_cache_bytes*2/1024/1024:.0f} MB (in + out)")

        # Trace and compile
        builder = ModelBuilder(model=wrapper)
        print("\nTracing...")
        builder.trace(kwargs=trace_kwargs, tag="decode")

        print("Compiling...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        traced = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save compiled model
        output_path = f"{args.compiled_models_dir}/decoder_v3_tp_rolling"
        os.makedirs(output_path, exist_ok=True)
        print(f"\nSaving to {output_path}...")
        traced.save(os.path.join(output_path, "nxd_model.pt"))

        # Save sharded weights per TP rank
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        original_decoder = vae.decoder
        print("Sharding and saving weights...")
        for rank in range(tp_degree):
            ckpt = create_sharded_checkpoint(original_decoder, wrapper, tp_degree, rank, dtype)
            save_file(ckpt, os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors"))
            if rank == 0:
                print(f"  {len(ckpt)} parameters per rank")
        print(f"  Saved {tp_degree} rank checkpoints")

        # Save config
        config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "decoder_frames": decoder_frames,
            "in_channels": 48,
            "out_channels": 12,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "bfloat16",
            "rolling_cache": True,
            "tp_rolling": True,
            "num_cache_tensors": NUM_FEAT_CACHE,
        }
        save_model_config(output_path, config)

        # ========== Compile post_quant_conv (float32, unsharded) ==========
        latent_frames = (args.num_frames - 1) // 4 + 1
        print("\nCompiling post_quant_conv (float32)...")

        class PostQuantConvWrapper(nn.Module):
            def __init__(self, pqc):
                super().__init__()
                self.conv = pqc
            def forward(self, x):
                return self.conv(x)

        pqc_wrapper = PostQuantConvWrapper(vae.post_quant_conv)
        pqc_input = torch.rand(
            (batch_size, 48, latent_frames, latent_height, latent_width),
            dtype=torch.float32)

        pqc_builder = ModelBuilder(model=pqc_wrapper)
        pqc_builder.trace(kwargs={"x": pqc_input}, tag="conv")
        traced_pqc = pqc_builder.compile(
            compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
            compiler_workdir=args.compiler_workdir,
        )

        pqc_output_path = f"{args.compiled_models_dir}/post_quant_conv_v3"
        os.makedirs(pqc_output_path, exist_ok=True)
        traced_pqc.save(os.path.join(pqc_output_path, "nxd_model.pt"))

        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        save_file(pqc_wrapper.state_dict(),
                  os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

        save_model_config(pqc_output_path, {
            "batch_size": batch_size,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "in_channels": 48,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "float32",
        })

        print("\n" + "=" * 60)
        print("Compilation Complete!")
        print(f"Decoder (TP={tp_degree} rolling): {output_path}")
        print(f"post_quant_conv: {pqc_output_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--decoder_frames", type=int, default=2)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    parser.add_argument("--cache_dir", type=str, default="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir")
    args = parser.parse_args()

    compile_decoder_tp_rolling(args)
