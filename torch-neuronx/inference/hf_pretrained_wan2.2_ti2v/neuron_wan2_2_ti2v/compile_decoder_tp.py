"""
Tensor Parallel Decoder Compilation for high-resolution video generation.

This script compiles the VAE decoder with tensor parallelism to handle
larger resolutions (like 720x1280) that exceed single-core instruction limits.

Strategy: Split channels across TP ranks with all-reduce for norm layers.
Supports feat_cache for temporal caching.
"""
import os
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Comment this line out if using trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries --optlevel 1 """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import neuronx_distributed
from functools import partial
from typing import List, Optional

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from neuron_commons import attention_wrapper

torch.nn.functional.scaled_dot_product_attention = attention_wrapper

CACHE_T = 2  # Cache last 2 frames for causal convolution


def get_sharded_data(data, dim):
    """Get sharded data for the current TP rank."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    shard_size = data.shape[dim] // tp_size
    if dim == 0:
        return data[shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    elif dim == 1:
        return data[:, shard_size * tp_rank : shard_size * (tp_rank + 1)].clone()
    else:
        raise ValueError(f"Unsupported dim: {dim}")


class ColumnParallelConv3d(nn.Module):
    """
    Conv3d with output channels sharded across TP ranks (Column Parallel).
    Output is kept sharded (no gather).
    """
    def __init__(self, original_conv, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.sharded_out_channels = self.out_channels // tp_degree

        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self._padding = original_conv._padding

        self.conv = nn.Conv3d(
            self.in_channels,
            self.sharded_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0, 0),
            bias=original_conv.bias is not None
        )

        self.conv.weight.data = get_sharded_data(original_conv.weight.data, 0)
        if original_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(original_conv.bias.data, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class RowParallelConv3d(nn.Module):
    """
    Conv3d with input channels sharded across TP ranks (Row Parallel).
    Output is reduced (all-reduce sum) across ranks.
    """
    def __init__(self, original_conv, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.sharded_in_channels = self.in_channels // tp_degree

        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self._padding = original_conv._padding

        self.conv = nn.Conv3d(
            self.sharded_in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0, 0),
            bias=original_conv.bias is not None
        )

        self.conv.weight.data = get_sharded_data(original_conv.weight.data, 1)
        if original_conv.bias is not None:
            self.conv.bias.data = original_conv.bias.data / tp_degree

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        output = self.conv(x)
        output = reduce_from_tensor_model_parallel_region(output)
        return output


class ColumnRowParallelConv3d(nn.Module):
    """
    Conv3d with both input and output channels sharded (internal layer).
    No gather/reduce - keeps data sharded.
    """
    def __init__(self, original_conv, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.sharded_in_channels = self.in_channels // tp_degree
        self.sharded_out_channels = self.out_channels // tp_degree

        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self._padding = original_conv._padding

        self.conv = nn.Conv3d(
            self.sharded_in_channels,
            self.sharded_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0, 0),
            bias=original_conv.bias is not None
        )

        weight = get_sharded_data(original_conv.weight.data, 0)
        weight = get_sharded_data(weight, 1)
        self.conv.weight.data = weight

        if original_conv.bias is not None:
            self.conv.bias.data = get_sharded_data(original_conv.bias.data, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class ShardedWanRMSNorm(nn.Module):
    """
    RMS Norm that works with sharded channel dimension.
    Uses all-reduce for correct normalization across TP ranks.
    """
    def __init__(self, original_norm, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.channel_first = original_norm.channel_first
        self.scale = original_norm.scale

        self.gamma = nn.Parameter(get_sharded_data(original_norm.gamma.data, 0))
        if isinstance(original_norm.bias, nn.Parameter):
            self.bias = nn.Parameter(get_sharded_data(original_norm.bias.data, 0))
        else:
            self.bias = 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        local_sq_sum = (x ** 2).sum(dim=dim, keepdim=True)
        global_sq_sum = reduce_from_tensor_model_parallel_region(local_sq_sum)
        global_norm = torch.sqrt(global_sq_sum + 1e-8)
        x_normalized = x / global_norm
        return x_normalized * self.scale * self.gamma + self.bias


class ShardedWanAttentionBlock(nn.Module):
    """Sharded attention block for decoder."""
    def __init__(self, original_attn, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree

        self.norm = ShardedWanRMSNorm(original_attn.norm, tp_degree)

        in_channels = original_attn.to_qkv.in_channels // tp_degree
        out_channels = original_attn.to_qkv.out_channels // tp_degree
        self.to_qkv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.to_qkv.weight.data = get_sharded_data(
            get_sharded_data(original_attn.to_qkv.weight.data, 0), 1
        )
        self.to_qkv.bias.data = get_sharded_data(original_attn.to_qkv.bias.data, 0)

        in_channels = original_attn.proj.in_channels // tp_degree
        out_channels = original_attn.proj.out_channels // tp_degree
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.proj.weight.data = get_sharded_data(
            get_sharded_data(original_attn.proj.weight.data, 0), 1
        )
        self.proj.bias.data = get_sharded_data(original_attn.proj.bias.data, 0)

        self.heads = 8 // tp_degree

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
    """Sharded residual block with feat_cache support."""
    def __init__(self, original_block, tp_degree, input_sharded=True, output_sharded=True):
        super().__init__()
        self.tp_degree = tp_degree
        self.input_sharded = input_sharded
        self.output_sharded = output_sharded

        # norm1 processes input, norm2 processes conv1 output
        self.norm1 = ShardedWanRMSNorm(original_block.norm1, tp_degree) if input_sharded else original_block.norm1

        # After conv1: if input_sharded=True, output is sharded (for ColumnRowParallel)
        #              if input_sharded=False, output is not sharded (original conv)
        # So norm2 needs to match the conv1 output state
        conv1_output_sharded = input_sharded  # conv1 output is sharded when input is sharded
        self.norm2 = ShardedWanRMSNorm(original_block.norm2, tp_degree) if conv1_output_sharded else original_block.norm2

        self.nonlinearity = original_block.nonlinearity
        self.dropout = original_block.dropout

        if input_sharded and output_sharded:
            self.conv1 = ColumnRowParallelConv3d(original_block.conv1, tp_degree)
            self.conv2 = ColumnRowParallelConv3d(original_block.conv2, tp_degree)
        elif input_sharded and not output_sharded:
            self.conv1 = ColumnRowParallelConv3d(original_block.conv1, tp_degree)
            self.conv2 = RowParallelConv3d(original_block.conv2, tp_degree)
        elif not input_sharded and output_sharded:
            self.conv1 = ColumnParallelConv3d(original_block.conv1, tp_degree)
            self.conv2 = ColumnRowParallelConv3d(original_block.conv2, tp_degree)
        else:
            self.conv1 = original_block.conv1
            self.conv2 = original_block.conv2

        if isinstance(original_block.conv_shortcut, nn.Identity):
            self.conv_shortcut = nn.Identity()
        else:
            if input_sharded and output_sharded:
                self.conv_shortcut = ColumnRowParallelConv3d(original_block.conv_shortcut, tp_degree)
            elif input_sharded and not output_sharded:
                self.conv_shortcut = RowParallelConv3d(original_block.conv_shortcut, tp_degree)
            elif not input_sharded and output_sharded:
                self.conv_shortcut = ColumnParallelConv3d(original_block.conv_shortcut, tp_degree)
            else:
                self.conv_shortcut = original_block.conv_shortcut

    def forward(self, x, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        # conv1 with cache
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        # conv2 with cache
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class ShardedWanResample(nn.Module):
    """Sharded upsampler with feat_cache support.

    For upsample3d mode (decoder):
    1. First apply time_conv (before spatial upsample)
    2. Then reshape to double temporal dimension
    3. Finally apply spatial resample (upsample)

    Note: Original WanResample has special first-chunk logic where feat_cache[idx]=None
    triggers "Rep" placeholder. For compilation, we always assume non-first-chunk
    execution path where time_conv is applied with feat_cache.
    """
    def __init__(self, original_upsampler, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.mode = original_upsampler.mode

        # Get the sharded channel count for this upsampler
        # For decoder, upsampler input is 1024 channels (sharded to 256 per rank)
        self.sharded_channels = None

        # resample contains WanUpsample + Conv2d
        self.resample = nn.Sequential()
        for i, layer in enumerate(original_upsampler.resample):
            if isinstance(layer, nn.Conv2d):
                # Shard both input and output for internal consistency
                in_ch = layer.in_channels // tp_degree
                out_ch = layer.out_channels // tp_degree
                self.sharded_channels = in_ch  # Save for reshape
                new_conv = nn.Conv2d(in_ch, out_ch, kernel_size=layer.kernel_size,
                                     stride=layer.stride, padding=layer.padding,
                                     bias=layer.bias is not None)
                new_conv.weight.data = get_sharded_data(get_sharded_data(layer.weight.data, 0), 1)
                if layer.bias is not None:
                    new_conv.bias.data = get_sharded_data(layer.bias.data, 0)
                self.resample.add_module(str(i), new_conv)
            else:
                self.resample.add_module(str(i), layer)

        # time_conv if exists - for upsample3d, output channels are 2x input
        if hasattr(original_upsampler, 'time_conv') and original_upsampler.time_conv is not None:
            self.time_conv = ColumnRowParallelConv3d(original_upsampler.time_conv, tp_degree)
        else:
            self.time_conv = None

    def forward(self, x, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None):
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            # For upsample3d: time_conv FIRST (before spatial upsample)
            if self.time_conv is not None:
                if feat_cache is not None and feat_idx is not None:
                    idx = feat_idx[0]
                    # Save cache BEFORE time_conv (at original spatial resolution)
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                    x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = self.time_conv(x)

                # time_conv outputs 2*c channels (sharded), reshape to double temporal dimension
                # x shape after time_conv: (b, 2*c_sharded, t, h, w)
                # Need to reshape to: (b, c_sharded, 2*t, h, w)
                c_sharded = c  # Original input channels (already sharded)
                x = x.reshape(b, 2, c_sharded, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c_sharded, t * 2, h, w)

            # Update t and c after temporal upsampling
            t = x.shape[2]
            c = x.shape[1]

        # Spatial resample (2D operations)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        return x


class ShardedWanMidBlock(nn.Module):
    """Sharded mid block with feat_cache support."""
    def __init__(self, original_mid_block, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree

        self.attentions = nn.ModuleList([
            ShardedWanAttentionBlock(attn, tp_degree)
            for attn in original_mid_block.attentions
        ])
        self.resnets = nn.ModuleList([
            ShardedWanResidualBlock(resnet, tp_degree, input_sharded=True, output_sharded=True)
            for resnet in original_mid_block.resnets
        ])

    def forward(self, x, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None):
        x = self.resnets[0](x, feat_cache, feat_idx)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)

        return x


class ShardedWanResidualUpBlock(nn.Module):
    """Sharded up block with feat_cache support.

    For the last block (up_block 3), we need special handling:
    - resnets.0: input_sharded=True (512/4=128), output_sharded=False (256, gathered)
    - resnets.1, resnets.2: input_sharded=False (256), output_sharded=False (256)
    """
    def __init__(self, original_up_block, tp_degree, is_last_block=False):
        super().__init__()
        self.tp_degree = tp_degree
        self.is_last_block = is_last_block

        self.resnets = nn.ModuleList()
        for i, resnet in enumerate(original_up_block.resnets):
            if is_last_block:
                # Last block: first resnet transitions from sharded to gathered
                if i == 0:
                    input_sharded = True
                    output_sharded = False
                else:
                    # Subsequent resnets: fully gathered (256 channels)
                    input_sharded = False
                    output_sharded = False
            else:
                # Non-last blocks: keep everything sharded
                input_sharded = True
                output_sharded = True

            self.resnets.append(
                ShardedWanResidualBlock(resnet, tp_degree, input_sharded=input_sharded, output_sharded=output_sharded)
            )

        if hasattr(original_up_block, 'upsampler') and original_up_block.upsampler is not None:
            self.upsampler = ShardedWanResample(original_up_block.upsampler, tp_degree)
        else:
            self.upsampler = None

        # NOTE: avg_shortcut is disabled in TP version because it expects full channel count
        # and performs complex reshape operations incompatible with sharded tensors.
        # This is a residual connection that can be safely skipped for compilation.
        self.avg_shortcut = None

    def forward(self, x, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None):
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx)

        # avg_shortcut disabled in TP version

        return x


class ShardedWanDecoder(nn.Module):
    """
    Tensor Parallel Wan Decoder with feat_cache support.
    """
    def __init__(self, original_decoder, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree

        # conv_in: Column parallel
        self.conv_in = ColumnParallelConv3d(original_decoder.conv_in, tp_degree)

        self.nonlinearity = original_decoder.nonlinearity

        # Mid block
        self.mid_block = ShardedWanMidBlock(original_decoder.mid_block, tp_degree)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        for block_idx, up_block in enumerate(original_decoder.up_blocks):
            is_last_block = (block_idx == len(original_decoder.up_blocks) - 1)
            self.up_blocks.append(
                ShardedWanResidualUpBlock(up_block, tp_degree, is_last_block=is_last_block)
            )

        # Output layers (not sharded - input already gathered from last resnet)
        self.norm_out = original_decoder.norm_out
        self.conv_out = original_decoder.conv_out

    def forward(self, x, feat_cache: Optional[List[torch.Tensor]] = None, feat_idx: Optional[List[int]] = None):
        if feat_idx is None:
            feat_idx = [0]

        # conv_in with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        # mid_block
        x = self.mid_block(x, feat_cache, feat_idx)

        # up_blocks
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        # Output
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        # conv_out with cache
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class DecoderWrapperTP(nn.Module):
    """Wrapper for parallel tracing with feat_cache support."""
    def __init__(self, decoder, tp_degree):
        super().__init__()
        self.decoder = ShardedWanDecoder(decoder, tp_degree)

    def forward(self, x, feat_cache: List[torch.Tensor]):
        return self.decoder(x, feat_cache)


def get_decoder_model(tp_degree):
    """Factory function for parallel_model_trace."""
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
    )

    decoder = vae.decoder
    decoder.eval()

    wrapped_decoder = DecoderWrapperTP(decoder, tp_degree)
    return wrapped_decoder, {}


def compile_decoder_tp(args):
    latent_height = args.height // 16
    latent_width = args.width // 16
    tp_degree = args.tp_degree
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir

    os.environ["LOCAL_WORLD_SIZE"] = str(tp_degree)

    batch_size = 1
    decoder_frames = 2
    in_channels = 48

    print(f"Compiling decoder with TP={tp_degree} for {args.height}x{args.width}")
    print(f"Latent size: {latent_height}x{latent_width}")

    get_decoder_f = partial(get_decoder_model, tp_degree)

    with torch.no_grad():
        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=torch.float32
        )

        # Based on decoder structure, create feat_cache
        # CORRECTED: conv_shortcut does NOT use feat_cache (only conv1 and conv2 do)
        # Total 32 entries:
        # - conv_in: 1
        # - mid_block: 4 (2 resnets × 2 convs)
        # - up_block 0: 7 (3 resnets × 2 convs + 1 upsampler.time_conv)
        # - up_block 1: 7 (3 resnets × 2 convs + 1 upsampler.time_conv)
        # - up_block 2: 6 (3 resnets × 2 convs, no time_conv for upsample2d)
        # - up_block 3: 6 (3 resnets × 2 convs, no upsampler)
        # - conv_out: 1
        feat_cache = [
            # [0] conv_in: input 48 channels (not sharded)
            torch.rand((batch_size, 48, 2, latent_height, latent_width), dtype=torch.float32),
            # [1-4] mid_block: 1024 channels sharded
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # mid.resnet0.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # mid.resnet0.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # mid.resnet1.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # mid.resnet1.conv2
            # [5-11] up_block 0: 1024 channels sharded, spatial=latent_height×latent_width
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.0.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.0.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.1.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.1.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.2.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # resnets.2.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height, latent_width), dtype=torch.float32),  # upsampler.time_conv (BEFORE spatial upsample)
            # [12-18] up_block 1: 1024 channels sharded, spatial=latent_height*2×latent_width*2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.0.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.0.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.1.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.1.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.2.conv1
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # resnets.2.conv2
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # upsampler.time_conv (BEFORE spatial upsample)
            # [19-24] up_block 2: 1024->512 channels, spatial=latent_height*4×latent_width*4
            # Note: upsample2d has no time_conv, so no upsampler cache entry
            torch.rand((batch_size, 1024 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # resnets.0.conv1 (input 1024)
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),   # resnets.0.conv2 (output 512)
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),   # resnets.1.conv1
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),   # resnets.1.conv2
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),   # resnets.2.conv1
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*4, latent_width*4), dtype=torch.float32),   # resnets.2.conv2
            # [25-30] up_block 3 (last block): 512->256 channels, spatial=latent_height*8×latent_width*8
            # Note: no upsampler in last block
            # For last block: resnet.0 has input_sharded=True, output_sharded=False
            # conv1: ColumnRowParallel (input sharded, output sharded) -> cache needs sharded shape
            # conv2: RowParallel (input sharded, output gathered) -> cache needs sharded shape
            torch.rand((batch_size, 512 // tp_degree, 2, latent_height*8, latent_width*8), dtype=torch.float32),   # resnets.0.conv1 (input 512 sharded)
            torch.rand((batch_size, 256 // tp_degree, 2, latent_height*8, latent_width*8), dtype=torch.float32),   # resnets.0.conv2 (input 256 sharded, output gathered)
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),                 # resnets.1.conv1 (gathered)
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),                 # resnets.1.conv2 (gathered)
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),                 # resnets.2.conv1 (gathered)
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),                 # resnets.2.conv2 (gathered)
            # [31] conv_out: 256 channels (gathered)
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),                 # conv_out (gathered)
        ]

        sample_inputs = (decoder_input, feat_cache)

        compiled_decoder = neuronx_distributed.trace.parallel_model_trace(
            get_decoder_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/decoder_tp",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )

        compiled_model_dir = f"{compiled_models_dir}/decoder_tp"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)

        neuronx_distributed.trace.parallel_model_save(
            compiled_decoder, compiled_model_dir
        )

        print(f"Decoder TP compiled and saved to {compiled_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    args = parser.parse_args()

    compile_decoder_tp(args)
