"""
Wan2.2 VAE Decoder Compilation - Optimized V2.

Key optimizations:
1. Optimized compiler flags (-O1)
2. Standard attention (NKI Flash Attention disabled - causes OOM due to spill buffers)

Note: world_size must match the transformer's NxDParallelState context.
The decoder weights are duplicated (not sharded) across all ranks.
"""
import os
import json
import math

os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"  # Required for NKI
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Optimized compiler flags
compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference -O1 --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import List, Optional

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file

# Import NKI Flash Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

print("NKI Flash Attention kernel loaded for decoder optimization")


def nki_flash_attention_2d(query, key, value):
    """
    NKI Flash Attention for 2D spatial attention (used in VAE decoder mid_block).

    Args:
        query: [B*T, H, N, D] where N = height * width
        key: [B*T, H, N, D]
        value: [B*T, H, N, D]

    Returns:
        attention output [B*T, H, N, D]
    """
    bt, n_head, seq_len, d_head = query.shape

    # Reshape for NKI kernel: (B*H, D, S) for Q/K, (B*H, S, D) for V
    q = query.reshape(bt * n_head, seq_len, d_head).permute(0, 2, 1).contiguous()  # [B*H, D, S]
    k = key.reshape(bt * n_head, seq_len, d_head).permute(0, 2, 1).contiguous()    # [B*H, D, S]
    v = value.reshape(bt * n_head, seq_len, d_head).contiguous()                    # [B*H, S, D]

    attn_output = torch.zeros((bt * n_head, seq_len, d_head), dtype=q.dtype, device=q.device)
    scale = 1.0 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    # Reshape back: [B*H, S, D] -> [B*T, H, N, D]
    return attn_output.reshape(bt, n_head, seq_len, d_head)


class NKIAttentionBlock(nn.Module):
    """
    Attention block with NKI Flash Attention for VAE decoder.
    Replaces the original WanAttentionBlock's scaled_dot_product_attention.
    """
    def __init__(self, original_attn):
        super().__init__()
        self.norm = original_attn.norm
        self.to_qkv = original_attn.to_qkv
        self.proj = original_attn.proj

        # Get number of heads from channel configuration
        # to_qkv outputs 3 * channels, so channels = out_channels / 3
        out_channels = original_attn.to_qkv.out_channels
        self.channels = out_channels // 3
        self.heads = 8  # Wan decoder uses 8 heads
        self.head_dim = self.channels // self.heads

    def forward(self, x):
        batch, channels, frames, height, width = x.shape

        # Process each frame as 2D attention
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

        # Normalize
        x_normed = self.norm(x_2d)

        # QKV projection
        qkv = self.to_qkv(x_normed)  # [B*T, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B*T, C, H, W]

        # Reshape for attention: [B*T, H, N, D] where N = H*W
        seq_len = height * width
        q = q.reshape(batch * frames, self.heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        k = k.reshape(batch * frames, self.heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        v = v.reshape(batch * frames, self.heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        # NKI Flash Attention
        attn_out = nki_flash_attention_2d(q, k, v)

        # Reshape back: [B*T, H, N, D] -> [B*T, C, H, W]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(batch * frames, self.channels, height, width)

        # Output projection
        out = self.proj(attn_out)

        # Reshape to 5D and add residual
        out = out.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

        return x + out


def replace_attention_with_nki(decoder):
    """Replace mid_block attention with NKI version."""
    if hasattr(decoder, 'mid_block') and hasattr(decoder.mid_block, 'attentions'):
        for i, attn in enumerate(decoder.mid_block.attentions):
            if attn is not None:
                decoder.mid_block.attentions[i] = NKIAttentionBlock(attn)
                print(f"  Replaced mid_block.attentions[{i}] with NKI version")
    return decoder


class DecoderWrapper(nn.Module):
    """
    Wrapper for VAE decoder to handle feat_cache as individual tensor arguments.
    ModelBuilder requires all inputs to be tensors.
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x,
                feat_cache_0, feat_cache_1, feat_cache_2, feat_cache_3, feat_cache_4,
                feat_cache_5, feat_cache_6, feat_cache_7, feat_cache_8, feat_cache_9,
                feat_cache_10, feat_cache_11, feat_cache_12, feat_cache_13, feat_cache_14,
                feat_cache_15, feat_cache_16, feat_cache_17, feat_cache_18, feat_cache_19,
                feat_cache_20, feat_cache_21, feat_cache_22, feat_cache_23, feat_cache_24,
                feat_cache_25, feat_cache_26, feat_cache_27, feat_cache_28, feat_cache_29,
                feat_cache_30, feat_cache_31, feat_cache_32, feat_cache_33):
        feat_cache = [
            feat_cache_0, feat_cache_1, feat_cache_2, feat_cache_3, feat_cache_4,
            feat_cache_5, feat_cache_6, feat_cache_7, feat_cache_8, feat_cache_9,
            feat_cache_10, feat_cache_11, feat_cache_12, feat_cache_13, feat_cache_14,
            feat_cache_15, feat_cache_16, feat_cache_17, feat_cache_18, feat_cache_19,
            feat_cache_20, feat_cache_21, feat_cache_22, feat_cache_23, feat_cache_24,
            feat_cache_25, feat_cache_26, feat_cache_27, feat_cache_28, feat_cache_29,
            feat_cache_30, feat_cache_31, feat_cache_32, feat_cache_33
        ]
        return self.decoder(x, feat_cache)


class PostQuantConvWrapper(nn.Module):
    """Wrapper for post_quant_conv."""
    def __init__(self, post_quant_conv):
        super().__init__()
        self.conv = post_quant_conv

    def forward(self, x):
        return self.conv(x)


def save_model_config(output_path, config):
    """Save model configuration."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_v2_optimized(args):
    """Compile optimized VAE decoder using ModelBuilder."""
    latent_height = args.height // 16
    latent_width = args.width // 16
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = 2  # CACHE_T=2
    latent_frames = (args.num_frames - 1) // 4 + 1
    in_channels = 48

    print("=" * 60)
    print("Wan2.2 VAE Decoder V2 Optimized Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"num_frames={args.num_frames} -> latent_frames={latent_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Compiler: -O1 optimization")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
    )

    # Compile with matching world_size/tp_degree to be compatible with transformer
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Decoder ==========
        print("\nPreparing decoder...")
        decoder = vae.decoder
        decoder.eval()

        # NOTE: NKI Flash Attention disabled - causes OOM due to large spill buffers
        # The decoder mid_block attention has small sequence length (32x32=1024),
        # so standard SDPA is efficient enough.
        # decoder = replace_attention_with_nki(decoder)

        # Prepare inputs
        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=torch.float32
        )

        # Create feat_cache
        feat_cache = [
            torch.rand((batch_size, 48, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),
            torch.rand((batch_size, 1024, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),
            torch.rand((batch_size, 512, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),
            torch.rand((batch_size, 12, 2, latent_height*8, latent_width*8), dtype=torch.float32),
        ]

        # Wrap decoder
        decoder_wrapper = DecoderWrapper(decoder)

        # Build trace kwargs
        trace_kwargs = {"x": decoder_input}
        for i, fc in enumerate(feat_cache):
            trace_kwargs[f"feat_cache_{i}"] = fc

        # Initialize ModelBuilder
        print("\nInitializing ModelBuilder for decoder...")
        decoder_builder = ModelBuilder(model=decoder_wrapper)

        print("Tracing decoder...")
        decoder_builder.trace(
            kwargs=trace_kwargs,
            tag="decode",
        )

        print("Compiling decoder...")
        traced_decoder = decoder_builder.compile()

        # Save decoder
        decoder_output_path = f"{compiled_models_dir}/decoder_v2"
        os.makedirs(decoder_output_path, exist_ok=True)
        print(f"Saving decoder to {decoder_output_path}...")
        traced_decoder.save(os.path.join(decoder_output_path, "nxd_model.pt"))

        # Save weights (single checkpoint, will be duplicated at runtime)
        print("Saving decoder weights...")
        decoder_weights_path = os.path.join(decoder_output_path, "weights")
        os.makedirs(decoder_weights_path, exist_ok=True)
        decoder_checkpoint = decoder_wrapper.state_dict()
        save_file(decoder_checkpoint, os.path.join(decoder_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        decoder_config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "latent_frames": latent_frames,
            "decoder_frames": decoder_frames,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "nki_flash_attention": False,  # Disabled to avoid OOM
        }
        save_model_config(decoder_output_path, decoder_config)

        # ========== Compile post_quant_conv ==========
        print("\nCompiling post_quant_conv...")
        post_quant_conv_wrapper = PostQuantConvWrapper(vae.post_quant_conv)

        post_quant_conv_input = torch.rand(
            (batch_size, in_channels, latent_frames, latent_height, latent_width),
            dtype=torch.float32
        )

        pqc_builder = ModelBuilder(model=post_quant_conv_wrapper)

        print("Tracing post_quant_conv...")
        pqc_builder.trace(
            kwargs={"x": post_quant_conv_input},
            tag="conv",
        )

        print("Compiling post_quant_conv...")
        traced_pqc = pqc_builder.compile()

        # Save post_quant_conv
        pqc_output_path = f"{compiled_models_dir}/post_quant_conv_v2"
        os.makedirs(pqc_output_path, exist_ok=True)
        print(f"Saving post_quant_conv to {pqc_output_path}...")
        traced_pqc.save(os.path.join(pqc_output_path, "nxd_model.pt"))

        # Save weights
        print("Saving post_quant_conv weights...")
        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        pqc_checkpoint = post_quant_conv_wrapper.state_dict()
        save_file(pqc_checkpoint, os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        pqc_config = {
            "batch_size": batch_size,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
        }
        save_model_config(pqc_output_path, pqc_config)

        print("\n" + "=" * 60)
        print("Compilation Complete!")
        print("=" * 60)
        print(f"Decoder saved to: {decoder_output_path}")
        print(f"post_quant_conv saved to: {pqc_output_path}")
        print("\nConfiguration:")
        print(f"  - World size: {world_size}, TP: {tp_degree}")
        print("  - Compiler: -O1 optimization")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Optimized Wan2.2 VAE Decoder")
    parser.add_argument("--height", type=int, default=512, help="Height of generated video")
    parser.add_argument("--width", type=int, default=512, help="Width of generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--tp_degree", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--world_size", type=int, default=8, help="World size (must match transformer)")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    args = parser.parse_args()

    compile_decoder_v2_optimized(args)
