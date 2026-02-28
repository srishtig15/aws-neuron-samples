"""
Compile Wan 3D VAE Decoder for RollingForcing on Neuron (NoCache pattern).

Adapted from Wan2.2's compile_decoder_v3_nocache.py.
- Resolution: 480x832 -> latent 60x104
- feat_cache internalized as registered buffers (no per-call transfer)
- Decoder processes 2 frames at a time: input [1, 16, 2, 60, 104]
  (CACHE_T=2 in WanDecoder3d requires >= 2 temporal frames for XLA tracing)
- Also compiles post_quant_conv

Wan2.1 VAE (1.3B): base_dim=96, dim_mult=[1,2,4,4], z_dim=16
  Decoder channels: 384 -> 384 -> 192 -> 96
  32 feat_cache entries (indices 11, 18 are temporal upsample markers)
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = " --target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers import AutoencoderKLWan
import torch
import torch.nn as nn
import argparse
from functools import reduce
import operator

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file


def get_feat_cache_shapes(batch_size, latent_height, latent_width, dtype=torch.bfloat16):
    """Return the 32 feat_cache tensor shapes for the Wan2.1 decoder.

    Wan2.1 VAE: base_dim=96, dim_mult=[1,2,4,4], z_dim=16
    Decoder channels: 384 -> 384 -> 192 -> 96
    Spatial: lh x lw -> 2*lh x 2*lw -> 4*lh x 4*lw -> 8*lh x 8*lw

    Indices 11 and 18 are temporal upsample caches (originally 'Rep' markers).
    We register them as zero tensors so the decoder takes the cached code path.
    """
    lh, lw = latent_height, latent_width
    return [
        # 0: conv_in cache
        (batch_size, 16, 2, lh, lw),
        # 1-4: mid_block resnets (2 resnets × 2 convs each)
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        # 5-10: up_block_0 resnets (3 resnets × 2 convs each), spatial lh×lw
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        (batch_size, 384, 2, lh, lw),
        # 11: up_block_0 temporal upsample cache (originally 'Rep')
        (batch_size, 384, 2, lh, lw),
        # 12-17: up_block_1 resnets, spatial 2*lh × 2*lw
        (batch_size, 192, 2, lh*2, lw*2),       # first resnet has channel change
        (batch_size, 384, 2, lh*2, lw*2),
        (batch_size, 384, 2, lh*2, lw*2),
        (batch_size, 384, 2, lh*2, lw*2),
        (batch_size, 384, 2, lh*2, lw*2),
        (batch_size, 384, 2, lh*2, lw*2),
        # 18: up_block_1 temporal upsample cache (originally 'Rep')
        (batch_size, 384, 2, lh*2, lw*2),
        # 19-24: up_block_2 resnets, spatial 4*lh × 4*lw
        (batch_size, 192, 2, lh*4, lw*4),
        (batch_size, 192, 2, lh*4, lw*4),
        (batch_size, 192, 2, lh*4, lw*4),
        (batch_size, 192, 2, lh*4, lw*4),
        (batch_size, 192, 2, lh*4, lw*4),
        (batch_size, 192, 2, lh*4, lw*4),
        # 25-30: up_block_3 resnets, spatial 8*lh × 8*lw
        (batch_size, 96, 2, lh*8, lw*8),
        (batch_size, 96, 2, lh*8, lw*8),
        (batch_size, 96, 2, lh*8, lw*8),
        (batch_size, 96, 2, lh*8, lw*8),
        (batch_size, 96, 2, lh*8, lw*8),
        (batch_size, 96, 2, lh*8, lw*8),
        # 31: conv_out cache
        (batch_size, 96, 2, lh*8, lw*8),
    ]


class DecoderWrapperNoCache(nn.Module):
    """
    Decoder wrapper with feat_cache as registered buffers.

    Eliminates ~960MB per-call data transfer by keeping feat_cache on device.
    Only x (~300KB) is transferred per call.
    """
    NUM_FEAT_CACHE = 32

    def __init__(self, decoder, feat_cache_shapes, dtype=torch.bfloat16):
        super().__init__()
        self.decoder = decoder
        for i, shape in enumerate(feat_cache_shapes):
            self.register_buffer(f'feat_cache_{i}', torch.zeros(shape, dtype=dtype))

    def forward(self, x):
        feat_cache = [
            getattr(self, f'feat_cache_{i}')
            for i in range(self.NUM_FEAT_CACHE)
        ]
        return self.decoder(x, feat_cache)


class PostQuantConvWrapper(nn.Module):
    def __init__(self, post_quant_conv):
        super().__init__()
        self.conv = post_quant_conv

    def forward(self, x):
        return self.conv(x)


def save_model_config(output_path, config):
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_nocache(args):
    """Compile VAE decoder with NoCache pattern for RollingForcing."""
    latent_height = args.height // 8  # Wan2.1 VAE spatial downscale = 8
    latent_width = args.width // 8
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = args.decoder_frames
    in_channels = 16  # Wan2.1 latent channels = 16
    dtype = torch.bfloat16

    print("=" * 60)
    print("RollingForcing VAE Decoder NoCache Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"Decoder frames: {decoder_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Key: feat_cache as buffers -> only 1 input argument")
    print("=" * 60)

    # Load VAE from Wan2.1
    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )

    # Patch WanUpsample2d to use 'nearest' instead of 'nearest-exact'
    # (Neuron doesn't support _upsample_nearest_exact2d)
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample
    _orig_upsample_forward = WanUpsample.forward
    def _patched_upsample_forward(self, x):
        # Replace nearest-exact with nearest for Neuron compatibility
        return torch.nn.functional.interpolate(
            x.float(), scale_factor=2.0, mode='nearest').type_as(x)
    WanUpsample.forward = _patched_upsample_forward
    print("  Patched WanUpsample: nearest-exact -> nearest")

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Decoder (bfloat16, 1 input arg) ==========
        print("\nPreparing decoder (bfloat16, no external feat_cache)...")
        decoder = vae.decoder.to(dtype).eval()

        feat_cache_shapes = get_feat_cache_shapes(batch_size, latent_height, latent_width, dtype)
        wrapper = DecoderWrapperNoCache(decoder, feat_cache_shapes, dtype)

        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=dtype,
        )

        print(f"  Input: {decoder_input.shape} ({decoder_input.nelement()*2/1024:.0f}KB)")
        print(f"  Buffers: {sum(reduce(operator.mul, s) for s in feat_cache_shapes)*2/1024/1024:.0f}MB (on device)")

        builder = ModelBuilder(model=wrapper)
        print("Tracing...")
        builder.trace(kwargs={"x": decoder_input}, tag="decode")

        print("Compiling...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        # Use absolute path to avoid doubled --logfile path in neuronx-cc
        abs_compiler_workdir = os.path.abspath(args.compiler_workdir)
        traced = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=abs_compiler_workdir,
        )

        # Save
        output_path = f"{compiled_models_dir}/decoder_nocache"
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving to {output_path}...")
        traced.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)
        checkpoint = wrapper.state_dict()
        save_file(checkpoint, os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save config
        config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "decoder_frames": decoder_frames,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "bfloat16",
            "nocache": True,
        }
        save_model_config(output_path, config)

        # ========== Compile post_quant_conv (float32) ==========
        latent_frames = (args.num_frames - 1) // 4 + 1
        print("\nCompiling post_quant_conv (float32)...")
        pqc_wrapper = PostQuantConvWrapper(vae.post_quant_conv)
        pqc_input = torch.rand(
            (batch_size, in_channels, latent_frames, latent_height, latent_width),
            dtype=torch.float32,
        )

        pqc_builder = ModelBuilder(model=pqc_wrapper)
        pqc_builder.trace(kwargs={"x": pqc_input}, tag="conv")
        traced_pqc = pqc_builder.compile(
            compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
            compiler_workdir=abs_compiler_workdir,
        )

        pqc_output_path = f"{compiled_models_dir}/post_quant_conv"
        os.makedirs(pqc_output_path, exist_ok=True)
        traced_pqc.save(os.path.join(pqc_output_path, "nxd_model.pt"))

        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        pqc_checkpoint = pqc_wrapper.state_dict()
        save_file(pqc_checkpoint, os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

        pqc_config = {
            "batch_size": batch_size,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "in_channels": in_channels,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "float32",
        }
        save_model_config(pqc_output_path, pqc_config)

        print("\n" + "=" * 60)
        print("Compilation Complete!")
        print(f"Decoder: {output_path}")
        print(f"post_quant_conv: {pqc_output_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--decoder_frames", type=int, default=2)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    parser.add_argument("--cache_dir", type=str,
                       default="/opt/dlami/nvme/rolling_forcing_hf_cache")
    args = parser.parse_args()

    compile_decoder_nocache(args)
