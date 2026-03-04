"""
Wan2.2 T2V-A14B VAE Decoder Compilation - NoCache (Zero-argument feat_cache).

Adapted from TI2V compile_decoder_v3_nocache.py for T2V-A14B:
- in_channels = 16 (was 48 for TI2V)
- Resolution: 480x832 (latent 30x52, was 512x512 / 32x32)
- feat_cache shapes computed dynamically from VAE decoder structure

Key insight: feat_cache is internalized as registered buffers (loaded once to device),
reducing NxDModel arguments from 35 to 1. Only x (~300KB) is transferred per call.
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
import argparse
from functools import reduce
import operator

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file


def compute_feat_cache_shapes(decoder, batch_size, in_channels, latent_height, latent_width, dtype=torch.bfloat16):
    """
    Compute feat_cache shapes dynamically by running a dummy forward pass.

    This avoids hardcoding channel widths that differ between TI2V-5B (z_dim=48)
    and T2V-A14B (z_dim=16) VAEs.
    """
    decoder_cpu = decoder.cpu().eval()

    # Create dummy input
    dummy_x = torch.zeros(batch_size, in_channels, 2, latent_height, latent_width, dtype=torch.float32)

    # Initialize feat_cache with None to discover shapes
    num_caches = 34  # Wan VAE decoder uses 34 feat_cache entries
    feat_cache = [None] * num_caches

    # Run forward pass to populate feat_cache shapes
    with torch.no_grad():
        try:
            _ = decoder_cpu(dummy_x, feat_cache=feat_cache)
        except Exception:
            pass  # Some decoders may error but still populate feat_cache

    # Extract shapes from populated feat_cache
    shapes = []
    for i, cache in enumerate(feat_cache):
        if cache is not None and isinstance(cache, torch.Tensor):
            # Use the actual shape from the forward pass, but fix batch and temporal dims
            shape = list(cache.shape)
            shape[0] = batch_size
            shape[2] = 2  # CACHE_T=2
            shapes.append(tuple(shape))
        else:
            # This entry wasn't used or is a non-tensor sentinel; provide a small placeholder
            shapes.append((batch_size, 1, 2, 1, 1))

    print(f"  Computed {len(shapes)} feat_cache shapes dynamically")
    for i, s in enumerate(shapes):
        print(f"    [{i:2d}] {s}")

    return shapes


def get_feat_cache_shapes_and_none_indices(batch_size, in_channels, latent_height, latent_width, dtype=torch.bfloat16):
    """
    Get feat_cache shapes and None indices for Wan2.2-T2V-A14B VAE (z_dim=16, base_dim=96).

    Actual architecture (from dynamic discovery):
    - conv_in: 16 channels (z_dim)
    - mid_block: 384 channels (base_dim*4) — 4 entries (2 resnets × 2 convs)
    - up_block_0: 384 channels — 6 entries (3 resnets × 2 convs)
    - up_block_0 upsampler: None entry (first-chunk 'Rep' sentinel)
    - up_block_1: 192/384 channels, 2x spatial — 6 entries
    - up_block_1 upsampler: None entry (first-chunk 'Rep' sentinel)
    - up_block_2: 192 channels, 4x spatial — 6 entries
    - up_block_3: 96 channels, 8x spatial — 7 entries (3 resnets × 2 + conv_out)
    - Entries 32, 33: unused (None)

    Returns:
        (shapes, none_indices): shapes list and set of indices that should be None
    """
    lh, lw = latent_height, latent_width

    # Indices that should be None (upsampler 'Rep' sentinels + unused entries)
    none_indices = {11, 18, 32, 33}

    shapes = [
        (batch_size, 16, 2, lh, lw),            # 0: conv_in
        (batch_size, 384, 2, lh, lw),            # 1: mid_block resnet_0 conv1
        (batch_size, 384, 2, lh, lw),            # 2: mid_block resnet_0 conv2
        (batch_size, 384, 2, lh, lw),            # 3: mid_block resnet_1 conv1
        (batch_size, 384, 2, lh, lw),            # 4: mid_block resnet_1 conv2
        (batch_size, 384, 2, lh, lw),            # 5: up_block_0 resnet_0 conv1
        (batch_size, 384, 2, lh, lw),            # 6: up_block_0 resnet_0 conv2
        (batch_size, 384, 2, lh, lw),            # 7: up_block_0 resnet_1 conv1
        (batch_size, 384, 2, lh, lw),            # 8: up_block_0 resnet_1 conv2
        (batch_size, 384, 2, lh, lw),            # 9: up_block_0 resnet_2 conv1
        (batch_size, 384, 2, lh, lw),            # 10: up_block_0 resnet_2 conv2
        None,                                     # 11: up_block_0 upsampler (Rep)
        (batch_size, 192, 2, lh*2, lw*2),        # 12: up_block_1 resnet_0 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 13: up_block_1 resnet_0 conv2
        (batch_size, 384, 2, lh*2, lw*2),        # 14: up_block_1 resnet_1 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 15: up_block_1 resnet_1 conv2
        (batch_size, 384, 2, lh*2, lw*2),        # 16: up_block_1 resnet_2 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 17: up_block_1 resnet_2 conv2
        None,                                     # 18: up_block_1 upsampler (Rep)
        (batch_size, 192, 2, lh*4, lw*4),        # 19: up_block_2 resnet_0 conv1
        (batch_size, 192, 2, lh*4, lw*4),        # 20: up_block_2 resnet_0 conv2
        (batch_size, 192, 2, lh*4, lw*4),        # 21: up_block_2 resnet_1 conv1
        (batch_size, 192, 2, lh*4, lw*4),        # 22: up_block_2 resnet_1 conv2
        (batch_size, 192, 2, lh*4, lw*4),        # 23: up_block_2 resnet_2 conv1
        (batch_size, 192, 2, lh*4, lw*4),        # 24: up_block_2 resnet_2 conv2
        (batch_size, 96, 2, lh*8, lw*8),         # 25: up_block_3 resnet_0 conv1
        (batch_size, 96, 2, lh*8, lw*8),         # 26: up_block_3 resnet_0 conv2
        (batch_size, 96, 2, lh*8, lw*8),         # 27: up_block_3 resnet_1 conv1
        (batch_size, 96, 2, lh*8, lw*8),         # 28: up_block_3 resnet_1 conv2
        (batch_size, 96, 2, lh*8, lw*8),         # 29: up_block_3 resnet_2 conv1
        (batch_size, 96, 2, lh*8, lw*8),         # 30: up_block_3 resnet_2 conv2
        (batch_size, 96, 2, lh*8, lw*8),         # 31: conv_out input cache
        None,                                     # 32: unused
        None,                                     # 33: unused
    ]

    return shapes, none_indices


class DecoderWrapperNoCache(nn.Module):
    """
    Decoder wrapper with feat_cache as registered buffers (not input arguments).

    Eliminates ~960MB per-call data transfer by keeping feat_cache on device.
    Only x is transferred per call.

    Some feat_cache entries are None (upsampler 'Rep' sentinels for first-chunk
    behavior, and unused entries). These are NOT registered as buffers.
    """

    def __init__(self, decoder, feat_cache_shapes, none_indices=None, dtype=torch.bfloat16):
        super().__init__()
        self.decoder = decoder
        self.num_feat_cache = len(feat_cache_shapes)
        self.none_indices = none_indices or set()

        # Register feat_cache as persistent buffers (skip None entries)
        for i, shape in enumerate(feat_cache_shapes):
            if i not in self.none_indices and shape is not None:
                self.register_buffer(f'feat_cache_{i}', torch.zeros(shape, dtype=dtype))

    def forward(self, x):
        feat_cache = []
        for i in range(self.num_feat_cache):
            if i in self.none_indices:
                feat_cache.append(None)
            else:
                feat_cache.append(getattr(self, f'feat_cache_{i}'))
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
    latent_height = args.height // 8  # T2V-A14B VAE scale_factor_spatial=8
    latent_width = args.width // 8
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = args.decoder_frames
    in_channels = 16  # T2V-A14B uses z_dim=16 (was 48 for TI2V)
    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 T2V-A14B VAE Decoder NoCache Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"in_channels (z_dim): {in_channels}")
    print(f"Decoder frames: {decoder_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Key: feat_cache as buffers -> only 1 input argument")
    print("=" * 60)

    if not args.compile_post_quant_conv:
        # ========== Compile Decoder ==========
        print("\nLoading VAE...")
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=args.cache_dir,
        )

        with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
            print("\nGetting feat_cache shapes...")
            feat_cache_shapes, none_indices = get_feat_cache_shapes_and_none_indices(
                batch_size, in_channels, latent_height, latent_width, dtype
            )
            print(f"  {len(feat_cache_shapes)} entries, {len(none_indices)} None indices: {sorted(none_indices)}")
            for i, s in enumerate(feat_cache_shapes):
                print(f"    [{i:2d}] {'None' if s is None else s}")

            print("\nPreparing decoder (bfloat16, no external feat_cache)...")
            decoder = vae.decoder.to(dtype).eval()
            wrapper = DecoderWrapperNoCache(decoder, feat_cache_shapes, none_indices, dtype)

            decoder_input = torch.rand(
                (batch_size, in_channels, decoder_frames, latent_height, latent_width),
                dtype=dtype,
            )

            print(f"  Input: {decoder_input.shape} ({decoder_input.nelement()*2/1024:.0f}KB)")
            buffer_elements = sum(reduce(operator.mul, s) for s in feat_cache_shapes if s is not None)
            print(f"  Buffers: {buffer_elements*2/1024/1024:.0f}MB (on device)")

            builder = ModelBuilder(model=wrapper)
            print("Tracing...")
            builder.trace(kwargs={"x": decoder_input}, tag="decode")

            print("Compiling...")
            compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
            traced = builder.compile(
                compiler_args=compile_args,
                compiler_workdir=args.compiler_workdir,
            )

            # Save
            output_path = f"{compiled_models_dir}/decoder_nocache"
            os.makedirs(output_path, exist_ok=True)
            print(f"Saving to {output_path}...")
            traced.save(os.path.join(output_path, "nxd_model.pt"))

            # Save weights (includes decoder weights + feat_cache buffers)
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

            print(f"\nDecoder saved to {output_path}")

    else:
        # ========== Compile post_quant_conv only ==========
        print("\nLoading VAE for post_quant_conv...")
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=args.cache_dir,
        )

        with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
            latent_frames = (args.num_frames - 1) // 4 + 1
            print(f"\nCompiling post_quant_conv (float32)...")
            pqc_wrapper = PostQuantConvWrapper(vae.post_quant_conv)
            pqc_input = torch.rand(
                (batch_size, in_channels, latent_frames, latent_height, latent_width),
                dtype=torch.float32,
            )

            pqc_builder = ModelBuilder(model=pqc_wrapper)
            pqc_builder.trace(kwargs={"x": pqc_input}, tag="conv")
            traced_pqc = pqc_builder.compile(
                compiler_args="--model-type=unet-inference -O1 --auto-cast=none",
                compiler_workdir=args.compiler_workdir,
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

            print(f"\npost_quant_conv saved to {pqc_output_path}")

    print("\n" + "=" * 60)
    print("Compilation Complete!")
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
    parser.add_argument("--cache_dir", type=str, default="/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir")
    parser.add_argument("--compile_post_quant_conv", action="store_true",
                        help="Only compile post_quant_conv (skip decoder)")
    args = parser.parse_args()

    compile_decoder_nocache(args)
