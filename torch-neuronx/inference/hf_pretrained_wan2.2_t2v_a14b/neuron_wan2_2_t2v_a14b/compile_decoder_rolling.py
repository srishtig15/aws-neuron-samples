"""
Wan2.2 T2V-A14B VAE Decoder Compilation - Rolling feat_cache.

Unlike NoCache mode (feat_cache as zero buffers), this approach passes
feat_cache as explicit inputs AND outputs:
  Inputs:  x [B,C,T,H,W] + 34 cache tensors
  Outputs: video [B,3,T*4,H*8,W*8] + 34 updated cache tensors

This allows carrying temporal context between decoder calls, eliminating
the flickering artifacts caused by zero temporal context in NoCache mode.

Trade-off: ~1.8GB extra transfer per decoder call (in + out), but produces
temporally coherent video matching CPU VAE decode quality.
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


def get_feat_cache_shapes(batch_size, latent_height, latent_width, dtype=torch.bfloat16):
    """
    Return the 34 feat_cache tensor shapes for the Wan2.2-T2V-A14B VAE decoder.

    ALL 34 entries must be zero tensors (not None). Passing zero tensors ensures
    the temporal upsample path (t -> t*2) is traced and compiled correctly.
    """
    lh, lw = latent_height, latent_width

    return [
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
        (batch_size, 384, 2, lh, lw),            # 11: up_block_0 upsampler
        (batch_size, 192, 2, lh*2, lw*2),        # 12: up_block_1 resnet_0 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 13: up_block_1 resnet_0 conv2
        (batch_size, 384, 2, lh*2, lw*2),        # 14: up_block_1 resnet_1 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 15: up_block_1 resnet_1 conv2
        (batch_size, 384, 2, lh*2, lw*2),        # 16: up_block_1 resnet_2 conv1
        (batch_size, 384, 2, lh*2, lw*2),        # 17: up_block_1 resnet_2 conv2
        (batch_size, 384, 2, lh*2, lw*2),        # 18: up_block_1 upsampler
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
        (batch_size, 96, 2, lh*8, lw*8),         # 32: placeholder (unused)
        (batch_size, 12, 2, lh*8, lw*8),         # 33: placeholder (unused)
    ]


class DecoderWrapperRolling(nn.Module):
    """
    Decoder wrapper with feat_cache as explicit inputs AND outputs.

    Unlike NoCache (feat_cache as zero buffers), this carries temporal context
    between calls by returning the updated feat_cache as additional outputs.

    Forward signature: (x, c0, c1, ..., c33) -> (output, c0, c1, ..., c33)
    - 35 inputs (x + 34 caches)
    - 35 outputs (video + 34 updated caches)
    """
    NUM_FEAT_CACHE = 34

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

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
        output = self.decoder(x, feat_cache)
        # decoder replaces feat_cache list elements in-place (feat_cache[idx] = new_tensor)
        # Return updated cache tensors as additional outputs
        return tuple([output] + feat_cache)


def save_model_config(output_path, config):
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_rolling(args):
    latent_height = args.height // 8
    latent_width = args.width // 8
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = args.decoder_frames
    in_channels = 16
    dtype = torch.bfloat16

    print("=" * 60)
    print("Wan2.2 T2V-A14B VAE Decoder Rolling Cache Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {latent_height}x{latent_width}")
    print(f"in_channels (z_dim): {in_channels}")
    print(f"Decoder frames: {decoder_frames}")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Key: feat_cache as I/O -> 35 inputs, 35 outputs")
    print("=" * 60)

    print("\nLoading VAE...")
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("\nGetting feat_cache shapes...")
        feat_cache_shapes = get_feat_cache_shapes(
            batch_size, latent_height, latent_width, dtype
        )
        print(f"  {len(feat_cache_shapes)} entries (ALL tensors, as I/O)")
        total_cache_bytes = 0
        for i, s in enumerate(feat_cache_shapes):
            size_mb = reduce(operator.mul, s) * 2 / 1024 / 1024
            total_cache_bytes += reduce(operator.mul, s) * 2
            print(f"    [{i:2d}] {s}  ({size_mb:.1f} MB)")
        print(f"  Total cache: {total_cache_bytes/1024/1024:.0f} MB per direction")

        print("\nPreparing decoder (bfloat16, rolling feat_cache)...")
        decoder = vae.decoder.to(dtype).eval()
        wrapper = DecoderWrapperRolling(decoder)

        # Build trace kwargs: x + 34 cache tensors
        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=dtype,
        )
        trace_kwargs = {"x": decoder_input}
        for i, shape in enumerate(feat_cache_shapes):
            trace_kwargs[f"c{i}"] = torch.zeros(shape, dtype=dtype)

        print(f"  Input x: {decoder_input.shape} ({decoder_input.nelement()*2/1024:.0f} KB)")
        print(f"  Cache inputs: 34 tensors ({total_cache_bytes/1024/1024:.0f} MB)")
        print(f"  Total I/O per call: ~{total_cache_bytes*2/1024/1024:.0f} MB (in + out)")

        builder = ModelBuilder(model=wrapper)
        print("Tracing (this may take a while due to 35 I/O tensors)...")
        builder.trace(kwargs=trace_kwargs, tag="decode")

        print("Compiling...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=none"
        traced = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{compiled_models_dir}/{args.output_subdir}"
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving to {output_path}...")
        traced.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights (decoder parameters only, no buffers)
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
            "rolling_cache": True,
            "num_cache_tensors": len(feat_cache_shapes),
        }
        save_model_config(output_path, config)

        print(f"\nDecoder (rolling) saved to {output_path}")

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
    parser.add_argument("--output_subdir", type=str, default="decoder_rolling",
                        help="Subdirectory name within compiled_models_dir (default: decoder_rolling)")
    parser.add_argument("--cache_dir", type=str, default="/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir")
    args = parser.parse_args()

    compile_decoder_rolling(args)
