"""
Wan2.2 VAE Decoder Compilation using Model Builder V2 API.

This script uses the new ModelBuilder API for single-device (non-TP) compilation.
Note: For high-resolution (720x1280), use compile_decoder_tp.py with tensor parallelism.
"""
import os
import json

os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Comment this line out if using trn1/inf2
compiler_flags = """ --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan
import torch
import argparse
from torch import nn
from typing import List

from neuronx_distributed import ModelBuilder, NxDModel, NxDParallelState
from safetensors.torch import save_file

from neuron_commons import attention_wrapper, f32Wrapper

torch.nn.functional.scaled_dot_product_attention = attention_wrapper


class DecoderWrapper(nn.Module):
    """Wrapper for VAE decoder to handle feat_cache as individual tensor arguments.

    ModelBuilder V2 API requires all inputs to be tensors, so we accept 34 separate
    feat_cache tensors and reconstruct the list inside forward().
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
    """Save model configuration for loading."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder_v2(args):
    """Compile VAE decoder using Model Builder V2 API."""
    latent_height = args.height // 16
    latent_width = args.width // 16
    compiled_models_dir = args.compiled_models_dir
    world_size = args.world_size  # Must match transformer for compatibility
    tp_degree = args.tp_degree

    batch_size = 1
    decoder_frames = 2  # Decoder needs CACHE_T=2 frames
    latent_frames = (args.num_frames - 1) // 4 + 1
    in_channels = 48

    print(f"num_frames={args.num_frames} -> latent_frames={latent_frames}")
    print(f"Latent size: {latent_height}x{latent_width}")
    print(f"World size: {world_size}, TP: {tp_degree}")

    # Load VAE
    print("Loading VAE...")
    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir="wan2.2_ti2v_hf_cache_dir"
    )

    # Use NxDParallelState with matching world_size for compatibility with other models
    # Even though decoder uses TP=1, world_size must match to share process groups
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Decoder ==========
        print("Compiling decoder...")
        decoder = vae.decoder
        decoder.eval()

        # Prepare decoder inputs
        decoder_input = torch.rand(
            (batch_size, in_channels, decoder_frames, latent_height, latent_width),
            dtype=torch.float32
        )

        # Create feat_cache based on decoder structure
        feat_cache = [
            torch.rand((batch_size, 48, 2, latent_height, latent_width), dtype=torch.float32),  # 0: conv_in
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 1: mid_block.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 2: mid_block.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 3: mid_block.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 4: mid_block.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 5: up_blocks.0.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 6: up_blocks.0.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 7: up_blocks.0.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 8: up_blocks.0.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 9: up_blocks.0.resnets.2.conv1
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 10: up_blocks.0.resnets.2.conv2
            torch.rand((batch_size, 1024, 2, latent_height, latent_width), dtype=torch.float32),  # 11: up_blocks.0.upsampler.time_conv
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 12: up_blocks.1.resnets.0.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 13: up_blocks.1.resnets.0.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 14: up_blocks.1.resnets.1.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 15: up_blocks.1.resnets.1.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 16: up_blocks.1.resnets.2.conv1
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 17: up_blocks.1.resnets.2.conv2
            torch.rand((batch_size, 1024, 2, latent_height*2, latent_width*2), dtype=torch.float32),  # 18: up_blocks.1.upsampler.time_conv
            torch.rand((batch_size, 1024, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 19: up_blocks.2.resnets.0.conv1
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 20: up_blocks.2.resnets.0.conv2
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 21: up_blocks.2.resnets.0.conv_shortcut
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 22: up_blocks.2.resnets.1.conv1
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 23: up_blocks.2.resnets.1.conv2
            torch.rand((batch_size, 512, 2, latent_height*4, latent_width*4), dtype=torch.float32),  # 24: up_blocks.2.resnets.2.conv1
            torch.rand((batch_size, 512, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 25: up_blocks.2.resnets.2.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 26: up_blocks.3.resnets.0.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 27: up_blocks.3.resnets.0.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 28: up_blocks.3.resnets.0.conv_shortcut
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 29: up_blocks.3.resnets.1.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 30: up_blocks.3.resnets.1.conv2
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 31: up_blocks.3.resnets.2.conv1
            torch.rand((batch_size, 256, 2, latent_height*8, latent_width*8), dtype=torch.float32),  # 32: up_blocks.3.resnets.2.conv2 (dummy)
            torch.rand((batch_size, 12, 2, latent_height*8, latent_width*8), dtype=torch.float32),   # 33: conv_out (dummy)
        ]

        # Wrap decoder
        decoder_wrapper = DecoderWrapper(decoder)

        # Initialize ModelBuilder for decoder
        decoder_builder = ModelBuilder(model=decoder_wrapper)

        print("Tracing decoder...")
        # Build kwargs with individual feat_cache tensors
        trace_kwargs = {"x": decoder_input}
        for i, fc in enumerate(feat_cache):
            trace_kwargs[f"feat_cache_{i}"] = fc

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

        # Save decoder weights (tp_degree=1, no sharding needed)
        print("Saving decoder weights...")
        decoder_weights_path = os.path.join(decoder_output_path, "weights")
        os.makedirs(decoder_weights_path, exist_ok=True)
        decoder_checkpoint = decoder_wrapper.state_dict()
        save_file(decoder_checkpoint, os.path.join(decoder_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save decoder config
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

        # Save post_quant_conv weights (tp_degree=1, no sharding needed)
        print("Saving post_quant_conv weights...")
        pqc_weights_path = os.path.join(pqc_output_path, "weights")
        os.makedirs(pqc_weights_path, exist_ok=True)
        pqc_checkpoint = post_quant_conv_wrapper.state_dict()
        save_file(pqc_checkpoint, os.path.join(pqc_weights_path, "tp0_sharded_checkpoint.safetensors"))

        # Save post_quant_conv config
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

        print(f"\nDone! Models saved to {compiled_models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 VAE Decoder using Model Builder V2")
    parser.add_argument("--height", type=int, default=512, help="Height of generated video")
    parser.add_argument("--width", type=int, default=512, help="Width of generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in generated video")
    parser.add_argument("--tp_degree", type=int, default=4, help="Tensor parallelism degree")
    parser.add_argument("--world_size", type=int, default=8, help="World size (must match transformer)")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    args = parser.parse_args()

    compile_decoder_v2(args)
