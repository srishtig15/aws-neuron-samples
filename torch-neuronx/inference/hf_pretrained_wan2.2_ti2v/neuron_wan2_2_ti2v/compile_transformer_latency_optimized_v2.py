"""
Wan2.2 Transformer Compilation using Model Builder V2 API.

This script uses the new ModelBuilder API instead of the deprecated parallel_model_trace.
Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/
"""
import os
import json

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"  # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"  # Comment this line out if using trn1/inf2
compiler_flags = """ --target=trn2 --lnc=2 --internal-hlo2tensorizer-options='--fuse-dot-logistic=false' --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import AutoencoderKLWan, WanPipeline
import torch
import argparse

from torch import nn
import torch.nn.functional as F

from neuronx_distributed import ModelBuilder, NxDModel, NxDParallelState
from neuronx_distributed.parallel_layers import parallel_state

# Reuse existing sharding functions
from neuron_parallel_utils import shard_transformer3d_attn, shard_transformer_feedforward


class TracingTransformerWrapperV2(nn.Module):
    """Wrapper for tracing with Model Builder V2."""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )


def shard_transformer_for_v2(transformer, tp_degree):
    """
    Shard transformer blocks using existing sharding functions.
    This reuses the proven sharding logic from neuron_parallel_utils.py.
    """
    for block_idx, block in enumerate(transformer.blocks):
        print(f"Sharding block {block_idx + 1}/{len(transformer.blocks)}")

        # Shard attention layers using existing function
        block.attn1 = shard_transformer3d_attn(tp_degree, block.attn1)
        block.attn2 = shard_transformer3d_attn(tp_degree, block.attn2)

        # Shard feed-forward layer using existing function
        block.ffn = shard_transformer_feedforward(block.ffn)

    return transformer


def save_model_config(output_path, config):
    """Save model configuration for loading."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_transformer_v2(args):
    """Compile transformer using Model Builder V2 API."""
    tp_degree = args.tp_degree
    latent_height = args.height // 16
    latent_width = args.width // 16
    max_sequence_length = args.max_sequence_length
    hidden_size = 4096
    compiled_models_dir = args.compiled_models_dir
    batch_size = 1
    latent_frames = (args.num_frames - 1) // 4 + 1
    in_channels = 48

    print(f"num_frames={args.num_frames} -> latent_frames={latent_frames}")

    # Calculate sequence length after patch embedding
    patch_size_t, patch_size_h, patch_size_w = 1, 2, 2
    seq_len = (latent_frames // patch_size_t) * (latent_height // patch_size_h) * (latent_width // patch_size_w)
    print(f"seq_len: {seq_len}")

    # Prepare sample inputs
    sample_hidden_states = torch.ones(
        (batch_size, in_channels, latent_frames, latent_height, latent_width),
        dtype=torch.bfloat16
    )
    sample_encoder_hidden_states = torch.ones(
        (batch_size, max_sequence_length, hidden_size),
        dtype=torch.bfloat16
    )
    sample_timestep = torch.ones((batch_size, seq_len), dtype=torch.float32)

    # Use NxDParallelState context manager (Model Builder V2 style)
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        print("Loading model...")
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir="wan2.2_ti2v_hf_cache_dir"
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir="wan2.2_ti2v_hf_cache_dir"
        )

        print("Sharding transformer blocks...")
        pipe.transformer = shard_transformer_for_v2(pipe.transformer, tp_degree)

        # Wrap for tracing
        model = TracingTransformerWrapperV2(pipe.transformer)

        print("Initializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "timestep": sample_timestep,
                "encoder_hidden_states": sample_encoder_hidden_states,
            },
            tag="inference",  # Note: "forward" is reserved, use different name
        )

        print("Compiling model...")
        traced_model = builder.compile()

        # Save model
        output_path = f"{compiled_models_dir}/transformer_v2"
        os.makedirs(output_path, exist_ok=True)

        print(f"Saving compiled model to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save config for loading
        config = {
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "latent_frames": latent_frames,
            "seq_len": seq_len,
            "max_sequence_length": max_sequence_length,
            "tp_degree": tp_degree,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
        }
        save_model_config(output_path, config)

        print(f"Done! Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 Transformer using Model Builder V2")
    parser.add_argument("--height", type=int, default=720, help="Height of generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in generated video")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max sequence length for text encoder")
    parser.add_argument("--tp_degree", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models", help="Output directory")
    args = parser.parse_args()

    compile_transformer_v2(args)
