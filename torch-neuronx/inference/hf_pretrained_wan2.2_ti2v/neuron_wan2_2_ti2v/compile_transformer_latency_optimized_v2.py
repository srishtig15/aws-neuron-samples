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

from neuronx_distributed import ModelBuilder, NxDModel, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import load_file, save_file

# Reuse existing sharding functions
from neuron_parallel_utils import shard_transformer3d_attn, shard_transformer_feedforward


def fix_norm_weights_per_rank(weights_path, unsharded_norm_weights, tp_degree):
    """
    Fix norm_k/norm_q weights for each rank after shard_checkpoint.

    The issue: During V2 compilation, parallel_state.get_tensor_model_parallel_rank()
    always returns 0 inside NxDParallelState. This means all norm weights are sharded
    using rank 0's slice. shard_checkpoint doesn't re-shard DistributedRMSNorm weights,
    so all ranks end up with the SAME (incorrect) weights.

    This function manually fixes each rank's norm weights with the correct slice.
    It also handles the padding case where norm weights may be padded before sharding.
    """
    print(f"Fixing norm weights for {tp_degree} ranks...")

    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)

        # Fix each norm weight
        fixed_count = 0
        for key, unsharded_weight in unsharded_norm_weights.items():
            if key in ckpt:
                ckpt_shape = ckpt[key].shape[0]
                unsharded_dim = unsharded_weight.shape[0]

                # Calculate expected shard size without padding
                expected_shard_size = unsharded_dim // tp_degree

                # Check if padding was applied
                if ckpt_shape == expected_shard_size:
                    # No padding case: directly slice unsharded weights
                    start = expected_shard_size * rank
                    end = expected_shard_size * (rank + 1)
                    correct_slice = unsharded_weight[start:end].clone()
                else:
                    # Padding case: need to pad unsharded weights first, then slice
                    # The checkpoint has padded shape, calculate total padded dim
                    total_padded_dim = ckpt_shape * tp_degree

                    # Create padded version with ones (default for RMSNorm)
                    padded_weight = torch.ones(total_padded_dim, dtype=unsharded_weight.dtype)
                    padded_weight[:unsharded_dim] = unsharded_weight

                    # Now slice for this rank
                    start = ckpt_shape * rank
                    end = ckpt_shape * (rank + 1)
                    correct_slice = padded_weight[start:end].clone()

                # Verify the shape matches
                if correct_slice.shape == ckpt[key].shape:
                    ckpt[key] = correct_slice
                    fixed_count += 1
                else:
                    print(f"Warning: Shape mismatch for {key}: expected {correct_slice.shape}, got {ckpt[key].shape}")

        # Save the fixed checkpoint
        save_file(ckpt, ckpt_path)
        print(f"  Rank {rank}: Fixed {fixed_count} norm weights")

    print("Norm weights fixed for all ranks.")


def make_rope_buffers_persistent(transformer):
    """
    Re-register RoPE buffers as persistent so they appear in state_dict().

    The Wan transformer's RoPE buffers are registered with persistent=False,
    which means they don't appear in state_dict(). During shard_checkpoint,
    keys not in state_dict() get removed. By re-registering them as persistent,
    they will be included in the checkpoint and loaded correctly.
    """
    if hasattr(transformer, 'rope'):
        rope = transformer.rope
        # Get current buffer values
        if hasattr(rope, 'freqs_cos') and rope.freqs_cos is not None:
            freqs_cos = rope.freqs_cos.clone()
            # Delete the non-persistent buffer
            del rope._buffers['freqs_cos']
            # Re-register as persistent
            rope.register_buffer('freqs_cos', freqs_cos, persistent=True)
            print("Made rope.freqs_cos persistent")

        if hasattr(rope, 'freqs_sin') and rope.freqs_sin is not None:
            freqs_sin = rope.freqs_sin.clone()
            # Delete the non-persistent buffer
            del rope._buffers['freqs_sin']
            # Re-register as persistent
            rope.register_buffer('freqs_sin', freqs_sin, persistent=True)
            print("Made rope.freqs_sin persistent")


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
            cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir="/opt/dlami/nvme/wan2.2_ti2v_hf_cache_dir"
        )

        # Make RoPE buffers persistent BEFORE saving state dict
        # This ensures they are included in state_dict() and not removed by shard_checkpoint
        print("Making RoPE buffers persistent...")
        make_rope_buffers_persistent(pipe.transformer)

        # Save UNSHARDED state dict BEFORE sharding (for shard_checkpoint later)
        # Now includes rope.freqs_cos and rope.freqs_sin
        unsharded_transformer_state = pipe.transformer.state_dict()

        # Collect UNSHARDED norm weights for later fixing
        # These need to be manually sharded per rank after shard_checkpoint
        unsharded_norm_weights = {}
        for key, value in unsharded_transformer_state.items():
            if 'norm_k.weight' in key or 'norm_q.weight' in key:
                # Store with the wrapper prefix that will be used in checkpoint
                unsharded_norm_weights[f"transformer.{key}"] = value.clone()
        print(f"Collected {len(unsharded_norm_weights)} unsharded norm weights for per-rank sharding")

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

        # Save sharded weights
        print("Saving sharded weights...")
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Get the full state dict from the sharded model (wrapper)
        sharded_model_state = model.state_dict()

        # Build checkpoint: use unsharded values for weights (to be properly sharded)
        # RoPE buffers are now persistent and included in unsharded_transformer_state
        unsharded_checkpoint = {}
        unsharded_keys = {"transformer." + k for k in unsharded_transformer_state.keys()}

        # Debug: Check if RoPE keys are included
        rope_keys_in_unsharded = [k for k in unsharded_transformer_state.keys() if 'rope' in k.lower()]
        print(f"RoPE keys in unsharded state: {rope_keys_in_unsharded}")

        # Add weights from state_dict
        # Note: norm_k/norm_q weights are already sharded in the model and won't be
        # re-sharded by shard_checkpoint (only ColumnParallelLinear/RowParallelLinear are).
        # So we use the sharded values directly for those.
        for key, sharded_value in sharded_model_state.items():
            # Check if this is a norm weight that's already been sharded
            is_already_sharded_norm = 'norm_k.weight' in key or 'norm_q.weight' in key

            if key in unsharded_keys and not is_already_sharded_norm:
                # Use unsharded value (will be sharded by shard_checkpoint)
                orig_key = key.replace("transformer.", "", 1)
                unsharded_checkpoint[key] = unsharded_transformer_state[orig_key].clone()
            else:
                # Use value from sharded model (already sharded norm weights, buffers, etc.)
                unsharded_checkpoint[key] = sharded_value.clone()

        # Debug: Check RoPE keys in final checkpoint
        rope_keys_in_checkpoint = [k for k in unsharded_checkpoint.keys() if 'rope' in k.lower()]
        print(f"RoPE keys in checkpoint: {rope_keys_in_checkpoint}")
        print(f"Total checkpoint keys: {len(unsharded_checkpoint)}")

        # Use shard_checkpoint with checkpoint - it will shard parallel layer weights per rank
        shard_checkpoint(
            checkpoint=unsharded_checkpoint,
            model=model,
            start_rank=0,
            end_rank=tp_degree - 1,
            serialize_path=weights_path,
        )

        # Fix norm weights per rank
        # shard_checkpoint doesn't properly handle DistributedRMSNorm weights because
        # parallel_state.get_tensor_model_parallel_rank() returns 0 during NxDParallelState.
        # This manually fixes each rank's norm weights with the correct slice.
        fix_norm_weights_per_rank(weights_path, unsharded_norm_weights, tp_degree)

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
