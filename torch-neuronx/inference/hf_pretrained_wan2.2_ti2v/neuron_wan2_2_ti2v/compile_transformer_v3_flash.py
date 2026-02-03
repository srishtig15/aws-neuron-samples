"""
Wan2.2 Transformer Compilation with NKI Flash Attention (V3 Flash).

This script uses:
- NKI Flash Attention for optimal performance
- TP=8 for model sharding (same as V2)
- NO Context Parallel (avoids collective operation serialization issues)

Key differences from v3_cp:
- Uses standard TP=8 without sequence parallelism
- No scatter/gather collectives that cause serialization issues
- Simpler deployment while still getting NKI performance benefits
"""

import os
import json
import math

# Environment setup for NKI and Trainium2
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"  # Required for NKI
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags optimized for NKI
compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse
from typing import Optional, Tuple

from diffusers import AutoencoderKLWan, WanPipeline

# ModelBuilder imports
from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import load_file, save_file

# Import sharding utilities from existing module
from neuron_parallel_utils import shard_transformer3d_attn, shard_transformer_feedforward

# Import NKI Flash Attention
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

print("NKI Flash Attention kernel loaded successfully")


def pad_to_multiple(tensor, dim, multiple):
    """Pad tensor along dimension to be divisible by multiple."""
    size = tensor.shape[dim]
    if size % multiple == 0:
        return tensor, 0
    pad_size = multiple - (size % multiple)

    # Create padding tuple (reversed order for F.pad)
    ndim = tensor.dim()
    pad = [0] * (2 * ndim)
    pad_idx = 2 * (ndim - 1 - dim)
    pad[pad_idx + 1] = pad_size  # Pad at the end

    return torch.nn.functional.pad(tensor, pad), pad_size


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, S, D]
        key: [B, H, S, D]
        value: [B, H, S, D]

    Returns:
        attention output [B, H, S, D]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    # NKI requires seq_len divisible by 512
    ALIGNMENT = 512

    # Pad Q, K, V to be divisible by 512
    query_padded, q_pad = pad_to_multiple(query, dim=2, multiple=ALIGNMENT)
    key_padded, k_pad = pad_to_multiple(key, dim=2, multiple=ALIGNMENT)
    value_padded, v_pad = pad_to_multiple(value, dim=2, multiple=ALIGNMENT)

    padded_q_len = query_padded.shape[2]
    padded_k_len = key_padded.shape[2]
    padded_v_len = value_padded.shape[2]

    # Reshape for NKI kernel: (B*H, D, S) for Q/K, (B*H, S, D) for V
    q = query_padded.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, padded_q_len))
    k = key_padded.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, padded_k_len))
    v = value_padded.clone().reshape((bs * n_head, padded_v_len, d_head))

    attn_output = torch.zeros((bs * n_head, padded_q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    result = attn_output.reshape((bs, n_head, padded_q_len, d_head))

    # Remove padding
    if q_pad > 0:
        result = result[:, :, :q_len, :]

    return result


def apply_rotary_emb_nki(hidden_states, freqs):
    """
    Apply rotary embeddings with pre-computed cos/sin tensors.

    Args:
        hidden_states: [batch, heads, seq_len, head_dim]
        freqs: tuple of (cos, sin)

    Returns:
        Tensor with rotary embeddings applied
    """
    cos, sin = freqs
    batch, heads, seq_len, head_dim = hidden_states.shape
    dtype = hidden_states.dtype
    half_head_dim = head_dim // 2

    # Reshape hidden_states: split into real/imag pairs
    x = hidden_states.float().reshape(batch, heads, seq_len, half_head_dim, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    # Handle various RoPE shapes
    if cos.dim() == 4:
        if cos.shape[1] == 1 and cos.shape[2] == seq_len:
            if cos.shape[3] != half_head_dim:
                cos = cos[..., :half_head_dim]
                sin = sin[..., :half_head_dim]
        elif cos.shape[1] == seq_len and cos.shape[2] == 1:
            cos = cos.permute(0, 2, 1, 3)
            sin = sin.permute(0, 2, 1, 3)
            if cos.shape[3] != half_head_dim:
                cos = cos[..., :half_head_dim]
                sin = sin[..., :half_head_dim]
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        if cos.shape[3] != half_head_dim:
            cos = cos[..., :half_head_dim]
            sin = sin[..., :half_head_dim]
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        if cos.shape[3] != half_head_dim:
            cos = cos[..., :half_head_dim]
            sin = sin[..., :half_head_dim]

    # Complex multiplication
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    # Re-interleave
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    return out.to(dtype)


class NKIWanSelfAttention(nn.Module):
    """
    NKI Flash Attention for Wan2.2 Self-Attention (attn1).

    Uses NKI kernel for optimal performance on Trainium2.
    No Context Parallel - standard TP=8 sharding.
    """

    def __init__(self, orig_attn):
        super().__init__()
        self.heads = orig_attn.heads

        # Copy projections (already sharded for TP)
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # QK normalization
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with NKI Flash Attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Apply QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Reshape to [B, H, S, D]
        head_dim = query.shape[-1] // self.heads
        query = query.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)

        # Apply RoPE
        if rotary_emb is not None:
            query = apply_rotary_emb_nki(query, rotary_emb)
            key = apply_rotary_emb_nki(key, rotary_emb)

        # NKI Flash Attention
        hidden_states = nki_flash_attention(query, key, value)

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class NKIWanCrossAttention(nn.Module):
    """
    NKI Flash Attention for Wan2.2 Cross-Attention (attn2).

    Uses NKI kernel for optimal performance.
    """

    def __init__(self, orig_attn):
        super().__init__()
        self.heads = orig_attn.heads

        # Copy projections
        self.to_q = orig_attn.to_q
        self.to_k = orig_attn.to_k
        self.to_v = orig_attn.to_v
        self.to_out = orig_attn.to_out

        # QK normalization
        self.norm_q = orig_attn.norm_q if hasattr(orig_attn, 'norm_q') else None
        self.norm_k = orig_attn.norm_k if hasattr(orig_attn, 'norm_k') else None

        # I2V projections
        self.add_k_proj = orig_attn.add_k_proj if hasattr(orig_attn, 'add_k_proj') else None
        self.add_v_proj = orig_attn.add_v_proj if hasattr(orig_attn, 'add_v_proj') else None
        self.norm_added_k = orig_attn.norm_added_k if hasattr(orig_attn, 'norm_added_k') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with NKI Flash Attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Handle I2V image context
        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Query from video
        query = self.to_q(hidden_states)

        # K/V from text
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Apply QK normalization
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Reshape to [B, H, S, D]
        head_dim = query.shape[-1] // self.heads
        query = query.view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Handle I2V image attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj(encoder_hidden_states_img)
            key_img = self.norm_added_k(key_img)
            value_img = self.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value_img = value_img.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            hidden_states_img = nki_flash_attention(query, key_img, value_img)
            hidden_states_img = hidden_states_img.transpose(1, 2).reshape(batch_size, seq_len, -1)
            hidden_states_img = hidden_states_img.to(query.dtype)

        # NKI Flash Attention for text
        hidden_states = nki_flash_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = hidden_states.to(query.dtype)

        # Combine
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class NeuronWanTransformerV3Flash(nn.Module):
    """
    Neuron-optimized Wan2.2 Transformer with NKI Flash Attention.

    Features:
    - TP=8 for model parameter sharding
    - NKI Flash Attention for optimal performance
    - No Context Parallel (avoids collective serialization issues)
    """

    def __init__(self, original_transformer, tp_degree):
        super().__init__()

        self.config = original_transformer.config
        self.tp_degree = tp_degree

        # Patch embedding
        self.patch_embedding = original_transformer.patch_embedding

        # Condition embedder
        self.condition_embedder = original_transformer.condition_embedder

        # Transformer blocks with TP sharding
        self.blocks = nn.ModuleList()
        for i, block in enumerate(original_transformer.blocks):
            # Shard attention and FFN with TP
            block.attn1 = shard_transformer3d_attn(tp_degree, block.attn1)
            block.attn2 = shard_transformer3d_attn(tp_degree, block.attn2)
            block.ffn = shard_transformer_feedforward(block.ffn)
            self.blocks.append(block)

            if (i + 1) % 8 == 0:
                print(f"  Sharded block {i+1}/{len(original_transformer.blocks)}")

        # Replace attention with NKI versions
        self._replace_attention()

        # Output layers
        self.norm_out = original_transformer.norm_out
        self.proj_out = original_transformer.proj_out
        self.scale_shift_table = original_transformer.scale_shift_table

        self.patch_size = original_transformer.config.patch_size

    def _replace_attention(self):
        """Replace attention modules with NKI versions."""
        for block in self.blocks:
            block.attn1 = NKIWanSelfAttention(block.attn1)
            block.attn2 = NKIWanCrossAttention(block.attn2)
        print(f"Replaced attention with NKI Flash Attention on {len(self.blocks)} blocks")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_cos: torch.Tensor,
        rotary_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with NKI Flash Attention."""

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Condition embedding
        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Process through blocks
        for block in self.blocks:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                block.scale_shift_table + timestep_proj.float()
            ).chunk(6, dim=1)

            # Self-attention with RoPE
            norm_hidden = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            rotary_emb = (rotary_emb_cos, rotary_emb_sin)
            attn_output = block.attn1(hidden_states=norm_hidden, rotary_emb=rotary_emb)
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

            # Cross-attention
            norm_hidden = block.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = block.attn2(hidden_states=norm_hidden, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output

            # Feed-forward
            norm_hidden = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
            ff_output = block.ffn(norm_hidden)
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        # Output
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        output = self.proj_out(hidden_states)

        # Unpatchify
        output = output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


class TracingWrapper(nn.Module):
    """Wrapper for tracing with ModelBuilder."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin):
        return self.transformer(
            hidden_states, timestep, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin
        )


def compute_rope(transformer, latent_frames, latent_height, latent_width, in_channels=48):
    """Compute RoPE using transformer's rope module."""
    batch_size = 1
    dummy_hidden = torch.zeros(
        batch_size, in_channels, latent_frames, latent_height, latent_width,
        dtype=torch.float32
    )

    print(f"  Computing RoPE for shape: {dummy_hidden.shape}")
    rotary_emb = transformer.rope(dummy_hidden)

    if isinstance(rotary_emb, tuple):
        freqs_cos, freqs_sin = rotary_emb
        print(f"  RoPE cos shape: {freqs_cos.shape}")
        print(f"  RoPE sin shape: {freqs_sin.shape}")
    else:
        raise ValueError("Unexpected rope output format")

    return freqs_cos, freqs_sin


def fix_norm_weights_per_rank(weights_path, unsharded_norm_weights, tp_degree):
    """Fix norm_k/norm_q weights for each rank after shard_checkpoint."""
    print(f"Fixing norm weights for {tp_degree} ranks...")

    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)

        fixed_count = 0
        for key, unsharded_weight in unsharded_norm_weights.items():
            if key in ckpt:
                ckpt_shape = ckpt[key].shape[0]
                unsharded_dim = unsharded_weight.shape[0]
                expected_shard_size = unsharded_dim // tp_degree

                if ckpt_shape == expected_shard_size:
                    start = expected_shard_size * rank
                    end = expected_shard_size * (rank + 1)
                    correct_slice = unsharded_weight[start:end].clone()
                else:
                    total_padded_dim = ckpt_shape * tp_degree
                    padded_weight = torch.ones(total_padded_dim, dtype=unsharded_weight.dtype)
                    padded_weight[:unsharded_dim] = unsharded_weight
                    start = ckpt_shape * rank
                    end = ckpt_shape * (rank + 1)
                    correct_slice = padded_weight[start:end].clone()

                if correct_slice.shape == ckpt[key].shape:
                    ckpt[key] = correct_slice
                    fixed_count += 1

        save_file(ckpt, ckpt_path)
        print(f"  Rank {rank}: Fixed {fixed_count} norm weights")


def compile_transformer_v3_flash(args):
    """Compile transformer with NKI Flash Attention using ModelBuilder API."""

    tp_degree = args.tp_degree

    latent_height = args.height // 16
    latent_width = args.width // 16
    latent_frames = (args.num_frames - 1) // 4 + 1
    max_sequence_length = args.max_sequence_length
    hidden_size = 4096
    batch_size = 1
    in_channels = 48

    print("=" * 60)
    print("Wan2.2 Transformer V3 Flash Compilation (NKI, TP=8)")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}, Frames: {args.num_frames}")
    print(f"Latent: {latent_frames}x{latent_height}x{latent_width}")
    print(f"TP degree: {tp_degree}")
    print(f"NKI Flash Attention: Enabled")
    print("=" * 60)

    # Sample inputs
    sample_hidden_states = torch.randn(
        batch_size, in_channels, latent_frames, latent_height, latent_width,
        dtype=torch.bfloat16
    )
    sample_encoder_hidden_states = torch.randn(
        batch_size, max_sequence_length, hidden_size,
        dtype=torch.bfloat16
    )
    sample_timestep = torch.randn(batch_size, dtype=torch.float32)

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        print("\nLoading model...")
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=args.cache_dir
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir
        )

        # Compute RoPE
        print("\nComputing RoPE...")
        rotary_emb_cos, rotary_emb_sin = compute_rope(
            pipe.transformer, latent_frames, latent_height, latent_width
        )
        rotary_emb_cos = rotary_emb_cos.to(torch.bfloat16)
        rotary_emb_sin = rotary_emb_sin.to(torch.bfloat16)

        # Save unsharded state dict
        unsharded_state = pipe.transformer.state_dict()

        # Collect unsharded norm weights
        unsharded_norm_weights = {}
        for key, value in unsharded_state.items():
            if 'norm_k.weight' in key or 'norm_q.weight' in key:
                unsharded_norm_weights[f"transformer.{key}"] = value.clone()
        print(f"Collected {len(unsharded_norm_weights)} unsharded norm weights")

        # Create Neuron transformer
        print(f"\nCreating Neuron transformer (TP={tp_degree})...")
        neuron_transformer = NeuronWanTransformerV3Flash(pipe.transformer, tp_degree)
        neuron_transformer = neuron_transformer.to(torch.bfloat16)
        neuron_transformer.eval()

        # Wrap for tracing
        model = TracingWrapper(neuron_transformer)

        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden_states,
                "timestep": sample_timestep,
                "encoder_hidden_states": sample_encoder_hidden_states,
                "rotary_emb_cos": rotary_emb_cos,
                "rotary_emb_sin": rotary_emb_sin,
            },
            tag="inference",
        )

        print("Compiling model...")
        compile_args = "--model-type=transformer -O1 --auto-cast=none"
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save
        output_path = f"{args.compiled_models_dir}/transformer_v3_flash"
        os.makedirs(output_path, exist_ok=True)

        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {}
        for key, value in model.state_dict().items():
            orig_key = key.replace("transformer.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        print("Sharding weights...")
        shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            serialize_path=weights_path,
        )

        # Fix norm weights
        fix_norm_weights_per_rank(weights_path, unsharded_norm_weights, tp_degree)

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "max_sequence_length": max_sequence_length,
            "tp_degree": tp_degree,
            "nki_flash_attention": True,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save RoPE cache
        torch.save({
            "rotary_emb_cos": rotary_emb_cos,
            "rotary_emb_sin": rotary_emb_sin,
        }, os.path.join(output_path, "rope_cache.pt"))

        print("\nCompilation complete!")
        print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Wan2.2 Transformer with NKI Flash Attention")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max text sequence length")
    parser.add_argument("--tp_degree", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models_v3_flash", help="Output directory")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir_v3_flash", help="Compiler workdir")
    parser.add_argument("--cache_dir", type=str, default="wan2.2_ti2v_hf_cache_dir", help="HuggingFace cache dir")
    args = parser.parse_args()

    compile_transformer_v3_flash(args)
