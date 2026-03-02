"""
Compile CausalWanModel transformer for RollingForcing on Neuron.

Core compilation script that adapts the CausalWanModel (1.3B) for Trainium2:
- Replaces complex-valued RoPE with real-valued cos/sin
- Replaces flash_attn/flex_attention with NKI Flash Attention
- Eliminates dynamic KV cache (full-sequence processing)
- Preserves per-frame timestep modulation via CausalHead
- TP=4, NKI Flash Attention, bf16

Forward signature:
    hidden_states:          [B, C=16, F=21, H=60, W=104]
    timestep:               [B, F=21] per-frame timesteps
    encoder_hidden_states:  [B, 512, 4096] text embeddings
    rope_cos:               [1, 32760, 1, 128] pre-computed
    rope_sin:               [1, 32760, 1, 128] pre-computed
"""
import os
import sys
import json
import math

# Environment setup
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = " --target=trn2 --lnc=2 --model-type=transformer -O1 --auto-cast=none --enable-fast-loading-neuron-binaries "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Optional, Tuple

from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from safetensors.torch import load_file, save_file

from neuron_commons import nki_flash_attention, nki_flash_attention_causal, local_rms_norm, apply_rotary_emb
from neuron_parallel_utils import (
    get_sharded_data,
    shard_causal_wan_self_attention,
    shard_causal_wan_cross_attention,
    shard_causal_wan_ffn,
)

# Add RollingForcing repo to path for model loading
sys.path.insert(0, "/tmp/RollingForcing")


# ========================
# Neuron Model Components
# ========================

def sinusoidal_embedding_1d(dim, position):
    """Sinusoidal embedding matching Wan's implementation.
    Uses float32 instead of float64 for Neuron compatibility.
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class NeuronCausalSelfAttention(nn.Module):
    """
    Neuron-compatible CausalWanSelfAttention using NKI Flash Attention.

    Replaces:
    - Complex RoPE → real-valued cos/sin (pre-computed, passed as input)
    - flex_attention → NKI flash attention with causal mask
    - Dynamic KV cache → removed (full-sequence processing)

    Uses causal masking so that clean frames (anchor + working cache) at the
    start of the sequence don't attend to noisy frames later in the sequence.
    This preserves clean frame K/V quality, matching the original pipeline's
    KV cache behavior where clean frames were processed in isolation.
    """

    def __init__(self, orig_self_attn, tp_degree=4):
        super().__init__()
        # After TP sharding: num_heads is local (e.g., 12/4 = 3)
        self.head_dim = orig_self_attn.head_dim
        self.num_heads = orig_self_attn.num_heads // tp_degree

        # Copy projections (already TP-sharded)
        self.q = orig_self_attn.q
        self.k = orig_self_attn.k
        self.v = orig_self_attn.v
        self.o = orig_self_attn.o

        # QK normalization weights (already sharded)
        self.norm_q = orig_self_attn.norm_q
        self.norm_k = orig_self_attn.norm_k

    def forward(self, x, rope_cos, rope_sin):
        """
        Args:
            x: [B, S, dim] hidden states
            rope_cos: [1, S, 1, head_dim] pre-computed cosines
            rope_sin: [1, S, 1, head_dim] pre-computed sines
        Returns:
            [B, S, dim] output
        """
        B, S, _ = x.shape

        # QKV projections → [B, S, local_inner_dim] where local = num_heads_local * head_dim
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # QK normalization (local, no all-reduce)
        q = local_rms_norm(q, self.norm_q.weight, self.norm_q.eps)
        k = local_rms_norm(k, self.norm_k.weight, self.norm_k.eps)

        # Reshape to [B, H_local, S, D] for attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply real-valued RoPE
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Pad sequence to multiple of 2048 for flash_fwd kernel requirement
        target_s = ((S + 2047) // 2048) * 2048
        pad_s = target_s - S
        if pad_s > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_s))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_s))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_s))

        # NKI Flash Attention with causal mask.
        # The original CausalWanModel uses block-causal attention via flex_attention
        # (bidirectional within 3-frame blocks, causal between blocks). Standard
        # causal is the best approximation because it preserves clean anchor frame
        # representations (placed first in sequence, only attend to other clean
        # frames) while allowing noisy frames to see all preceding context.
        # Tested: bidirectional attention corrupts clean anchor representations
        # through noisy attention, yielding worse quality (35.7 vs 64.5 pixel_std).
        out = nki_flash_attention_causal(q, k, v)

        # Remove padding
        if pad_s > 0:
            out = out[:, :, :S, :]

        # Reshape back and output projection
        out = out.transpose(1, 2).reshape(B, S, -1)
        out = out.to(x.dtype)
        out = self.o(out)
        return out


class NeuronCrossAttention(nn.Module):
    """
    Neuron-compatible WanT2VCrossAttention using NKI Flash Attention.

    Q from video tokens, K/V from text tokens.
    No RoPE, no causal mask.
    """

    def __init__(self, orig_cross_attn, tp_degree=4):
        super().__init__()
        # After TP sharding: local head count
        self.head_dim = orig_cross_attn.head_dim
        self.num_heads = orig_cross_attn.num_heads // tp_degree

        # Copy projections (already TP-sharded)
        self.q = orig_cross_attn.q
        self.k = orig_cross_attn.k
        self.v = orig_cross_attn.v
        self.o = orig_cross_attn.o

        # QK normalization (already sharded)
        self.norm_q = orig_cross_attn.norm_q
        self.norm_k = orig_cross_attn.norm_k

    def forward(self, x, context):
        """
        Args:
            x: [B, S, dim] video hidden states
            context: [B, T, dim] text embeddings (already projected)
        Returns:
            [B, S, dim] output
        """
        B, S, _ = x.shape

        q = self.q(x)
        k = self.k(context)
        v = self.v(context)

        # QK normalization
        q = local_rms_norm(q, self.norm_q.weight, self.norm_q.eps)
        k = local_rms_norm(k, self.norm_k.weight, self.norm_k.eps)

        # Reshape to [B, H, S/T, D]
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Pad Q sequence to next power of 2 for NKI flash attention
        S_q = q.shape[2]
        pad_q = _next_power_of_2(S_q) - S_q
        if pad_q > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_q))

        # NKI Flash Attention (no mask for cross-attention)
        out = nki_flash_attention(q, k, v)

        # Remove Q padding
        if pad_q > 0:
            out = out[:, :, :S_q, :]

        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        out = out.to(x.dtype)
        out = self.o(out)
        return out


class NeuronCausalWanModel(nn.Module):
    """
    Neuron-optimized CausalWanModel for full-sequence processing.

    Key differences from original:
    1. No KV cache - processes full visible sequence each call
    2. Real-valued RoPE (cos/sin) instead of complex
    3. NKI Flash Attention instead of flex_attention
    4. Per-frame timestep modulation preserved
    """

    def __init__(self, original_model, tp_degree):
        super().__init__()

        self.tp_degree = tp_degree

        # Model config
        self.dim = original_model.dim           # 1536
        self.freq_dim = original_model.freq_dim # 256
        self.text_len = original_model.text_len # 512
        self.out_dim = original_model.out_dim   # 16
        self.num_heads = original_model.num_heads  # 12
        self.head_dim = self.dim // self.num_heads  # 128
        self.patch_size = original_model.patch_size  # (1, 2, 2)

        # Embeddings (not sharded)
        self.patch_embedding = original_model.patch_embedding
        self.text_embedding = original_model.text_embedding
        self.time_embedding = original_model.time_embedding
        self.time_projection = original_model.time_projection

        # Transformer blocks with TP sharding
        self.blocks = nn.ModuleList()
        for i, block in enumerate(original_model.blocks):
            # Shard attention and FFN
            block.self_attn = shard_causal_wan_self_attention(tp_degree, block.self_attn)
            block.cross_attn = shard_causal_wan_cross_attention(tp_degree, block.cross_attn)
            block.ffn = shard_causal_wan_ffn(tp_degree, block.ffn)
            self.blocks.append(block)
            if (i + 1) % 10 == 0:
                print(f"  Sharded block {i+1}/{len(original_model.blocks)}")

        # Replace attention with Neuron-compatible versions
        self._replace_attention()

        # Output head
        self.head_norm = original_model.head.norm
        self.head_linear = original_model.head.head
        self.head_modulation = original_model.head.modulation

        # Per-head count after TP
        self.tp_num_heads = self.num_heads // tp_degree

    def _replace_attention(self):
        """Replace attention modules with Neuron NKI versions."""
        for i, block in enumerate(self.blocks):
            block.self_attn = NeuronCausalSelfAttention(block.self_attn, self.tp_degree)
            block.cross_attn = NeuronCrossAttention(block.cross_attn, self.tp_degree)
        print(f"Replaced {len(self.blocks)} blocks with Neuron attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C=16, F, H, W] input video latents
            timestep: [B, F] per-frame timesteps (0 for clean, t for noisy)
            encoder_hidden_states: [B, 512, 4096] raw text embeddings
            rope_cos: [1, seq_len, 1, head_dim] pre-computed RoPE cosines
            rope_sin: [1, seq_len, 1, head_dim] pre-computed RoPE sines
        Returns:
            [B, C=16, F, H, W] predicted flow
        """
        B, C, F, H, W = hidden_states.shape
        p_t, p_h, p_w = self.patch_size

        # 1. Patch embedding: [B, C, F, H, W] -> [B, dim, F/pt, H/ph, W/pw] -> [B, seq_len, dim]
        x = self.patch_embedding(hidden_states)
        post_f = F // p_t
        post_h = H // p_h
        post_w = W // p_w
        x = x.flatten(2).transpose(1, 2)  # [B, seq_len, dim]

        # 2. Text embedding: [B, 512, 4096] -> [B, 512, dim]
        context = self.text_embedding(encoder_hidden_states)

        # 3. Per-frame time embedding
        # timestep: [B, F] -> sinusoidal -> time_embedding -> time_projection
        # -> [B, F, 6, dim]
        t_flat = timestep.flatten()  # [B*F]
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t_flat).type_as(x)  # [B*F, freq_dim]
        e = self.time_embedding(t_emb)  # [B*F, dim]
        e0 = self.time_projection(e)  # [B*F, dim*6]
        e0 = e0.unflatten(1, (6, self.dim))  # [B*F, 6, dim]
        e0 = e0.unflatten(0, (B, F))  # [B, F, 6, dim]

        # For CausalHead: e_head = e reshaped to [B, F, 1, dim]
        # (time_embedding output before time_projection, for head modulation)
        e_for_head = e.unflatten(0, (B, F)).unsqueeze(2)  # [B, F, 1, dim]

        # frame_seqlen = tokens per frame
        frame_seqlen = post_h * post_w  # 30 * 52 = 1560

        # 4. Transformer blocks
        for block in self.blocks:
            # Per-frame modulation: modulation is [1, 6, dim], e0 is [B, F, 6, dim]
            mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            # Each mod[i] is [B, F, 1, dim]

            # Self-attention with per-frame scale/shift/gate
            x_norm = block.norm1(x)  # [B, seq_len, dim]
            # Unflatten to [B, F, frame_seqlen, dim], apply per-frame modulation
            x_norm = x_norm.unflatten(1, (F, frame_seqlen))
            x_norm = (x_norm * (1 + mod[1]) + mod[0]).flatten(1, 2)

            attn_out = block.self_attn(x_norm, rope_cos, rope_sin)

            # Gate and residual
            x_unflatten = x.unflatten(1, (F, frame_seqlen))
            attn_out_unflatten = attn_out.unflatten(1, (F, frame_seqlen))
            x = (x_unflatten.float() + attn_out_unflatten * mod[2]).flatten(1, 2).type_as(x)

            # Cross-attention (no per-frame modulation in original)
            x_norm = block.norm3(x)
            cross_out = block.cross_attn(x_norm, context)
            x = x + cross_out

            # FFN with per-frame modulation
            x_norm = block.norm2(x)
            x_norm = x_norm.unflatten(1, (F, frame_seqlen))
            x_norm = (x_norm * (1 + mod[4]) + mod[3]).flatten(1, 2)
            ff_out = block.ffn(x_norm)
            x_unflatten = x.unflatten(1, (F, frame_seqlen))
            ff_out_unflatten = ff_out.unflatten(1, (F, frame_seqlen))
            x = (x_unflatten.float() + ff_out_unflatten.float() * mod[5]).flatten(1, 2).type_as(x)

        # 5. CausalHead: per-frame modulation + norm + linear
        head_mod = (self.head_modulation.unsqueeze(1) + e_for_head).chunk(2, dim=2)
        # head_mod[0], head_mod[1] are [B, F, 1, dim]
        x_head = self.head_norm(x)
        x_head = x_head.unflatten(1, (F, frame_seqlen))
        x_head = self.head_linear(x_head * (1 + head_mod[1]) + head_mod[0])
        # x_head: [B, F, frame_seqlen, out_dim * prod(patch_size)]

        # 6. Unpatchify: [B, F, frame_seqlen, C_out*pt*ph*pw] -> [B, C_out, F*pt, H, W]
        # The head linear outputs 64 values per token (pt*ph*pw*C = 1*2*2*16).
        # Original einops: rearrange('b (f h w) (p q r c) -> b c (f p) (h q) (w r)')
        # In (p q r c), c=out_dim is LAST (fastest varying). Must match this view order.
        x_head = x_head.view(B, post_f, post_h, post_w, p_t, p_h, p_w, self.out_dim)
        output = x_head.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        output = output.reshape(B, self.out_dim, post_f * p_t, post_h * p_h, post_w * p_w)

        return output


class TracingWrapper(nn.Module):
    """Wrapper for tracing with ModelBuilder."""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                rope_cos, rope_sin):
        return self.transformer(
            hidden_states, timestep, encoder_hidden_states,
            rope_cos, rope_sin)


# ========================
# Weight Loading
# ========================

def load_rolling_forcing_weights(ckpt_path, model_prefix="model."):
    """
    Load RollingForcing DMD checkpoint and extract CausalWanModel weights.

    The checkpoint contains WanDiffusionWrapper state_dict with keys like:
    "model.patch_embedding.weight", "model.blocks.0.self_attn.q.weight", etc.
    We strip the model_prefix to get CausalWanModel keys.
    """
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The checkpoint may have 'generator' or 'generator_ema' keys
    if isinstance(ckpt, dict):
        if 'generator_ema' in ckpt:
            state_dict = ckpt['generator_ema']
            print("  Using 'generator_ema' weights")
        elif 'generator' in ckpt:
            state_dict = ckpt['generator']
            print("  Using 'generator' weights")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("  Using 'state_dict' weights")
        else:
            state_dict = ckpt
            print("  Using raw checkpoint dict")
    else:
        state_dict = ckpt

    # Strip model prefix and/or FSDP wrapper prefix
    fsdp_prefix = "_fsdp_wrapped_module."
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        # Strip model prefix first (e.g. "model.blocks.0..." -> "blocks.0...")
        if key.startswith(model_prefix):
            key = key[len(model_prefix):]
        # Strip FSDP wrapper prefix (e.g. "_fsdp_wrapped_module.blocks.0..." -> "blocks.0...")
        if key.startswith(fsdp_prefix):
            key = key[len(fsdp_prefix):]
        cleaned[key] = v

    print(f"  Loaded {len(cleaned)} weight tensors")
    return cleaned


# ========================
# Compilation
# ========================

def compile_transformer(args):
    """Compile CausalWanModel transformer with TP using ModelBuilder API."""

    tp_degree = args.tp_degree
    world_size = args.world_size

    # Dimensions for 480x832 resolution
    latent_height = args.height // 8   # 60 (VAE downscale 8x)
    latent_width = args.width // 8     # 104
    max_frames = args.max_frames       # 21
    text_len = 512
    text_dim = 4096
    batch_size = 1
    in_channels = 16

    # After patching: (1,2,2)
    post_f = max_frames  # patch_t = 1
    post_h = latent_height // 2  # 30
    post_w = latent_width // 2   # 52
    seq_len = post_f * post_h * post_w  # 21 * 30 * 52 = 32760

    print("=" * 60)
    print("RollingForcing CausalWanModel Compilation")
    print("=" * 60)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent: {max_frames}x{latent_height}x{latent_width}")
    print(f"Post-patch: {post_f}x{post_h}x{post_w}")
    print(f"Sequence length: {seq_len}")
    print(f"TP degree: {tp_degree}, World size: {world_size}")
    print("=" * 60)

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # Load CausalWanModel
        # Monkey-patch torch.cuda to avoid CUDA init in RollingForcing's t5.py
        print("\nLoading CausalWanModel...")
        import torch.cuda
        _orig_current_device = torch.cuda.current_device
        _orig_is_available = torch.cuda.is_available
        torch.cuda.current_device = lambda: 0
        torch.cuda.is_available = lambda: False
        try:
            from wan.modules.causal_model import CausalWanModel
        finally:
            torch.cuda.current_device = _orig_current_device
            torch.cuda.is_available = _orig_is_available

        original_model = CausalWanModel(
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=1536,
            ffn_dim=8960,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=12,
            num_layers=30,
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )

        # Load DMD weights
        ckpt_weights = load_rolling_forcing_weights(args.checkpoint_path)
        missing, unexpected = original_model.load_state_dict(ckpt_weights, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    {k}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"    {k}")

        original_model = original_model.to(torch.bfloat16).eval()

        # Save unsharded state for weight fixing
        unsharded_state = original_model.state_dict()

        # Collect norm weights that need manual sharding
        unsharded_norm_weights = {}
        for key, value in unsharded_state.items():
            if 'norm_k.weight' in key or 'norm_q.weight' in key:
                unsharded_norm_weights[f"transformer.{key}"] = value.clone()

        # Create Neuron model with TP sharding
        print(f"\nCreating NeuronCausalWanModel (TP={tp_degree})...")
        neuron_model = NeuronCausalWanModel(original_model, tp_degree)
        neuron_model = neuron_model.to(torch.bfloat16).eval()

        # Pre-compute RoPE for max sequence
        print("\nPre-computing RoPE...")
        from neuron_rope import compute_wan_rope_3d
        rope_cos, rope_sin = compute_wan_rope_3d(
            num_frames=max_frames,
            height=post_h,
            width=post_w,
            head_dim=original_model.dim // original_model.num_heads,
        )
        rope_cos = rope_cos.to(torch.bfloat16)
        rope_sin = rope_sin.to(torch.bfloat16)
        print(f"  RoPE cos: {rope_cos.shape}, sin: {rope_sin.shape}")

        # Sample inputs for tracing
        sample_hidden = torch.randn(
            batch_size, in_channels, max_frames, latent_height, latent_width,
            dtype=torch.bfloat16)
        sample_timestep = torch.zeros(batch_size, max_frames, dtype=torch.bfloat16)
        sample_text = torch.randn(
            batch_size, text_len, text_dim, dtype=torch.bfloat16)

        # Wrap and trace
        model = TracingWrapper(neuron_model)
        print("\nInitializing ModelBuilder...")
        builder = ModelBuilder(model=model)

        print("Tracing model...")
        builder.trace(
            kwargs={
                "hidden_states": sample_hidden,
                "timestep": sample_timestep,
                "encoder_hidden_states": sample_text,
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
            },
            tag="inference",
        )

        print("Compiling model...")
        compile_args = "--model-type=transformer -O1 --auto-cast=none"
        # Use absolute path for compiler_workdir to avoid doubled --logfile path
        # in neuronx-cc (relative path gets resolved against compiler's own workdir)
        abs_compiler_workdir = os.path.abspath(args.compiler_workdir)
        traced_model = builder.compile(
            compiler_args=compile_args,
            compiler_workdir=abs_compiler_workdir,
        )

        # Save compiled model
        output_path = f"{args.compiled_models_dir}/transformer"
        os.makedirs(output_path, exist_ok=True)
        print(f"\nSaving to {output_path}...")
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        # Save weights
        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        checkpoint = {}
        for key, value in model.state_dict().items():
            orig_key = key.replace("transformer.", "", 1)
            if orig_key in unsharded_state:
                val = unsharded_state[orig_key].clone()
            else:
                val = value.clone()
            if val.dtype == torch.float32:
                val = val.to(torch.bfloat16)
            checkpoint[key] = val

        print("Sharding weights...")
        shard_checkpoint(
            checkpoint=checkpoint,
            model=model,
            serialize_path=weights_path,
        )

        # Post-process: remove master_weight, fix norm weights
        print("Post-processing sharded checkpoints...")
        for rank in range(tp_degree):
            shard_file = os.path.join(
                weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                continue
            shard_data = dict(load_file(shard_file))
            cleaned = {k: v for k, v in shard_data.items()
                      if 'master_weight' not in k}
            save_file(cleaned, shard_file)

        # Fix norm weights
        _fix_norm_weights(weights_path, unsharded_norm_weights, tp_degree)

        # Save config
        config = {
            "height": args.height,
            "width": args.width,
            "max_frames": max_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "seq_len": seq_len,
            "text_len": text_len,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dim": 1536,
            "num_heads": 12,
            "num_layers": 30,
            "head_dim": 128,
            "patch_size": [1, 2, 2],
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save RoPE cache
        torch.save({
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        }, os.path.join(output_path, "rope_cache.pt"))

        print("\nCompilation complete!")
        print(f"Model saved to: {output_path}")


def _fix_norm_weights(weights_path, unsharded_norm_weights, tp_degree):
    """Fix norm_q/norm_k weights that shard_checkpoint doesn't handle."""
    print(f"Fixing norm weights for {tp_degree} ranks...")
    for rank in range(tp_degree):
        ckpt_path = os.path.join(
            weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = dict(load_file(ckpt_path))
        fixed = 0
        for key, unsharded in unsharded_norm_weights.items():
            if key in ckpt:
                shard_size = unsharded.shape[0] // tp_degree
                start = shard_size * rank
                end = shard_size * (rank + 1)
                ckpt[key] = unsharded[start:end].to(torch.bfloat16).clone()
                fixed += 1
        save_file(ckpt, ckpt_path)
        print(f"  Rank {rank}: Fixed {fixed} norm weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile CausalWanModel transformer for Neuron")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_frames", type=int, default=21)
    parser.add_argument("--tp_degree", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str,
                       default="/opt/dlami/nvme/rolling_forcing_hf_cache/rolling_forcing/checkpoints/rolling_forcing_dmd.pt")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="compiler_workdir")
    args = parser.parse_args()

    compile_transformer(args)
