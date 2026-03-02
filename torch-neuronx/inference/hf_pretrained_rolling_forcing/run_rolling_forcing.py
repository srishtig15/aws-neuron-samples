"""
RollingForcing Neuron Inference Script.

Runs the full rolling forcing pipeline on Trainium2:
1. Text encoding (compiled UMT5-XXL, TP=8)
2. Rolling forcing denoising loop (compiled CausalWanModel, TP=4)
3. VAE decoding (compiled Wan 3D VAE, TP=8)

Key design: Full-sequence processing (no KV cache).
- Maintain clean latent frames on host
- For each rolling window: concatenate anchor + working cache + current noisy → up to 21 frames
- Run compiled transformer on padded full sequence with per-frame timesteps
- Extract output for current frames only
- Eliminate the cache-update model call entirely (saves 50% model calls)
"""
import os

# Neuron runtime environment (must be set before torch import)
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import sys
import time
import json
import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file

# Add RollingForcing repo for scheduler
sys.path.insert(0, "/tmp/RollingForcing")

from neuron_rolling_forcing.neuron_rope import compute_rope_for_rolling_window

# ========================
# Constants
# ========================
MAX_FRAMES = 21
BLOCK_SIZE = 3          # num_frame_per_block
FRAME_SEQ = 1560        # 30 * 52 tokens per frame
MAX_SEQ_LEN = MAX_FRAMES * FRAME_SEQ  # 32760
POST_H = 30             # latent_height / patch_h
POST_W = 52             # latent_width / patch_w
HEAD_DIM = 128


# ========================
# Flow Matching Scheduler (copied from RollingForcing)
# ========================
class FlowMatchScheduler:
    """Minimal flow matching scheduler for inference."""

    def __init__(self, shift=5.0, sigma_min=0.0, sigma_max=1.0,
                 num_train_timesteps=1000, extra_one_step=True):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.extra_one_step = extra_one_step
        self.set_timesteps(1000, training=True)

    def set_timesteps(self, num_steps, training=False):
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                self.sigma_min + (self.sigma_max - self.sigma_min),
                self.sigma_min, num_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                self.sigma_min + (self.sigma_max - self.sigma_min),
                self.sigma_min, num_steps)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def add_noise(self, original_samples, noise, timestep):
        """x_t = (1-sigma) * x0 + sigma * noise"""
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    def convert_flow_to_x0(self, flow_pred, xt, timestep):
        """x0 = xt - sigma * flow_pred, where flow_pred = noise - x0"""
        self.sigmas = self.sigmas.to(flow_pred.device)
        self.timesteps = self.timesteps.to(flow_pred.device)
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma * flow_pred).type_as(flow_pred)

    def get_sigma(self, timestep_val):
        """Get sigma for a single warped timestep value."""
        tid = torch.argmin((self.timesteps - timestep_val).abs())
        return self.sigmas[tid].item()


# ========================
# Weight Loading Helpers
# ========================

def load_model_config(model_path):
    """Load config.json from compiled model directory."""
    with open(os.path.join(model_path, "config.json"), "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
    """Load TP sharded weights from safetensors files.

    Filters out master_weight tensors which are artifacts from shard_checkpoint()
    and not actual model parameters.
    """
    weights_path = os.path.join(model_path, "weights")
    sharded_weights = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        if rank == 0:
            removed = len(raw_ckpt) - len(ckpt)
            if removed > 0:
                print(f"  Filtered {removed} master_weight tensors ({len(ckpt)} keys remaining)")
        sharded_weights.append(ckpt)
    return sharded_weights


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    """Duplicate TP checkpoints for CP ranks.

    For world_size = tp_degree * cp_degree, creates one checkpoint per rank:
    Rank 0 = tp0, Rank 1 = tp1, ..., Rank tp_degree = tp0 (CP group 1), etc.
    """
    cp_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            cp_checkpoints.append(tp_checkpoints[tp_rank])
    return cp_checkpoints


def load_duplicated_weights(model_path, world_size):
    """Load single checkpoint and duplicate for all ranks.

    For models like decoder that don't use actual TP sharding,
    we load tp0 checkpoint and duplicate for all world_size ranks.
    """
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)
    sharded_weights = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded_weights.append(ckpt)
    return sharded_weights


# ========================
# Neuron Model Wrappers
# ========================

class NeuronTransformerWrapper:
    """Wrapper for compiled Neuron CausalWanModel transformer."""

    def __init__(self, compiled_model_dir, tp_degree=4):
        from neuronx_distributed import NxDModel

        transformer_path = os.path.join(compiled_model_dir, "transformer")
        config = load_model_config(transformer_path)
        tp_degree = config["tp_degree"]
        world_size = config["world_size"]
        cp_degree = world_size // tp_degree

        print(f"  Loading transformer (TP={tp_degree}, CP={cp_degree}, world_size={world_size})...")

        # Load NxDModel
        nxd_model_path = os.path.join(transformer_path, "nxd_model.pt")
        self.model = NxDModel.load(nxd_model_path)

        # Load TP-sharded weights and duplicate for CP ranks
        tp_weights = load_sharded_weights(transformer_path, tp_degree)
        if cp_degree > 1:
            weights = prepare_cp_checkpoints(tp_weights, tp_degree, cp_degree)
        else:
            weights = tp_weights
        self.model.set_weights(weights)
        self.model.to_neuron()

        # Load RoPE cache for default positions
        rope_data = torch.load(
            os.path.join(transformer_path, "rope_cache.pt"),
            map_location="cpu")
        self.default_rope_cos = rope_data["rope_cos"]
        self.default_rope_sin = rope_data["rope_sin"]
        print(f"  Transformer loaded. RoPE: cos={self.default_rope_cos.shape}")

    def __call__(self, hidden_states, timestep, encoder_hidden_states,
                 rope_cos, rope_sin):
        """
        Args:
            hidden_states: [1, 16, F, 60, 104] (will be padded to 21 frames)
            timestep: [1, F] per-frame timesteps
            encoder_hidden_states: [1, 512, 4096]
            rope_cos: [1, seq, 1, 128]
            rope_sin: [1, seq, 1, 128]
        Returns:
            flow_pred: [1, 16, F, 60, 104] (unpadded)
        """
        B, C, F, H, W = hidden_states.shape

        # Pad to MAX_FRAMES if needed
        if F < MAX_FRAMES:
            pad_f = MAX_FRAMES - F
            # Pad with random noise at the max timestep instead of zeros at t=0.
            # Zero-padding with t=0 tells the model "these are clean frames with zero
            # content", which dilutes attention and weakens predictions for real frames
            # (especially severe for early windows with only 3-6 real frames).
            # Noise-padding with the same high timestep makes padded frames look like
            # additional noisy input, so the model treats them similarly to real frames.
            pad_noise = torch.randn(B, C, pad_f, H, W, dtype=hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, pad_noise], dim=2)
            max_t = timestep.max().item()
            timestep = F_module.pad(timestep, (0, pad_f), value=max_t)

        # Pad RoPE to MAX_SEQ_LEN if needed
        rope_seq = rope_cos.shape[1]
        if rope_seq < MAX_SEQ_LEN:
            pad_seq = MAX_SEQ_LEN - rope_seq
            rope_cos = F_module.pad(rope_cos, (0, 0, 0, 0, 0, pad_seq))
            rope_sin = F_module.pad(rope_sin, (0, 0, 0, 0, 0, pad_seq))

        # Convert to bf16
        hidden_states = hidden_states.to(torch.bfloat16)
        timestep = timestep.to(torch.bfloat16)
        encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
        rope_cos = rope_cos.to(torch.bfloat16)
        rope_sin = rope_sin.to(torch.bfloat16)

        # Run compiled model
        output = self.model(
            hidden_states, timestep, encoder_hidden_states,
            rope_cos, rope_sin)
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Extract valid frames
        return output[:, :, :F, :, :]


class NeuronTextEncoderWrapper:
    """Wrapper for compiled Neuron UMT5-XXL text encoder."""

    def __init__(self, compiled_model_dir, tp_degree=8):
        import neuronx_distributed
        self.model = neuronx_distributed.trace.parallel_model_load(
            os.path.join(compiled_model_dir, "text_encoder"))

    def __call__(self, text_input_ids, attention_mask):
        """
        Args:
            text_input_ids: [1, 512] int64
            attention_mask: [1, 512] int64
        Returns:
            prompt_embeds: [1, 512, 4096]
        """
        output = self.model(text_input_ids, attention_mask)
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        if isinstance(output, dict):
            # parallel_model_load returns dict with 'last_hidden_state' or numeric keys
            if 'last_hidden_state' in output:
                return output['last_hidden_state']
            # Try first value
            return next(iter(output.values()))
        return output[0] if isinstance(output, (tuple, list)) else output


class NeuronVAEDecoderWrapper:
    """Wrapper for compiled Neuron VAE decoder with NoCache pattern."""

    def __init__(self, compiled_model_dir, tp_degree=8):
        from neuronx_distributed import NxDModel

        # Load decoder
        decoder_path = os.path.join(compiled_model_dir, "decoder_nocache")
        decoder_config = load_model_config(decoder_path)
        decoder_world_size = decoder_config.get("world_size", 8)
        print(f"  Loading decoder (world_size={decoder_world_size})...")

        self.decoder = NxDModel.load(os.path.join(decoder_path, "nxd_model.pt"))
        decoder_weights = load_duplicated_weights(decoder_path, decoder_world_size)
        self.decoder.set_weights(decoder_weights)
        self.decoder.to_neuron()
        print("  Decoder loaded.")

        # Load post_quant_conv
        pqc_path = os.path.join(compiled_model_dir, "post_quant_conv")
        pqc_config = load_model_config(pqc_path)
        pqc_world_size = pqc_config.get("world_size", 8)
        print(f"  Loading post_quant_conv (world_size={pqc_world_size})...")

        self.post_quant_conv = NxDModel.load(os.path.join(pqc_path, "nxd_model.pt"))
        pqc_weights = load_duplicated_weights(pqc_path, pqc_world_size)
        self.post_quant_conv.set_weights(pqc_weights)
        self.post_quant_conv.to_neuron()
        print("  post_quant_conv loaded.")

        # VAE normalization constants (from Wan2.1)
        self.mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=torch.float32).view(1, 16, 1, 1, 1)
        self.std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], dtype=torch.float32).view(1, 16, 1, 1, 1)

    def decode(self, latents):
        """
        Decode latents to pixel space.

        Args:
            latents: [1, F, 16, 60, 104] (frames, channels, h, w)
        Returns:
            video: [1, F, 3, 480, 832]
        """
        # Convert from [B, F, C, H, W] to [B, C, F, H, W]
        z = latents.permute(0, 2, 1, 3, 4).float()

        # Un-normalize
        z = z * self.std.to(z.device) + self.mean.to(z.device)

        # post_quant_conv on full volume
        z = self.post_quant_conv(z)
        if isinstance(z, (tuple, list)):
            z = z[0]

        # Decode 2 frames at a time with NoCache decoder
        # (compiled with decoder_frames=2 to satisfy CACHE_T=2 requirement)
        z_bf16 = z.to(torch.bfloat16)
        num_frames = z.shape[2]
        decoded_frames = []
        for f in range(0, num_frames, 2):
            end_f = min(f + 2, num_frames)
            chunk = z_bf16[:, :, f:end_f, :, :]
            # Pad last chunk to 2 frames if odd number of frames
            if chunk.shape[2] < 2:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, 1))
            decoded = self.decoder(chunk)  # [1, 3, 2, 480, 832]
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            # Only keep valid frames
            valid = end_f - f
            decoded_frames.append(decoded[:, :, :valid, :, :].float())

        video = torch.cat(decoded_frames, dim=2)  # [1, 3, F, 480, 832]
        video = video.clamp(-1, 1)

        # Convert to [B, F, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        return video


# ========================
# Rolling Forcing Loop
# ========================

F_module = torch.nn.functional


def build_rolling_windows(num_blocks, num_denoising_steps):
    """Build rolling forcing window schedule."""
    window_length = num_denoising_steps
    num_windows = num_blocks + window_length - 1
    windows = []
    for w in range(num_windows):
        start = max(0, w - window_length + 1)
        end = min(num_blocks - 1, w)
        windows.append((start, end))
    return windows


def build_shared_timesteps(denoising_step_list, num_frame_per_block, device):
    """Build per-frame timestep tensor for full rolling window."""
    num_steps = len(denoising_step_list)
    total_frames = num_steps * num_frame_per_block
    timestep = torch.ones(1, total_frames, device=device, dtype=torch.float32)
    # From clean to noisy (reversed list)
    for idx, t in enumerate(reversed(denoising_step_list)):
        start = idx * num_frame_per_block
        end = (idx + 1) * num_frame_per_block
        timestep[:, start:end] *= t
    return timestep


def run_rolling_forcing(args):
    """Main inference function."""

    device = torch.device("cpu")  # Host device for orchestration
    compiled_dir = args.compiled_models_dir

    print("=" * 60)
    print("RollingForcing Neuron Inference")
    print("=" * 60)

    # Load compiled models
    print("\nLoading compiled models...")
    t0 = time.time()
    text_encoder = NeuronTextEncoderWrapper(compiled_dir)
    transformer = NeuronTransformerWrapper(compiled_dir)
    vae_decoder = NeuronVAEDecoderWrapper(compiled_dir)
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer",
        cache_dir=args.cache_dir)
    tokens = tokenizer(
        args.prompt, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt")
    text_input_ids = tokens.input_ids.to(torch.int64)
    attention_mask = tokens.attention_mask.to(torch.int64)

    # Text encoding (positive prompt)
    print("Running text encoder...")
    t0 = time.time()
    prompt_embeds = text_encoder(text_input_ids, attention_mask)
    print(f"  Positive text encoding: {time.time()-t0:.1f}s, shape: {prompt_embeds.shape}")

    # Text encoding (negative prompt for CFG)
    negative_prompt_embeds = None
    if args.guidance_scale > 1.0:
        print(f"  Encoding negative prompt (guidance_scale={args.guidance_scale})...")
        neg_tokens = tokenizer(
            args.negative_prompt, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt")
        neg_input_ids = neg_tokens.input_ids.to(torch.int64)
        neg_attention_mask = neg_tokens.attention_mask.to(torch.int64)
        t0 = time.time()
        negative_prompt_embeds = text_encoder(neg_input_ids, neg_attention_mask)
        print(f"  Negative text encoding: {time.time()-t0:.1f}s, shape: {negative_prompt_embeds.shape}")

    # Setup dimensions
    num_frames = args.num_frames  # Total output frames (e.g., 81)
    latent_frames = (num_frames - 1) // 4 + 1  # e.g., 21 for 81 frames
    latent_h, latent_w = 60, 104
    num_channels = 16

    # For rolling forcing: all frames are generated
    num_blocks = latent_frames // BLOCK_SIZE
    assert latent_frames % BLOCK_SIZE == 0, \
        f"latent_frames {latent_frames} must be divisible by BLOCK_SIZE {BLOCK_SIZE}"

    # Denoising steps (from RollingForcing config)
    # The DMD model uses 5 denoising steps
    denoising_step_list = torch.tensor(
        args.denoising_step_list, dtype=torch.float32)

    # Warp timesteps using scheduler (shift=5.0, sigma_min=0.0 matching reference config)
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    warped_steps = []
    timesteps_1000 = torch.cat([
        scheduler.timesteps.cpu(),
        torch.tensor([0], dtype=torch.float32)])
    for step in denoising_step_list:
        warped_steps.append(timesteps_1000[1000 - int(step.item())].item())
    denoising_step_list = torch.tensor(warped_steps, dtype=torch.float32)
    num_denoising_steps = len(denoising_step_list)
    print(f"\nDenoising steps (warped): {denoising_step_list.tolist()}")

    # Pre-compute sigma values for each denoising step (for ODE stepping)
    denoising_sigmas = torch.tensor(
        [scheduler.get_sigma(t) for t in denoising_step_list],
        dtype=torch.float32)
    print(f"Denoising sigmas: {denoising_sigmas.tolist()}")

    # Build rolling windows
    windows = build_rolling_windows(num_blocks, num_denoising_steps)
    print(f"Total blocks: {num_blocks}, Windows: {len(windows)}")

    # Initialize noise
    torch.manual_seed(args.seed)
    noise = torch.randn(1, latent_frames, num_channels, latent_h, latent_w)

    # Shared timestep pattern
    shared_timestep = build_shared_timesteps(
        denoising_step_list, BLOCK_SIZE, device)

    # Output and noisy cache
    output = torch.zeros(1, latent_frames, num_channels, latent_h, latent_w)
    noisy_cache = torch.zeros_like(output)

    # pred_x0 cache: stores each block's latest predicted x0 (clean estimate).
    # Updated after EVERY window (not just finalization), mimicking the original
    # pipeline's KV cache behavior where clean representations are available
    # from the very first window.
    # Key = block_idx, Value = [1, BLOCK_SIZE, C, H, W] tensor
    pred_x0_cache = {}
    # Track which block indices are finalized (completed all denoising steps)
    finalized_blocks = set()

    # Denoising loop
    cfg_str = f" (CFG scale={args.guidance_scale})" if negative_prompt_embeds is not None else " (no CFG)"
    print(f"\nStarting rolling forcing denoising...{cfg_str}")
    total_t0 = time.time()

    for window_idx, (start_block, end_block) in enumerate(windows):
        w_t0 = time.time()

        current_start = start_block * BLOCK_SIZE
        current_end = (end_block + 1) * BLOCK_SIZE
        current_num_frames = current_end - current_start
        rolling_len = num_denoising_steps * BLOCK_SIZE

        # 1. Build noisy input for current window
        if current_num_frames == rolling_len or current_start == 0:
            # Normal case or start: use cached noisy + fresh noise for last block
            noisy_input = torch.cat([
                noisy_cache[:, current_start:current_end - BLOCK_SIZE],
                noise[:, current_end - BLOCK_SIZE:current_end],
            ], dim=1)
        else:
            # End of video: use cached noisy
            noisy_input = noisy_cache[:, current_start:current_end]

        # 2. Build per-frame timesteps for noisy frames
        if current_num_frames == rolling_len:
            current_timestep = shared_timestep
        elif current_start == 0:
            current_timestep = shared_timestep[:, -current_num_frames:]
        elif current_end == latent_frames:
            current_timestep = shared_timestep[:, :current_num_frames]
        else:
            raise ValueError(f"Unexpected window: start={current_start}, end={current_end}")

        # 3. Build full visible sequence with clean pred_x0 context
        # Use pred_x0_cache to provide clean frame context, matching the original
        # pipeline's KV cache mechanism. Clean frames go FIRST (causal attention
        # ensures they only attend to other clean frames, preserving quality).
        #
        # Selection strategy (matching original):
        # - Block 0 as "attention sink" (always included if cached)
        # - Recent cached blocks as "working cache" (most relevant context)
        # - Blocks IN the current noisy window can also be included as clean
        #   context (original does this too: anchor block 0 appears in both
        #   clean KV cache and noisy input simultaneously)
        anchor_frames = None
        anchor_positions = torch.tensor([], dtype=torch.long)
        n_anchor = 0

        if pred_x0_cache:
            max_anchor_frames = MAX_FRAMES - current_num_frames
            if max_anchor_frames >= BLOCK_SIZE:
                anchor_block_indices = []

                # Always include block 0 (attention sink) if cached and NOT
                # in current window. Original uses local_start_index==0 to skip
                # cache context entirely for windows starting at block 0.
                if 0 in pred_x0_cache and start_block > 0:
                    anchor_block_indices.append(0)

                # Add cached blocks before the current window (working cache)
                for blk in sorted(pred_x0_cache.keys()):
                    if blk in anchor_block_indices:
                        continue
                    if blk >= start_block:
                        break  # Only include blocks before current window
                    total_anchor = (len(anchor_block_indices) + 1) * BLOCK_SIZE
                    if total_anchor <= max_anchor_frames:
                        anchor_block_indices.append(blk)

                if anchor_block_indices:
                    anchor_list = []
                    anchor_pos_list = []
                    for blk in sorted(anchor_block_indices):
                        # Use pred_x0_cache (latest clean estimate) instead of output
                        anchor_list.append(pred_x0_cache[blk])
                        blk_start_f = blk * BLOCK_SIZE
                        anchor_pos_list.extend(range(blk_start_f, blk_start_f + BLOCK_SIZE))

                    anchor_frames = torch.cat(anchor_list, dim=1)
                    anchor_positions = torch.tensor(anchor_pos_list, dtype=torch.long)
                    n_anchor = anchor_frames.shape[1]

        # Build visible input: [clean pred_x0 at t=0 | noisy current window]
        if anchor_frames is not None and n_anchor > 0:
            full_input = torch.cat([anchor_frames, noisy_input], dim=1)
            anchor_timestep = torch.zeros(1, n_anchor, dtype=torch.float32)
            full_timestep = torch.cat([anchor_timestep, current_timestep], dim=1)
        else:
            full_input = noisy_input
            full_timestep = current_timestep

        total_visible = full_input.shape[1]
        visible_input = full_input.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

        # 4. Compute RoPE for this window (with anchor positions)
        current_positions = torch.arange(current_start, current_end, dtype=torch.long)
        rope_cos, rope_sin = compute_rope_for_rolling_window(
            anchor_frame_positions=anchor_positions,
            working_frame_positions=torch.tensor([], dtype=torch.long),
            current_frame_positions=current_positions,
            height=POST_H,
            width=POST_W,
            max_frames=MAX_FRAMES,
            head_dim=HEAD_DIM,
        )

        # 5. Run compiled transformer
        if negative_prompt_embeds is not None:
            # Classifier-free guidance: run twice
            flow_pred_cond = transformer(
                visible_input, full_timestep,
                prompt_embeds, rope_cos, rope_sin)
            flow_pred_uncond = transformer(
                visible_input, full_timestep,
                negative_prompt_embeds, rope_cos, rope_sin)
            flow_pred = flow_pred_uncond + args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
        else:
            flow_pred = transformer(
                visible_input, full_timestep,
                prompt_embeds, rope_cos, rope_sin)

        # 6. Convert flow prediction to x0 for CURRENT frames only (skip anchor)
        # Extract only the current window frames from the full output
        flow_pred_all = flow_pred.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        flow_pred_current = flow_pred_all[:, n_anchor:n_anchor + current_num_frames]
        pred_x0 = scheduler.convert_flow_to_x0(
            flow_pred_current.flatten(0, 1),
            noisy_input.flatten(0, 1),
            current_timestep.flatten(0, 1),
        ).unflatten(0, (1, current_num_frames))

        # 7. Store output
        output[:, current_start:current_end] = pred_x0

        # 8. Update pred_x0_cache and noisy cache for next windows
        with torch.no_grad():
            # Update pred_x0_cache for the FIRST block of the current window
            # (matching original's cache update which only processes the first block)
            first_blk_x0 = pred_x0[:, :BLOCK_SIZE].detach().clone()
            pred_x0_cache[start_block] = first_blk_x0

            for block_idx in range(start_block, end_block + 1):
                blk_start = (block_idx - start_block) * BLOCK_SIZE
                blk_end = blk_start + BLOCK_SIZE
                block_t = current_timestep[:, blk_start:blk_end].mean().item()

                # Find which denoising step this corresponds to
                matches = torch.abs(denoising_step_list - block_t) < 1e-4
                step_idx = torch.nonzero(matches, as_tuple=True)[0]
                if len(step_idx) == 0 or step_idx[0] >= len(denoising_step_list) - 1:
                    # This block is at its final denoising step → mark as finalized
                    finalized_blocks.add(block_idx)
                    # Also update pred_x0_cache for finalized blocks
                    block_x0 = pred_x0[:, blk_start:blk_end].detach().clone()
                    pred_x0_cache[block_idx] = block_x0
                    continue

                next_t = denoising_step_list[step_idx[0] + 1]

                # Deterministic ODE step for re-noising:
                #   x_{t-1} = xt + (sigma_{t-1} - sigma_t) * flow
                # This avoids catastrophic cancellation in x0 = xt - sigma*flow
                # and preserves the signal through intermediate steps.
                sigma_current = scheduler.get_sigma(block_t)
                sigma_next = scheduler.get_sigma(next_t)
                flow_block = flow_pred_current[:, blk_start:blk_end].float()
                noisy_block = noisy_input[:, blk_start:blk_end].float()

                noisy_cache[:, block_idx*BLOCK_SIZE:(block_idx+1)*BLOCK_SIZE] = \
                    noisy_block + (sigma_next - sigma_current) * flow_block

        w_time = time.time() - w_t0
        anchor_str = f", clean={n_anchor}f" if n_anchor > 0 else ""
        cache_str = f", cache={len(pred_x0_cache)}blk"
        print(f"  Window {window_idx}/{len(windows)-1}: "
              f"blocks [{start_block}-{end_block}], "
              f"frames [{current_start}-{current_end}]{anchor_str}{cache_str}, "
              f"{w_time:.2f}s")

    total_time = time.time() - total_t0
    print(f"\nDenoising complete: {total_time:.1f}s "
          f"({total_time/len(windows):.2f}s/window)")

    # Save raw latents for debugging
    if args.save_latents:
        latent_path = args.output_path.replace('.mp4', '_latents.pt')
        torch.save(output, latent_path)
        print(f"\nSaved raw latents to {latent_path}, shape: {output.shape}")

    # VAE decoding
    if args.decode_cpu:
        print("\nDecoding with CPU VAE (full temporal upsampling)...")
        video = decode_with_cpu_vae(output, args.cache_dir)
    else:
        print("\nDecoding with Neuron VAE...")
        t0 = time.time()
        video = vae_decoder.decode(output)
        print(f"  VAE decoding: {time.time()-t0:.1f}s")

    # Normalize to [0, 1]
    video = (video * 0.5 + 0.5).clamp(0, 1)

    # Save video
    print(f"\nSaving video to {args.output_path}...")
    save_video(video[0], args.output_path, fps=args.fps)
    print("Done!")


def decode_with_cpu_vae(latents, cache_dir):
    """
    Decode latents using the original Wan2.1 VAE on CPU.

    This provides full temporal upsampling (21 latent frames → 81 pixel frames)
    without the limitations of the Neuron NoCache decoder.

    Args:
        latents: [1, F_latent, 16, 60, 104] latent tensor
        cache_dir: HuggingFace cache directory containing the Wan2.1 model
    Returns:
        video: [1, F_pixel, 3, 480, 832] in [-1, 1]
    """
    from diffusers import AutoencoderKLWan
    t0 = time.time()

    print("  Loading AutoencoderKLWan from cache...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    ).eval()

    # VAE normalization constants (from Wan2.1)
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=torch.float32).view(1, 16, 1, 1, 1)

    # Convert from [B, F, C, H, W] to [B, C, F, H, W]
    z = latents.permute(0, 2, 1, 3, 4).float()

    # Un-normalize
    z = z * std + mean

    print(f"  Latent shape: {z.shape}, running VAE decode on CPU...")
    with torch.no_grad():
        # AutoencoderKLWan.decode expects [B, C, F, H, W]
        decoded = vae.decode(z).sample  # [B, 3, F_pixel, H, W]

    decoded = decoded.clamp(-1, 1)
    # Convert to [B, F, C, H, W]
    video = decoded.permute(0, 2, 1, 3, 4)
    print(f"  CPU VAE decode: {time.time()-t0:.1f}s, output shape: {video.shape}")

    del vae
    return video


def save_video(video_tensor, output_path, fps=16):
    """
    Save video tensor to file.

    Args:
        video_tensor: [F, C, H, W] in [0, 1]
        output_path: output file path
        fps: frames per second
    """
    try:
        import imageio
        frames = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
        # [F, C, H, W] -> [F, H, W, C]
        frames = np.transpose(frames, (0, 2, 3, 1))
        imageio.mimwrite(output_path, frames, fps=fps)
        print(f"  Saved {len(frames)} frames to {output_path}")
    except ImportError:
        # Fallback: save as individual frames
        os.makedirs(output_path.replace('.mp4', '_frames'), exist_ok=True)
        for i, frame in enumerate(video_tensor):
            from PIL import Image
            img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(
                os.path.join(output_path.replace('.mp4', '_frames'), f'{i:04d}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RollingForcing Neuron Inference")
    parser.add_argument("--prompt", type=str,
                       default="A cat walking gracefully across a sunlit garden path")
    parser.add_argument("--num_frames", type=int, default=81,
                       help="Total output video frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--compiled_models_dir", type=str, default="compiled_models")
    parser.add_argument("--cache_dir", type=str,
                       default="/opt/dlami/nvme/rolling_forcing_hf_cache")
    parser.add_argument("--denoising_step_list", type=int, nargs="+",
                       default=[1000, 800, 600, 400, 200],
                       help="Denoising step indices (from noisy to clean)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Classifier-free guidance scale (0=uncond only, 1=cond only, >1=amplified)")
    parser.add_argument("--negative_prompt", type=str,
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                       help="Negative prompt for classifier-free guidance")
    parser.add_argument("--decode_cpu", action="store_true",
                       help="Decode latents on CPU using original VAE "
                            "(full temporal upsampling, 21→81 frames)")
    parser.add_argument("--save_latents", action="store_true",
                       help="Save raw latent output to .pt file for debugging")
    args = parser.parse_args()

    run_rolling_forcing(args)
