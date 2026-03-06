"""
Wan2.2 T2V-A14B Inference with MoE (Mixture of Experts) Transformer Switching.

This model uses TWO transformers (both WanTransformer3DModel with 40 heads, 40 layers)
that are selected based on the denoising timestep:
- transformer (high-noise expert): used when timestep >= 875 (~12.5% of steps)
- transformer_2 (low-noise expert): used when timestep < 875 (~87.5% of steps)

Architecture:
- TP=4, CP=2 (world_size=8) for each transformer
- Both transformers share the SAME compiled NEFF (identical architecture)
- MoE weight swap uses NxDModel.replace_weights() (no unload/reload needed)
- Staged pipeline:
  Phase 1: Text Encoder → encode prompt → unload
  Phase 2: Load 1 NxDModel → high-noise steps → replace_weights → low-noise steps
  Phase 3: Decoder + PostQuantConv → decode latents → save video

Usage:
    python run_wan2.2_t2v_a14b.py \
        --compiled_models_dir /opt/dlami/nvme/compiled_models_t2v_a14b \
        --prompt "A cat walking on the grass"
"""
# IMPORTANT: Set environment variables BEFORE any imports
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import argparse
import gc
import json
import numpy as np
import random
import time
import torch
import torch_neuronx

from neuronx_distributed import NxDModel
from safetensors.torch import load_file

from neuron_wan2_2_t2v_a14b.neuron_commons import (
    InferenceTextEncoderWrapperV2,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_model_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_sharded_weights(model_path, tp_degree):
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


def fix_norm_weights_from_pipeline(sharded_weights, pipe_transformer, expected_norm_size):
    unsharded_norms = {}
    for key, value in pipe_transformer.state_dict().items():
        if 'norm_k.weight' in key or 'norm_q.weight' in key:
            unsharded_norms[f"transformer.{key}"] = value.clone()
    if not unsharded_norms:
        return
    num_ranks = len(sharded_weights)
    norm_tp = unsharded_norms[list(unsharded_norms.keys())[0]].shape[0] // expected_norm_size
    fixed_count = 0
    for rank in range(num_ranks):
        ckpt = sharded_weights[rank]
        norm_rank = rank % norm_tp
        for norm_key, full_weight in unsharded_norms.items():
            if norm_key in ckpt and ckpt[norm_key].shape[0] != expected_norm_size:
                start = norm_rank * expected_norm_size
                ckpt[norm_key] = full_weight[start:start + expected_norm_size].to(ckpt[norm_key].dtype).clone()
                fixed_count += 1
    if fixed_count > 0:
        print(f"  Fixed {fixed_count} norm weights to size {expected_norm_size} (norm_tp={norm_tp})")


def load_duplicated_weights(model_path, world_size):
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)
    sharded_weights = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded_weights.append(ckpt)
    return sharded_weights


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    world_size = tp_degree * cp_degree
    sharded_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
            sharded_checkpoints.append(ckpt)
    print(f"  Prepared {len(sharded_checkpoints)} checkpoints (TP={tp_degree}, CP={cp_degree})")
    return sharded_checkpoints


# ============================================================
# Phase 1: Text Encoding
# ============================================================
def phase_text_encoding_cpu(pipe, seqlen, prompt, negative_prompt):
    """Encode prompt using CPU text encoder."""
    print("Encoding prompt on CPU...")
    device = torch.device("cpu")
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=seqlen,
        device=device,
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}, negative: {negative_prompt_embeds.shape}")
    print(f"  Text encoding done in {time.time() - t0:.1f}s")
    return prompt_embeds, negative_prompt_embeds


def phase_text_encoding_neuron(pipe, compiled_models_dir, seqlen, prompt, negative_prompt):
    """
    Encode prompt using Neuron-compiled text encoder (TI2V pattern).

    Uses InferenceTextEncoderWrapperV2 to wrap NxDModel and replaces
    pipe.text_encoder, then calls pipe.encode_prompt() which handles
    tokenization and attention mask application correctly.
    """
    te_path = f"{compiled_models_dir}/text_encoder"
    te_config = load_model_config(te_path)
    te_tp_degree = te_config["tp_degree"]
    te_world_size = te_config["world_size"]

    # Step 1: Create wrapper (TI2V pattern)
    text_encoder_wrapper = InferenceTextEncoderWrapperV2(
        torch.bfloat16, pipe.text_encoder, seqlen
    )

    # Step 2: Load NxDModel (no start_rank/local_ranks_size, like TI2V)
    print(f"Loading text encoder (TP={te_tp_degree}, world_size={te_world_size})...")
    t0 = time.time()
    te_nxd = NxDModel.load(os.path.join(te_path, "nxd_model.pt"))

    # Step 3: Load weights with prepare_cp_checkpoints (like TI2V)
    tp_checkpoints = load_sharded_weights(te_path, te_tp_degree)
    if te_world_size > te_tp_degree:
        cp_degree = te_world_size // te_tp_degree
        checkpoints = prepare_cp_checkpoints(tp_checkpoints, te_tp_degree, cp_degree)
    else:
        checkpoints = tp_checkpoints

    te_nxd.set_weights(checkpoints)
    te_nxd.to_neuron()
    print(f"  Text encoder loaded in {time.time() - t0:.1f}s")

    # Step 4: Replace wrapper's encoder with NxDModel (like TI2V)
    text_encoder_wrapper.t = te_nxd

    # Step 5: Replace pipe's text encoder (like TI2V)
    original_text_encoder = pipe.text_encoder
    pipe.text_encoder = text_encoder_wrapper

    # Step 6: Use pipe.encode_prompt (handles tokenization + mask correctly)
    t_enc = time.time()
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=seqlen,
        device=torch.device("cpu"),
    )
    print(f"  prompt_embeds: {prompt_embeds.shape}, negative: {negative_prompt_embeds.shape}")
    print(f"  Text encoding done in {time.time() - t_enc:.1f}s (load: {t_enc - t0:.1f}s)")

    # Step 7: Restore and clean up
    pipe.text_encoder = original_text_encoder
    del te_nxd, tp_checkpoints, checkpoints, text_encoder_wrapper
    gc.collect()

    return prompt_embeds, negative_prompt_embeds


def phase_text_encoding(pipe, compiled_models_dir, seqlen, prompt, negative_prompt,
                        use_neuron=False):
    """Encode prompt. Uses CPU by default, Neuron with --neuron_text_encoder flag."""
    print("\n" + "="*60)
    mode = "Neuron" if use_neuron else "CPU"
    print(f"PHASE 1: Text Encoding ({mode})")
    print("="*60)

    if use_neuron:
        return phase_text_encoding_neuron(pipe, compiled_models_dir, seqlen, prompt, negative_prompt)
    else:
        return phase_text_encoding_cpu(pipe, seqlen, prompt, negative_prompt)


# ============================================================
# Phase 2: Denoising with MoE Transformer Swap (replace_weights)
# ============================================================
def load_and_prepare_weights(compiled_path, pipe_transformer, tp_degree, cp_degree, label="transformer"):
    """Load and prepare sharded+CP weights from a compiled transformer directory."""
    print(f"Loading weights for {label}...")
    tp_checkpoints = load_sharded_weights(compiled_path, tp_degree)

    # Detect and fix norm weights
    hidden_size = 5120
    ideal_norm_size = hidden_size // tp_degree
    expected_norm_size = None
    for key in tp_checkpoints[0]:
        if 'norm_q.weight' in key:
            actual_size = tp_checkpoints[0][key].shape[0]
            if actual_size != ideal_norm_size and actual_size != hidden_size:
                expected_norm_size = actual_size
            break
    if expected_norm_size is not None:
        fix_norm_weights_from_pipeline(tp_checkpoints, pipe_transformer, expected_norm_size)

    cp_checkpoints = prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree)
    return cp_checkpoints


def run_transformer_step(nxd_model, hidden_states, timestep, encoder_hidden_states,
                          rotary_emb_cos, rotary_emb_sin):
    """Run a single transformer forward pass."""
    if timestep is not None:
        if timestep.dim() > 1:
            timestep = timestep.flatten()[0:1]
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(torch.float32)

    output = nxd_model(
        hidden_states, timestep, encoder_hidden_states,
        rotary_emb_cos, rotary_emb_sin,
    )

    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def phase_denoising(pipe, compiled_models_dir, prompt_embeds, negative_prompt_embeds,
                     args, generator=None):
    """Run the denoising loop with MoE transformer swap using replace_weights()."""
    print("\n" + "="*60)
    print("PHASE 2: Denoising (MoE via replace_weights)")
    print("="*60)

    DTYPE = torch.bfloat16
    device = torch.device("cpu")

    prompt_embeds = prompt_embeds.to(DTYPE)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)

    # Prepare scheduler and timesteps
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latents
    in_channels = pipe.transformer.config.in_channels if pipe.transformer is not None else 16
    latents = pipe.prepare_latents(
        1, in_channels, args.height, args.width, args.num_frames,
        torch.float32, device, generator, None,
    )
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    # MoE boundary
    boundary_timestep = 0.875 * 1000  # 875.0
    guidance_scale = args.guidance_scale

    # Determine switch point
    switch_idx = None
    for i, t in enumerate(timesteps):
        if t < boundary_timestep:
            switch_idx = i
            break
    if switch_idx is None:
        switch_idx = len(timesteps)

    print(f"Total steps: {len(timesteps)}, switch at step {switch_idx} (boundary={boundary_timestep})")
    print(f"  transformer_1 (high-noise): steps 0-{switch_idx-1} ({switch_idx} steps)")
    print(f"  transformer_2 (low-noise):  steps {switch_idx}-{len(timesteps)-1} ({len(timesteps) - switch_idx} steps)")

    transformer_1_path = f"{compiled_models_dir}/transformer"
    transformer_2_path = f"{compiled_models_dir}/transformer_2"

    # Load config from transformer_1
    config = load_model_config(transformer_1_path)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]

    # Load RoPE (same for both transformers - same architecture)
    rope_cache = torch.load(os.path.join(transformer_1_path, "rope_cache.pt"))
    rotary_emb_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
    rotary_emb_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)

    # Prepare weights for both transformers
    t1_weights = load_and_prepare_weights(
        transformer_1_path, pipe.transformer, tp_degree, cp_degree, "transformer_1"
    )
    t2_weights = load_and_prepare_weights(
        transformer_2_path, pipe.transformer_2, tp_degree, cp_degree, "transformer_2"
    )

    # Load ONE NxDModel (from transformer_1's NEFF)
    nxd_model_path = os.path.join(transformer_1_path, "nxd_model.pt")
    nxd_model = NxDModel.load(nxd_model_path, start_rank=0, local_ranks_size=world_size)

    # Initialize with transformer_1 weights
    print(f"\nLoading NxDModel with transformer_1 weights (TP={tp_degree}, CP={cp_degree})...")
    nxd_model.set_weights(t1_weights)
    t0 = time.time()
    nxd_model.to_neuron()
    print(f"  NxDModel loaded to NeuronCores in {time.time() - t0:.1f}s")

    # ---- Run high-noise steps with transformer_1 weights ----
    print(f"\nRunning {switch_idx} high-noise denoising steps (transformer_1)...")
    t0 = time.time()
    for i in range(switch_idx):
        t = timesteps[i]
        latent_input = latents.to(DTYPE)

        if pipe.config.expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            ts = t.expand(latents.shape[0])

        noise_pred = run_transformer_step(nxd_model, latent_input, ts, prompt_embeds,
                                           rotary_emb_cos, rotary_emb_sin)
        noise_uncond = run_transformer_step(nxd_model, latent_input, ts, negative_prompt_embeds,
                                             rotary_emb_cos, rotary_emb_sin)
        # CFG in float32 to reduce accumulated precision errors
        noise_pred = noise_uncond.float() + guidance_scale * (noise_pred.float() - noise_uncond.float())

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        elapsed = time.time() - t0
        print(f"  Step {i+1}/{switch_idx} (t={t.item():.0f}) - {elapsed:.1f}s elapsed")

    print(f"High-noise phase done. {switch_idx} steps in {time.time() - t0:.1f}s")

    # ---- Swap weights to transformer_2 using replace_weights() ----
    remaining_steps = len(timesteps) - switch_idx
    if remaining_steps > 0:
        print(f"\nSwapping weights to transformer_2 via replace_weights()...")
        t_swap = time.time()
        nxd_model.replace_weights(t2_weights)
        swap_time = time.time() - t_swap
        print(f"  Weight swap completed in {swap_time:.1f}s")

        # Free transformer_1 weights from CPU
        del t1_weights
        gc.collect()

        # ---- Run low-noise steps with transformer_2 weights ----
        print(f"\nRunning {remaining_steps} low-noise denoising steps (transformer_2)...")
        t0 = time.time()
        for i in range(switch_idx, len(timesteps)):
            t = timesteps[i]
            latent_input = latents.to(DTYPE)

            if pipe.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                ts = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                ts = t.expand(latents.shape[0])

            noise_pred = run_transformer_step(nxd_model, latent_input, ts, prompt_embeds,
                                               rotary_emb_cos, rotary_emb_sin)
            noise_uncond = run_transformer_step(nxd_model, latent_input, ts, negative_prompt_embeds,
                                                 rotary_emb_cos, rotary_emb_sin)
            # CFG in float32 to reduce accumulated precision errors
            noise_pred = noise_uncond.float() + guidance_scale * (noise_pred.float() - noise_uncond.float())

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            step_num = i - switch_idx + 1
            if step_num % 10 == 0 or i == len(timesteps) - 1:
                elapsed = time.time() - t0
                print(f"  Step {step_num}/{remaining_steps} (t={t.item():.0f}) - {elapsed:.1f}s elapsed")

        print(f"Low-noise phase done. {remaining_steps} steps in {time.time() - t0:.1f}s")

    # Cleanup
    del nxd_model, t2_weights
    gc.collect()

    return latents


# ============================================================
# Phase 3: VAE Decoding (Neuron)
# ============================================================
def phase_vae_decode(pipe, compiled_models_dir, latents, num_frames=81):
    """
    Decode latents using compiled Neuron VAE (post_quant_conv + chunked decoder).

    Pipeline:
    1. Denormalize latents using VAE config (latents_mean/latents_std)
    2. Run Neuron post_quant_conv on full latent volume (float32, all 21 frames)
    3. Run Neuron decoder in chunks of decoder_frames (default 2) latent frames
       Each chunk produces decoder_frames × 4 video frames (4× temporal upsample)
    4. Concatenate and trim to expected frame count

    Decoder modes (auto-detected based on compiled model):
    - Rolling cache (decoder_rolling/): feat_cache carried between chunks as I/O.
      Flicker-free but ~1.8GB extra transfer per chunk.
    - NoCache (decoder_nocache/): feat_cache as zero buffers. Fast but flickering.
    """
    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (Neuron)")
    print("="*60)

    # Load VAE config for denormalization constants
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
        cache_dir=HUGGINGFACE_CACHE_DIR,
    )

    # Denormalize latents
    latents = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    print(f"Denormalized latents: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    del vae
    gc.collect()

    # ---- Step 1: Load and run Neuron post_quant_conv ----
    pqc_path = f"{compiled_models_dir}/post_quant_conv"
    pqc_config = load_model_config(pqc_path)
    pqc_world_size = pqc_config["world_size"]

    print(f"\nLoading post_quant_conv (world_size={pqc_world_size})...")
    t0 = time.time()
    pqc_nxd = NxDModel.load(
        os.path.join(pqc_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=pqc_world_size,
    )
    pqc_weights = load_duplicated_weights(pqc_path, pqc_world_size)
    pqc_nxd.set_weights(pqc_weights)
    pqc_nxd.to_neuron()
    print(f"  post_quant_conv loaded in {time.time() - t0:.1f}s")

    print(f"Running post_quant_conv on {latents.shape}...")
    t0 = time.time()
    z = pqc_nxd(latents)
    if isinstance(z, (tuple, list)):
        z = z[0]
    z = z.to(torch.float32)
    print(f"  post_quant_conv done in {time.time() - t0:.1f}s, output: {z.shape}")

    del pqc_nxd, pqc_weights
    gc.collect()

    # ---- Step 2: Load and run Neuron decoder ----
    # Try rolling cache first (flicker-free), fall back to nocache
    rolling_path = f"{compiled_models_dir}/decoder_rolling"
    nocache_path = f"{compiled_models_dir}/decoder_nocache"

    if os.path.exists(os.path.join(rolling_path, "nxd_model.pt")):
        decoder_path = rolling_path
        is_rolling = True
    else:
        decoder_path = nocache_path
        is_rolling = False

    decoder_config = load_model_config(decoder_path)
    decoder_world_size = decoder_config["world_size"]
    decoder_frames = decoder_config.get("decoder_frames", 2)
    mode_str = "rolling cache" if is_rolling else "nocache"

    print(f"\nLoading decoder [{mode_str}] (world_size={decoder_world_size}, compiled_frames={decoder_frames})...")
    t0 = time.time()
    decoder_nxd = NxDModel.load(
        os.path.join(decoder_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=decoder_world_size,
    )
    decoder_weights = load_duplicated_weights(decoder_path, decoder_world_size)
    decoder_nxd.set_weights(decoder_weights)
    decoder_nxd.to_neuron()
    print(f"  Decoder loaded in {time.time() - t0:.1f}s")

    z_bf16 = z.to(torch.bfloat16)
    num_latent_frames = z_bf16.shape[2]
    decoded_frames = []

    num_chunks = (num_latent_frames + decoder_frames - 1) // decoder_frames
    print(f"Decoding {num_latent_frames} latent frames in {num_chunks} chunks of {decoder_frames} [{mode_str}]...")
    decode_start = time.time()

    # Initialize rolling cache (zero tensors, same shapes as compiled)
    if is_rolling:
        from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes
        latent_h, latent_w = z_bf16.shape[3], z_bf16.shape[4]
        cache_shapes = get_feat_cache_shapes(1, latent_h, latent_w)
        caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]
        print(f"  Rolling cache initialized: {len(caches)} tensors")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * decoder_frames
        end = min(start + decoder_frames, num_latent_frames)
        chunk = z_bf16[:, :, start:end, :, :]

        # Pad last chunk if needed
        if chunk.shape[2] < decoder_frames:
            pad_frames = decoder_frames - chunk.shape[2]
            padding = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
            chunk = torch.cat([chunk, padding], dim=2)

        if is_rolling:
            # Rolling mode: pass cache as inputs, get updated cache as outputs
            results = decoder_nxd(chunk, *caches)
            if isinstance(results, (tuple, list)):
                output = results[0]
                caches = [r.to(torch.bfloat16) for r in results[1:1 + len(cache_shapes)]]
            else:
                output = results
        else:
            # NoCache mode: just pass x
            output = decoder_nxd(chunk)
            if isinstance(output, (list, tuple)):
                output = output[0]

        output = output.to(torch.float32)

        # Each chunk of decoder_frames latent frames -> decoder_frames * 4 video frames
        actual_latent = end - start
        video_frames_from_chunk = actual_latent * 4
        output = output[:, :, :video_frames_from_chunk]
        decoded_frames.append(output)

        elapsed = time.time() - decode_start
        print(f"  Chunk {chunk_idx+1}/{num_chunks}: latent [{start}:{end}] -> {output.shape[2]} video frames ({elapsed:.1f}s)")

    video = torch.cat(decoded_frames, dim=2)  # [B, C, total_video_frames, H, W]
    decode_time = time.time() - decode_start

    # Trim to expected number of video frames
    if video.shape[2] > num_frames:
        print(f"  Trimming {video.shape[2]} -> {num_frames} video frames")
        video = video[:, :, :num_frames]

    print(f"  Total decode time: {decode_time:.1f}s, output: {video.shape} [{mode_str}]")

    del decoder_nxd, decoder_weights, z, z_bf16, decoded_frames
    gc.collect()

    # Post-processing: [B, C, F, H, W] -> numpy [F, H, W, C], float [0,1]
    # NOTE: export_to_video expects float [0,1] ndarray and does *255 internally.
    # Do NOT pass uint8, otherwise it double-multiplies by 255!
    video = video[0]  # [C, F, H, W]
    video = video.permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
    video = ((video + 1.0) / 2.0).clip(0, 1)
    print(f"Output video: shape={video.shape}, dtype={video.dtype}")

    return video


# ============================================================
# Main
# ============================================================
DEFAULT_COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models_t2v_a14b"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"
SEED = 42


def main(args):
    total_start = time.time()
    set_seed(SEED)

    DTYPE = torch.bfloat16
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Load base pipeline (CPU)
    print("Loading base pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=DTYPE, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    compiled_models_dir = args.compiled_models_dir
    seqlen = args.max_sequence_length

    # Phase 1: Text Encoding
    t1_start = time.time()
    prompt_embeds, negative_prompt_embeds = phase_text_encoding(
        pipe, compiled_models_dir, seqlen, args.prompt, args.negative_prompt,
        use_neuron=args.neuron_text_encoder,
    )
    t1_time = time.time() - t1_start

    # Phase 2: Denoising with MoE swap
    print("\nStarting denoising...")
    generator = torch.Generator().manual_seed(SEED)
    t2_start = time.time()
    latents = phase_denoising(
        pipe, compiled_models_dir, prompt_embeds, negative_prompt_embeds, args, generator
    )
    t2_time = time.time() - t2_start
    print(f"\nTotal denoising time: {t2_time:.2f}s")
    print(f"Per step (including swap): {t2_time / args.num_inference_steps:.3f}s")

    # Save latents for debugging
    torch.save(latents, "debug_latents.pt")
    print(f"Saved debug_latents.pt: shape={latents.shape}, dtype={latents.dtype}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Phase 3: VAE Decode
    t3_start = time.time()
    video = phase_vae_decode(pipe, compiled_models_dir, latents, num_frames=args.num_frames)
    t3_time = time.time() - t3_start

    # Save video - video is numpy float [0,1] shape [F, H, W, C]
    output_path = args.output
    frames = [video[i] for i in range(video.shape[0])]
    print(f"Exporting {len(frames)} frames, shape={frames[0].shape}, dtype={frames[0].dtype}")
    export_to_video(frames, output_path, fps=24)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Video saved to: {output_path} ({len(frames)} frames)")
    print(f"{'='*60}")
    print(f"  Phase 1 - Text Encoding:  {t1_time:.1f}s")
    print(f"  Phase 2 - Denoising:      {t2_time:.1f}s")
    print(f"  Phase 3 - VAE Decode:     {t3_time:.1f}s")
    print(f"  Total (incl. loading):    {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 T2V-A14B Inference with MoE Transformers")
    parser.add_argument("--compiled_models_dir", type=str, default=DEFAULT_COMPILED_MODELS_DIR)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--output", type=str, default="output_t2v_a14b.mp4")
    parser.add_argument("--neuron_text_encoder", action="store_true",
                        help="Use Neuron-compiled text encoder (experimental, default: CPU)")
    args = parser.parse_args()

    main(args)
