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
def phase_text_encoding(pipe, compiled_models_dir, seqlen, prompt, negative_prompt):
    """Encode prompt using CPU text encoder (avoids NeuronCore NEFF residual)."""
    print("\n" + "="*60)
    print("PHASE 1: Text Encoding (CPU)")
    print("="*60)

    # Use CPU text encoder to avoid leaving residual NEFF on NeuronCores
    # that would conflict with transformer loading (HBM limit)
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
# Phase 3: VAE Decoding
# ============================================================
def phase_vae_decode(pipe, compiled_models_dir, latents):
    """
    Pure CPU VAE decode with fresh float32 VAE (avoids bfloat16 weight truncation).
    """
    print("\n" + "="*60)
    print("PHASE 3: VAE Decoding (Fresh float32 VAE)")
    print("="*60)

    # Load a FRESH VAE in float32 to avoid bfloat16 weight truncation
    # (pipeline.from_pretrained casts all components to bfloat16)
    print("Loading fresh float32 VAE...")
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

    # Decode
    print("Decoding latents...")
    decode_start = time.time()
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    decode_time = time.time() - decode_start
    print(f"  VAE decode completed in {decode_time:.1f}s")

    # Post-processing: [B, C, F, H, W] -> numpy [F, H, W, C], float [0,1]
    # NOTE: export_to_video expects float [0,1] ndarray and does *255 internally.
    # Do NOT pass uint8, otherwise it double-multiplies by 255!
    video = video[0]  # [C, F, H, W]
    video = video.permute(1, 2, 3, 0).float().cpu().numpy()  # [F, H, W, C]
    video = ((video + 1.0) / 2.0).clip(0, 1)
    print(f"Decoded video: shape={video.shape}, dtype={video.dtype}")

    del vae
    gc.collect()

    return video


# ============================================================
# Main
# ============================================================
DEFAULT_COMPILED_MODELS_DIR = "/opt/dlami/nvme/compiled_models_t2v_a14b"
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"
SEED = 42


def main(args):
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
    prompt_embeds, negative_prompt_embeds = phase_text_encoding(
        pipe, compiled_models_dir, seqlen, args.prompt, args.negative_prompt
    )

    # Phase 2: Denoising with MoE swap
    print("\nStarting denoising...")
    generator = torch.Generator().manual_seed(SEED)
    denoise_start = time.time()
    latents = phase_denoising(
        pipe, compiled_models_dir, prompt_embeds, negative_prompt_embeds, args, generator
    )
    denoise_time = time.time() - denoise_start
    print(f"\nTotal denoising time: {denoise_time:.2f}s")
    print(f"Per step (including swap): {denoise_time / args.num_inference_steps:.3f}s")

    # Save latents for debugging
    torch.save(latents, "debug_latents.pt")
    print(f"Saved debug_latents.pt: shape={latents.shape}, dtype={latents.dtype}, range=[{latents.min():.3f}, {latents.max():.3f}]")

    # Phase 3: VAE Decode
    video = phase_vae_decode(pipe, compiled_models_dir, latents)

    # Save video - video is numpy float [0,1] shape [F, H, W, C]
    output_path = args.output
    frames = [video[i] for i in range(video.shape[0])]
    print(f"Exporting {len(frames)} frames, shape={frames[0].shape}, dtype={frames[0].dtype}")
    export_to_video(frames, output_path, fps=24)
    print(f"\nVideo saved to: {output_path} ({len(frames)} frames)")


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
    args = parser.parse_args()

    main(args)
