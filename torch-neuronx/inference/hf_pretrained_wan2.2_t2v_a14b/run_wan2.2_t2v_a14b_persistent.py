"""
Wan2.2 T2V-A14B Persistent Mode: ALL models stay loaded on separate NCs.

Eliminates all model load/swap overhead by keeping text encoder, both transformers,
and VAE decoder on separate NeuronCores simultaneously.

NC allocation (480P, transformer ws=8):
  NCs  0-7:   Text encoder (persistent)
  NCs  8-15:  Transformer_1 (persistent)
  NCs 16-23:  Transformer_2 (persistent)
  NCs 24-31:  VAE Decoder + PQC (persistent)
  Total: 32 NCs

NC allocation (720P, transformer ws=16):
  NCs  0-15:  Transformer_1 (persistent, must be 16-aligned for CCOM topology)
  NCs 16-31:  Transformer_2 (persistent, must be 16-aligned for CCOM topology)
  NCs 32-39:  Text encoder (persistent)
  NCs 40-47:  8x Tile Decoder ws=1 (persistent, PQC on CPU)
  Total: 48 NCs

Supports --num_runs for multi-inference benchmarking. First run is warmup,
subsequent runs are timed and averaged.
"""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import random
import shutil
import torch

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.pipelines.wan import WanPipeline
from diffusers.utils import export_to_video

SEED = 42
HUGGINGFACE_CACHE_DIR = "/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_config(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


# ============================================================
# Persistent Worker Manager
# ============================================================

class PersistentWorker:
    """Manages a long-running persistent worker subprocess."""

    def __init__(self, label, visible_cores):
        self.label = label
        self.visible_cores = visible_cores
        self.proc = None
        self.tmpdir = tempfile.mkdtemp(prefix=f"pw_{label}_")

    def _make_env(self, cwd, num_cores):
        neuron_vars = {
            "NEURON_RT_NUM_CORES": str(num_cores),
            "NEURON_RT_VIRTUAL_CORE_SIZE": "2",
            "NEURON_RT_VISIBLE_CORES": self.visible_cores,
            "NEURON_RT_INSPECT_ENABLE": "0",
            "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
            "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
            "NEURON_RT_PROFILING_MODE": "0",
        }
        env_config_path = os.path.join(self.tmpdir, "env.json")
        with open(env_config_path, "w") as f:
            json.dump(neuron_vars, f)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Clear all existing Neuron env vars first
        for k in list(env.keys()):
            if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
                del env[k]
        # Set Neuron env vars in process environment (needed for runtime init)
        env.update(neuron_vars)
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = cwd
        elif cwd not in env["PYTHONPATH"]:
            env["PYTHONPATH"] = cwd + ":" + env["PYTHONPATH"]
        return env, env_config_path

    def launch(self, cwd, cmd_args, num_cores):
        env, env_config_path = self._make_env(cwd, num_cores)
        full_cmd = [sys.executable] + cmd_args + ["--env_config", env_config_path]
        self.proc = subprocess.Popen(
            full_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, cwd=cwd, env=env, bufsize=1, text=True,
        )
        print(f"  [{self.label}] Launched PID {self.proc.pid} on NCs {self.visible_cores}")

    def wait_ready(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                print(f"  [{self.label}] {line}")
            if line == "READY":
                return
        raise RuntimeError(f"[{self.label}] exited before READY")

    def send_command(self, cmd, input_data=None):
        """Send command, optionally save input data, wait for DONE."""
        parts = cmd.split()
        if input_data is not None:
            input_path = parts[1]
            torch.save(input_data, input_path)

        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

        for line in self.proc.stdout:
            line = line.strip()
            if line:
                print(f"  [{self.label}] {line}")
            if line == "DONE":
                output_path = parts[2]
                return torch.load(output_path, weights_only=False)
            if line == "ERROR":
                raise RuntimeError(f"[{self.label}] error")
        raise RuntimeError(f"[{self.label}] exited during command")

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============================================================
# Single Inference Run
# ============================================================

def _run_text_encoding(pipe, args, te_worker, run_idx):
    """Phase 1: Text encoding via persistent worker."""
    seqlen = args.max_sequence_length
    text_inputs = pipe.tokenizer(
        args.prompt, padding="max_length", max_length=seqlen,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )
    neg_text_inputs = pipe.tokenizer(
        args.negative_prompt, padding="max_length", max_length=seqlen,
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )

    te_input = os.path.join(te_worker.tmpdir, f"enc_in_{run_idx}.pt")
    te_output = os.path.join(te_worker.tmpdir, f"enc_out_{run_idx}.pt")
    te_result = te_worker.send_command(
        f"encode {te_input} {te_output}",
        {
            "prompt_input_ids": text_inputs.input_ids,
            "prompt_attention_mask": text_inputs.attention_mask,
            "neg_input_ids": neg_text_inputs.input_ids,
            "neg_attention_mask": neg_text_inputs.attention_mask,
        }
    )
    return te_result["prompt_embeds"], te_result["negative_prompt_embeds"]


def _run_denoising(pipe, args, t1_worker, t2_worker, prompt_embeds,
                    negative_prompt_embeds, run_idx):
    """Phase 2: Denoising via persistent transformer workers."""
    DTYPE = torch.bfloat16
    device = torch.device("cpu")

    prompt_embeds = prompt_embeds.to(DTYPE)
    negative_prompt_embeds = negative_prompt_embeds.to(DTYPE)

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    generator = torch.Generator().manual_seed(SEED)
    in_channels = pipe.transformer.config.in_channels if pipe.transformer is not None else 16
    latents = pipe.prepare_latents(
        1, in_channels, args.height, args.width, args.num_frames,
        torch.float32, device, generator, None,
    )
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    boundary_timestep = 0.875 * 1000
    switch_idx = None
    for i, t in enumerate(timesteps):
        if t < boundary_timestep:
            switch_idx = i
            break
    if switch_idx is None:
        switch_idx = len(timesteps)

    scheduler_config = dict(pipe.scheduler.config)
    scheduler_state = {
        "scheduler_order_list": getattr(pipe.scheduler, 'order_list', None),
        "scheduler_model_outputs": getattr(pipe.scheduler, 'model_outputs', None),
        "scheduler_timestep_list": getattr(pipe.scheduler, 'timestep_list', None),
        "scheduler_lower_order_nums": getattr(pipe.scheduler, 'lower_order_nums', None),
        "scheduler_sample": getattr(pipe.scheduler, 'sample', None),
    }

    base_denoise_data = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "timesteps": timesteps,
        "expand_timesteps": pipe.config.expand_timesteps,
        "mask": mask,
        "scheduler_config": scheduler_config,
    }

    # T1: high-noise steps
    t1_in = os.path.join(t1_worker.tmpdir, f"in_{run_idx}.pt")
    t1_out = os.path.join(t1_worker.tmpdir, f"out_{run_idx}.pt")
    r1 = t1_worker.send_command(
        f"denoise {t1_in} {t1_out}",
        {**base_denoise_data, **scheduler_state,
         "latents": latents, "step_start": 0, "step_end": switch_idx,
         "guidance_scale": args.guidance_scale}
    )

    # T2: low-noise steps
    remaining = len(timesteps) - switch_idx
    if remaining > 0:
        t2_state = {
            "scheduler_model_outputs": r1.get("scheduler_model_outputs"),
            "scheduler_timestep_list": r1.get("scheduler_timestep_list"),
            "scheduler_lower_order_nums": r1.get("scheduler_lower_order_nums"),
            "scheduler_sample": r1.get("scheduler_sample"),
        }
        t2_in = os.path.join(t2_worker.tmpdir, f"in_{run_idx}.pt")
        t2_out = os.path.join(t2_worker.tmpdir, f"out_{run_idx}.pt")
        r2 = t2_worker.send_command(
            f"denoise {t2_in} {t2_out}",
            {**base_denoise_data, **t2_state,
             "latents": r1["latents"], "step_start": switch_idx, "step_end": len(timesteps),
             "guidance_scale": args.guidance_scale_2}
        )
        latents = r2["latents"]
    else:
        latents = r1["latents"]

    phase1_time = r1["phase_time"]
    phase2_time = r2["phase_time"] if remaining > 0 else 0
    return latents, phase1_time, phase2_time


def _denormalize_latents(pipe, latents):
    """Denormalize latents using VAE config."""
    vae_config = pipe.vae.config
    latents = latents.to(torch.float32)
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1)
    return latents / latents_std + latents_mean


def _compute_tile_positions(full_h, full_w, tile_h, tile_w):
    """Compute tile positions with overlap for full coverage."""
    if full_h <= tile_h:
        h_positions = [0]
    else:
        h_positions = [0]
        while h_positions[-1] + tile_h < full_h:
            h_positions.append(min(h_positions[-1] + (full_h - tile_h),
                                    full_h - tile_h))
            if h_positions[-1] + tile_h >= full_h:
                break
        # Ensure last tile reaches the edge
        if h_positions[-1] + tile_h < full_h:
            h_positions.append(full_h - tile_h)

    if full_w <= tile_w:
        w_positions = [0]
    else:
        w_positions = [0]
        while w_positions[-1] + tile_w < full_w:
            w_positions.append(min(w_positions[-1] + (full_w - tile_w),
                                    full_w - tile_w))
            if w_positions[-1] + tile_w >= full_w:
                break
        if w_positions[-1] + tile_w < full_w:
            w_positions.append(full_w - tile_w)

    # Spread evenly
    if len(h_positions) >= 2:
        n = len(h_positions)
        stride = (full_h - tile_h) / (n - 1)
        h_positions = [round(i * stride) for i in range(n)]
    if len(w_positions) >= 2:
        n = len(w_positions)
        stride = (full_w - tile_w) / (n - 1)
        w_positions = [round(i * stride) for i in range(n)]

    positions = [(hi, wi) for hi in h_positions for wi in w_positions]
    return positions, h_positions, w_positions


def _blend_tiles(tile_results, tile_positions, tile_h_px, tile_w_px,
                  full_h_px, full_w_px, num_frames):
    """Blend decoded tiles using linear ramp weights in overlap regions."""
    # Determine overlap from tile positions
    h_positions_px = sorted(set(hi * 8 for hi, _ in tile_positions))
    w_positions_px = sorted(set(wi * 8 for _, wi in tile_positions))

    # Build per-tile weight maps
    output = torch.zeros(num_frames, full_h_px, full_w_px, 3)
    weight_sum = torch.zeros(1, full_h_px, full_w_px, 1)

    for idx, (hi_lat, wi_lat) in enumerate(tile_positions):
        hi_px = hi_lat * 8
        wi_px = wi_lat * 8
        tile_video = tile_results[idx]  # [T, tile_h_px, tile_w_px, 3]

        # Trim to actual output size (edge tiles may have been padded)
        actual_h = min(tile_h_px, full_h_px - hi_px)
        actual_w = min(tile_w_px, full_w_px - wi_px)
        tile_video = tile_video[:num_frames, :actual_h, :actual_w, :]

        # Create weight map with linear ramps at overlapping edges
        w_h = torch.ones(actual_h)
        w_w = torch.ones(actual_w)

        # Ramp at top edge (if not first row)
        if hi_px > 0:
            overlap_top = (h_positions_px[0] + tile_h_px - hi_px) if len(h_positions_px) > 1 else 0
            if overlap_top <= 0:
                for prev_h in h_positions_px:
                    if prev_h < hi_px and prev_h + tile_h_px > hi_px:
                        overlap_top = prev_h + tile_h_px - hi_px
                        break
            if overlap_top > 0:
                ramp = torch.linspace(0, 1, overlap_top)
                w_h[:overlap_top] = ramp

        # Ramp at bottom edge (if not last row)
        if hi_px + tile_h_px < full_h_px:
            overlap_bottom = 0
            for next_h in h_positions_px:
                if next_h > hi_px and next_h < hi_px + tile_h_px:
                    overlap_bottom = hi_px + tile_h_px - next_h
                    break
            if overlap_bottom > 0:
                ramp = torch.linspace(1, 0, overlap_bottom)
                w_h[-overlap_bottom:] = ramp

        # Ramp at left edge
        if wi_px > 0:
            overlap_left = 0
            for prev_w in w_positions_px:
                if prev_w < wi_px and prev_w + tile_w_px > wi_px:
                    overlap_left = prev_w + tile_w_px - wi_px
                    break
            if overlap_left > 0:
                ramp = torch.linspace(0, 1, overlap_left)
                w_w[:overlap_left] = ramp

        # Ramp at right edge
        if wi_px + tile_w_px < full_w_px:
            overlap_right = 0
            for next_w in w_positions_px:
                if next_w > wi_px and next_w < wi_px + tile_w_px:
                    overlap_right = wi_px + tile_w_px - next_w
                    break
            if overlap_right > 0:
                ramp = torch.linspace(1, 0, overlap_right)
                w_w[-overlap_right:] = ramp

        # 2D weight = outer product
        weight_2d = w_h.unsqueeze(1) * w_w.unsqueeze(0)  # [H, W]

        output[:, hi_px:hi_px+actual_h, wi_px:wi_px+actual_w, :] += \
            tile_video * weight_2d.unsqueeze(0).unsqueeze(-1)
        weight_sum[0, hi_px:hi_px+actual_h, wi_px:wi_px+actual_w, :] += \
            weight_2d.unsqueeze(-1)

    # Normalize
    weight_sum = weight_sum.clamp(min=1e-6)
    output = output / weight_sum
    return output


def run_single_inference(pipe, compiled_models_dir, args,
                          te_worker, t1_worker, t2_worker, dec_worker,
                          run_idx=0):
    """Run one complete inference (480P mode: single decoder). Returns timing dict."""
    set_seed(SEED)
    timings = {}

    # Phase 1: Text Encoding
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = _run_text_encoding(
        pipe, args, te_worker, run_idx)
    timings["text_encoding"] = time.time() - t0

    # Phase 2: Denoising
    t0 = time.time()
    latents, phase1_time, phase2_time = _run_denoising(
        pipe, args, t1_worker, t2_worker,
        prompt_embeds, negative_prompt_embeds, run_idx)
    timings["denoising"] = time.time() - t0
    timings["phase1"] = phase1_time
    timings["phase2"] = phase2_time

    # Phase 3: VAE Decode (single decoder)
    t0 = time.time()
    latents = _denormalize_latents(pipe, latents)

    dec_in = os.path.join(dec_worker.tmpdir, f"in_{run_idx}.pt")
    dec_out = os.path.join(dec_worker.tmpdir, f"out_{run_idx}.pt")
    dec_result = dec_worker.send_command(
        f"decode {dec_in} {dec_out}",
        {"latents_f32": latents, "num_frames": args.num_frames}
    )
    video = dec_result["video"].numpy()
    timings["vae_decode"] = time.time() - t0
    timings["decode_pure"] = dec_result["decode_time"]

    timings["total"] = timings["text_encoding"] + timings["denoising"] + timings["vae_decode"]
    return video, timings


def run_single_inference_tiled(pipe, compiled_models_dir, args,
                                te_worker, t1_worker, t2_worker, dec_workers,
                                tile_config, run_idx=0):
    """Run one complete inference (720P mode: parallel tiled decode). Returns timing dict."""
    set_seed(SEED)
    timings = {}

    # Phase 1: Text Encoding
    t0 = time.time()
    prompt_embeds, negative_prompt_embeds = _run_text_encoding(
        pipe, args, te_worker, run_idx)
    timings["text_encoding"] = time.time() - t0

    # Phase 2: Denoising
    t0 = time.time()
    latents, phase1_time, phase2_time = _run_denoising(
        pipe, args, t1_worker, t2_worker,
        prompt_embeds, negative_prompt_embeds, run_idx)
    timings["denoising"] = time.time() - t0
    timings["phase1"] = phase1_time
    timings["phase2"] = phase2_time

    # Phase 3: VAE Decode (parallel tiled)
    t0 = time.time()
    latents = _denormalize_latents(pipe, latents)

    # PQC on CPU
    t_pqc = time.time()
    with torch.no_grad():
        z = pipe.vae.post_quant_conv(latents)
    z = z.to(torch.bfloat16)
    pqc_time = time.time() - t_pqc
    print(f"  PQC on CPU: {pqc_time:.2f}s, shape={z.shape}")

    # Compute tile positions in latent space
    full_h, full_w = z.shape[3], z.shape[4]
    tile_h_lat = tile_config["height"] // 8
    tile_w_lat = tile_config["width"] // 8
    tile_positions, _, _ = _compute_tile_positions(
        full_h, full_w, tile_h_lat, tile_w_lat)

    num_tiles = len(tile_positions)
    print(f"  Tiling: {full_h}x{full_w} -> {num_tiles} tiles of {tile_h_lat}x{tile_w_lat}")

    # Extract tiles
    tiles = []
    for hi, wi in tile_positions:
        tile = z[:, :, :, hi:hi+tile_h_lat, wi:wi+tile_w_lat]
        # Pad if tile is smaller than expected (edge tiles)
        if tile.shape[3] < tile_h_lat or tile.shape[4] < tile_w_lat:
            padded = torch.zeros(1, z.shape[1], z.shape[2], tile_h_lat, tile_w_lat,
                                  dtype=tile.dtype)
            padded[:, :, :, :tile.shape[3], :tile.shape[4]] = tile
            tile = padded
        tiles.append(tile)

    # Send tiles to workers in parallel
    def decode_one_tile(worker, tile_data, idx):
        in_path = os.path.join(worker.tmpdir, f"tile_in_{run_idx}.pt")
        out_path = os.path.join(worker.tmpdir, f"tile_out_{run_idx}.pt")
        return worker.send_command(
            f"decode_tile {in_path} {out_path}",
            {"z_tile": tile_data}
        )

    tile_results_raw = [None] * num_tiles
    t_decode = time.time()
    with ThreadPoolExecutor(max_workers=num_tiles) as executor:
        futures = {}
        for idx in range(num_tiles):
            worker = dec_workers[idx % len(dec_workers)]
            f = executor.submit(decode_one_tile, worker, tiles[idx], idx)
            futures[f] = idx
        for f in futures:
            idx = futures[f]
            tile_results_raw[idx] = f.result()
    decode_time = time.time() - t_decode
    print(f"  Parallel tile decode: {decode_time:.1f}s ({num_tiles} tiles)")

    # Post-process tiles: [1, C, T_px, tile_H_px, tile_W_px] -> [T, H, W, C]
    tile_h_px = tile_h_lat * 8
    tile_w_px = tile_w_lat * 8
    full_h_px = args.height
    full_w_px = args.width
    num_frames = args.num_frames

    tile_videos = []
    for idx in range(num_tiles):
        tv = tile_results_raw[idx]["tile_video"]  # [1, C, T_px, H_px, W_px]
        tv = tv[0].permute(1, 2, 3, 0).float()  # [T, H, W, C]
        tv = (tv + 1.0) / 2.0
        tile_videos.append(tv)

    # Blend tiles
    video = _blend_tiles(tile_videos, tile_positions, tile_h_px, tile_w_px,
                          full_h_px, full_w_px, num_frames)
    video = video.clamp(0, 1).numpy()

    timings["vae_decode"] = time.time() - t0
    timings["decode_pure"] = decode_time
    timings["pqc_time"] = pqc_time

    timings["total"] = timings["text_encoding"] + timings["denoising"] + timings["vae_decode"]
    return video, timings


# ============================================================
# Main
# ============================================================

def main(args):
    total_start = time.time()

    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    print("Loading base pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=HUGGINGFACE_CACHE_DIR
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=HUGGINGFACE_CACHE_DIR
    )

    compiled_models_dir = args.compiled_models_dir
    transformer_1_path = os.path.join(compiled_models_dir, "transformer")
    transformer_2_path = os.path.join(compiled_models_dir, "transformer_2")
    te_path = os.path.join(compiled_models_dir, "text_encoder")

    t_config = load_model_config(transformer_1_path)
    t_ws = t_config["world_size"]
    te_config = load_model_config(te_path)
    te_ws = te_config["world_size"]

    # Detect tile decode mode
    tile_mode = False
    tile_config = None
    tile_decoder_path = args.tile_decoder_dir
    if tile_decoder_path and os.path.exists(os.path.join(tile_decoder_path, "nxd_model.pt")):
        tile_mode = True
    else:
        # Check compiled_models_dir for decoder_tile_ws1
        tile_decoder_path = os.path.join(compiled_models_dir, "decoder_tile_ws1")
        if os.path.exists(os.path.join(tile_decoder_path, "nxd_model.pt")):
            # Only use tile mode if no full-res decoder exists
            has_full_decoder = os.path.exists(
                os.path.join(compiled_models_dir, "decoder_rolling", "nxd_model.pt"))
            tile_mode = not has_full_decoder
    if tile_mode:
        tile_config = load_model_config(tile_decoder_path)

    # NC layout
    # For 720P (ws=16), transformers MUST use 16-NC-aligned blocks (0-15, 16-31, etc.)
    # due to CCOM topology constraints on trn2.
    if tile_mode and t_ws > 8:
        # 720P layout: transformers on aligned blocks, TE and decoders after
        t1_start = 0
        t1_cores = f"{t1_start}-{t1_start + t_ws - 1}"
        t2_start = t_ws  # 16
        t2_cores = f"{t2_start}-{t2_start + t_ws - 1}"
        te_start = t2_start + t_ws  # 32
        te_cores = f"{te_start}-{te_start + te_ws - 1}"
        dec_start = te_start + te_ws  # 40
        num_tile_workers = 8
        dec_cores_str = f"{dec_start}-{dec_start + num_tile_workers - 1}"
        total_ncs = dec_start + num_tile_workers
    else:
        # 480P layout: TE first, then transformers, then decoder
        te_cores = "0-7"
        t1_start = 8
        t1_cores = f"{t1_start}-{t1_start + t_ws - 1}"
        t2_start = t1_start + t_ws
        t2_cores = f"{t2_start}-{t2_start + t_ws - 1}"
        dec_start = t2_start + t_ws
        if tile_mode:
            num_tile_workers = 8
            dec_cores_str = f"{dec_start}-{dec_start + num_tile_workers - 1}"
            total_ncs = dec_start + num_tile_workers
        else:
            dec_cores_str = f"{dec_start}-{dec_start + 7}"
            total_ncs = dec_start + 8

    print(f"\n{'='*60}")
    print(f"Persistent Mode — ALL models co-resident")
    print(f"{'='*60}")
    print(f"  Text Encoder:   NCs {te_cores} (ws={te_ws})")
    print(f"  Transformer_1:  NCs {t1_cores} (ws={t_ws})")
    print(f"  Transformer_2:  NCs {t2_cores} (ws={t_ws})")
    if tile_mode:
        print(f"  Tile Decoders:  NCs {dec_cores_str} ({num_tile_workers}x ws=1, PQC on CPU)")
    else:
        print(f"  VAE Decoder:    NCs {dec_cores_str} (ws=8)")
    print(f"  Total: {total_ncs} / 64 NCs")
    print(f"  Runs: 1 warmup + {args.num_runs} timed")
    print(f"{'='*60}")

    cwd = os.path.dirname(os.path.abspath(__file__))

    # Launch persistent workers
    # For 720P (tile_mode with ws=16 transformers), launch sequentially to avoid
    # CCOM topology conflicts during NCCL ring setup. For 480P, parallel is fine.
    print("\nLaunching persistent workers...")
    t_launch = time.time()

    te_worker = PersistentWorker("TE", te_cores)
    t1_worker = PersistentWorker("T1", t1_cores)
    t2_worker = PersistentWorker("T2", t2_cores)

    if tile_mode:
        # Sequential launch for 720P: transformers first with warmup to establish NCCL
        # rings before other Neuron processes load (avoids CCOM topology conflicts)
        print("  720P mode: sequential launch with NCCL warmup (T1 -> T2 -> TE -> decoders)")
        t1_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                               "--transformer_path", transformer_1_path, "--warmup"], t_ws)
        t1_worker.wait_ready()

        t2_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                               "--transformer_path", transformer_2_path, "--warmup"], t_ws)
        t2_worker.wait_ready()

        te_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_text_encoder_worker",
                               "--te_path", te_path], te_ws)
        te_worker.wait_ready()

        dec_workers = []
        for i in range(num_tile_workers):
            nc = dec_start + i
            w = PersistentWorker(f"DEC{i}", f"{nc}-{nc}")
            w.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_tile_decoder_worker",
                           "--decoder_path", tile_decoder_path], 1)
            dec_workers.append(w)
        for w in dec_workers:
            w.wait_ready()
        dec_worker = None
    else:
        # Parallel launch for 480P (ws=8 transformers, no CCOM conflicts)
        te_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_text_encoder_worker",
                               "--te_path", te_path], te_ws)
        t1_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                               "--transformer_path", transformer_1_path], t_ws)
        t2_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                               "--transformer_path", transformer_2_path], t_ws)
        dec_worker = PersistentWorker("DEC", dec_cores_str)
        dec_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_decoder_worker",
                                "--compiled_models_dir", compiled_models_dir], 8)
        dec_workers = None

        te_worker.wait_ready()
        t1_worker.wait_ready()
        t2_worker.wait_ready()
        dec_worker.wait_ready()

    load_time = time.time() - t_launch
    print(f"\nAll models loaded in {load_time:.1f}s (parallel)")

    # ---- Inference function based on mode ----
    def do_inference(run_idx):
        if tile_mode:
            return run_single_inference_tiled(
                pipe, compiled_models_dir, args,
                te_worker, t1_worker, t2_worker, dec_workers,
                tile_config, run_idx=run_idx)
        else:
            return run_single_inference(
                pipe, compiled_models_dir, args,
                te_worker, t1_worker, t2_worker, dec_worker,
                run_idx=run_idx)

    # ---- Warmup run ----
    print(f"\n{'='*60}")
    print("WARMUP RUN")
    print(f"{'='*60}")
    video, warmup_timings = do_inference(run_idx=0)
    print(f"\nWarmup: {warmup_timings['total']:.1f}s "
          f"(te={warmup_timings['text_encoding']:.1f}s, "
          f"denoise={warmup_timings['denoising']:.1f}s, "
          f"decode={warmup_timings['vae_decode']:.1f}s)")

    # Save warmup video
    frames = [video[i] for i in range(video.shape[0])]
    export_to_video(frames, args.output, fps=16)
    print(f"Warmup video saved to: {args.output}")

    # ---- Timed runs ----
    all_timings = []
    for run in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{args.num_runs}")
        print(f"{'='*60}")
        video, timings = do_inference(run_idx=run + 1)
        all_timings.append(timings)
        print(f"\nRun {run+1}: {timings['total']:.1f}s "
              f"(te={timings['text_encoding']:.1f}s, "
              f"denoise={timings['denoising']:.1f}s, "
              f"decode={timings['vae_decode']:.1f}s)")

        # Save last run's video
        frames = [video[i] for i in range(video.shape[0])]
        output_name = args.output.replace(".mp4", f"_run{run+1}.mp4")
        export_to_video(frames, output_name, fps=16)

    # Shutdown workers
    te_worker.shutdown()
    t1_worker.shutdown()
    t2_worker.shutdown()
    if tile_mode:
        for w in dec_workers:
            w.shutdown()
    else:
        dec_worker.shutdown()

    # ---- Summary ----
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS (excluding model loading)")
    print(f"{'='*60}")
    print(f"Model load time (one-time): {load_time:.1f}s")
    print(f"Warmup: {warmup_timings['total']:.1f}s")
    print()

    if all_timings:
        for key in ["text_encoding", "denoising", "vae_decode", "total"]:
            vals = [t[key] for t in all_timings]
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            print(f"  {key:20s}: avg={avg:.1f}s  min={mn:.1f}s  max={mx:.1f}s")
        print()
        for i, t in enumerate(all_timings):
            print(f"  Run {i+1}: total={t['total']:.1f}s  "
                  f"(te={t['text_encoding']:.1f}s  "
                  f"phase1={t['phase1']:.1f}s  phase2={t['phase2']:.1f}s  "
                  f"decode={t['vae_decode']:.1f}s)")

    print(f"\n  Wall clock total: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 T2V-A14B Persistent Mode (all models co-resident)")
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models_t2v_a14b")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale_2", type=float, default=3.0)
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--output", type=str, default="output_persistent.mp4")
    parser.add_argument("--tile_decoder_dir", type=str, default=None,
                        help="Path to decoder_tile_ws1 dir (for 720P parallel tiled decode)")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of timed runs (after warmup)")
    args = parser.parse_args()

    main(args)
