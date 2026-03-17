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
  NCs  0-7:   Text encoder (persistent)
  NCs  8-23:  Transformer_1 (persistent)
  NCs 24-39:  Transformer_2 (persistent)
  NCs 40-47:  VAE Decoder tiles 8x ws=1 (subprocess, cache reset needed)
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
        env_config_path = os.path.join(self.tmpdir, "env.json")
        with open(env_config_path, "w") as f:
            json.dump({
                "NEURON_RT_NUM_CORES": str(num_cores),
                "NEURON_RT_VIRTUAL_CORE_SIZE": "2",
                "NEURON_RT_VISIBLE_CORES": self.visible_cores,
                "NEURON_RT_INSPECT_ENABLE": "0",
                "NEURON_RT_INSPECT_DEVICE_PROFILE": "0",
                "NEURON_RT_INSPECT_SYSTEM_PROFILE": "0",
                "NEURON_RT_PROFILING_MODE": "0",
            }, f)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for k in list(env.keys()):
            if k.startswith("NEURON_RT_") or k == "NEURON_LOGICAL_NC_CONFIG":
                del env[k]
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

def run_single_inference(pipe, compiled_models_dir, args,
                          te_worker, t1_worker, t2_worker, dec_worker,
                          run_idx=0):
    """Run one complete inference using persistent workers. Returns timing dict."""
    set_seed(SEED)

    seqlen = args.max_sequence_length
    timings = {}

    # ---- Phase 1: Text Encoding ----
    t0 = time.time()

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
    prompt_embeds = te_result["prompt_embeds"]
    negative_prompt_embeds = te_result["negative_prompt_embeds"]
    timings["text_encoding"] = time.time() - t0

    # ---- Phase 2: Denoising ----
    t0 = time.time()
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

    # Phase 1: T1 high-noise steps
    t1_in = os.path.join(t1_worker.tmpdir, f"in_{run_idx}.pt")
    t1_out = os.path.join(t1_worker.tmpdir, f"out_{run_idx}.pt")
    r1 = t1_worker.send_command(
        f"denoise {t1_in} {t1_out}",
        {**base_denoise_data, **scheduler_state,
         "latents": latents, "step_start": 0, "step_end": switch_idx,
         "guidance_scale": args.guidance_scale}
    )

    # Phase 2: T2 low-noise steps
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

    timings["denoising"] = time.time() - t0
    timings["phase1"] = r1["phase_time"]
    timings["phase2"] = r2["phase_time"] if remaining > 0 else 0

    # ---- Phase 3: VAE Decode ----
    t0 = time.time()

    # Denormalize latents
    vae_config = pipe.vae.config
    latents = latents.to(torch.float32)
    latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean

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

    # NC layout
    te_cores = "0-7"
    t1_start = 8
    t1_cores = f"{t1_start}-{t1_start + t_ws - 1}"
    t2_start = t1_start + t_ws
    t2_cores = f"{t2_start}-{t2_start + t_ws - 1}"
    dec_start = t2_start + t_ws
    dec_cores = f"{dec_start}-{dec_start + 7}"

    total_ncs = dec_start + 8
    print(f"\n{'='*60}")
    print(f"Persistent Mode — ALL models co-resident")
    print(f"{'='*60}")
    print(f"  Text Encoder:   NCs {te_cores} (ws={te_ws})")
    print(f"  Transformer_1:  NCs {t1_cores} (ws={t_ws})")
    print(f"  Transformer_2:  NCs {t2_cores} (ws={t_ws})")
    print(f"  VAE Decoder:    NCs {dec_cores} (ws=8)")
    print(f"  Total: {total_ncs} / 64 NCs")
    print(f"  Runs: 1 warmup + {args.num_runs} timed")
    print(f"{'='*60}")

    cwd = os.path.dirname(os.path.abspath(__file__))

    # Launch all 4 persistent workers in parallel
    print("\nLaunching all persistent workers...")
    t_launch = time.time()

    te_worker = PersistentWorker("TE", te_cores)
    t1_worker = PersistentWorker("T1", t1_cores)
    t2_worker = PersistentWorker("T2", t2_cores)
    dec_worker = PersistentWorker("DEC", dec_cores)

    te_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_text_encoder_worker",
                           "--te_path", te_path], te_ws)
    t1_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                           "--transformer_path", transformer_1_path], t_ws)
    t2_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_denoise_worker",
                           "--transformer_path", transformer_2_path], t_ws)
    dec_worker.launch(cwd, ["-m", "neuron_wan2_2_t2v_a14b.persistent_decoder_worker",
                            "--compiled_models_dir", compiled_models_dir], 8)

    # Wait for all to be ready
    te_worker.wait_ready()
    t1_worker.wait_ready()
    t2_worker.wait_ready()
    dec_worker.wait_ready()

    load_time = time.time() - t_launch
    print(f"\nAll models loaded in {load_time:.1f}s (parallel)")

    # ---- Warmup run ----
    print(f"\n{'='*60}")
    print("WARMUP RUN")
    print(f"{'='*60}")
    video, warmup_timings = run_single_inference(
        pipe, compiled_models_dir, args,
        te_worker, t1_worker, t2_worker, dec_worker, run_idx=0,
    )
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
        video, timings = run_single_inference(
            pipe, compiled_models_dir, args,
            te_worker, t1_worker, t2_worker, dec_worker, run_idx=run + 1,
        )
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
    parser.add_argument("--num_runs", type=int, default=3, help="Number of timed runs (after warmup)")
    args = parser.parse_args()

    main(args)
