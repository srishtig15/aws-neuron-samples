"""
GPU Benchmark for RollingForcing on H100.

Runs RollingForcing inference using the original CausalInferencePipeline
with timing for performance comparison against Neuron.

Usage:
    source /opt/pytorch/bin/activate
    cd /tmp/RollingForcing
    python /tmp/run_rolling_forcing_gpu_bench.py \
        --checkpoint_path /tmp/rolling_forcing_dmd.pt \
        --prompt "A cat walking gracefully across a sunlit garden path" \
        --num_warmup 1 --num_runs 3
"""
import os
import sys
import time
import argparse
import torch
import imageio
import numpy as np

sys.path.insert(0, "/tmp/RollingForcing")

from omegaconf import OmegaConf
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="A cat walking gracefully across a sunlit garden path")
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="/tmp/gpu_output.mp4")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Total output video frames")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    # Load config
    config = OmegaConf.load("/tmp/RollingForcing/configs/rolling_forcing_dmd.yaml")
    default_config = OmegaConf.load("/tmp/RollingForcing/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.num_frames = args.num_frames

    # Compute num_output_frames (latent frames)
    num_output_frames = (args.num_frames - 1) // 4 + 1  # 21 for 81 frames

    print("=" * 60)
    print("RollingForcing GPU Benchmark (H100)")
    print("=" * 60)
    print(f"Resolution: {config.height}x{config.width}")
    print(f"Frames: {args.num_frames} (latent: {num_output_frames})")
    print(f"Denoising steps: {config.denoising_step_list}")
    print(f"Warmup: {args.num_warmup}, Runs: {args.num_runs}")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    t0 = time.time()
    from pipeline import CausalInferencePipeline
    pipeline = CausalInferencePipeline(config, device=device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if 'generator_ema' in state_dict:
        sd = state_dict['generator_ema']
    elif 'generator' in state_dict:
        sd = state_dict['generator']
    else:
        sd = state_dict

    # Clean FSDP prefixes
    cleaned = OrderedDict()
    for k, v in sd.items():
        cleaned[k.replace("_fsdp_wrapped_module.", "")] = v
    pipeline.generator.load_state_dict(cleaned)

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    print(f"Pipeline loaded in {time.time()-t0:.1f}s")

    # GPU memory
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"GPU memory: {mem:.1f} GB")

    # Encode text
    print(f"\nPrompt: {args.prompt}")
    prompt_embeds = pipeline.text_encoder.encode([args.prompt])
    if isinstance(prompt_embeds, (list, tuple)):
        prompt_embeds = prompt_embeds[0]

    # Warmup + timed runs
    all_times = []
    for run in range(args.num_warmup + args.num_runs):
        is_warmup = run < args.num_warmup
        label = f"Warmup {run+1}" if is_warmup else f"Run {run - args.num_warmup + 1}"

        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t_start = time.time()

        # Run inference
        video = pipeline.inference_rolling_forcing(
            prompt_embeds=prompt_embeds,
            num_output_frames=num_output_frames,
            profile=False,
        )

        torch.cuda.synchronize()
        t_end = time.time()
        elapsed = t_end - t_start

        if not is_warmup:
            all_times.append(elapsed)

        print(f"  {label}: {elapsed:.2f}s")

    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    if all_times:
        avg = sum(all_times) / len(all_times)
        print(f"Average: {avg:.2f}s")
        print(f"Min: {min(all_times):.2f}s")
        print(f"Max: {max(all_times):.2f}s")
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_mem:.1f} GB")

    # Save last video
    if video is not None:
        print(f"\nSaving to {args.output_path}...")
        # video expected to be [B, F, C, H, W] in [0, 1] or [-1, 1]
        v = video[0].float().cpu()
        if v.min() < 0:
            v = (v + 1) / 2
        v = v.clamp(0, 1)
        frames = (v.numpy() * 255).astype(np.uint8)
        # [F, C, H, W] -> [F, H, W, C]
        frames = np.transpose(frames, (0, 2, 3, 1))
        imageio.mimwrite(args.output_path, frames, fps=16)
        print(f"Saved {len(frames)} frames")

    print("Done!")


if __name__ == "__main__":
    main()
