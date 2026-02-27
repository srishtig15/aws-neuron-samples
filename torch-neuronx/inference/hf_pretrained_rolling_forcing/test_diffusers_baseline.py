"""
Diffusers WanPipeline baseline test.

Runs the standard Wan2.1-T2V pipeline on CPU to get a reference output.
Uses 3 latent frames (9 pixel frames) with reduced steps for speed.

This tells us:
1. Whether the base model generates good content with the same VAE
2. What expected latent statistics look like
3. Whether the VAE decoding is correct
"""
import os
import sys

# Disable Neuron/XLA to run on pure CPU
os.environ["NEURON_RT_NUM_CORES"] = "0"
os.environ["NEURON_RT_VISIBLE_CORES"] = ""
os.environ["PJRT_DEVICE"] = "CPU"

import time
import torch
import numpy as np

# Monkey-patch xm.mark_step to avoid Neuron initialization
try:
    import torch_xla.core.xla_model as xm
    _orig_mark_step = xm.mark_step
    xm.mark_step = lambda *args, **kwargs: None
    print("  Patched xm.mark_step for CPU execution")
except ImportError:
    pass

print("=" * 60)
print("DIFFUSERS BASELINE: Wan2.1-T2V-1.3B CPU Pipeline")
print("=" * 60)

cache_dir = "/opt/dlami/nvme/rolling_forcing_hf_cache"

# Load the standard Diffusers pipeline
print("\n[1] Loading WanPipeline...")
t0 = time.time()

from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.float32,
    cache_dir=cache_dir,
)
print(f"  Pipeline loaded in {time.time()-t0:.1f}s")

# Generate with few steps for speed
prompt = "A cat walking on the grass"
num_frames = 9   # Minimal frames for speed (produces 3 latent frames)
height = 480
width = 832
num_steps = 20   # Reduced from 50 for speed (CPU is slow)

print(f"\n[2] Generating video...")
print(f"  Prompt: {prompt}")
print(f"  Frames: {num_frames}, Steps: {num_steps}")
print(f"  Resolution: {height}x{width}")

t0 = time.time()
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=5.0,  # Standard CFG for base model
        generator=torch.Generator().manual_seed(42),
        output_type="pt",
    )

gen_time = time.time() - t0
print(f"  Generation time: {gen_time:.1f}s")

# Extract video
if hasattr(output, 'frames'):
    video = output.frames
elif isinstance(output, dict) and 'frames' in output:
    video = output['frames']
else:
    video = output[0] if isinstance(output, (tuple, list)) else output

# Handle different output formats
if isinstance(video, list):
    # List of PIL images - convert to tensor
    from PIL import Image
    frames_np = np.array([np.array(f) for f in video[0]])
    video_tensor = torch.from_numpy(frames_np).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0)  # [1, F, H, W, C]
    print(f"  Video (from PIL): {video_tensor.shape}")
elif isinstance(video, torch.Tensor):
    video_tensor = video
    print(f"  Video (tensor): {video_tensor.shape}")
elif isinstance(video, np.ndarray):
    video_tensor = torch.from_numpy(video).float()
    if video_tensor.max() > 1.0:
        video_tensor = video_tensor / 255.0
    print(f"  Video (numpy): {video_tensor.shape}")
else:
    print(f"  Unknown output type: {type(video)}")
    print(f"  Repr: {repr(video)[:200]}")
    sys.exit(1)

# Save video
output_path = os.path.expanduser("~/Downloads/output_diffusers_baseline.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

import imageio

# Ensure correct shape for saving: [F, H, W, C]
if video_tensor.ndim == 5:
    # [B, F, C, H, W] or [B, F, H, W, C]
    vid = video_tensor[0]
    if vid.shape[1] == 3:  # [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)
elif video_tensor.ndim == 4:
    vid = video_tensor
    if vid.shape[1] == 3:  # [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)
else:
    vid = video_tensor

frames_out = (vid.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
imageio.mimwrite(output_path, frames_out, fps=16)
print(f"\n[3] Saved {len(frames_out)} frames to {output_path}")

# Statistics
print(f"\n[4] Video statistics:")
print(f"  Shape: {frames_out.shape}")
print(f"  Pixel range: [{frames_out.min()}, {frames_out.max()}]")
print(f"  Mean: {frames_out.mean():.1f}, Std: {frames_out.std():.1f}")

# Check for grid artifacts
frame0 = frames_out[0].astype(np.float32)
H_px, W_px = frame0.shape[:2]
for ch_name, ch in [('R', 0), ('G', 1), ('B', 2)]:
    ch_data = frame0[:, :, ch]
    h_even_odd = np.abs(ch_data[0::2, :] - ch_data[1::2, :]).mean()
    h_adj = np.abs(ch_data[:-1, :] - ch_data[1:, :]).mean()
    w_even_odd = np.abs(ch_data[:, 0::2] - ch_data[:, 1::2]).mean()
    w_adj = np.abs(ch_data[:, :-1] - ch_data[:, 1:]).mean()
    h_ratio = h_even_odd / max(h_adj, 1e-8)
    w_ratio = w_even_odd / max(w_adj, 1e-8)
    print(f"  {ch_name}: h_grid_ratio={h_ratio:.4f}, w_grid_ratio={w_ratio:.4f}")

print("\nDone!")
