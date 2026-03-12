"""Quick test: load saved latents and run only the tiled VAE decode phase."""
import os
import sys
import json
import time

# Set Neuron env vars before any imports
compiled_dir = "/opt/dlami/nvme/compiled_models_t2v_a14b_720p"
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "8-15"
os.environ.setdefault("NEURON_RT_INSPECT_ENABLE", "0")
os.environ.setdefault("NEURON_RT_INSPECT_DEVICE_PROFILE", "0")
os.environ.setdefault("NEURON_RT_INSPECT_SYSTEM_PROFILE", "0")
os.environ.setdefault("NEURON_RT_PROFILING_MODE", "0")

import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch_neuronx
from neuronx_distributed import NxDModel
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes

def load_model_config(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)

def load_duplicated_weights(model_path, world_size):
    base = load_file(os.path.join(model_path, "weights", "tp0_sharded_checkpoint.safetensors"))
    return [{k: v.clone() for k, v in base.items()} for _ in range(world_size)]

# Load latents
print("Loading saved latents...")
latents = torch.load("debug_latents.pt", weights_only=True)
print(f"  Latents: {latents.shape}, dtype={latents.dtype}")

# Load VAE config for denormalization
model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
cache_dir = "/opt/dlami/nvme/wan2.2_t2v_a14b_hf_cache_dir"
print("Loading VAE...")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=cache_dir)

# Denormalize
vae_config = vae.config
latents = latents.to(torch.float32)
latents_mean = torch.tensor(vae_config.latents_mean).view(1, vae_config.z_dim, 1, 1, 1)
latents_std = 1.0 / torch.tensor(vae_config.latents_std).view(1, vae_config.z_dim, 1, 1, 1)
latents = latents / latents_std + latents_mean
print(f"Denormalized: {latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")

# PQC on CPU
print("Running post_quant_conv on CPU...")
t0 = time.time()
vae.post_quant_conv.to(torch.float32)
with torch.no_grad():
    z = vae.post_quant_conv(latents)
print(f"  PQC done in {time.time()-t0:.1f}s, output: {z.shape}")

z_bf16 = z.to(torch.bfloat16)
del z, latents
gc.collect()

B, C, T, H_lat, W_lat = z_bf16.shape
num_frames = 81

# Tile params
tile_h, tile_w = 60, 104
overlap_h = tile_h // 4  # 15
overlap_w = tile_w // 4  # 26
stride_h = tile_h - overlap_h  # 45
stride_w = tile_w - overlap_w  # 78

# Fixed tile position computation
h_starts = []
h = 0
while h < H_lat:
    h_starts.append(h)
    if h + tile_h >= H_lat:
        break
    h += stride_h
w_starts = []
w = 0
while w < W_lat:
    w_starts.append(w)
    if w + tile_w >= W_lat:
        break
    w += stride_w

n_tiles = len(h_starts) * len(w_starts)
print(f"\nTiling: latent {H_lat}x{W_lat} -> {len(h_starts)}x{len(w_starts)} = {n_tiles} tiles")
for hi, hs in enumerate(h_starts):
    for wi, ws in enumerate(w_starts):
        ah = min(tile_h, H_lat - hs)
        aw = min(tile_w, W_lat - ws)
        print(f"  Tile ({hi},{wi}): latent [{hs}:{hs+ah}, {ws}:{ws+aw}] (actual {ah}x{aw})")

# Load decoder
decoder_path = f"{compiled_dir}/decoder_rolling_480p"
decoder_config = load_model_config(decoder_path)
decoder_world_size = decoder_config["world_size"]
decoder_frames = decoder_config.get("decoder_frames", 2)

print(f"\nLoading 480P decoder (world_size={decoder_world_size}, frames={decoder_frames})...")
t0 = time.time()
decoder_nxd = NxDModel.load(os.path.join(decoder_path, "nxd_model.pt"), start_rank=0, local_ranks_size=decoder_world_size)
decoder_weights = load_duplicated_weights(decoder_path, decoder_world_size)
decoder_nxd.set_weights(decoder_weights)
decoder_nxd.to_neuron()
load_time = time.time() - t0
print(f"  Loaded in {load_time:.1f}s")

H_pix = H_lat * 8
W_pix = W_lat * 8
overlap_h_pix = overlap_h * 8
overlap_w_pix = overlap_w * 8

output_acc = torch.zeros(3, num_frames, H_pix, W_pix, dtype=torch.float32)
weight_acc = torch.zeros(H_pix, W_pix, dtype=torch.float32)

num_chunks = (T + decoder_frames - 1) // decoder_frames
decode_start = time.time()

for hi, h_start in enumerate(h_starts):
    for wi, w_start in enumerate(w_starts):
        actual_h = min(tile_h, H_lat - h_start)
        actual_w = min(tile_w, W_lat - w_start)

        z_tile = z_bf16[:, :, :, h_start:h_start+actual_h, w_start:w_start+actual_w]
        if actual_h < tile_h or actual_w < tile_w:
            z_tile = F.pad(z_tile, (0, tile_w - actual_w, 0, tile_h - actual_h))

        cache_shapes = get_feat_cache_shapes(1, tile_h, tile_w)
        caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]

        tile_frames = []
        tile_start = time.time()
        for chunk_idx in range(num_chunks):
            t_start = chunk_idx * decoder_frames
            t_end = min(t_start + decoder_frames, T)
            chunk = z_tile[:, :, t_start:t_end, :, :]
            if chunk.shape[2] < decoder_frames:
                pad_t = decoder_frames - chunk.shape[2]
                chunk = torch.cat([chunk, chunk[:, :, -1:, :, :].expand(-1, -1, pad_t, -1, -1)], dim=2)
            results = decoder_nxd(chunk, *caches)
            if isinstance(results, (tuple, list)):
                output = results[0]
                caches = [r.to(torch.bfloat16) for r in results[1:1+len(cache_shapes)]]
            else:
                output = results
            output = output.to(torch.float32)
            actual_t = t_end - t_start
            output = output[:, :, :actual_t * 4]
            tile_frames.append(output)

        tile_video = torch.cat(tile_frames, dim=2)
        if tile_video.shape[2] > num_frames:
            tile_video = tile_video[:, :, :num_frames]

        actual_h_pix = actual_h * 8
        actual_w_pix = actual_w * 8
        tile_video = tile_video[:, :, :, :actual_h_pix, :actual_w_pix]

        h_weight = torch.ones(actual_h_pix)
        w_weight = torch.ones(actual_w_pix)
        if hi > 0:
            ramp = min(overlap_h_pix, actual_h_pix)
            h_weight[:ramp] = torch.linspace(0, 1, ramp + 2)[1:-1]
        if hi < len(h_starts) - 1:
            ramp = min(overlap_h_pix, actual_h_pix)
            h_weight[-ramp:] = torch.linspace(1, 0, ramp + 2)[1:-1]
        if wi > 0:
            ramp = min(overlap_w_pix, actual_w_pix)
            w_weight[:ramp] = torch.linspace(0, 1, ramp + 2)[1:-1]
        if wi < len(w_starts) - 1:
            ramp = min(overlap_w_pix, actual_w_pix)
            w_weight[-ramp:] = torch.linspace(1, 0, ramp + 2)[1:-1]
        weight_2d = h_weight.unsqueeze(1) * w_weight.unsqueeze(0)

        h_pix = h_start * 8
        w_pix = w_start * 8
        output_acc[:, :, h_pix:h_pix+actual_h_pix, w_pix:w_pix+actual_w_pix] += (
            tile_video[0] * weight_2d.unsqueeze(0).unsqueeze(0)
        )
        weight_acc[h_pix:h_pix+actual_h_pix, w_pix:w_pix+actual_w_pix] += weight_2d

        tile_time = time.time() - tile_start
        elapsed = time.time() - decode_start
        print(f"  Tile ({hi},{wi}) done: {num_chunks} chunks, {tile_time:.1f}s ({elapsed:.1f}s total)")

        del tile_video, tile_frames, z_tile, caches
        gc.collect()

decode_time = time.time() - decode_start
video = output_acc / weight_acc.unsqueeze(0).unsqueeze(0).clamp(min=1e-6)
print(f"\nTotal tiled decode: {decode_time:.1f}s (load: {load_time:.1f}s)")

video = video.permute(1, 2, 3, 0).float().cpu().numpy()
video = ((video + 1.0) / 2.0).clip(0, 1)
print(f"Output: {video.shape}")

frames = [video[i] for i in range(video.shape[0])]
export_to_video(frames, "output_tiled_test.mp4", fps=16)
print(f"Saved output_tiled_test.mp4 ({len(frames)} frames)")
