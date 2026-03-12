"""
Subprocess worker for running VAE decode on Neuron.

Supports both stateful (on-device cache) and legacy (I/O cache) rolling decoders.
Auto-detects mode from compiled model's config.json.

Usage (called programmatically):
    python -m neuron_wan2_2_t2v_a14b.decoder_worker <input.pt> <output.pt> <env.json>
"""
import json
import os
import sys
import time

input_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3] if len(sys.argv) > 3 else None
if config_path:
    with open(config_path) as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        os.environ[k] = str(v)

import gc
import torch
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file


def load_duplicated_weights(model_path, world_size):
    weights_path = os.path.join(model_path, "weights")
    base_ckpt_path = os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors")
    base_ckpt = load_file(base_ckpt_path)
    sharded = []
    for rank in range(world_size):
        ckpt = {k: v.clone() for k, v in base_ckpt.items()}
        sharded.append(ckpt)
    return sharded


def main():
    t_total = time.time()

    data = torch.load(input_path, weights_only=False)
    compiled_models_dir = data["compiled_models_dir"]
    num_frames = data["num_frames"]

    # ---- Load and run post_quant_conv ----
    pqc_path = f"{compiled_models_dir}/post_quant_conv"
    pqc_config_file = os.path.join(pqc_path, "config.json")
    with open(pqc_config_file) as f:
        pqc_config = json.load(f)
    pqc_world_size = pqc_config["world_size"]

    print(f"  Loading post_quant_conv (world_size={pqc_world_size})...")
    t0 = time.time()
    pqc_nxd = NxDModel.load(
        os.path.join(pqc_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=pqc_world_size,
    )
    pqc_weights = load_duplicated_weights(pqc_path, pqc_world_size)
    pqc_nxd.set_weights(pqc_weights)
    pqc_nxd.to_neuron()
    print(f"  post_quant_conv loaded in {time.time() - t0:.1f}s")

    latents_f32 = data["latents_f32"]
    print(f"  Running post_quant_conv on {latents_f32.shape}...")
    t0 = time.time()
    z = pqc_nxd(latents_f32)
    if isinstance(z, (tuple, list)):
        z = z[0]
    z = z.to(torch.float32)
    print(f"  post_quant_conv done in {time.time() - t0:.1f}s, output: {z.shape}")
    del pqc_nxd, pqc_weights
    gc.collect()

    # ---- Load and run decoder ----
    rolling_path = f"{compiled_models_dir}/decoder_rolling"
    nocache_path = f"{compiled_models_dir}/decoder_nocache"

    if os.path.exists(os.path.join(rolling_path, "nxd_model.pt")):
        decoder_path = rolling_path
        is_rolling = True
    else:
        decoder_path = nocache_path
        is_rolling = False

    decoder_config_file = os.path.join(decoder_path, "config.json")
    with open(decoder_config_file) as f:
        decoder_config = json.load(f)
    decoder_world_size = decoder_config["world_size"]
    decoder_frames = decoder_config.get("decoder_frames", 2)
    is_stateful = decoder_config.get("stateful", False)
    mode_str = "stateful rolling" if is_stateful else ("rolling cache" if is_rolling else "nocache")

    print(f"  Loading decoder [{mode_str}] (world_size={decoder_world_size}, frames={decoder_frames})...")
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
    print(f"  Decoding {num_latent_frames} latent frames in {num_chunks} chunks [{mode_str}]...")
    decode_start = time.time()

    if is_rolling and not is_stateful:
        # Legacy mode: initialize caches on host
        from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes
        latent_h, latent_w = z_bf16.shape[3], z_bf16.shape[4]
        cache_shapes = get_feat_cache_shapes(1, latent_h, latent_w)
        caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]

    for chunk_idx in range(num_chunks):
        start = chunk_idx * decoder_frames
        end = min(start + decoder_frames, num_latent_frames)
        chunk = z_bf16[:, :, start:end, :, :]

        if chunk.shape[2] < decoder_frames:
            pad_frames = decoder_frames - chunk.shape[2]
            padding = chunk[:, :, -1:, :, :].expand(-1, -1, pad_frames, -1, -1)
            chunk = torch.cat([chunk, padding], dim=2)

        if is_stateful:
            # Stateful: only pass x, cache stays on device
            output = decoder_nxd(chunk)
            if isinstance(output, (list, tuple)):
                output = output[0]
        elif is_rolling:
            # Legacy rolling: pass x + caches, get back output + updated caches
            results = decoder_nxd(chunk, *caches)
            if isinstance(results, (tuple, list)):
                output = results[0]
                caches = [r.to(torch.bfloat16) for r in results[1:1 + len(cache_shapes)]]
            else:
                output = results
        else:
            # NoCache
            output = decoder_nxd(chunk)
            if isinstance(output, (list, tuple)):
                output = output[0]

        output = output.to(torch.float32)
        actual_latent = end - start
        video_frames_from_chunk = actual_latent * 4
        output = output[:, :, :video_frames_from_chunk]
        decoded_frames.append(output)

        elapsed = time.time() - decode_start
        print(f"    Chunk {chunk_idx+1}/{num_chunks}: [{start}:{end}] -> {output.shape[2]} frames ({elapsed:.1f}s)")

    video = torch.cat(decoded_frames, dim=2)
    decode_time = time.time() - decode_start

    if video.shape[2] > num_frames:
        video = video[:, :, :num_frames]

    print(f"  Decode done: {decode_time:.1f}s, output: {video.shape} [{mode_str}]")

    # Post-process to numpy
    video = video[0].permute(1, 2, 3, 0).float().cpu().numpy()
    video = ((video + 1.0) / 2.0).clip(0, 1)

    import numpy as np
    torch.save({
        "video": torch.from_numpy(video),
        "decode_time": decode_time,
    }, output_path)

    print(f"  Worker total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
