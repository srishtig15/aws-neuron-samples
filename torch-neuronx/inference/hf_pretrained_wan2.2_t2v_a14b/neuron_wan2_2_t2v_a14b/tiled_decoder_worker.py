"""
Subprocess worker for decoding a single spatial tile on one Neuron Core.

Usage (called programmatically):
    python -m neuron_wan2_2_t2v_a14b.tiled_decoder_worker <input.pt> <output.pt> <env.json>

Input .pt contains:
    - tile_data: [1, 16, T, tile_h, tile_w] bfloat16 tensor
    - decoder_path: str (path to compiled decoder_tile_ws1/)
    - decoder_frames: int
    - num_video_frames: int

Output .pt contains:
    - tile_video: [1, 3, F, tile_h_pix, tile_w_pix] float32 tensor
    - decode_time: float
    - load_time: float
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

import torch
import torch_neuronx
from neuronx_distributed import NxDModel
from safetensors.torch import load_file


def load_duplicated_weights(model_path, world_size):
    weights_path = os.path.join(model_path, "weights")
    base_ckpt = load_file(os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors"))
    return [{k: v.clone() for k, v in base_ckpt.items()} for _ in range(world_size)]


def main():
    t_total = time.time()
    data = torch.load(input_path, weights_only=False)

    tile_data = data["tile_data"]  # [1, 16, T, H, W] bfloat16
    decoder_path = data["decoder_path"]
    decoder_frames = data["decoder_frames"]
    num_video_frames = data["num_video_frames"]
    tile_id = data.get("tile_id", "?")

    with open(os.path.join(decoder_path, "config.json")) as f:
        config = json.load(f)
    world_size = config["world_size"]
    is_stateful = config.get("stateful", False)

    print(f"  [Tile {tile_id}] Loading decoder (ws={world_size}, stateful={is_stateful})...")
    t0 = time.time()
    nxd = NxDModel.load(
        os.path.join(decoder_path, "nxd_model.pt"),
        start_rank=0, local_ranks_size=world_size,
    )
    weights = load_duplicated_weights(decoder_path, world_size)
    nxd.set_weights(weights)
    nxd.to_neuron()
    load_time = time.time() - t0
    print(f"  [Tile {tile_id}] Decoder loaded in {load_time:.1f}s")

    T = tile_data.shape[2]
    num_chunks = (T + decoder_frames - 1) // decoder_frames
    decoded = []
    decode_start = time.time()

    if not is_stateful:
        from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes
        lh, lw = tile_data.shape[3], tile_data.shape[4]
        cache_shapes = get_feat_cache_shapes(1, lh, lw)
        caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]

    for i in range(num_chunks):
        ts = i * decoder_frames
        te = min(ts + decoder_frames, T)
        chunk = tile_data[:, :, ts:te, :, :]
        if chunk.shape[2] < decoder_frames:
            pad = chunk[:, :, -1:].expand(-1, -1, decoder_frames - chunk.shape[2], -1, -1)
            chunk = torch.cat([chunk, pad], dim=2)

        if is_stateful:
            out = nxd(chunk)
            if isinstance(out, (list, tuple)):
                out = out[0]
        else:
            results = nxd(chunk, *caches)
            if isinstance(results, (list, tuple)):
                out = results[0]
                caches = [r.to(torch.bfloat16) for r in results[1:1 + len(cache_shapes)]]
            else:
                out = results

        out = out.to(torch.float32)
        actual = te - ts
        out = out[:, :, :actual * 4]
        decoded.append(out)

    video = torch.cat(decoded, dim=2)
    if video.shape[2] > num_video_frames:
        video = video[:, :, :num_video_frames]

    decode_time = time.time() - decode_start
    print(f"  [Tile {tile_id}] Decode done: {num_chunks} chunks, {decode_time:.1f}s (total {time.time()-t_total:.1f}s)")

    torch.save({
        "tile_video": video,
        "decode_time": decode_time,
        "load_time": load_time,
    }, output_path)


if __name__ == "__main__":
    main()
