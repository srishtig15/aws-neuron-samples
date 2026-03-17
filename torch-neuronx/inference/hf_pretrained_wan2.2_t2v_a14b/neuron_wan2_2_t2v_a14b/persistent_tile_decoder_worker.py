"""
Persistent tile decoder worker: loads ws=1 stateful decoder, decodes tiles.

Protocol:
  Startup: loads model, prints "READY"
  Request: "decode_tile <input.pt> <output.pt>" -> runs decode, prints "DONE"
  Shutdown: "exit"
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--env_config", default=None)
    args = parser.parse_args()

    if args.env_config:
        with open(args.env_config) as f:
            env_config = json.load(f)
        for k, v in env_config.items():
            os.environ[k] = str(v)

    import torch
    import torch_neuronx
    from neuronx_distributed import NxDModel
    from safetensors.torch import load_file

    decoder_path = args.decoder_path
    with open(os.path.join(decoder_path, "config.json")) as f:
        config = json.load(f)
    dec_ws = config["world_size"]
    dec_frames = config.get("decoder_frames", 2)
    is_stateful = config.get("stateful", False)

    print(f"[TileDec] Loading decoder (ws={dec_ws}, stateful={is_stateful})...", flush=True)
    t0 = time.time()

    dec_nxd = NxDModel.load(os.path.join(decoder_path, "nxd_model.pt"),
                             start_rank=0, local_ranks_size=dec_ws)
    weights_path = os.path.join(decoder_path, "weights", "tp0_sharded_checkpoint.safetensors")
    base_weights = load_file(weights_path)
    dec_weights = [{k: v.clone() for k, v in base_weights.items()} for _ in range(dec_ws)]
    dec_nxd.set_weights(dec_weights)
    dec_nxd.to_neuron()

    load_time = time.time() - t0
    print(f"[TileDec] Loaded in {load_time:.1f}s", flush=True)

    print("READY", flush=True)

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        if parts[0] == "exit":
            print("[TileDec] Exiting.", flush=True)
            break

        if parts[0] == "decode_tile" and len(parts) == 3:
            input_path, output_path = parts[1], parts[2]
            try:
                data = torch.load(input_path, weights_only=False)
                z_tile = data["z_tile"]  # [1, C, T, H, W] bf16
                num_latent = z_tile.shape[2]

                # Reset cache for fresh decode
                if is_stateful:
                    t_reset = time.time()
                    dec_nxd.replace_weights(dec_weights)
                    print(f"[TileDec] Cache reset: {time.time()-t_reset:.1f}s", flush=True)

                # Rolling decode
                num_chunks = (num_latent + dec_frames - 1) // dec_frames
                decoded = []

                decode_start = time.time()
                for ci in range(num_chunks):
                    ts = ci * dec_frames
                    te = min(ts + dec_frames, num_latent)
                    chunk = z_tile[:, :, ts:te, :, :]
                    if chunk.shape[2] < dec_frames:
                        pad = chunk[:, :, -1:].expand(-1, -1, dec_frames - chunk.shape[2], -1, -1)
                        chunk = torch.cat([chunk, pad], dim=2)

                    out = dec_nxd(chunk)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    out = out.to(torch.float32)
                    actual = te - ts
                    out = out[:, :, :actual * 4]
                    decoded.append(out)

                tile_video = torch.cat(decoded, dim=2)
                decode_time = time.time() - decode_start

                print(f"[TileDec] Decoded: {decode_time:.1f}s ({num_chunks} chunks), shape={tile_video.shape}", flush=True)

                torch.save({
                    "tile_video": tile_video,
                    "decode_time": decode_time,
                }, output_path)

                print("DONE", flush=True)

            except Exception as e:
                print(f"[TileDec] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("ERROR", flush=True)
        else:
            print(f"[TileDec] Unknown: {cmd}", flush=True)


if __name__ == "__main__":
    main()
