"""
Persistent VAE decoder worker: loads PQC + decoder once, processes multiple requests.

Protocol:
  Startup: loads models, prints "READY"
  Request: "decode <input.pt> <output.pt>" -> runs decode, prints "DONE"
  Shutdown: "exit"

Cache reset between inferences: re-calls set_weights() + to_neuron() to zero the
stateful rolling cache buffers (NEFF stays in memory, only device transfer needed).
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled_models_dir", required=True)
    parser.add_argument("--env_config", default=None)
    args = parser.parse_args()

    if args.env_config:
        with open(args.env_config) as f:
            env_config = json.load(f)
        for k, v in env_config.items():
            os.environ[k] = str(v)

    import gc
    import torch
    import torch_neuronx
    from neuronx_distributed import NxDModel
    from safetensors.torch import load_file

    compiled_models_dir = args.compiled_models_dir

    # ---- Load post_quant_conv ----
    pqc_path = os.path.join(compiled_models_dir, "post_quant_conv")
    with open(os.path.join(pqc_path, "config.json")) as f:
        pqc_config = json.load(f)
    pqc_ws = pqc_config["world_size"]

    print(f"[Decoder] Loading post_quant_conv (ws={pqc_ws})...", flush=True)
    t0 = time.time()
    pqc_nxd = NxDModel.load(os.path.join(pqc_path, "nxd_model.pt"),
                             start_rank=0, local_ranks_size=pqc_ws)
    pqc_weights_path = os.path.join(pqc_path, "weights", "tp0_sharded_checkpoint.safetensors")
    pqc_base = load_file(pqc_weights_path)
    pqc_weights = [{k: v.clone() for k, v in pqc_base.items()} for _ in range(pqc_ws)]
    pqc_nxd.set_weights(pqc_weights)
    pqc_nxd.to_neuron()
    print(f"[Decoder] PQC loaded in {time.time()-t0:.1f}s", flush=True)

    # ---- Load decoder ----
    rolling_path = os.path.join(compiled_models_dir, "decoder_rolling")
    with open(os.path.join(rolling_path, "config.json")) as f:
        dec_config = json.load(f)
    dec_ws = dec_config["world_size"]
    dec_frames = dec_config.get("decoder_frames", 2)
    is_stateful = dec_config.get("stateful", False)

    print(f"[Decoder] Loading decoder (ws={dec_ws}, stateful={is_stateful})...", flush=True)
    t0 = time.time()
    dec_nxd = NxDModel.load(os.path.join(rolling_path, "nxd_model.pt"),
                             start_rank=0, local_ranks_size=dec_ws)
    dec_weights_path = os.path.join(rolling_path, "weights", "tp0_sharded_checkpoint.safetensors")
    dec_base = load_file(dec_weights_path)
    dec_weights = [{k: v.clone() for k, v in dec_base.items()} for _ in range(dec_ws)]
    dec_nxd.set_weights(dec_weights)
    dec_nxd.to_neuron()
    dec_load_time = time.time() - t0
    print(f"[Decoder] Decoder loaded in {dec_load_time:.1f}s", flush=True)

    if not is_stateful:
        from neuron_wan2_2_t2v_a14b.compile_decoder_rolling import get_feat_cache_shapes

    print("READY", flush=True)

    # Command loop
    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        if parts[0] == "exit":
            print("[Decoder] Exiting.", flush=True)
            break

        if parts[0] == "decode" and len(parts) == 3:
            input_path, output_path = parts[1], parts[2]
            try:
                data = torch.load(input_path, weights_only=False)
                latents_f32 = data["latents_f32"]
                num_frames = data["num_frames"]

                # Run PQC
                t0 = time.time()
                z = pqc_nxd(latents_f32)
                if isinstance(z, (tuple, list)):
                    z = z[0]
                z = z.to(torch.float32)
                pqc_time = time.time() - t0
                print(f"[Decoder] PQC: {pqc_time:.1f}s", flush=True)

                # Reset decoder cache for fresh inference
                if is_stateful:
                    t_reset = time.time()
                    dec_nxd.replace_weights(dec_weights)
                    reset_time = time.time() - t_reset
                    print(f"[Decoder] Cache reset (replace_weights): {reset_time:.1f}s", flush=True)

                # Rolling decode
                z_bf16 = z.to(torch.bfloat16)
                num_latent = z_bf16.shape[2]
                num_chunks = (num_latent + dec_frames - 1) // dec_frames
                decoded = []

                if not is_stateful:
                    lh, lw = z_bf16.shape[3], z_bf16.shape[4]
                    cache_shapes = get_feat_cache_shapes(1, lh, lw)
                    caches = [torch.zeros(s, dtype=torch.bfloat16) for s in cache_shapes]

                decode_start = time.time()
                for ci in range(num_chunks):
                    ts = ci * dec_frames
                    te = min(ts + dec_frames, num_latent)
                    chunk = z_bf16[:, :, ts:te, :, :]
                    if chunk.shape[2] < dec_frames:
                        pad = chunk[:, :, -1:].expand(-1, -1, dec_frames - chunk.shape[2], -1, -1)
                        chunk = torch.cat([chunk, pad], dim=2)

                    if is_stateful:
                        out = dec_nxd(chunk)
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                    else:
                        results = dec_nxd(chunk, *caches)
                        if isinstance(results, (list, tuple)):
                            out = results[0]
                            caches = [r.to(torch.bfloat16) for r in results[1:1+len(cache_shapes)]]
                        else:
                            out = results

                    out = out.to(torch.float32)
                    actual = te - ts
                    out = out[:, :, :actual * 4]
                    decoded.append(out)

                video = torch.cat(decoded, dim=2)
                decode_time = time.time() - decode_start

                if video.shape[2] > num_frames:
                    video = video[:, :, :num_frames]

                # Post-process
                video = video[0].permute(1, 2, 3, 0).float().cpu().numpy()
                video = ((video + 1.0) / 2.0).clip(0, 1)

                print(f"[Decoder] Decode: {decode_time:.1f}s ({num_chunks} chunks), output: {video.shape}", flush=True)

                torch.save({
                    "video": torch.from_numpy(video),
                    "decode_time": decode_time,
                    "pqc_time": pqc_time,
                }, output_path)

                print("DONE", flush=True)

            except Exception as e:
                print(f"[Decoder] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("ERROR", flush=True)
        else:
            print(f"[Decoder] Unknown: {cmd}", flush=True)


if __name__ == "__main__":
    main()
