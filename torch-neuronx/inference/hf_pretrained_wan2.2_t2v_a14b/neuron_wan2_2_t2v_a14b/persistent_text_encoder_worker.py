"""
Persistent text encoder worker: loads once, processes multiple requests.

Protocol:
  Startup: loads model, prints "READY"
  Request: "encode <input.pt> <output.pt>" -> runs encoding, prints "DONE"
  Shutdown: "exit"
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--te_path", required=True)
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

    te_path = args.te_path
    with open(os.path.join(te_path, "config.json")) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    te_ws = config["world_size"]
    cp_degree = te_ws // tp_degree

    print(f"[TextEnc] Loading (TP={tp_degree}, ws={te_ws})...", flush=True)
    t0 = time.time()

    # Load weights
    weights_path = os.path.join(te_path, "weights")
    tp_ckpts = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        tp_ckpts.append(load_file(ckpt_path))

    if cp_degree > 1:
        checkpoints = []
        for cp_rank in range(cp_degree):
            for tp_rank in range(tp_degree):
                world_rank = cp_rank * tp_degree + tp_rank
                ckpt = {k: v.clone() for k, v in tp_ckpts[tp_rank].items()}
                ckpt["text_encoder.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
                checkpoints.append(ckpt)
    else:
        checkpoints = tp_ckpts

    te_nxd = NxDModel.load(os.path.join(te_path, "nxd_model.pt"),
                            start_rank=0, local_ranks_size=te_ws)
    te_nxd.set_weights(checkpoints)
    te_nxd.to_neuron()
    print(f"[TextEnc] Loaded in {time.time()-t0:.1f}s", flush=True)

    def extract_hidden_state(result):
        if isinstance(result, dict):
            for key in ["last_hidden_state", "hidden_states"]:
                if key in result:
                    val = result[key]
                    return val[-1] if isinstance(val, (list, tuple)) else val
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    print("READY", flush=True)

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        if parts[0] == "exit":
            print("[TextEnc] Exiting.", flush=True)
            break

        if parts[0] == "encode" and len(parts) == 3:
            input_path, output_path = parts[1], parts[2]
            try:
                data = torch.load(input_path, weights_only=False)
                t0 = time.time()

                prompt_out = te_nxd(data["prompt_input_ids"], data["prompt_attention_mask"])
                prompt_embeds = extract_hidden_state(prompt_out)

                neg_out = te_nxd(data["neg_input_ids"], data["neg_attention_mask"])
                neg_embeds = extract_hidden_state(neg_out)

                enc_time = time.time() - t0
                print(f"[TextEnc] Encoded in {enc_time:.2f}s, shape={prompt_embeds.shape}", flush=True)

                torch.save({
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": neg_embeds,
                    "encode_time": enc_time,
                }, output_path)

                print("DONE", flush=True)

            except Exception as e:
                print(f"[TextEnc] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("ERROR", flush=True)
        else:
            print(f"[TextEnc] Unknown: {cmd}", flush=True)


if __name__ == "__main__":
    main()
