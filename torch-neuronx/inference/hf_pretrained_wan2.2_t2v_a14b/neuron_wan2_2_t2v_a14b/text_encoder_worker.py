"""
Subprocess worker for running text encoding on Neuron.

Receives tokenized input_ids and attention_mask, runs through Neuron text encoder.

Usage (called programmatically):
    python -m neuron_wan2_2_t2v_a14b.text_encoder_worker <input.pt> <output.pt> <env.json>
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


def load_sharded_weights(model_path, tp_degree):
    weights_path = os.path.join(model_path, "weights")
    sharded = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        sharded.append(ckpt)
    return sharded


def prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree):
    sharded = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            ckpt["transformer.global_rank.rank"] = torch.tensor([world_rank], dtype=torch.int32)
            sharded.append(ckpt)
    return sharded


def extract_hidden_state(result):
    """Extract last_hidden_state from NxDModel output (mirrors InferenceTextEncoderWrapperV2)."""
    if isinstance(result, dict):
        return result.get('last_hidden_state', result.get(0))
    elif isinstance(result, (tuple, list)):
        return result[0]
    return result


def main():
    t_total = time.time()

    data = torch.load(input_path, weights_only=False)
    te_path = data["te_path"]
    seqlen = data["seqlen"]
    # Pre-tokenized inputs from main process
    prompt_input_ids = data["prompt_input_ids"]          # [1, seqlen]
    prompt_attention_mask = data["prompt_attention_mask"]  # [1, seqlen]
    neg_input_ids = data["neg_input_ids"]                # [1, seqlen]
    neg_attention_mask = data["neg_attention_mask"]        # [1, seqlen]

    # Load config
    config_file = os.path.join(te_path, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    te_world_size = config["world_size"]

    # Load NxDModel
    print(f"  Loading text encoder NxDModel (TP={tp_degree}, world_size={te_world_size})...")
    t0 = time.time()
    te_nxd = NxDModel.load(os.path.join(te_path, "nxd_model.pt"))
    tp_checkpoints = load_sharded_weights(te_path, tp_degree)
    if te_world_size > tp_degree:
        cp_degree = te_world_size // tp_degree
        checkpoints = prepare_cp_checkpoints(tp_checkpoints, tp_degree, cp_degree)
    else:
        checkpoints = tp_checkpoints
    te_nxd.set_weights(checkpoints)
    te_nxd.to_neuron()
    load_time = time.time() - t0
    print(f"  Text encoder loaded in {load_time:.1f}s")

    # Run encoding
    t_enc = time.time()
    raw_prompt = te_nxd(prompt_input_ids, prompt_attention_mask)
    prompt_embeds = extract_hidden_state(raw_prompt)

    raw_neg = te_nxd(neg_input_ids, neg_attention_mask)
    negative_prompt_embeds = extract_hidden_state(raw_neg)

    enc_time = time.time() - t_enc
    print(f"  prompt_embeds: {prompt_embeds.shape}, negative: {negative_prompt_embeds.shape}")
    print(f"  Encoding done in {enc_time:.1f}s")

    torch.save({
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "load_time": load_time,
        "enc_time": enc_time,
    }, output_path)

    print(f"  Worker total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
