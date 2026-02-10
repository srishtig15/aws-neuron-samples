"""
Minimal diagnostic script to isolate the replica groups assertion error.

This script:
1. Checks checkpoint contents (master_weight tensors, global_rank, etc.)
2. Tries loading ONLY the transformer NxDModel (no other imports)
3. Compares with qwen checkpoint structure

Usage:
    python debug_minimal_load.py --compiled_models_dir /opt/dlami/nvme/compiled_models_v3_cp
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import argparse
import json
import sys
import torch
from safetensors.torch import load_file


def check_checkpoint_contents(weights_path, tp_degree, label=""):
    """Analyze checkpoint contents for issues."""
    print(f"\n{'='*60}")
    print(f"Checkpoint Analysis: {label}")
    print(f"{'='*60}")

    for rank in range(tp_degree):
        shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        if not os.path.exists(shard_file):
            print(f"  WARNING: {shard_file} not found!")
            continue

        ckpt = load_file(shard_file)
        total_keys = len(ckpt)
        master_weight_keys = [k for k in ckpt.keys() if 'master_weight' in k]
        global_rank_keys = [k for k in ckpt.keys() if 'global_rank' in k]
        normal_keys = [k for k in ckpt.keys() if 'master_weight' not in k and 'global_rank' not in k]

        total_size = sum(v.numel() * v.element_size() for v in ckpt.values())
        master_weight_size = sum(ckpt[k].numel() * ckpt[k].element_size() for k in master_weight_keys)
        normal_size = sum(ckpt[k].numel() * ckpt[k].element_size() for k in normal_keys)

        print(f"\n  tp{rank}: {total_keys} total keys, {total_size/1e9:.3f} GB")
        print(f"    Normal params:    {len(normal_keys)} keys, {normal_size/1e9:.3f} GB")
        print(f"    master_weight:    {len(master_weight_keys)} keys, {master_weight_size/1e9:.3f} GB")
        print(f"    global_rank:      {len(global_rank_keys)} keys")

        if master_weight_keys:
            print(f"    >>> ISSUE: master_weight tensors found! These should be removed.")
            print(f"    First 5 master_weight keys:")
            for k in master_weight_keys[:5]:
                print(f"      {k}: shape={ckpt[k].shape}, dtype={ckpt[k].dtype}")

        if global_rank_keys:
            for k in global_rank_keys:
                print(f"    global_rank key: {k} = {ckpt[k]}")

        # Check dtypes
        dtypes = {}
        for k, v in ckpt.items():
            dt = str(v.dtype)
            if dt not in dtypes:
                dtypes[dt] = 0
            dtypes[dt] += 1
        print(f"    Dtypes: {dtypes}")


def try_minimal_load(compiled_models_dir):
    """Try loading ONLY the transformer NxDModel with minimal setup."""
    print(f"\n{'='*60}")
    print(f"Minimal NxDModel Load Test")
    print(f"{'='*60}")

    v3_cp_path = f"{compiled_models_dir}/transformer_v3_cp"

    # Load config
    config_path = os.path.join(v3_cp_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    print(f"Config: TP={tp_degree}, CP={cp_degree}, world_size={world_size}")

    # Load and prepare checkpoints
    weights_path = os.path.join(v3_cp_path, "weights")

    # Load TP checkpoints
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_path)
        tp_checkpoints.append(ckpt)

    # Check for master_weight tensors
    master_weight_count = sum(1 for k in tp_checkpoints[0].keys() if 'master_weight' in k)
    print(f"\nmaster_weight tensors in checkpoint: {master_weight_count}")

    # Test 1: Try loading WITH master_weight tensors (original behavior)
    # Test 2: Try loading WITHOUT master_weight tensors (cleaned, like qwen)

    for test_name, clean_master_weights in [("WITH master_weights", False), ("WITHOUT master_weights (cleaned)", True)]:
        print(f"\n--- Test: {test_name} ---")

        # Prepare CP checkpoints
        sharded_checkpoints = []
        for cp_rank in range(cp_degree):
            for tp_rank in range(tp_degree):
                world_rank = cp_rank * tp_degree + tp_rank

                if clean_master_weights:
                    ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()
                            if 'master_weight' not in k}
                else:
                    ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}

                # Set unique global_rank
                global_rank_key = "transformer.global_rank.rank"
                if global_rank_key in ckpt:
                    ckpt[global_rank_key] = torch.tensor([world_rank], dtype=torch.int32)

                sharded_checkpoints.append(ckpt)

        print(f"  Prepared {len(sharded_checkpoints)} checkpoints")
        print(f"  Keys per checkpoint: {len(sharded_checkpoints[0])}")

        # Try loading NxDModel
        try:
            from neuronx_distributed import NxDModel

            nxd_model_path = os.path.join(v3_cp_path, "nxd_model.pt")
            print(f"  Loading NxDModel from {nxd_model_path}...")
            nxd_model = NxDModel.load(nxd_model_path)

            print(f"  Setting weights...")
            nxd_model.set_weights(sharded_checkpoints)

            print(f"  Moving to Neuron...")
            nxd_model.to_neuron()

            print(f"  SUCCESS! NxDModel loaded with {test_name}")

            # Quick test inference
            rope_cache_path = os.path.join(v3_cp_path, "rope_cache.pt")
            rope_cache = torch.load(rope_cache_path)
            rotary_emb_cos = rope_cache["rotary_emb_cos"].to(torch.bfloat16)
            rotary_emb_sin = rope_cache["rotary_emb_sin"].to(torch.bfloat16)

            # Create dummy inputs for a quick test
            seq_len = config["seq_len"]
            cp_seq_len = seq_len  # full sequence before CP splitting
            sample_hidden = torch.randn(1, cp_seq_len, 1536, dtype=torch.bfloat16)
            sample_timestep = torch.tensor([500.0], dtype=torch.float32)
            sample_encoder_hidden = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

            print(f"  Running test inference...")
            output = nxd_model(
                sample_hidden,
                sample_timestep,
                sample_encoder_hidden,
                rotary_emb_cos,
                rotary_emb_sin,
            )
            print(f"  Inference SUCCESS! Output type: {type(output)}")
            return True

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled_models_dir", type=str,
                        default="/opt/dlami/nvme/compiled_models_v3_cp")
    parser.add_argument("--qwen_compiled_dir", type=str,
                        default="/opt/dlami/nvme/compiled_models/transformer_v3_cp",
                        help="Qwen transformer_v3_cp dir for comparison")
    args = parser.parse_args()

    # Step 1: Analyze wan2.2 checkpoints
    wan_weights = os.path.join(args.compiled_models_dir, "transformer_v3_cp", "weights")
    check_checkpoint_contents(wan_weights, tp_degree=4, label="Wan2.2 V3 CP")

    # Step 2: Analyze qwen checkpoints if available
    qwen_weights = os.path.join(args.qwen_compiled_dir, "weights")
    if os.path.exists(qwen_weights):
        check_checkpoint_contents(qwen_weights, tp_degree=4, label="Qwen V3 CP")
    else:
        print(f"\nQwen checkpoint dir not found at {qwen_weights}, skipping comparison")

    # Step 3: Try minimal load
    try_minimal_load(args.compiled_models_dir)


if __name__ == "__main__":
    main()
