"""
Step-by-step diagnostic to identify exact failure point.
Adds explicit flush prints between each NxDModel operation.

Since the assertion error causes an immediate abort (not a Python exception),
we use sys.stdout.flush() to ensure prints are visible before the crash.
"""
import os
import sys

os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import json
import torch
from safetensors.torch import load_file


def main():
    compiled_models_dir = "/opt/dlami/nvme/compiled_models_v3_cp"
    v3_cp_path = f"{compiled_models_dir}/transformer_v3_cp"

    # Load config
    with open(os.path.join(v3_cp_path, "config.json"), "r") as f:
        config = json.load(f)
    tp_degree = config["tp_degree"]
    cp_degree = config["cp_degree"]
    world_size = config["world_size"]
    print(f"Config: TP={tp_degree}, CP={cp_degree}, world_size={world_size}")
    sys.stdout.flush()

    # Load checkpoints
    weights_path = os.path.join(v3_cp_path, "weights")
    tp_checkpoints = []
    for rank in range(tp_degree):
        ckpt_path = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
        raw_ckpt = load_file(ckpt_path)
        # Filter master_weight
        ckpt = {k: v for k, v in raw_ckpt.items() if 'master_weight' not in k}
        tp_checkpoints.append(ckpt)
    print(f"Loaded {len(tp_checkpoints)} TP checkpoints, {len(tp_checkpoints[0])} keys each")
    sys.stdout.flush()

    # Prepare CP checkpoints
    sharded_checkpoints = []
    for cp_rank in range(cp_degree):
        for tp_rank in range(tp_degree):
            world_rank = cp_rank * tp_degree + tp_rank
            ckpt = {k: v.clone() for k, v in tp_checkpoints[tp_rank].items()}
            global_rank_key = "transformer.global_rank.rank"
            if global_rank_key in ckpt:
                ckpt[global_rank_key] = torch.tensor([world_rank], dtype=torch.int32)
            sharded_checkpoints.append(ckpt)
    print(f"Prepared {len(sharded_checkpoints)} CP checkpoints")
    sys.stdout.flush()

    # Step 1: NxDModel.load()
    print("\n>>> STEP 1: NxDModel.load()...")
    sys.stdout.flush()
    from neuronx_distributed import NxDModel
    nxd_model_path = os.path.join(v3_cp_path, "nxd_model.pt")
    nxd_model = NxDModel.load(nxd_model_path)
    print(f"    NxDModel.load() SUCCEEDED. world_size={nxd_model.world_size}, start_rank={nxd_model.start_rank}, local_ranks_size={nxd_model.local_ranks_size}")
    sys.stdout.flush()

    # Step 2: set_weights()
    print("\n>>> STEP 2: nxd_model.set_weights()...")
    sys.stdout.flush()
    nxd_model.set_weights(sharded_checkpoints)
    print("    set_weights() SUCCEEDED")
    sys.stdout.flush()

    # Step 3: to_neuron()
    print("\n>>> STEP 3: nxd_model.to_neuron()...")
    sys.stdout.flush()
    nxd_model.to_neuron()
    print("    to_neuron() SUCCEEDED")
    sys.stdout.flush()

    print("\n>>> ALL STEPS PASSED! Transformer loaded successfully.")


if __name__ == "__main__":
    main()
