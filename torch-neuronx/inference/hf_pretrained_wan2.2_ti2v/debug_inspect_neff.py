"""
Inspect the compiled NEFF's HLO to find all collective operations
and their replica group configurations.

This tells us exactly what replica groups the NEFF expects at runtime.
"""
import os
import sys

os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import json
import torch


def inspect_neff(nxd_model_path, label=""):
    print(f"\n{'='*60}")
    print(f"Inspecting NEFF: {label}")
    print(f"{'='*60}")
    sys.stdout.flush()

    from neuronx_distributed import NxDModel
    nxd_model = NxDModel.load(nxd_model_path)
    print(f"  world_size={nxd_model.world_size}, start_rank={nxd_model.start_rank}")

    # Get available keys (model names)
    keys = nxd_model.get_available_keys()
    print(f"  Available keys: {keys}")

    for key in keys:
        print(f"\n  --- Key: {key} ---")

        # Get metaneff
        try:
            metaneff = nxd_model.get_metaneff(key)
            print(f"  MetaNeff input tensors: {len(metaneff.input_tensors)}")
            print(f"  MetaNeff output tensors: {len(metaneff.output_tensors)}")

            # Check for state tensors
            state_count = sum(1 for t in metaneff.input_tensors
                            if t.type == 2)  # INPUT_STATE type
            weight_count = sum(1 for t in metaneff.input_tensors
                             if t.type == 1)  # INPUT_WEIGHT type
            input_count = sum(1 for t in metaneff.input_tensors
                            if t.type == 0)  # INPUT type
            print(f"  Input types: {input_count} inputs, {weight_count} weights, {state_count} states")
        except Exception as e:
            print(f"  MetaNeff error: {e}")

        # Get HLO and analyze collective ops
        try:
            hlo = nxd_model.get_hlo(key)
            print(f"\n  HLO Module: {hlo.name}")
            print(f"  HLO computations: {len(hlo.computations)}")

            # Search all instructions for collective operations
            collective_ops = []
            for comp in hlo.computations:
                for inst in comp.instructions:
                    opcode = inst.opcode
                    if opcode in ['all-reduce', 'all-gather', 'reduce-scatter',
                                  'collective-permute', 'all-to-all',
                                  'collective-broadcast']:
                        replica_groups = []
                        for group in inst.replica_groups:
                            replica_groups.append(list(group.replica_ids))
                        collective_ops.append({
                            'name': inst.name,
                            'opcode': opcode,
                            'replica_groups': replica_groups,
                        })

            print(f"\n  Collective operations found: {len(collective_ops)}")

            # Group by opcode
            by_opcode = {}
            for op in collective_ops:
                opcode = op['opcode']
                if opcode not in by_opcode:
                    by_opcode[opcode] = []
                by_opcode[opcode].append(op)

            for opcode, ops in sorted(by_opcode.items()):
                print(f"\n  {opcode}: {len(ops)} operations")
                # Show unique replica group configurations
                unique_groups = set()
                for op in ops:
                    groups_str = str(op['replica_groups'])
                    unique_groups.add(groups_str)
                print(f"    Unique replica group configs: {len(unique_groups)}")
                for g in sorted(unique_groups):
                    print(f"      {g}")

            # Find max group_id referenced
            max_group_count = 0
            for op in collective_ops:
                num_groups = len(op['replica_groups'])
                if num_groups > max_group_count:
                    max_group_count = num_groups
            print(f"\n  Max replica groups in any operation: {max_group_count}")

        except Exception as e:
            print(f"  HLO error: {e}")
            import traceback
            traceback.print_exc()

    sys.stdout.flush()


def main():
    # Inspect wan2.2 v3_cp transformer
    wan_path = "/opt/dlami/nvme/compiled_models_v3_cp/transformer_v3_cp/nxd_model.pt"
    if os.path.exists(wan_path):
        inspect_neff(wan_path, "Wan2.2 V3 CP Transformer")

    # Inspect wan2.2 v3_flash transformer if available
    flash_path = "/opt/dlami/nvme/compiled_models_v3_flash/transformer_v3_flash/nxd_model.pt"
    if os.path.exists(flash_path):
        inspect_neff(flash_path, "Wan2.2 V3 Flash Transformer")

    # Check for any qwen compiled models
    qwen_paths = [
        "/opt/dlami/nvme/qwen_compiled/transformer_v3_cp/nxd_model.pt",
        "/home/ubuntu/aws-neuron-samples/torch-neuronx/inference/hf_pretrained_qwen_image_edit/compiled_models/transformer_v3_cp/nxd_model.pt",
    ]
    for qp in qwen_paths:
        if os.path.exists(qp):
            inspect_neff(qp, f"Qwen V3 CP Transformer ({qp})")


if __name__ == "__main__":
    main()
