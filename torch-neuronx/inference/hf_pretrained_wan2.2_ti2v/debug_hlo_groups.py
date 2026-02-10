"""
Detailed HLO analysis: identify which all-reduce operations have
the incorrect [[0,1,2,3]] group config vs the correct [[0,1,2,3],[4,5,6,7]].
"""
import os
import sys

os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch


def analyze_hlo(nxd_model_path, label=""):
    print(f"\n{'='*60}")
    print(f"HLO Analysis: {label}")
    print(f"{'='*60}")

    from neuronx_distributed import NxDModel
    nxd_model = NxDModel.load(nxd_model_path)

    for key in nxd_model.get_available_keys():
        hlo = nxd_model.get_hlo(key)

        # Collect all collective ops with details
        correct_ops = []  # [[0,1,2,3],[4,5,6,7]]
        incorrect_ops = []  # [[0,1,2,3]] only

        for comp in hlo.computations:
            for inst in comp.instructions:
                if inst.opcode == 'all-reduce':
                    groups = [list(g.replica_ids) for g in inst.replica_groups]
                    groups_str = str(groups)
                    if groups == [[0, 1, 2, 3], [4, 5, 6, 7]]:
                        correct_ops.append(inst.name)
                    elif groups == [[0, 1, 2, 3]]:
                        incorrect_ops.append(inst.name)
                    else:
                        print(f"  UNEXPECTED group config: {groups_str} in {inst.name}")

        print(f"\n  Total all-reduce ops: {len(correct_ops) + len(incorrect_ops)}")
        print(f"  Correct [[0,1,2,3],[4,5,6,7]]: {len(correct_ops)}")
        print(f"  INCORRECT [[0,1,2,3]]: {len(incorrect_ops)}")

        if incorrect_ops:
            print(f"\n  Incorrect operation names (first 20):")
            for name in incorrect_ops[:20]:
                print(f"    {name}")
            if len(incorrect_ops) > 20:
                print(f"    ... and {len(incorrect_ops) - 20} more")

        # Also check if incorrect ops share common patterns
        if incorrect_ops:
            print(f"\n  Pattern analysis of incorrect op names:")
            patterns = {}
            for name in incorrect_ops:
                # Extract base pattern (remove numbers)
                parts = name.split('.')
                if len(parts) > 1:
                    base = '.'.join(parts[:-1])
                else:
                    base = name
                if base not in patterns:
                    patterns[base] = 0
                patterns[base] += 1
            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                print(f"    {pattern}: {count}")


def main():
    wan_path = "/opt/dlami/nvme/compiled_models_v3_cp/transformer_v3_cp/nxd_model.pt"
    if os.path.exists(wan_path):
        analyze_hlo(wan_path, "Wan2.2 V3 CP")


if __name__ == "__main__":
    main()
