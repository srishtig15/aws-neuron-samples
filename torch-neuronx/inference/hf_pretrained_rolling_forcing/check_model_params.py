"""Check NxDModel parameter names vs weight file keys."""
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
from safetensors.torch import load_file
from neuronx_distributed import NxDModel

compiled_dir = "compiled_models/transformer"

# Load compiled model (without weights)
model = NxDModel.load(os.path.join(compiled_dir, "nxd_model.pt"))

# Get expected parameter names from the compiled model
model_param_names = model.get_parameter_names() if hasattr(model, 'get_parameter_names') else None
print(f"model.get_parameter_names: {model_param_names is not None}")

# Try to get state dict info
if hasattr(model, 'state_dict_info'):
    print(f"model.state_dict_info: available")

# Check the weight file keys
ckpt = load_file(os.path.join(compiled_dir, "weights/tp0_sharded_checkpoint.safetensors"))
weight_keys = sorted(ckpt.keys())

print(f"\nWeight file keys (first 20 of {len(weight_keys)}):")
for k in weight_keys[:20]:
    print(f"  {k}: {ckpt[k].shape}")

# Load weights and set
tp_weights = []
for rank in range(4):
    raw = load_file(os.path.join(compiled_dir, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
    tp_weights.append({k: v for k, v in raw.items() if 'master_weight' not in k})

# Duplicate for CP=2
weights = []
for cp in range(2):
    for tp in range(4):
        weights.append(tp_weights[tp])

print(f"\nNumber of weight dicts: {len(weights)}")
print(f"Keys per dict: {len(weights[0])}")

# Try setting weights and check the result
try:
    model.set_weights(weights)
    print("\nset_weights succeeded")
except Exception as e:
    print(f"\nset_weights FAILED: {e}")

try:
    model.to_neuron()
    print("to_neuron succeeded")
except Exception as e:
    print(f"to_neuron FAILED: {e}")
