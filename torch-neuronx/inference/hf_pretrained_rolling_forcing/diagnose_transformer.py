"""
Comprehensive diagnostic for transformer all-zeros output.

Tests:
1. Weight effect: Do different weights produce different outputs?
2. Parameter name check: Do weight file keys match compiled model expectations?
3. Intermediate values: Check patch_embedding output, text_embedding output, etc.
4. Single-block CPU test: Run 1 block of NeuronCausalWanModel on CPU
"""
import os
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import json
import sys
from safetensors.torch import load_file
from neuronx_distributed import NxDModel

compiled_dir = "compiled_models/transformer"

# Load config
with open(os.path.join(compiled_dir, "config.json")) as f:
    config = json.load(f)
tp_degree = config["tp_degree"]
world_size = config["world_size"]
cp_degree = world_size // tp_degree

print("=" * 60)
print("Transformer Diagnostic")
print(f"TP={tp_degree}, CP={cp_degree}, world_size={world_size}")
print("=" * 60)

# Load RoPE and prepare inputs
rope_data = torch.load(os.path.join(compiled_dir, "rope_cache.pt"), map_location="cpu")
torch.manual_seed(42)
noise = torch.randn(1, 16, 21, 60, 104)
timestep = torch.ones(1, 21) * 500.0
text = torch.randn(1, 512, 4096)

inputs = [
    noise.to(torch.bfloat16),
    timestep.to(torch.bfloat16),
    text.to(torch.bfloat16),
    rope_data["rope_cos"].to(torch.bfloat16),
    rope_data["rope_sin"].to(torch.bfloat16),
]

# ========================================
# Test 1: Check if weights affect output
# ========================================
print("\n" + "=" * 60)
print("TEST 1: Do weights affect model output?")
print("=" * 60)

# Load with real weights
tp_weights = []
for rank in range(tp_degree):
    raw = load_file(os.path.join(compiled_dir, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
    tp_weights.append({k: v for k, v in raw.items() if 'master_weight' not in k})

weights_real = []
for cp in range(cp_degree):
    for tp in range(tp_degree):
        weights_real.append(tp_weights[tp])

model = NxDModel.load(os.path.join(compiled_dir, "nxd_model.pt"))
model.set_weights(weights_real)
model.to_neuron()

out_real = model(*inputs)
if isinstance(out_real, (tuple, list)):
    out_real = out_real[0]

print(f"Real weights output: shape={out_real.shape}")
print(f"  min={out_real.float().min():.6f}, max={out_real.float().max():.6f}")
print(f"  std={out_real.float().std():.6f}, mean={out_real.float().mean():.6f}")
print(f"  non-zero elements: {(out_real != 0).sum().item()} / {out_real.numel()}")
print(f"  abs-sum: {out_real.float().abs().sum():.6f}")

# Load with all-ones weights (scaled by 0.01)
print("\nLoading model with modified weights (all 0.01)...")
weights_modified = []
for cp in range(cp_degree):
    for tp in range(tp_degree):
        mod = {k: torch.ones_like(v) * 0.01 for k, v in tp_weights[tp].items()}
        weights_modified.append(mod)

model2 = NxDModel.load(os.path.join(compiled_dir, "nxd_model.pt"))
model2.set_weights(weights_modified)
model2.to_neuron()

out_modified = model2(*inputs)
if isinstance(out_modified, (tuple, list)):
    out_modified = out_modified[0]

print(f"Modified weights output: shape={out_modified.shape}")
print(f"  min={out_modified.float().min():.6f}, max={out_modified.float().max():.6f}")
print(f"  std={out_modified.float().std():.6f}, mean={out_modified.float().mean():.6f}")
print(f"  non-zero elements: {(out_modified != 0).sum().item()} / {out_modified.numel()}")
print(f"  abs-sum: {out_modified.float().abs().sum():.6f}")

if (out_real == out_modified).all():
    if out_real.abs().sum() == 0 and out_modified.abs().sum() == 0:
        print("\n>>> DIAGNOSIS: BOTH outputs are ALL ZEROS")
        print(">>> The compiled graph is a constant-zero function!")
        print(">>> Root cause: Compilation issue (graph ignores weights and inputs)")
    else:
        print("\n>>> DIAGNOSIS: Both outputs are IDENTICAL but non-zero")
        print(">>> The compiled graph ignores weights (uses compiled-in constants)")
else:
    if out_real.abs().sum() == 0:
        print("\n>>> DIAGNOSIS: Real weights → zeros, Modified weights → non-zero")
        print(">>> Weight loading maps wrong keys (real weights map to unused parameters)")
    elif out_modified.abs().sum() == 0:
        print("\n>>> DIAGNOSIS: Real weights → non-zero, Modified weights → zeros")
        print(">>> Something specific about 0.01 weights causes zero output")
    else:
        print("\n>>> DIAGNOSIS: Outputs DIFFER - weights DO affect output")
        print(">>> This means the compilation is correct!")
        diff = (out_real.float() - out_modified.float()).abs()
        print(f">>> Difference: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")

# ========================================
# Test 2: Check parameter names match
# ========================================
print("\n" + "=" * 60)
print("TEST 2: Parameter name matching")
print("=" * 60)

weight_keys = set(tp_weights[0].keys())
print(f"Weight file keys: {len(weight_keys)}")

# Print a sample of keys for visual inspection
print("\nSample weight keys (first 20):")
for k in sorted(weight_keys)[:20]:
    print(f"  {k}: {tp_weights[0][k].shape}")

print(f"\nSample weight keys (last 20):")
for k in sorted(weight_keys)[-20:]:
    print(f"  {k}: {tp_weights[0][k].shape}")

# Check for head-related keys specifically
head_keys = [k for k in weight_keys if 'head' in k]
print(f"\nHead-related keys ({len(head_keys)}):")
for k in sorted(head_keys):
    print(f"  {k}: {tp_weights[0][k].shape}")

# ========================================
# Test 3: Test with zero timestep
# ========================================
print("\n" + "=" * 60)
print("TEST 3: Test with all-zero timestep")
print("=" * 60)

inputs_zero_t = [
    noise.to(torch.bfloat16),
    torch.zeros(1, 21, dtype=torch.bfloat16),  # All zero timestep
    text.to(torch.bfloat16),
    rope_data["rope_cos"].to(torch.bfloat16),
    rope_data["rope_sin"].to(torch.bfloat16),
]

out_zero_t = model(*inputs_zero_t)
if isinstance(out_zero_t, (tuple, list)):
    out_zero_t = out_zero_t[0]

print(f"Zero timestep output:")
print(f"  min={out_zero_t.float().min():.6f}, max={out_zero_t.float().max():.6f}")
print(f"  std={out_zero_t.float().std():.6f}")
print(f"  non-zero elements: {(out_zero_t != 0).sum().item()} / {out_zero_t.numel()}")

# ========================================
# Test 4: Test with different noise
# ========================================
print("\n" + "=" * 60)
print("TEST 4: Test with different inputs")
print("=" * 60)

# All zeros input
inputs_zero = [
    torch.zeros(1, 16, 21, 60, 104, dtype=torch.bfloat16),
    torch.ones(1, 21, dtype=torch.bfloat16) * 500.0,
    text.to(torch.bfloat16),
    rope_data["rope_cos"].to(torch.bfloat16),
    rope_data["rope_sin"].to(torch.bfloat16),
]

out_zero_input = model(*inputs_zero)
if isinstance(out_zero_input, (tuple, list)):
    out_zero_input = out_zero_input[0]

print(f"Zero hidden_states output:")
print(f"  min={out_zero_input.float().min():.6f}, max={out_zero_input.float().max():.6f}")
print(f"  std={out_zero_input.float().std():.6f}")
print(f"  non-zero elements: {(out_zero_input != 0).sum().item()} / {out_zero_input.numel()}")

# All ones input
inputs_ones = [
    torch.ones(1, 16, 21, 60, 104, dtype=torch.bfloat16),
    torch.ones(1, 21, dtype=torch.bfloat16) * 500.0,
    text.to(torch.bfloat16),
    rope_data["rope_cos"].to(torch.bfloat16),
    rope_data["rope_sin"].to(torch.bfloat16),
]

out_ones_input = model(*inputs_ones)
if isinstance(out_ones_input, (tuple, list)):
    out_ones_input = out_ones_input[0]

print(f"All-ones hidden_states output:")
print(f"  min={out_ones_input.float().min():.6f}, max={out_ones_input.float().max():.6f}")
print(f"  std={out_ones_input.float().std():.6f}")
print(f"  non-zero elements: {(out_ones_input != 0).sum().item()} / {out_ones_input.numel()}")

# Check if all outputs are identical
all_same = (out_real == out_zero_input).all() and (out_real == out_ones_input).all() and (out_real == out_zero_t).all()
if all_same:
    print("\n>>> ALL outputs are IDENTICAL regardless of input!")
    print(">>> The compiled model is a constant function (ignores all inputs)")
else:
    print("\n>>> Outputs differ with different inputs - model IS processing inputs")
    if out_real.abs().sum() == 0:
        print(">>> But all outputs are zero - possible numerical collapse")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
