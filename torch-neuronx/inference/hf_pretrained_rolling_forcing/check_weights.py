"""Check sharded weight statistics."""
from safetensors.torch import load_file
import os

weights_path = "compiled_models/transformer/weights"
ckpt = load_file(os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors"))
print(f"Total keys: {len(ckpt)}")

for key in sorted(ckpt.keys())[:15]:
    t = ckpt[key]
    print(f"  {key}: shape={t.shape}, dtype={t.dtype}, min={t.min():.4f}, max={t.max():.4f}, std={t.std():.4f}")

n_zero = 0
for key, val in ckpt.items():
    if val.abs().sum() == 0:
        n_zero += 1
        if n_zero <= 5:
            print(f"  ALL ZEROS: {key} {val.shape}")
print(f"Total all-zero tensors: {n_zero}/{len(ckpt)}")
