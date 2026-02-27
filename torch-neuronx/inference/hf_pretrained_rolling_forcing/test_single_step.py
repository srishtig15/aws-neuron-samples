"""
Single-step denoising test to isolate transformer from rolling forcing loop.

Tests the compiled Neuron transformer in isolation:
1. All 21 frames at the SAME timestep (no rolling forcing complexity)
2. Converts flow prediction to x0 directly
3. Decodes with CPU Diffusers VAE

If this produces grids → problem is in the transformer itself
If this produces reasonable content → problem is in the rolling forcing loop
"""
import torch, sys, os, time
import numpy as np

# Neuron runtime environment
os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_rolling_forcing import (
    NeuronTransformerWrapper, NeuronTextEncoderWrapper,
    FlowMatchScheduler, save_video,
)
from neuron_rolling_forcing.neuron_rope import compute_wan_rope_3d

print("=" * 60)
print("SINGLE-STEP TEST: Isolate transformer from rolling forcing")
print("=" * 60)

compiled_dir = "compiled_models"
cache_dir = "/opt/dlami/nvme/rolling_forcing_hf_cache"

# 1. Load compiled models
print("\nLoading compiled models...")
t0 = time.time()
transformer = NeuronTransformerWrapper(compiled_dir)
text_encoder = NeuronTextEncoderWrapper(compiled_dir)
print(f"  Models loaded in {time.time()-t0:.1f}s")

# 2. Encode text prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer",
    cache_dir=cache_dir)
tokens = tokenizer("A cat walking on the grass", max_length=512, padding="max_length",
                   truncation=True, return_tensors="pt")
prompt_embeds = text_encoder(tokens.input_ids.to(torch.int64),
                             tokens.attention_mask.to(torch.int64))
if isinstance(prompt_embeds, (tuple, list)):
    prompt_embeds = prompt_embeds[0]
prompt_embeds = prompt_embeds.float()
print(f"Text embeds: {prompt_embeds.shape}")

# 3. Create input: all 21 frames at SAME timestep
B, C, F, H, W = 1, 16, 21, 60, 104
torch.manual_seed(42)
noise = torch.randn(B, C, F, H, W)

# Use the warped timestep for the noisiest step
scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
t_val = scheduler.timesteps[0].item()  # Highest/noisiest warped timestep
sigma = scheduler.sigmas[0].item()

timestep = torch.ones(B, F) * t_val
print(f"\nTimestep: {t_val:.2f}, sigma: {sigma:.6f}")

# 4. Compute RoPE for 21 frames at positions 0-20
rope_cos, rope_sin = compute_wan_rope_3d(
    num_frames=21, height=30, width=52, head_dim=128)

# 5. Run transformer - single forward pass
print(f"\nRunning transformer (all {F} frames at t={t_val:.1f})...")
t0 = time.time()
flow_pred = transformer(noise, timestep, prompt_embeds, rope_cos, rope_sin)
print(f"  Time: {time.time()-t0:.2f}s")
print(f"  Flow pred shape: {flow_pred.shape}")
print(f"  Flow pred stats: mean={flow_pred.float().mean():.6f}, std={flow_pred.float().std():.6f}")
print(f"  Flow pred range: [{flow_pred.float().min():.4f}, {flow_pred.float().max():.4f}]")

# 6. Convert to x0: x0 = xt - sigma * v
x0_pred = noise - sigma * flow_pred.float()
print(f"\n  x0 pred stats: mean={x0_pred.mean():.6f}, std={x0_pred.std():.6f}")
print(f"  x0 pred range: [{x0_pred.min():.4f}, {x0_pred.max():.4f}]")

# 7. Analyze spatial structure
print("\nSpatial analysis (per-channel):")
frame0 = x0_pred[0, :, 0].numpy()  # [C, H, W]
for ch in range(min(4, C)):
    ch_data = frame0[ch]
    h_diff = np.abs(ch_data[0::2, :] - ch_data[1::2, :]).mean()
    w_diff = np.abs(ch_data[:, 0::2] - ch_data[:, 1::2]).mean()
    std_val = np.std(ch_data)
    # Compute within-patch variance vs overall variance
    patch_vars = []
    for h in range(0, 60, 2):
        for w in range(0, 104, 2):
            patch = ch_data[h:h+2, w:w+2]
            patch_vars.append(np.var(patch))
    avg_patch_var = np.mean(patch_vars)
    overall_var = np.var(ch_data)
    ratio = avg_patch_var / max(overall_var, 1e-10)
    print(f"  Ch{ch}: h_diff={h_diff:.4f}, w_diff={w_diff:.4f}, "
          f"std={std_val:.4f}, patch_var_ratio={ratio:.4f}")

# Check temporal consistency
print("\nTemporal consistency:")
for f in range(min(5, F-1)):
    frame_diff = (x0_pred[0, :, f] - x0_pred[0, :, f+1]).abs().mean().item()
    print(f"  Frame {f}-{f+1} diff: {frame_diff:.6f}")

# 8. Decode with CPU VAE
print("\nDecoding x0 with CPU Diffusers VAE...")
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir=cache_dir,
).eval()

# Un-normalize using VAE config
mean_t = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
std_t = torch.tensor(vae.config.latents_std).reshape(1, 16, 1, 1, 1)
z = x0_pred.float() * std_t + mean_t

with torch.no_grad():
    decoded = vae.decode(z).sample

decoded = decoded.clamp(-1, 1)
video = (decoded * 0.5 + 0.5).clamp(0, 1)
video = video.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

# Save output
output_path = os.path.expanduser("~/Downloads/output_single_step.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
save_video(video[0], output_path, fps=16)
print(f"\nSaved {output_path} ({video.shape[1]} frames)")

# Also save the raw latent for analysis
torch.save({
    'x0_pred': x0_pred,
    'flow_pred': flow_pred.float(),
    'noise_input': noise,
    'timestep': t_val,
    'sigma': sigma,
}, '/tmp/single_step_diagnostic.pt')
print("Saved diagnostic data to /tmp/single_step_diagnostic.pt")
print("Done!")
