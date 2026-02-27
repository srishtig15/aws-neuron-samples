"""
5-step CPU denoising test WITHOUT rolling forcing.

This isolates whether the DMD model produces good output when run correctly,
vs whether the rolling forcing loop is the problem.

Tests:
1. Load DMD CausalWanModel on CPU
2. Run simple 5-step denoising loop (no rolling forcing, no KV cache)
3. All 21 frames at the SAME timestep per step (like a standard diffusion pipeline)
4. Decode with CPU Diffusers VAE
5. Save output video

If this produces good video -> problem is in rolling forcing loop
If this produces bad video -> problem is in model or model loading
"""
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/RollingForcing")

print("=" * 60)
print("5-STEP CPU DENOISING TEST (No Rolling Forcing)")
print("=" * 60)

# ========== 1. Load CausalWanModel on CPU ==========
print("\n[1] Loading CausalWanModel on CPU...")
t0 = time.time()

import torch.cuda
_orig_cd = torch.cuda.current_device
_orig_ia = torch.cuda.is_available
torch.cuda.current_device = lambda: 0
torch.cuda.is_available = lambda: False
try:
    from wan.modules.causal_model import CausalWanModel
    from wan.modules.model import sinusoidal_embedding_1d as original_sinusoidal
    from wan.modules.model import rope_apply
except Exception as e:
    print(f"  Import error: {e}")
    print("  Make sure /tmp/RollingForcing is cloned and accessible")
    sys.exit(1)
finally:
    torch.cuda.current_device = _orig_cd
    torch.cuda.is_available = _orig_ia

model = CausalWanModel(
    model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16,
    dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=12, num_layers=30, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)

ckpt_path = "/opt/dlami/nvme/rolling_forcing_hf_cache/rolling_forcing/rolling_forcing_dmd.pt"
print(f"  Loading weights from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get('generator_ema', ckpt.get('generator', ckpt))
cleaned = {}
for k, v in sd.items():
    key = k.replace("model.", "", 1).replace("_fsdp_wrapped_module.", "")
    cleaned[key] = v
missing, unexpected = model.load_state_dict(cleaned, strict=False)
print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    for k in missing[:5]:
        print(f"    Missing: {k}")

model = model.float().eval()
print(f"  Model loaded in {time.time()-t0:.1f}s")

# ========== 2. Setup scheduler ==========
print("\n[2] Setting up scheduler...")

class FlowMatchScheduler:
    def __init__(self, shift=5.0, num_train_timesteps=1000):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

scheduler = FlowMatchScheduler(shift=5.0)

# DMD denoising steps: [1000, 800, 600, 400, 200] (raw indices)
raw_steps = [1000, 800, 600, 400, 200]
warped_timesteps = []
warped_sigmas = []

full_timesteps = torch.cat([scheduler.timesteps, torch.tensor([0.0])])
for s in raw_steps:
    idx = 1000 - s
    t_val = full_timesteps[idx].item()
    # Find closest sigma
    sigma_idx = torch.argmin((scheduler.timesteps - t_val).abs())
    sigma_val = scheduler.sigmas[sigma_idx].item()
    warped_timesteps.append(t_val)
    warped_sigmas.append(sigma_val)

print(f"  Warped timesteps: {warped_timesteps}")
print(f"  Warped sigmas:    {warped_sigmas}")

# ========== 3. Encode text on CPU ==========
print("\n[3] Encoding text prompt...")
cache_dir = "/opt/dlami/nvme/rolling_forcing_hf_cache"

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer",
    cache_dir=cache_dir)

# Use the model's text_embedding to match dimensions
# But we need raw T5 embeddings first
# For simplicity, use random text embeddings (matched stats from real encoding)
# OR use the full UMT5 encoder
try:
    from transformers import UMT5EncoderModel
    print("  Loading UMT5-XXL text encoder on CPU...")
    t0 = time.time()
    text_encoder = UMT5EncoderModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="text_encoder",
        cache_dir=cache_dir, torch_dtype=torch.float32).eval()

    prompt = "A cat walking on the grass"
    tokens = tokenizer(prompt, max_length=512, padding="max_length",
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_output = text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask)
        prompt_embeds = text_output.last_hidden_state.float()
    print(f"  Text encoded in {time.time()-t0:.1f}s: {prompt_embeds.shape}")
    print(f"  Text stats: mean={prompt_embeds.mean():.4f}, std={prompt_embeds.std():.4f}")
    del text_encoder  # Free memory
except Exception as e:
    print(f"  Text encoder loading failed: {e}")
    print("  Using random text embeddings as fallback...")
    torch.manual_seed(123)
    prompt_embeds = torch.randn(1, 512, 4096) * 0.05

# ========== 4. Run 5-step denoising ==========
print("\n[4] Running 5-step denoising (all frames at same timestep per step)...")

B, C, F, H, W = 1, 16, 3, 60, 104  # Use 3 frames (1 block) for speed on CPU
torch.manual_seed(42)
noise = torch.randn(B, C, F, H, W)

# Start from pure noise
x_t = noise.clone()

# Model forward on CPU - manual implementation to avoid KV cache / attention mask issues
p_t, p_h, p_w = 1, 2, 2
post_f, post_h, post_w = F, H // 2, W // 2  # 3, 30, 52
frame_seqlen = post_h * post_w  # 1560
seq_len = post_f * frame_seqlen  # 4680 for 3 frames

# Pre-compute grid_sizes for unpatchify
grid_sizes = torch.tensor([[post_f, post_h, post_w]])

def model_forward_no_mask(model, x_input, timestep_val, text_embeds):
    """Run CausalWanModel forward WITHOUT causal mask on CPU.

    This matches the Neuron model behavior (no KV cache, no causal mask).
    Uses SDPA attention instead of flex_attention to avoid mask issues.
    """
    B_in = x_input.shape[0]
    F_in = x_input.shape[2]

    with torch.no_grad():
        # 1. Patch embedding
        x = model.patch_embedding(x_input)
        x = x.flatten(2).transpose(1, 2)  # [B, seq_len, dim]

        # 2. Text embedding
        context = model.text_embedding(text_embeds)

        # 3. Time embedding (same timestep for all frames)
        t_tensor = torch.ones(B_in, F_in) * timestep_val
        t_emb = original_sinusoidal(model.freq_dim, t_tensor.flatten()).float()
        e = model.time_embedding(t_emb)
        e0 = model.time_projection(e).unflatten(1, (6, model.dim)).unflatten(0, (B_in, F_in))
        e_for_head = e.unflatten(0, (B_in, F_in)).unsqueeze(2)

        # RoPE
        grid = torch.tensor([[post_f, post_h, post_w]])

        # 4. Transformer blocks
        for block_idx, block in enumerate(model.blocks):
            mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)

            # Self-attention
            x_norm = block.norm1(x)
            x_norm = x_norm.unflatten(1, (F_in, frame_seqlen))
            x_norm = (x_norm * (1 + mod[1]) + mod[0]).flatten(1, 2)

            q = block.self_attn.norm_q(block.self_attn.q(x_norm))
            k = block.self_attn.norm_k(block.self_attn.k(x_norm))
            v = block.self_attn.v(x_norm)

            q = q.view(B_in, seq_len, 12, 128)
            k = k.view(B_in, seq_len, 12, 128)
            v = v.view(B_in, seq_len, 12, 128)

            q = rope_apply(q, grid, model.freqs).float()
            k = rope_apply(k, grid, model.freqs).float()
            v = v.float()

            # Standard attention (no causal mask)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
            attn_out = attn_out.transpose(1, 2).reshape(B_in, seq_len, -1)
            attn_out = block.self_attn.o(attn_out.to(block.self_attn.o.weight.dtype))

            x_unflatten = x.unflatten(1, (F_in, frame_seqlen))
            attn_unflatten = attn_out.unflatten(1, (F_in, frame_seqlen))
            x = (x_unflatten.float() + attn_unflatten * mod[2]).flatten(1, 2).to(x.dtype)

            # Cross-attention
            x_norm = block.norm3(x)
            cq = block.cross_attn.norm_q(block.cross_attn.q(x_norm))
            ck = block.cross_attn.norm_k(block.cross_attn.k(context))
            cv = block.cross_attn.v(context)
            cq = cq.view(B_in, -1, 12, 128).transpose(1, 2).float()
            ck = ck.view(B_in, -1, 12, 128).transpose(1, 2).float()
            cv = cv.view(B_in, -1, 12, 128).transpose(1, 2).float()
            cross_out = torch.nn.functional.scaled_dot_product_attention(cq, ck, cv)
            cross_out = cross_out.transpose(1, 2).reshape(B_in, seq_len, -1)
            cross_out = block.cross_attn.o(cross_out.to(block.cross_attn.o.weight.dtype))
            x = x + cross_out

            # FFN
            x_norm = block.norm2(x)
            x_norm = x_norm.unflatten(1, (F_in, frame_seqlen))
            x_norm = (x_norm * (1 + mod[4]) + mod[3]).flatten(1, 2)
            ff_out = block.ffn(x_norm)
            x_unflatten = x.unflatten(1, (F_in, frame_seqlen))
            ff_unflatten = ff_out.unflatten(1, (F_in, frame_seqlen))
            x = (x_unflatten.float() + ff_unflatten.float() * mod[5]).flatten(1, 2).to(x.dtype)

            if block_idx == 0 or (block_idx + 1) % 10 == 0:
                print(f"    Block {block_idx}: mean={x.float().mean():.4f}, std={x.float().std():.4f}")

        # Head
        head_mod = (model.head.modulation.unsqueeze(1) + e_for_head).chunk(2, dim=2)
        x_head = model.head.norm(x)
        x_head = x_head.unflatten(1, (F_in, frame_seqlen))
        x_head = model.head.head(x_head * (1 + head_mod[1]) + head_mod[0])

        # Unpatchify
        output_list = model.unpatchify(x_head.view(B_in, -1, x_head.shape[-1]), grid)
        return output_list[0].unsqueeze(0)  # [1, C, F, H, W]


for step_idx in range(5):
    t_val = warped_timesteps[step_idx]
    sigma = warped_sigmas[step_idx]

    print(f"\n  Step {step_idx}: t={t_val:.1f}, sigma={sigma:.4f}")
    t0 = time.time()

    # Run model
    flow_pred = model_forward_no_mask(model, x_t, t_val, prompt_embeds)
    print(f"    Forward pass: {time.time()-t0:.1f}s")
    print(f"    Flow pred: mean={flow_pred.float().mean():.6f}, std={flow_pred.float().std():.6f}")

    # Convert to x0 using double precision (matching original)
    x0_pred = (x_t.double() - sigma * flow_pred.double()).float()
    print(f"    x0 pred: mean={x0_pred.mean():.6f}, std={x0_pred.std():.6f}")

    # Analyze within-patch bias at this step
    frame0_ch0 = x0_pred[0, 0, 0].numpy()
    patch_deviations = np.zeros(4)
    count = 0
    for h in range(0, H, 2):
        for w in range(0, W, 2):
            patch = frame0_ch0[h:h+2, w:w+2]
            pmean = patch.mean()
            patch_deviations[0] += patch[0, 0] - pmean
            patch_deviations[1] += patch[0, 1] - pmean
            patch_deviations[2] += patch[1, 0] - pmean
            patch_deviations[3] += patch[1, 1] - pmean
            count += 1
    patch_deviations /= count
    print(f"    Within-patch bias: d(0,0)={patch_deviations[0]:.4f}, "
          f"d(0,1)={patch_deviations[1]:.4f}, "
          f"d(1,0)={patch_deviations[2]:.4f}, "
          f"d(1,1)={patch_deviations[3]:.4f}")

    if step_idx < 4:
        # Re-noise to next sigma with fresh noise
        next_sigma = warped_sigmas[step_idx + 1]
        fresh_noise = torch.randn_like(x0_pred)
        x_t = ((1 - next_sigma) * x0_pred + next_sigma * fresh_noise).float()
        print(f"    Re-noised to sigma={next_sigma:.4f}: mean={x_t.mean():.6f}, std={x_t.std():.6f}")
    else:
        final_x0 = x0_pred
        print(f"    FINAL x0: mean={final_x0.mean():.6f}, std={final_x0.std():.6f}")

# ========== 5. Decode with VAE ==========
print("\n[5] Decoding with CPU Diffusers VAE...")
t0 = time.time()

from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
    torch_dtype=torch.float32, cache_dir=cache_dir).eval()

# Un-normalize
mean_t = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
std_t = torch.tensor(vae.config.latents_std).reshape(1, 16, 1, 1, 1)
z = final_x0.float() * std_t + mean_t

print(f"  Latent z: mean={z.mean():.4f}, std={z.std():.4f}, shape={z.shape}")

with torch.no_grad():
    decoded = vae.decode(z).sample

decoded = decoded.clamp(-1, 1)
video = (decoded * 0.5 + 0.5).clamp(0, 1)
video = video.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
print(f"  Decoded in {time.time()-t0:.1f}s: {video.shape}")

# Save
output_path = os.path.expanduser("~/Downloads/output_5step_cpu_no_rolling.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

import imageio
frames = (video[0].cpu().numpy() * 255).astype(np.uint8)
frames = np.transpose(frames, (0, 2, 3, 1))  # [F, H, W, C]
imageio.mimwrite(output_path, frames, fps=16)
print(f"\nSaved {len(frames)} frames to {output_path}")

# Also save diagnostic data
torch.save({
    'final_x0': final_x0,
    'noise': noise,
    'prompt_embeds': prompt_embeds,
    'warped_timesteps': warped_timesteps,
    'warped_sigmas': warped_sigmas,
}, '/tmp/5step_cpu_diagnostic.pt')
print("Saved diagnostic data to /tmp/5step_cpu_diagnostic.pt")
print("\nDone!")
