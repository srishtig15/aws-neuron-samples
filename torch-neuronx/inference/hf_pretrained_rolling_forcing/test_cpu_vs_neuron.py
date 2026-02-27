"""
Compare CPU manual forward (no causal mask) vs compiled Neuron model output.
This definitively tests whether the Neuron compilation preserves model correctness.
"""
import torch
import sys
import os
import math

os.environ["NEURON_RT_NUM_CORES"] = "8"
os.environ["LOCAL_WORLD_SIZE"] = "8"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp/RollingForcing")

# ========== Load original model on CPU ==========
print("=" * 60)
print("Loading original CausalWanModel on CPU...")
print("=" * 60)

import torch.cuda
_orig_cd = torch.cuda.current_device
_orig_ia = torch.cuda.is_available
torch.cuda.current_device = lambda: 0
torch.cuda.is_available = lambda: False
try:
    from wan.modules.causal_model import CausalWanModel
    from wan.modules.model import rope_apply, sinusoidal_embedding_1d as original_sinusoidal
finally:
    torch.cuda.current_device = _orig_cd
    torch.cuda.is_available = _orig_ia

cpu_model = CausalWanModel(
    model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16,
    dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=12, num_layers=30, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)

ckpt_path = "/opt/dlami/nvme/rolling_forcing_hf_cache/rolling_forcing/rolling_forcing_dmd.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get('generator_ema', ckpt.get('generator', ckpt))
cleaned = {}
for k, v in sd.items():
    key = k.replace("model.", "", 1).replace("_fsdp_wrapped_module.", "")
    cleaned[key] = v
cpu_model.load_state_dict(cleaned, strict=False)
cpu_model = cpu_model.float().eval()
print("  CPU model loaded.")

# ========== Prepare input ==========
B, C, F, H, W = 1, 16, 3, 60, 104
torch.manual_seed(42)
x_input = torch.randn(B, C, F, H, W, dtype=torch.float32)
t_input = torch.ones(B, F, dtype=torch.float32) * 1000.0
text_input = torch.randn(B, 512, 4096, dtype=torch.float32)

p_t, p_h, p_w = 1, 2, 2
post_f, post_h, post_w = F, H//2, W//2
frame_seqlen = post_h * post_w  # 1560
seq_len = post_f * frame_seqlen  # 4680

# ========== CPU manual forward (no causal mask, matching Neuron) ==========
print("\n" + "=" * 60)
print(f"CPU manual forward (no causal mask, {F} frames)...")
print("=" * 60)

with torch.no_grad():
    # 1. Patch embedding
    x = cpu_model.patch_embedding(x_input)
    x = x.flatten(2).transpose(1, 2)  # [B, seq_len, dim]

    # 2. Text embedding
    context = cpu_model.text_embedding(text_input)

    # 3. Time embedding
    t_emb = original_sinusoidal(cpu_model.freq_dim, t_input.flatten()).float()
    e = cpu_model.time_embedding(t_emb)
    e0 = cpu_model.time_projection(e).unflatten(1, (6, cpu_model.dim)).unflatten(0, (B, F))
    e_for_head = e.unflatten(0, (B, F)).unsqueeze(2)

    # RoPE
    grid_sizes = torch.tensor([[post_f, post_h, post_w]])

    # 4. Transformer blocks
    for block_idx, block in enumerate(cpu_model.blocks):
        mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)

        # Self-attention
        x_norm = block.norm1(x)
        x_norm = x_norm.unflatten(1, (F, frame_seqlen))
        x_norm = (x_norm * (1 + mod[1]) + mod[0]).flatten(1, 2)

        # QKV + norm
        q = block.self_attn.norm_q(block.self_attn.q(x_norm))
        k = block.self_attn.norm_k(block.self_attn.k(x_norm))
        v = block.self_attn.v(x_norm)

        q = q.view(B, seq_len, 12, 128)
        k = k.view(B, seq_len, 12, 128)
        v = v.view(B, seq_len, 12, 128)

        # RoPE
        q = rope_apply(q, grid_sizes, cpu_model.freqs).float()
        k = rope_apply(k, grid_sizes, cpu_model.freqs).float()
        v = v.float()

        # Attention (NO causal mask, matching Neuron)
        q_t = q.transpose(1, 2)  # [B, 12, S, 128]
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, -1)
        attn_out = block.self_attn.o(attn_out.to(block.self_attn.o.weight.dtype))

        # Residual + gate
        x_unflatten = x.unflatten(1, (F, frame_seqlen))
        attn_unflatten = attn_out.unflatten(1, (F, frame_seqlen))
        x = (x_unflatten.float() + attn_unflatten * mod[2]).flatten(1, 2).to(x.dtype)

        # Cross-attention
        x_norm = block.norm3(x)
        cq = block.cross_attn.norm_q(block.cross_attn.q(x_norm))
        ck = block.cross_attn.norm_k(block.cross_attn.k(context))
        cv = block.cross_attn.v(context)
        cq = cq.view(B, -1, 12, 128).transpose(1, 2).float()
        ck = ck.view(B, -1, 12, 128).transpose(1, 2).float()
        cv = cv.view(B, -1, 12, 128).transpose(1, 2).float()
        cross_out = torch.nn.functional.scaled_dot_product_attention(cq, ck, cv)
        cross_out = cross_out.transpose(1, 2).reshape(B, seq_len, -1)
        cross_out = block.cross_attn.o(cross_out.to(block.cross_attn.o.weight.dtype))
        x = x + cross_out

        # FFN
        x_norm = block.norm2(x)
        x_norm = x_norm.unflatten(1, (F, frame_seqlen))
        x_norm = (x_norm * (1 + mod[4]) + mod[3]).flatten(1, 2)
        ff_out = block.ffn(x_norm)
        x_unflatten = x.unflatten(1, (F, frame_seqlen))
        ff_unflatten = ff_out.unflatten(1, (F, frame_seqlen))
        x = (x_unflatten.float() + ff_unflatten.float() * mod[5]).flatten(1, 2).to(x.dtype)

        if (block_idx + 1) % 10 == 0 or block_idx == 0:
            print(f"  Block {block_idx}: x mean={x.float().mean():.6f}, std={x.float().std():.6f}")

    # CausalHead
    head_mod = (cpu_model.head.modulation.unsqueeze(1) + e_for_head).chunk(2, dim=2)
    x_head = cpu_model.head.norm(x)
    x_head = x_head.unflatten(1, (F, frame_seqlen))
    x_head = cpu_model.head.head(x_head * (1 + head_mod[1]) + head_mod[0])

    # Unpatchify (using original einsum)
    cpu_output_list = cpu_model.unpatchify(
        x_head.view(B, -1, x_head.shape[-1]),
        grid_sizes
    )
    cpu_output = cpu_output_list[0]  # [C, F, H, W]
    print(f"\n  CPU output: shape={cpu_output.shape}")
    print(f"  CPU output: mean={cpu_output.float().mean():.6f}, std={cpu_output.float().std():.6f}")

# ========== Neuron model ==========
print("\n" + "=" * 60)
print("Loading and running Neuron model...")
print("=" * 60)

from run_rolling_forcing import NeuronTransformerWrapper
from neuron_rolling_forcing.neuron_rope import compute_wan_rope_3d

neuron_wrapper = NeuronTransformerWrapper("compiled_models")
rope_cos, rope_sin = compute_wan_rope_3d(num_frames=F, height=post_h, width=post_w, head_dim=128)

neuron_output = neuron_wrapper(x_input, t_input, text_input, rope_cos, rope_sin)
neuron_out = neuron_output[0].float()  # [C, F, H, W]

print(f"  Neuron output: shape={neuron_out.shape}")
print(f"  Neuron output: mean={neuron_out.mean():.6f}, std={neuron_out.std():.6f}")

# ========== Compare ==========
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

diff = (neuron_out - cpu_output.float()).abs()
print(f"  Max diff:  {diff.max().item():.6f}")
print(f"  Mean diff: {diff.mean().item():.6f}")
print(f"  Relative diff: {(diff / (cpu_output.float().abs() + 1e-8)).mean().item():.6f}")

cos_sim = torch.nn.functional.cosine_similarity(
    neuron_out.flatten().unsqueeze(0),
    cpu_output.float().flatten().unsqueeze(0)).item()
print(f"  Cosine similarity: {cos_sim:.6f}")

# Per-channel comparison
print("\n  Per-channel cosine similarity:")
for ch in range(min(4, C)):
    ch_sim = torch.nn.functional.cosine_similarity(
        neuron_out[ch].flatten().unsqueeze(0),
        cpu_output[ch].float().flatten().unsqueeze(0)).item()
    print(f"    Ch{ch}: {ch_sim:.6f}")

# Save for debugging
torch.save({
    'cpu_output': cpu_output.float(),
    'neuron_output': neuron_out,
    'input': x_input,
    'timestep': t_input,
    'text_input': text_input,
}, '/tmp/cpu_vs_neuron_comparison.pt')
print("\nSaved comparison data to /tmp/cpu_vs_neuron_comparison.pt")
print("Done!")
