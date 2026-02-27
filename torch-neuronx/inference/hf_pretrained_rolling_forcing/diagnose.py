"""
Diagnostic script to isolate video quality issues.

Tests:
1. Save raw transformer latent output → decode with CPU Diffusers VAE
2. Test unpatchify round-trip (patchify → unpatchify should be identity)
3. Compare single forward pass: Neuron compiled vs CPU original model
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unpatchify_roundtrip():
    """Test that patchify (Conv3d) → unpatchify is consistent."""
    print("\n" + "="*60)
    print("TEST 1: Unpatchify Round-Trip")
    print("="*60)

    # Create a known pattern
    B, C, F, H, W = 1, 16, 3, 60, 104
    p_t, p_h, p_w = 1, 2, 2
    post_f, post_h, post_w = F // p_t, H // p_h, W // p_w  # 3, 30, 52

    # Create a simple pattern where we know the expected output
    # Each pixel has a unique value based on its position
    x = torch.zeros(B, C, F, H, W)
    for c in range(min(C, 3)):
        for f in range(F):
            for h in range(H):
                for w in range(W):
                    x[0, c, f, h, w] = c * 10000 + f * 1000 + h * 10 + w * 0.1

    # Simulate patchify: Conv3d with identity-like kernel
    # Actually, let's just test the unpatchify with a known flat vector
    # The head linear outputs [B, F, frame_seqlen, out_dim * pt * ph * pw]
    # where frame_seqlen = post_h * post_w = 1560, and output has 64 values per token

    out_dim = C  # 16
    frame_seqlen = post_h * post_w  # 1560

    # Create test tensor: for each token at (f, h, w), the 64 output values are:
    # (pt, ph, pw, c) → value = c * 10000 + f * 1000 + (h*p_h + ph) * 10 + (w*p_w + pw) * 0.1
    x_head = torch.zeros(B, post_f, frame_seqlen, out_dim * p_t * p_h * p_w)
    for f in range(post_f):
        for patch_idx in range(frame_seqlen):
            h = patch_idx // post_w
            w = patch_idx % post_w
            flat_idx = 0
            for pt in range(p_t):
                for ph in range(p_h):
                    for pw in range(p_w):
                        for c in range(out_dim):
                            pixel_f = f * p_t + pt
                            pixel_h = h * p_h + ph
                            pixel_w = w * p_w + pw
                            x_head[0, f, patch_idx, flat_idx] = c * 10000 + pixel_f * 1000 + pixel_h * 10 + pixel_w * 0.1
                            flat_idx += 1

    # Apply our unpatchify (current code: C last in view)
    x_test = x_head.view(B, post_f, post_h, post_w, p_t, p_h, p_w, out_dim)
    output = x_test.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    output = output.reshape(B, out_dim, post_f * p_t, post_h * p_h, post_w * p_w)

    # Verify specific pixel values
    errors = 0
    for c in range(min(3, out_dim)):
        for f in range(post_f):
            for h_idx in range(min(3, H)):
                for w_idx in range(min(3, W)):
                    expected = c * 10000 + f * 1000 + h_idx * 10 + w_idx * 0.1
                    actual = output[0, c, f, h_idx, w_idx].item()
                    if abs(expected - actual) > 0.01:
                        errors += 1
                        if errors <= 5:
                            print(f"  MISMATCH at (c={c}, f={f}, h={h_idx}, w={w_idx}): "
                                  f"expected={expected:.1f}, got={actual:.1f}")

    if errors == 0:
        print("  PASS: All pixel values match expected positions")
    else:
        print(f"  FAIL: {errors} mismatches found")

    return errors == 0


def test_cpu_forward_pass():
    """
    Run single forward pass on CPU with original model and compare.
    This tests the full transformer (minus NKI attention).
    """
    print("\n" + "="*60)
    print("TEST 2: CPU Forward Pass Comparison")
    print("="*60)

    from neuron_rolling_forcing.compile_transformer import (
        sinusoidal_embedding_1d, load_rolling_forcing_weights
    )

    # Monkey-patch torch.cuda
    import torch.cuda
    _orig_cd = torch.cuda.current_device
    _orig_ia = torch.cuda.is_available
    torch.cuda.current_device = lambda: 0
    torch.cuda.is_available = lambda: False
    try:
        from wan.modules.causal_model import CausalWanModel
    finally:
        torch.cuda.current_device = _orig_cd
        torch.cuda.is_available = _orig_ia

    # Create original model
    print("  Creating original CausalWanModel...")
    original_model = CausalWanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )

    # Load weights
    ckpt_path = "/opt/dlami/nvme/rolling_forcing_hf_cache/rolling_forcing/rolling_forcing_dmd.pt"
    print(f"  Loading weights from {ckpt_path}...")
    ckpt_weights = load_rolling_forcing_weights(ckpt_path)
    missing, unexpected = original_model.load_state_dict(ckpt_weights, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    original_model = original_model.float().eval()

    # Create small test input (3 frames to save memory)
    B, C, F, H, W = 1, 16, 3, 60, 104
    torch.manual_seed(42)
    hidden_states = torch.randn(B, C, F, H, W, dtype=torch.float32) * 0.1
    timestep = torch.tensor([[555.0, 555.0, 555.0]], dtype=torch.float32)
    text_embeds = torch.randn(B, 512, 4096, dtype=torch.float32) * 0.01

    # Run original model on CPU
    print("  Running original model on CPU (this may take a while)...")
    with torch.no_grad():
        # The original model expects different args, let me check what it needs
        # CausalWanModel.forward(self, x, timestep, context, ...)
        try:
            cpu_output = original_model(
                hidden_states, timestep, text_embeds,
            )
            print(f"  CPU output shape: {cpu_output.shape}")
            print(f"  CPU output stats: mean={cpu_output.mean():.6f}, std={cpu_output.std():.6f}")
            print(f"  CPU output range: [{cpu_output.min():.4f}, {cpu_output.max():.4f}]")

            # Check for spatial structure
            # Look at one frame, one channel
            frame0_ch0 = cpu_output[0, 0, 0].numpy()
            print(f"\n  Frame 0, Channel 0 stats:")
            print(f"    Shape: {frame0_ch0.shape}")
            print(f"    Unique values: {len(np.unique(frame0_ch0.flatten()[:1000]))}")

            # Check 2x2 patch consistency
            # If there's a 2x2 grid pattern, adjacent pixels within a patch would differ
            h_diff = np.abs(frame0_ch0[0::2, :] - frame0_ch0[1::2, :]).mean()
            w_diff = np.abs(frame0_ch0[:, 0::2] - frame0_ch0[:, 1::2]).mean()
            overall_std = np.std(frame0_ch0)
            print(f"    Even-odd row diff: {h_diff:.6f}")
            print(f"    Even-odd col diff: {w_diff:.6f}")
            print(f"    Overall std: {overall_std:.6f}")

            # Save for comparison
            torch.save({
                'input_hidden': hidden_states,
                'input_timestep': timestep,
                'input_text': text_embeds,
                'cpu_output': cpu_output,
            }, '/tmp/cpu_reference.pt')
            print("  Saved CPU reference to /tmp/cpu_reference.pt")

        except Exception as e:
            print(f"  ERROR running original model: {e}")
            import traceback
            traceback.print_exc()

            # Try to understand what forward() expects
            import inspect
            sig = inspect.signature(original_model.forward)
            print(f"\n  CausalWanModel.forward signature: {sig}")
            return False

    return True


def test_neuron_single_window():
    """
    Run a single window through compiled Neuron model and analyze output.
    """
    print("\n" + "="*60)
    print("TEST 3: Neuron Single Window Analysis")
    print("="*60)

    from neuron_rolling_forcing.neuron_rope import compute_wan_rope_3d

    # Load compiled model
    from neuronx_distributed.trace import NxDModel
    model_dir = "compiled_models/transformer"
    print(f"  Loading compiled model from {model_dir}...")

    model = NxDModel(model_dir)
    rope_data = torch.load(os.path.join(model_dir, "rope_cache.pt"))
    rope_cos = rope_data['rope_cos']
    rope_sin = rope_data['rope_sin']

    # Create test input (21 frames, padded)
    B, C, F, H, W = 1, 16, 21, 60, 104
    torch.manual_seed(42)

    # Use noise as input (simulating first denoising step)
    hidden_states = torch.randn(B, C, F, H, W) * 0.1
    timestep = torch.ones(B, F) * 555.0  # All frames at same timestep
    text_embeds = torch.randn(B, 512, 4096) * 0.01

    # Convert to bf16
    hidden_states = hidden_states.to(torch.bfloat16)
    timestep = timestep.to(torch.bfloat16)
    text_embeds = text_embeds.to(torch.bfloat16)
    rope_cos = rope_cos.to(torch.bfloat16)
    rope_sin = rope_sin.to(torch.bfloat16)

    print("  Running Neuron model...")
    output = model(hidden_states, timestep, text_embeds, rope_cos, rope_sin)
    if isinstance(output, (tuple, list)):
        output = output[0]

    output_f32 = output.float()
    print(f"  Output shape: {output_f32.shape}")
    print(f"  Output stats: mean={output_f32.mean():.6f}, std={output_f32.std():.6f}")
    print(f"  Output range: [{output_f32.min():.4f}, {output_f32.max():.4f}]")

    # Analyze spatial structure at patch level
    frame0 = output_f32[0, :, 0, :, :]  # [C, H, W]

    # Check if there's a 2x2 grid pattern
    for ch in range(min(3, C)):
        ch_data = frame0[ch].numpy()
        # Compare even vs odd rows and columns
        h_diff = np.abs(ch_data[0::2, :] - ch_data[1::2, :]).mean()
        w_diff = np.abs(ch_data[:, 0::2] - ch_data[:, 1::2]).mean()
        # Compare pixels within same patch
        patch_var = 0
        count = 0
        for h in range(0, 60, 2):
            for w in range(0, 104, 2):
                patch = ch_data[h:h+2, w:w+2]
                patch_var += np.var(patch)
                count += 1
        avg_patch_var = patch_var / count
        overall_var = np.var(ch_data)

        print(f"\n  Channel {ch}:")
        print(f"    Even-odd row diff: {h_diff:.6f}")
        print(f"    Even-odd col diff: {w_diff:.6f}")
        print(f"    Avg within-patch variance: {avg_patch_var:.6f}")
        print(f"    Overall variance: {overall_var:.6f}")
        print(f"    Ratio (patch_var/overall_var): {avg_patch_var/max(overall_var, 1e-10):.4f}")
        # If ratio << 1, patches are internally uniform = grid pattern

    # Check inter-frame consistency
    if output_f32.shape[2] >= 3:
        f01_diff = (output_f32[0, :, 0] - output_f32[0, :, 1]).abs().mean().item()
        f12_diff = (output_f32[0, :, 1] - output_f32[0, :, 2]).abs().mean().item()
        print(f"\n  Frame differences:")
        print(f"    Frame 0-1: {f01_diff:.6f}")
        print(f"    Frame 1-2: {f12_diff:.6f}")

    # Save output for further analysis
    torch.save({
        'neuron_output': output_f32,
        'input_hidden': hidden_states.float(),
        'input_timestep': timestep.float(),
    }, '/tmp/neuron_diagnostic.pt')
    print("\n  Saved Neuron output to /tmp/neuron_diagnostic.pt")

    return True


def test_decode_with_cpu_vae():
    """
    Take the inference output latents and decode with CPU Diffusers VAE.
    If grid persists → transformer issue. If grid disappears → VAE issue.
    """
    print("\n" + "="*60)
    print("TEST 4: Decode Latest Output with CPU Diffusers VAE")
    print("="*60)

    # Check if we have saved latent output
    latent_path = "/tmp/latest_latent_output.pt"
    if not os.path.exists(latent_path):
        print(f"  No saved latent output found at {latent_path}")
        print("  Running inference first to generate latents...")

        # Run inference and save latents
        return False

    latents = torch.load(latent_path, map_location='cpu')
    print(f"  Loaded latents: shape={latents.shape}, dtype={latents.dtype}")

    from diffusers import AutoencoderKLWan
    cache_dir = "/opt/dlami/nvme/rolling_forcing_hf_cache"

    print("  Loading Diffusers VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    ).eval()

    # Un-normalize
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                         0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, 16, 1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(1, 16, 1, 1, 1)

    z = latents.permute(0, 2, 1, 3, 4).float()  # [B, C, F, H, W]
    z = z * std + mean

    print(f"  Decoding with CPU VAE (shape: {z.shape})...")
    with torch.no_grad():
        decoded = vae.decode(z).sample

    decoded = decoded.clamp(-1, 1)
    video = (decoded * 0.5 + 0.5).clamp(0, 1)  # [B, C, F, H, W]
    video = video.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

    print(f"  Decoded video: shape={video.shape}")

    # Save as video
    from run_rolling_forcing import save_video
    save_video(video[0], "output_cpu_vae_test.mp4", fps=16)
    print("  Saved to output_cpu_vae_test.mp4")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "unpatchify", "cpu_forward", "neuron_window", "cpu_vae"])
    args = parser.parse_args()

    if args.test in ["all", "unpatchify"]:
        test_unpatchify_roundtrip()

    if args.test in ["all", "cpu_forward"]:
        test_cpu_forward_pass()

    if args.test in ["neuron_window"]:
        test_neuron_single_window()

    if args.test in ["cpu_vae"]:
        test_decode_with_cpu_vae()
