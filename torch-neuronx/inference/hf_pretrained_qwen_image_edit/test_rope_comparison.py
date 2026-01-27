#!/usr/bin/env python3
"""
Compare original diffusers QwenEmbedRope vs our NeuronQwenEmbedRope.
This verifies they produce equivalent outputs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# Import original diffusers RoPE
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope, apply_rotary_emb_qwen

# Import our Neuron-compatible RoPE
from neuron_qwen_image_edit.neuron_rope import NeuronQwenEmbedRope, apply_rotary_emb_neuron


def test_rope_equivalence():
    print("=" * 70)
    print("Testing RoPE Equivalence: Original vs Neuron")
    print("=" * 70)

    # Config matching QwenImage transformer
    theta = 10000
    axes_dim = [16, 24, 24]
    scale_rope = True

    # Create both RoPE implementations
    original_rope = QwenEmbedRope(theta=theta, axes_dim=axes_dim, scale_rope=scale_rope)
    neuron_rope = NeuronQwenEmbedRope(theta=theta, axes_dim=axes_dim, scale_rope=scale_rope)

    # Test inputs matching pipeline usage
    # Format 1: compilation format
    img_shapes_compile = [(2, 32, 32)]
    # Format 2: runtime format
    img_shapes_runtime = [[(1, 32, 32), (1, 32, 32)]]

    txt_seq_lens = [100]  # Text sequence length
    max_txt_seq_len = 100
    device = torch.device('cpu')

    print("\n[1] Testing with compilation format: [(2, 32, 32)]")
    print("-" * 50)

    # Original RoPE (returns complex tensors)
    orig_vid_freqs, orig_txt_freqs = original_rope(img_shapes_compile, txt_seq_lens, device)
    print(f"  Original vid_freqs: shape={orig_vid_freqs.shape}, dtype={orig_vid_freqs.dtype}")
    print(f"  Original txt_freqs: shape={orig_txt_freqs.shape}, dtype={orig_txt_freqs.dtype}")

    # Neuron RoPE (returns cos, sin tuples)
    neuron_vid_freqs, neuron_txt_freqs = neuron_rope(img_shapes_compile, max_txt_seq_len=max_txt_seq_len, device=device)
    print(f"  Neuron vid_freqs: cos={neuron_vid_freqs[0].shape}, sin={neuron_vid_freqs[1].shape}")
    print(f"  Neuron txt_freqs: cos={neuron_txt_freqs[0].shape}, sin={neuron_txt_freqs[1].shape}")

    # Convert original complex to cos/sin for comparison
    orig_vid_cos = orig_vid_freqs.real.float()
    orig_vid_sin = orig_vid_freqs.imag.float()
    neuron_vid_cos = neuron_vid_freqs[0].float()
    neuron_vid_sin = neuron_vid_freqs[1].float()

    # Compare
    vid_cos_match = torch.allclose(orig_vid_cos, neuron_vid_cos, atol=1e-5)
    vid_sin_match = torch.allclose(orig_vid_sin, neuron_vid_sin, atol=1e-5)

    print(f"\n  Video cos match: {vid_cos_match}")
    print(f"  Video sin match: {vid_sin_match}")
    if not vid_cos_match:
        cos_diff = (orig_vid_cos - neuron_vid_cos).abs().max()
        print(f"    Max cos diff: {cos_diff}")
    if not vid_sin_match:
        sin_diff = (orig_vid_sin - neuron_vid_sin).abs().max()
        print(f"    Max sin diff: {sin_diff}")

    print("\n[2] Testing with runtime format: [[(1, 32, 32), (1, 32, 32)]]")
    print("-" * 50)

    # Original RoPE with runtime format
    orig_vid_freqs2, orig_txt_freqs2 = original_rope(img_shapes_runtime, txt_seq_lens, device)
    print(f"  Original vid_freqs: shape={orig_vid_freqs2.shape}")

    # Neuron RoPE with runtime format
    neuron_vid_freqs2, neuron_txt_freqs2 = neuron_rope(img_shapes_runtime, max_txt_seq_len=max_txt_seq_len, device=device)
    print(f"  Neuron vid_freqs: cos={neuron_vid_freqs2[0].shape}")

    # Convert and compare
    orig_vid_cos2 = orig_vid_freqs2.real.float()
    orig_vid_sin2 = orig_vid_freqs2.imag.float()
    neuron_vid_cos2 = neuron_vid_freqs2[0].float()
    neuron_vid_sin2 = neuron_vid_freqs2[1].float()

    vid_cos_match2 = torch.allclose(orig_vid_cos2, neuron_vid_cos2, atol=1e-5)
    vid_sin_match2 = torch.allclose(orig_vid_sin2, neuron_vid_sin2, atol=1e-5)

    print(f"\n  Video cos match: {vid_cos_match2}")
    print(f"  Video sin match: {vid_sin_match2}")
    if not vid_cos_match2:
        cos_diff = (orig_vid_cos2 - neuron_vid_cos2).abs().max()
        print(f"    Max cos diff: {cos_diff}")
    if not vid_sin_match2:
        sin_diff = (orig_vid_sin2 - neuron_vid_sin2).abs().max()
        print(f"    Max sin diff: {sin_diff}")

    print("\n[3] Testing cross-format comparison")
    print("-" * 50)
    print("  Comparing: Original[(2,32,32)] vs Original[[(1,32,32),(1,32,32)]]")

    cross_cos_match = torch.allclose(orig_vid_cos, orig_vid_cos2, atol=1e-5)
    cross_sin_match = torch.allclose(orig_vid_sin, orig_vid_sin2, atol=1e-5)

    print(f"  Original formats match (cos): {cross_cos_match}")
    print(f"  Original formats match (sin): {cross_sin_match}")
    if not cross_cos_match or not cross_sin_match:
        print("  [INFO] Original RoPE produces DIFFERENT results for different formats!")
        cos_diff = (orig_vid_cos - orig_vid_cos2).abs()
        sin_diff = (orig_vid_sin - orig_vid_sin2).abs()
        print(f"    Max cos diff: {cos_diff.max()}")
        print(f"    Max sin diff: {sin_diff.max()}")
        print(f"    Positions with cos diff > 0.01: {(cos_diff > 0.01).sum().item()}")

    print("\n[4] Testing apply_rotary_emb equivalence")
    print("-" * 50)

    # Create test input
    batch_size = 1
    seq_len = 2048
    num_heads = 28
    head_dim = 64
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

    # Apply original RoPE (complex mode)
    orig_out = apply_rotary_emb_qwen(x, orig_vid_freqs, use_real=False)

    # Apply neuron RoPE (real mode with cos/sin)
    neuron_out = apply_rotary_emb_neuron(x, neuron_vid_freqs, use_real=True)

    # Compare
    out_match = torch.allclose(orig_out, neuron_out, atol=1e-4)
    cos_sim = F.cosine_similarity(orig_out.flatten().unsqueeze(0), neuron_out.flatten().unsqueeze(0)).item()
    max_diff = (orig_out - neuron_out).abs().max().item()

    print(f"  Output match (atol=1e-4): {out_match}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.6e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = vid_cos_match and vid_sin_match and vid_cos_match2 and vid_sin_match2 and out_match

    if cross_cos_match and cross_sin_match:
        print("[INFO] Original RoPE produces SAME output for both img_shapes formats")
    else:
        print("[WARNING] Original RoPE produces DIFFERENT output for different img_shapes formats!")
        print("         This means compilation format and runtime format are NOT equivalent!")

    if all_pass:
        print("[PASS] Neuron RoPE matches Original RoPE")
    else:
        print("[FAIL] Neuron RoPE does NOT match Original RoPE")
        print("       This could cause transformer output differences!")

    return all_pass


if __name__ == "__main__":
    test_rope_equivalence()
