"""
Real-valued RoPE computation for Wan CausalWanModel on Neuron.

The original Wan RoPE uses torch.polar (complex-valued), which is not supported
on Neuron. This module converts to real-valued cos/sin representation.

Wan's 3D RoPE uses a SHARED frequency base computed from the full head_dim=128,
then splits the 64 complex frequencies into three groups:
- temporal: 22 complex = 44 real dims (low frequencies, large wavelengths)
- spatial_h: 21 complex = 42 real dims (medium frequencies)
- spatial_w: 21 complex = 42 real dims (high frequencies)

IMPORTANT: All three axes share frequencies from the same base
  base_freqs = 1/theta^(arange(0, 128, 2) / 128)
NOT separate bases per axis. This ensures temporal uses low frequencies
and spatial uses progressively higher frequencies.

For each token at position (f, h, w), the full frequency vector is:
  freqs = concat(freqs_t[f], freqs_h[h], freqs_w[w])

Then: cos_emb = cos(freqs), sin_emb = sin(freqs)
"""
import torch
import numpy as np
from typing import Tuple


def compute_wan_rope_3d(
    num_frames: int,
    height: int,
    width: int,
    head_dim: int = 128,
    theta: float = 10000.0,
    frame_positions: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute real-valued 3D RoPE for Wan CausalWanModel.

    Matches the original rope_params + rope_apply from wan/modules/model.py:
    1. Compute 64 base frequencies from head_dim=128
    2. Split into temporal (22), spatial_h (21), spatial_w (21) complex frequencies
    3. For each token (f, h, w), compute angles and interleave for real-valued RoPE

    Args:
        num_frames: Number of frames (after patching, e.g. 21)
        height: Spatial height after patching (e.g. 30)
        width: Spatial width after patching (e.g. 52)
        head_dim: Head dimension (128)
        theta: RoPE theta
        frame_positions: Optional [num_frames] tensor of frame positions.
                        If None, uses 0..num_frames-1.

    Returns:
        rope_cos: [1, seq_len, 1, head_dim] cosine embeddings
        rope_sin: [1, seq_len, 1, head_dim] sine embeddings
        where seq_len = num_frames * height * width
    """
    d = head_dim  # 128

    # Step 1: Compute shared base frequencies (matching rope_params)
    # base_freqs[i] = 1/theta^(2i/128) for i=0..63
    max_pos = 1024
    base_freqs = 1.0 / torch.pow(
        theta, torch.arange(0, d, 2, dtype=torch.float64) / d)  # [64]

    # Step 2: Split into temporal, spatial_h, spatial_w (matching rope_apply)
    # c = head_dim // 2 = 64
    # split sizes: [c - 2*(c//3), c//3, c//3] = [22, 21, 21]
    c = d // 2  # 64
    n_t = c - 2 * (c // 3)  # 22 complex frequencies for temporal
    n_h = c // 3              # 21 complex frequencies for spatial_h
    n_w = c // 3              # 21 complex frequencies for spatial_w

    freq_t = base_freqs[:n_t]               # [22] — low frequencies
    freq_h = base_freqs[n_t:n_t + n_h]      # [21] — medium frequencies
    freq_w = base_freqs[n_t + n_h:]          # [21] — high frequencies

    # Step 3: Compute per-position angle tables
    # angles[pos] = pos * freq for each 1D position
    positions = torch.arange(max_pos, dtype=torch.float64)
    angles_t = torch.outer(positions, freq_t)  # [1024, 22]
    angles_h = torch.outer(positions, freq_h)  # [1024, 21]
    angles_w = torch.outer(positions, freq_w)  # [1024, 21]

    # Step 4: Interleave angles to match complex->real conversion
    # Original: view_as_complex(x.reshape(..., 2)) applies one angle per pair
    # Real equivalent: [θ0, θ0, θ1, θ1, ...] so each pair shares an angle
    angles_t_interleaved = angles_t.repeat_interleave(2, dim=-1)  # [1024, 44]
    angles_h_interleaved = angles_h.repeat_interleave(2, dim=-1)  # [1024, 42]
    angles_w_interleaved = angles_w.repeat_interleave(2, dim=-1)  # [1024, 42]

    # Frame positions
    if frame_positions is None:
        frame_positions = torch.arange(num_frames, dtype=torch.long)
    else:
        frame_positions = frame_positions.long()

    # Step 5: Build per-token frequency vector
    d_t = n_t * 2  # 44 real dims
    d_h = n_h * 2  # 42 real dims
    d_w = n_w * 2  # 42 real dims
    seq_len = num_frames * height * width
    all_freqs = torch.zeros(seq_len, d, dtype=torch.float64)

    idx = 0
    for f_idx in range(num_frames):
        f_pos = frame_positions[f_idx].item()
        for h in range(height):
            for w in range(width):
                all_freqs[idx, :d_t] = angles_t_interleaved[f_pos]
                all_freqs[idx, d_t:d_t + d_h] = angles_h_interleaved[h]
                all_freqs[idx, d_t + d_h:] = angles_w_interleaved[w]
                idx += 1

    # Convert to cos/sin: [1, seq_len, 1, head_dim]
    rope_cos = torch.cos(all_freqs).unsqueeze(0).unsqueeze(2).float()
    rope_sin = torch.sin(all_freqs).unsqueeze(0).unsqueeze(2).float()

    return rope_cos, rope_sin


def compute_rope_for_rolling_window(
    anchor_frame_positions: torch.Tensor,
    working_frame_positions: torch.Tensor,
    current_frame_positions: torch.Tensor,
    height: int,
    width: int,
    max_frames: int = 21,
    head_dim: int = 128,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE for a rolling forcing window with anchor repositioning.

    The full visible sequence is: [anchor(3) | working(variable) | current(3)] padded to max_frames.
    Each group has its own frame positions for RoPE.

    Args:
        anchor_frame_positions: [3] positions for anchor frames
        working_frame_positions: [N] positions for working cache frames
        current_frame_positions: [3] positions for current noisy frames
        height: Spatial height after patching (30)
        width: Spatial width after patching (52)
        max_frames: Maximum frames to pad to (21)
        head_dim: Head dimension (128)
        theta: RoPE theta

    Returns:
        rope_cos: [1, max_seq_len, 1, head_dim] where max_seq_len = max_frames * height * width
        rope_sin: [1, max_seq_len, 1, head_dim]
    """
    # Concatenate all frame positions
    all_positions = torch.cat([
        anchor_frame_positions,
        working_frame_positions,
        current_frame_positions,
    ])

    num_real_frames = len(all_positions)
    assert num_real_frames <= max_frames, \
        f"Total frames {num_real_frames} exceeds max_frames {max_frames}"

    # Pad frame positions with sequential positions after the last real frame.
    # Using position 0 would collide with real frame 0's temporal RoPE;
    # sequential positions give padded frames distinct, natural RoPE embeddings.
    if num_real_frames < max_frames:
        last_pos = all_positions[-1].item() if len(all_positions) > 0 else 0
        pad_count = max_frames - num_real_frames
        pad_positions = torch.arange(
            last_pos + 1, last_pos + 1 + pad_count, dtype=torch.long)
        all_positions = torch.cat([all_positions, pad_positions])

    return compute_wan_rope_3d(
        num_frames=max_frames,
        height=height,
        width=width,
        head_dim=head_dim,
        theta=theta,
        frame_positions=all_positions,
    )
