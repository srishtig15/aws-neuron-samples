"""
Real-valued RoPE computation for Wan CausalWanModel on Neuron.

The original Wan RoPE uses torch.polar (complex-valued), which is not supported
on Neuron. This module converts to real-valued cos/sin representation.

Wan's 3D RoPE splits head_dim=128 into:
- temporal: d - 4*(d//6) = 128 - 84 = 44
- spatial_h: 2*(d//6) = 42
- spatial_w: 2*(d//6) = 42

For each token at position (f, h, w), the full frequency vector is:
  freqs = concat(freqs_t[f], freqs_h[h], freqs_w[w])

Then: cos_emb = cos(freqs), sin_emb = sin(freqs)
"""
import torch
import numpy as np
from typing import Tuple


def _compute_freqs_1d(max_pos: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Compute 1D frequency table.

    Args:
        max_pos: Maximum position index
        dim: Dimension of frequency vector (half of the embedding dim for this axis)
        theta: Base for exponential frequency scaling

    Returns:
        freqs: [max_pos, dim] frequency angles
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    positions = torch.arange(max_pos, dtype=torch.float64)
    # [max_pos, dim//2] but we want [max_pos, dim] for interleaved
    angles = torch.outer(positions, freqs)  # [max_pos, dim//2]
    # Interleave: for each position, produce [a0, a0, a1, a1, ...] matching Wan's format
    angles_interleaved = angles.repeat_interleave(2, dim=-1)  # [max_pos, dim]
    return angles_interleaved


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
    d = head_dim
    # Wan's dimension split: temporal, spatial_h, spatial_w
    d_t = d - 4 * (d // 6)  # 128 - 84 = 44
    d_h = 2 * (d // 6)       # 42
    d_w = 2 * (d // 6)       # 42
    assert d_t + d_h + d_w == d, f"Dimension split error: {d_t}+{d_h}+{d_w} != {d}"

    # Compute 1D frequency tables
    max_pos = 1024  # Wan uses 1024 as max position
    freqs_t = _compute_freqs_1d(max_pos, d_t, theta)  # [1024, d_t]
    freqs_h = _compute_freqs_1d(max_pos, d_h, theta)  # [1024, d_h]
    freqs_w = _compute_freqs_1d(max_pos, d_w, theta)  # [1024, d_w]

    # Frame positions
    if frame_positions is None:
        frame_positions = torch.arange(num_frames, dtype=torch.long)
    else:
        frame_positions = frame_positions.long()

    # Build per-token frequency vector: for token at (f, h, w)
    # freq = concat(freqs_t[frame_positions[f]], freqs_h[h], freqs_w[w])
    seq_len = num_frames * height * width
    all_freqs = torch.zeros(seq_len, d, dtype=torch.float64)

    idx = 0
    for f_idx in range(num_frames):
        f_pos = frame_positions[f_idx].item()
        for h in range(height):
            for w in range(width):
                all_freqs[idx, :d_t] = freqs_t[f_pos]
                all_freqs[idx, d_t:d_t+d_h] = freqs_h[h]
                all_freqs[idx, d_t+d_h:] = freqs_w[w]
                idx += 1

    # Convert to cos/sin in Wan's expected format: [1, seq_len, 1, head_dim]
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

    # Pad frame positions if needed (padding frames get position 0, won't affect output
    # since they are masked/ignored in the final extraction)
    if num_real_frames < max_frames:
        pad_positions = torch.zeros(max_frames - num_real_frames, dtype=torch.long)
        all_positions = torch.cat([all_positions, pad_positions])

    return compute_wan_rope_3d(
        num_frames=max_frames,
        height=height,
        width=width,
        head_dim=head_dim,
        theta=theta,
        frame_positions=all_positions,
    )
