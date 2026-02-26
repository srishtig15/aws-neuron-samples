"""
Shared utilities for RollingForcing Neuron inference.

Includes:
- NKI Flash Attention kernel wrapper
- local_rms_norm (no cross-rank communication)
- apply_rotary_emb (real-valued cos/sin for Neuron)
- f32Wrapper for LayerNorm precision
- Inference wrapper classes
"""
import os
import math

import torch
import torch.nn as nn
from torch import Tensor

# NKI Flash Attention kernel
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, Q_len, D]
        key: [B, H, KV_len, D]
        value: [B, H, KV_len, D]

    Returns:
        attention output [B, H, Q_len, D]
    """
    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    # Reshape for NKI kernel: (B*H, D, S) for Q/K, (B*H, S, D) for V
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output,
                              kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output,
                        kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    return attn_output.reshape((bs, n_head, q_len, d_head))


def local_rms_norm(x, weight, eps=1e-6):
    """
    Apply RMSNorm locally without cross-rank all-reduce.

    Computes RMSNorm purely locally over the full local_inner_dim (H_local * D).
    The difference from global norm is negligible for QK-norm since each TP shard
    has a statistically similar distribution of activations.

    Args:
        x: [B, S, local_inner_dim] tensor
        weight: [local_inner_dim] parameter (already TP-sharded)
        eps: epsilon for numerical stability
    """
    dtype = x.dtype
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(dtype)


def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    """
    Apply rotary embeddings with pre-computed real-valued cos/sin tensors.

    Matches Wan's interleaved RoPE format:
    - x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    - out[..., 0::2] = x1 * cos - x2 * sin
    - out[..., 1::2] = x1 * sin + x2 * cos

    Args:
        hidden_states: [B, H, S, D] (transposed for NKI attention)
        freqs_cos: [1, S, 1, D] (Wan format)
        freqs_sin: [1, S, 1, D] (Wan format)

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    dtype = hidden_states.dtype

    # Unflatten last dim into pairs and separate
    # hidden_states: [B, H, S, D] -> x1, x2: [B, H, S, D//2]
    x1, x2 = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)

    # freqs_cos/sin: [1, S, 1, D] -> extract even/odd and permute to [1, 1, S, D//2]
    cos = freqs_cos[..., 0::2].permute(0, 2, 1, 3).float()  # [1, 1, S, D//2]
    sin = freqs_sin[..., 1::2].permute(0, 2, 1, 3).float()  # [1, 1, S, D//2]

    # Interleaved output
    out = torch.empty_like(hidden_states, dtype=torch.float32)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out.to(dtype)


class f32Wrapper(nn.Module):
    """Wrap a module to run in float32 precision (for LayerNorm)."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)


# SDPA replacement for text encoder (non-NKI path)
sdpa_original = torch.nn.functional.scaled_dot_product_attention

def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None,
                      is_causal=None, scale=None, enable_gqa=False):
    """SDPA wrapper that uses NKI-compatible attention when no mask is needed."""
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask,
                           dropout_p=dropout_p, is_causal=is_causal,
                           scale=scale, enable_gqa=enable_gqa)
    else:
        return _neuron_sdpa(query, key, value)


def _neuron_sdpa(query, key, value):
    """Neuron-compatible scaled dot-product attention."""
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape
        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])

    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)

    attn_out = torch.bmm(attention_probs, value)
    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out
