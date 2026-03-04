"""
Wrapper classes for Wan2.2 T2V-A14B inference.

Merged from TI2V neuron_commons.py and neuron_commons_v2.py.
Includes only wrappers needed for T2V (no I2V encoder/quant_conv wrappers).
"""
import time
import math

import torch
import torch.nn as nn
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
from transformers.models.umt5 import UMT5EncoderModel
from types import SimpleNamespace

_flash_fwd_call = nki_jit()(attention_isa_kernel)


# ============================================================
# Attention wrappers (for text encoder compilation)
# ============================================================

def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
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


sdpa_original = torch.nn.functional.scaled_dot_product_attention


def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None, scale=None, enable_gqa=False):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    else:
        return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)


# ============================================================
# Text Encoder Wrapper (V2 API)
# ============================================================

class InferenceTextEncoderWrapperV2(nn.Module):
    """Wrapper for text encoder with NxDModel V2 API."""

    def __init__(self, dtype, t: UMT5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        self.t = t

    def forward(self, text_input_ids, attention_mask=None):
        if hasattr(self.t, 'encode'):
            result = self.t.encode(
                text_input_ids=text_input_ids,
                attention_mask=attention_mask
            )
        else:
            result = self.t(text_input_ids, attention_mask)

        if isinstance(result, dict):
            last_hidden_state = result.get('last_hidden_state', result.get(0))
        elif isinstance(result, (tuple, list)):
            last_hidden_state = result[0]
        else:
            last_hidden_state = result

        return SimpleNamespace(last_hidden_state=last_hidden_state.to(self.dtype))


# ============================================================
# Decoder Wrapper (V3 NoCache)
# ============================================================

class DecoderWrapperV3NoCache(nn.Module):
    """
    Wrapper for V3 NoCache compiled decoder.

    The compiled model takes only x as input (no feat_cache arguments).
    feat_cache is internalized as registered buffers (zeros, loaded once to device).

    This eliminates ~960MB per-call data transfer. Only x is transferred per call.
    """

    def __init__(self, original_decoder, decoder_frames=2):
        super().__init__()
        self.original_decoder = original_decoder
        self.decoder_frames = decoder_frames
        self.nxd_model = None

    def forward(self, x, **kwargs):
        if 'feat_cache' not in kwargs:
            return self.original_decoder(x)

        _t0 = time.time()

        original_frame_count = x.shape[2]

        # Pad temporal dimension to decoder_frames if needed
        if x.shape[2] < self.decoder_frames:
            pad_frames = self.decoder_frames - x.shape[2]
            x = torch.cat([x] + [x[:, :, -1:]] * pad_frames, dim=2)

        x_bf16 = x.to(torch.bfloat16)

        _t1 = time.time()

        # NoCache: only pass x as input
        output = self.nxd_model(x_bf16)

        _t2 = time.time()

        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output.to(torch.float32)

        # Trim padded frames
        output_frames = original_frame_count * 4
        if output.shape[2] > output_frames:
            output = output[:, :, :output_frames]

        _t3 = time.time()

        print(f"[nocache] prep={_t1-_t0:.4f}s nxd_model={_t2-_t1:.4f}s postproc={_t3-_t2:.4f}s total={_t3-_t0:.4f}s frames={original_frame_count}")

        return output

    def clear_cache(self):
        pass


# ============================================================
# Post Quant Conv Wrapper (V2 API)
# ============================================================

class PostQuantConvWrapperV2(nn.Module):
    """Wrapper for V2/V3 compiled post_quant_conv using NxDModel."""

    def __init__(self, original_conv):
        super().__init__()
        self.original_conv = original_conv
        self.nxd_model = None

    def forward(self, x, **kwargs):
        output = self.nxd_model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def clear_cache(self):
        pass
